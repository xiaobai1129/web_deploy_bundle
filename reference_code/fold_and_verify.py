"""Stage 4 / Stage 6 Export & Folding Verification:
Loads a trained Client-L/S/P checkpoint (v1, v2, or v3), folds reparameterized branches
into deploy mode, verifies max absolute error < 1e-5 (FP32), verifies CPU 2s segment-prior
baking equivalence (< 5e-5 max / < 1e-6 mean FP32), verifies both deployed and baked FP16 vs FP32 PSNR >= 45 dB,
and exports deployable state_dict + metadata.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from research_core.client_archs.models import (
    ClientSRNet,
    V1_PRIOR_DIM,
    V2_PRIOR_DIM,
    V3_PRIOR_DIM,
    count_params_and_macs,
)
from scripts.eval_protocol.metrics import compute_psnr


def fold_and_verify(ckpt_path: Path, out_dir: Path) -> dict:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    tier = state.get("tier", "S")
    scale = float(state.get("scale", 2.0))
    use_prior = bool(state.get("use_prior", True))
    arch_version = str(state.get("arch_version", "v1"))
    default_pdim = V1_PRIOR_DIM if arch_version == "v1" else (V2_PRIOR_DIM if arch_version == "v2" else V3_PRIOR_DIM)
    prior_dim = int(state.get("prior_dim", default_pdim))

    train_model = ClientSRNet(
        tier=tier,
        scale=scale,
        use_prior=use_prior,
        use_rep=True,
        deploy=False,
        arch_version=arch_version,
    )
    weights = state.get("ema", state["model"])
    train_model.load_state_dict(weights)
    train_model.eval()

    deploy_model = ClientSRNet(
        tier=tier,
        scale=scale,
        use_prior=use_prior,
        use_rep=True,
        deploy=False,
        arch_version=arch_version,
    )
    deploy_model.load_state_dict(weights)
    deploy_model.eval()
    deploy_model.switch_to_deploy()

    h_in = 540 if abs(scale - 2.0) < 1e-4 else 720
    w_in = 960 if abs(scale - 2.0) < 1e-4 else 1280
    x = torch.rand(1, 1, h_in, w_in)
    if use_prior:
        if prior_dim <= 2:
            prior = torch.tensor([[0.55, 0.0]], dtype=torch.float32)
        elif prior_dim <= 5:
            prior = torch.tensor([[0.55, 0.0, 1.0, 0.42, 1.0]], dtype=torch.float32)
        else:
            prior = torch.tensor([[0.55, 0.0, 1.0, 0.42, 0.25, 0.0, 1.0]], dtype=torch.float32)
    else:
        prior = None

    prior_bake_err = 0.0
    baked_fp16_psnr = 0.0
    with torch.no_grad():
        y_train = train_model(x, prior)
        y_deploy_fp32 = deploy_model(x, prior)
        max_abs_err = float((y_train - y_deploy_fp32).abs().max().item())

        if arch_version in ("v2", "v3") and (use_prior or arch_version == "v3"):
            baked_model = deploy_model.bake_segment_prior(prior[0] if prior is not None else None)
            y_baked_fp32 = baked_model(x, None)
            prior_bake_diff = (y_deploy_fp32 - y_baked_fp32).abs()
            prior_bake_err = float(prior_bake_diff.max().item())
            prior_bake_mean_err = float(prior_bake_diff.mean().item())
            assert prior_bake_mean_err < 1e-6 and prior_bake_err < 5e-5, (
                f"{arch_version} CPU segment-prior baking error too high: "
                f"max={prior_bake_err}, mean={prior_bake_mean_err}"
            )

            baked_fp16 = copy.deepcopy(baked_model).half()
            y_baked_fp16 = baked_fp16(x.half(), None).float()
            baked_fp16_psnr = compute_psnr(
                (y_baked_fp16[0, 0].numpy() * 255.0).round().clip(0, 255),
                (y_baked_fp32[0, 0].numpy() * 255.0).round().clip(0, 255),
            )

        deploy_fp16 = ClientSRNet(
            tier=tier,
            scale=scale,
            use_prior=use_prior,
            use_rep=True,
            deploy=True,
            arch_version=arch_version,
        ).half()
        deploy_fp16.load_state_dict(deploy_model.state_dict())
        deploy_fp16.eval()
        y_deploy_fp16 = deploy_fp16(x.half(), prior.half() if prior is not None else None).float()
        fp16_psnr = compute_psnr(
            (y_deploy_fp16[0, 0].numpy() * 255.0).round().clip(0, 255),
            (y_deploy_fp32[0, 0].numpy() * 255.0).round().clip(0, 255),
        )
        if baked_fp16_psnr == 0.0:
            baked_fp16_psnr = fp16_psnr

    assert max_abs_err < 1e-5, f"Folding error too high: {max_abs_err}"
    assert fp16_psnr >= 45.0, f"FP16 vs FP32 PSNR too low: {fp16_psnr:.2f} dB"
    assert baked_fp16_psnr >= 45.0, f"Baked FP16 vs FP32 PSNR too low: {baked_fp16_psnr:.2f} dB"
    stats = count_params_and_macs(deploy_model, h_in, w_in)

    out_dir.mkdir(parents=True, exist_ok=True)
    deploy_path = out_dir / f"{ckpt_path.stem}_deploy.pt"
    torch.save({
        "state_dict": deploy_model.state_dict(),
        "tier": tier,
        "scale": scale,
        "use_prior": use_prior,
        "arch_version": arch_version,
        "prior_dim": prior_dim,
        "stats": stats,
        "fold_max_abs_err": max_abs_err,
        "prior_bake_max_abs_err": prior_bake_err,
        "fp16_vs_fp32_psnr": fp16_psnr,
        "baked_fp16_vs_fp32_psnr": baked_fp16_psnr,
    }, deploy_path)

    report = {
        "source_ckpt": str(ckpt_path),
        "deploy_ckpt": str(deploy_path),
        "tier": tier,
        "scale": scale,
        "arch_version": arch_version,
        "prior_dim": prior_dim,
        "stats": stats,
        "fold_max_abs_err": max_abs_err,
        "prior_bake_max_abs_err": prior_bake_err,
        "fp16_vs_fp32_psnr": round(fp16_psnr, 2),
        "baked_fp16_vs_fp32_psnr": round(baked_fp16_psnr, 2),
    }
    print(f"[Export] {json.dumps(report)}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("/root/workspace/video-chaofen/outputs/exported"))
    args = parser.parse_args()
    fold_and_verify(args.ckpt, args.out_dir)


if __name__ == "__main__":
    main()
