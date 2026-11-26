"""Stage 6: Pack folded Client-L/S/P FP16 weights into a compact binary blob (<420 KB for V3, <500 KB hard limit)
for mobile OpenGL ES 3.1/3.2 / Vulkan Compute Shader loading, plus symmetric per-channel NPU INT8 export.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch

from research_core.client_archs.models import ClientSRNet, V1_PRIOR_DIM, V2_PRIOR_DIM, V3_PRIOR_DIM
from scripts.eval_protocol.metrics import compute_psnr


def pack_deploy_weights(deploy_ckpt: Path, out_bin: Path, export_int8: bool = True) -> dict:
    payload = torch.load(deploy_ckpt, map_location="cpu", weights_only=False)
    sd = payload["state_dict"]
    arch_v = str(payload.get("arch_version", "v1")).lower()

    tensors = []
    manifest = []
    offset = 0
    for name, tensor in sd.items():
        arr = tensor.detach().cpu().to(torch.float16).numpy().copy()
        raw = arr.tobytes()
        manifest.append({
            "name": name,
            "shape": list(arr.shape),
            "dtype": "float16",
            "offset": offset,
            "nbytes": len(raw),
        })
        tensors.append(raw)
        offset += len(raw)

    meta_bytes = json.dumps({
        "tier": payload["tier"],
        "scale": payload["scale"],
        "use_prior": payload["use_prior"],
        "arch_version": arch_v,
        "prior_dim": payload.get("prior_dim", 2),
        "tensors": manifest,
    }, separators=(",", ":")).encode("utf-8")

    out_bin.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_bin.with_suffix(".bin.tmp")
    with open(tmp, "wb") as f:
        f.write(b"CSR1")
        f.write(struct.pack("<I", len(meta_bytes)))
        f.write(meta_bytes)
        for chunk in tensors:
            f.write(chunk)
    tmp.replace(out_bin)

    size_kb = out_bin.stat().st_size / 1024.0
    limit_kb = 420.0 if arch_v == "v3" else 500.0
    assert size_kb < limit_kb, f"SDK weight blob exceeds {limit_kb:.0f} KB limit: {size_kb:.2f} KB"

    res = {"out_bin": str(out_bin), "size_kb": round(size_kb, 2), "num_tensors": len(manifest)}
    if export_int8:
        int8_res = pack_npu_int8_weights(deploy_ckpt, out_bin.with_suffix(".int8.bin"))
        res.update({
            "out_int8_bin": int8_res["out_int8_bin"],
            "int8_size_kb": int8_res["int8_size_kb"],
            "int8_vs_fp32_psnr": int8_res["int8_vs_fp32_psnr"],
        })
    return res


def pack_npu_int8_weights(deploy_ckpt: Path, out_int8_bin: Path) -> dict:
    """Export symmetric per-channel INT8 weights + float16 scales for mobile NPU (Idea #320.2)."""
    payload = torch.load(deploy_ckpt, map_location="cpu", weights_only=False)
    sd = payload["state_dict"]
    tier = payload["tier"]
    scale = float(payload["scale"])
    use_prior = bool(payload["use_prior"])
    arch_v = str(payload.get("arch_version", "v1")).lower()

    deq_sd = {}
    tensors = []
    manifest = []
    offset = 0
    for name, tensor in sd.items():
        t_f = tensor.detach().cpu().float()
        if t_f.ndim >= 2 and ("weight" in name) and ("prelu" not in name):
            # Symmetric per-output-channel (and per-basis for 5D basis_weight) INT8 quantization
            if t_f.ndim == 5:
                reduce_dims = (2, 3, 4)
                scale_shape = (t_f.shape[0], t_f.shape[1])
            else:
                reduce_dims = tuple(range(1, t_f.ndim))
                scale_shape = (t_f.shape[0],)
            max_abs = t_f.abs().amax(dim=reduce_dims, keepdim=True).clamp(min=1e-7)
            best_scale = (max_abs / 127.0).clamp(min=1e-7).to(torch.float16)
            best_mse = None
            for clip_r in (1.0, 0.995, 0.99, 0.985, 0.98, 0.97, 0.96, 0.94, 0.92):
                cand_s = (max_abs * clip_r / 127.0).clamp(min=1e-7).to(torch.float16)
                cand_sf = cand_s.float().clamp(min=1e-7)
                cand_q = torch.round(t_f / cand_sf).clamp(-127, 127).to(torch.int8)
                mse = (t_f - cand_q.float() * cand_sf).square().mean(dim=reduce_dims, keepdim=True)
                if best_mse is None:
                    best_mse = mse
                    best_scale = cand_s
                else:
                    better = mse < best_mse
                    best_mse = torch.where(better, mse, best_mse)
                    best_scale = torch.where(better, cand_s, best_scale)
            ch_scale = best_scale
            q_i8 = torch.round(t_f / ch_scale.float().clamp(min=1e-7)).clamp(-127, 127).to(torch.int8)
            deq_sd[name] = (q_i8.float() * ch_scale.float().clamp(min=1e-7)).to(dtype=tensor.dtype)

            scale_view = ch_scale.reshape(scale_shape)
            raw_q = q_i8.numpy().copy().tobytes()
            raw_s = scale_view.numpy().copy().tobytes()
            manifest.append({"name": name, "shape": list(q_i8.shape), "dtype": "int8", "offset": offset, "nbytes": len(raw_q)})
            tensors.append(raw_q)
            offset += len(raw_q)
            manifest.append({"name": f"{name}.scale", "shape": list(scale_view.shape), "dtype": "float16", "offset": offset, "nbytes": len(raw_s)})
            tensors.append(raw_s)
            offset += len(raw_s)
        else:
            arr = t_f.to(torch.float16).numpy().copy()
            deq_sd[name] = torch.from_numpy(arr).to(dtype=tensor.dtype)
            raw = arr.tobytes()
            manifest.append({"name": name, "shape": list(arr.shape), "dtype": "float16", "offset": offset, "nbytes": len(raw)})
            tensors.append(raw)
            offset += len(raw)

    # Verify dequantized INT8 vs FP32 accuracy
    m_fp32 = ClientSRNet(tier=tier, scale=scale, use_prior=use_prior, use_rep=True, deploy=True, arch_version=arch_v).eval()
    m_fp32.load_state_dict(sd)
    m_i8 = ClientSRNet(tier=tier, scale=scale, use_prior=use_prior, use_rep=True, deploy=True, arch_version=arch_v).eval()
    m_i8.load_state_dict(deq_sd)
    pdim = int(payload.get("prior_dim", V1_PRIOR_DIM if arch_v == "v1" else (V2_PRIOR_DIM if arch_v == "v2" else V3_PRIOR_DIM)))
    x_t = torch.rand(1, 1, 96, 96)
    pr_t = torch.full((1, pdim), 0.5) if use_prior else None
    with torch.no_grad():
        y_ref = m_fp32(x_t, pr_t)
        y_i8 = m_i8(x_t, pr_t)
    i8_psnr = compute_psnr((y_i8[0, 0].numpy() * 255.0).round().clip(0, 255), (y_ref[0, 0].numpy() * 255.0).round().clip(0, 255))

    meta_bytes = json.dumps({
        "format": "CSR8_PER_CHANNEL",
        "tier": tier,
        "scale": scale,
        "use_prior": use_prior,
        "arch_version": arch_v,
        "prior_dim": pdim,
        "tensors": manifest,
    }, separators=(",", ":")).encode("utf-8")

    out_int8_bin.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_int8_bin.with_suffix(".bin.tmp")
    with open(tmp, "wb") as f:
        f.write(b"CSR8")
        f.write(struct.pack("<I", len(meta_bytes)))
        f.write(meta_bytes)
        for chunk in tensors:
            f.write(chunk)
    tmp.replace(out_int8_bin)
    return {
        "out_int8_bin": str(out_int8_bin),
        "int8_size_kb": round(out_int8_bin.stat().st_size / 1024.0, 2),
        "int8_vs_fp32_psnr": round(i8_psnr, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deploy-ckpt", type=Path, required=True)
    parser.add_argument("--out-bin", type=Path, required=True)
    args = parser.parse_args()
    res = pack_deploy_weights(args.deploy_ckpt, args.out_bin)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
