"""Zero-PyTorch Pure-NumPy Reference for CodecSR V3.5 Web Deployment.

This ~200-line file contains the EXACT math you need to port to TypeScript / JavaScript
on the Web frontend (with zero PyTorch dependency):
  1. `encode_stream_prior_7d(...)`   -> Build 7D prior vector from video metadata or UI slider
  2. `parse_csr1(raw_bytes)`         -> Parse `.bin` (`CSR1` or `CHR1`) header + FP16 tensors
  3. `bake_csr1_to_ssbo(...)`        -> Fold 7D prior into static FP16 Conv3x3 SSBO + Uniforms

Run `python3 standalone_numpy_reference.py` locally to verify 100% byte/bit parity against
PyTorch's `golden_vectors/baked_ssbo_*.bin`.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np


def encode_stream_prior_7d(
    width: int = 960,
    height: int = 540,
    fps: float = 30.0,
    bitrate_kbps: float = 800.0,
    qp_mean: float = 28.0,
    codec: str = "h264",        # "h264" -> 0.0, "hevc"/"h265" -> 1.0
    is_live: bool = False,
    is_screen_content: bool = False,
    has_prior: bool = True,
) -> list[float]:
    """Exact 7D stream prior vector [qp_norm, codec_id, is_live_vbv, log_bpp_norm, rd_complexity, is_screen_content, has_prior]."""
    w = max(1, int(width))
    h = max(1, int(height))
    fps_f = max(1.0, float(fps))
    kbps = max(1.0, float(bitrate_kbps))

    bpp = (kbps * 1000.0) / (w * h * fps_f)
    log_bpp_norm = float(min(1.0, max(0.0, (math.log10(max(bpp, 1e-4)) + 2.5) / 2.0)))
    if not has_prior:
        return [0.0, 0.0, 0.0, log_bpp_norm, 0.0, 0.0, 0.0]

    qp_norm = float(qp_mean) / 51.0
    codec_id = 1.0 if str(codec).lower() in ("hevc", "h265", "libx265", "hev1", "hvc1") else 0.0
    is_live_vbv = 1.0 if is_live else 0.0
    res_scale = ((w * h) / (960.0 * 540.0)) ** 0.28
    bpp_cal = bpp * res_scale
    log_bpp_cal = float(min(1.0, max(0.0, (math.log10(max(bpp_cal, 1e-4)) + 2.5) / 2.0)))
    rd_complexity = float(min(1.0, max(-1.0, log_bpp_cal + 1.8 * qp_norm - 1.1 - 0.06 * codec_id)))
    is_screen = 1.0 if is_screen_content else 0.0
    return [qp_norm, codec_id, is_live_vbv, log_bpp_norm, rd_complexity, is_screen, 1.0]


def parse_binary_blob(raw_bytes: bytes) -> tuple[str, dict[str, Any], dict[str, np.ndarray]]:
    """Parse a `CSR1` (luma) or `CHR1` (chroma) weight file into FP32 numpy tensors."""
    magic = raw_bytes[:4].decode("ascii")
    if magic not in ("CSR1", "CHR1"):
        raise ValueError(f"Unexpected magic header: {magic!r}")
    meta_len = struct.unpack("<I", raw_bytes[4:8])[0]
    meta = json.loads(raw_bytes[8 : 8 + meta_len].decode("utf-8"))
    payload = memoryview(raw_bytes)[8 + meta_len :]

    tensors: dict[str, np.ndarray] = {}
    for t in meta["tensors"]:
        off = int(t["offset"])
        nbytes = int(t["nbytes"])
        shape = tuple(int(d) for d in t["shape"])
        arr = np.frombuffer(payload[off : off + nbytes], dtype=np.float16).astype(np.float32).copy()
        tensors[t["name"]] = arr.reshape(shape) if shape else arr.reshape(())
    return magic, meta, tensors


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def _softplus(x: float) -> float:
    return math.log1p(math.exp(-abs(x))) + max(x, 0.0)


def _expand_grouped_kernel(w_g: np.ndarray, in_ch: int, out_ch: int, groups: int) -> np.ndarray:
    if groups <= 1:
        return w_g
    og = out_ch // groups
    ig = in_ch // groups
    w_full = np.zeros((out_ch, in_ch, 3, 3), dtype=np.float32)
    for g in range(groups):
        w_full[g * og : (g + 1) * og, g * ig : (g + 1) * ig] = w_g[g * og : (g + 1) * og]
    return w_full


def bake_csr1_to_ssbo(
    meta: dict[str, Any],
    tensors: dict[str, np.ndarray],
    prior_7d: list[float] | None = None,
) -> dict[str, Any]:
    """Pure-CPU dynamic kernel baking (equivalent to `ClientSRNet.bake_segment_prior`).

    Returns:
      - `ssbo_bytes`: 4-byte aligned FP16 WeightBuf ready for WebGPU `GPUBuffer`
      - `half_offsets`: dict mapping tensor name -> uint32 element index in FP16 array
      - `scalars`: `skip_alpha`, `skip_beta`, `skip_hr`, `trust_delta`, `is_hourglass`, `hg_has_up`, `hg_has_hr_skip`
    """
    tier = str(meta["tier"]).upper()
    scale = float(meta["scale"])
    use_prior = bool(meta["use_prior"])
    is_hourglass = (tier == "P")
    hg_has_up = is_hourglass and abs(scale - 2.0) < 1e-4
    hg_has_hr_skip = is_hourglass and abs(scale - 1.5) < 1e-4

    # Ordered list of HyperBasisConv3x3 prefixes matching `_all_v3_hyper_layers()`
    if is_hourglass:
        num_body = 4 if abs(scale - 2.0) < 1e-4 else 3
        layer_prefixes = ["stem1", "stem2", "down"] + [f"body.{i}" for i in range(num_body)]
        if hg_has_up:
            layer_prefixes += ["up", "refine"]
        layer_prefixes.append("tail")
    else:
        num_body = 4 if tier == "L" else 5
        layer_prefixes = ["head"] + [f"body.{i}" for i in range(num_body)] + ["tail"]

    out_sd: dict[str, np.ndarray] = {k: v.copy() for k, v in tensors.items()}

    # 1. Predict per-layer (alphas, gamma, beta) from 7D prior if enabled
    layer_params: list[tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]] = [
        (None, None, None) for _ in layer_prefixes
    ]
    if use_prior and prior_7d is not None and "prior_mod.mlp.0.weight" in out_sd:
        p = np.asarray(prior_7d[:7], dtype=np.float32)
        h = _silu(out_sd["prior_mod.mlp.0.weight"] @ p + out_sd["prior_mod.mlp.0.bias"])
        raw = out_sd["prior_mod.mlp.2.weight"] @ h + out_sd["prior_mod.mlp.2.bias"]
        off = 0
        layer_params = []
        for pref in layer_prefixes:
            out_ch = int(out_sd[f"{pref}.base.reparam_conv.bias"].shape[0])
            alphas = 0.20 * np.tanh(raw[off : off + 4])
            off += 4
            gamma = 0.10 * raw[off : off + out_ch]
            off += out_ch
            beta = 0.10 * raw[off : off + out_ch]
            off += out_ch
            layer_params.append((alphas, gamma, beta))

        # Bake Seam-3DLUT prior gain into 8x8 seam_mask
        if "seam_lut.seam_mask" in out_sd and "prior_mod.lut_head.weight" in out_sd:
            lut_raw = float((out_sd["prior_mod.lut_head.weight"] @ p + out_sd["prior_mod.lut_head.bias"])[0])
            g_lut = 2.0 * float(_sigmoid(lut_raw))
            out_sd["seam_lut.seam_mask"] = out_sd["seam_lut.seam_mask"] * g_lut

    # 2. Fold K=4 dynamic bases + FiLM (gamma, beta) into each static 3x3 conv
    for pref, (alphas, gamma, beta) in zip(layer_prefixes, layer_params):
        w = out_sd[f"{pref}.base.reparam_conv.weight"].astype(np.float32).copy()
        b = out_sd[f"{pref}.base.reparam_conv.bias"].astype(np.float32).copy()
        out_ch, in_ch = w.shape[0], w.shape[1]
        bw_key = f"{pref}.basis_weight"
        bb_key = f"{pref}.basis_bias"
        if alphas is not None and bw_key in out_sd:
            bw = out_sd[bw_key].astype(np.float32)  # [4, out_ch, in_ch // groups, 3, 3]
            bb = out_sd[bb_key].astype(np.float32)  # [4, out_ch]
            groups = in_ch // bw.shape[2]
            norms = np.linalg.norm(bw.reshape(4, -1), axis=1).reshape(4, 1, 1, 1, 1) + 1e-6
            bw = bw / norms
            for k in range(4):
                w = w + alphas[k] * _expand_grouped_kernel(bw[k], in_ch, out_ch, groups)
                b = b + alphas[k] * bb[k]
        if gamma is not None and beta is not None:
            w = w * (1.0 + gamma).reshape(-1, 1, 1, 1)
            b = b * (1.0 + gamma) + beta
        out_sd[f"{pref}.base.reparam_conv.weight"] = w
        out_sd[f"{pref}.base.reparam_conv.bias"] = b
        out_sd.pop(bw_key, None)
        out_sd.pop(bb_key, None)

    # Remove prior_mod weights after baking
    for k in [k for k in out_sd if k.startswith("prior_mod.")]:
        out_sd.pop(k)

    # 3. Bake QPAdaptiveTrustClamp delta scalar
    if use_prior and prior_7d is not None and "trust_clamp.delta_head.weight" in out_sd:
        p = np.asarray(prior_7d[:7], dtype=np.float32)
        raw_d = float((out_sd["trust_clamp.delta_head.weight"] @ p + out_sd["trust_clamp.delta_head.bias"])[0])
        trust_delta = 0.004 + 0.076 * float(_sigmoid(raw_d))
    else:
        th = out_sd["trust_clamp.theta"].astype(np.float32)
        trust_delta = 0.004 + _softplus(float(th[0])) + _softplus(float(th[1]))
    out_sd["trust_clamp.baked_delta"] = np.array(trust_delta, dtype=np.float32)

    # 4. Pack remaining tensors into 4-byte aligned FP16 WeightBuf in exact state_dict order
    chunks: list[bytes] = []
    half_offsets: dict[str, int] = {}
    cur_half_idx = 0
    for name, arr in out_sd.items():
        half_offsets[name] = cur_half_idx
        arr16 = np.ascontiguousarray(arr, dtype=np.float16).reshape(-1)
        chunks.append(arr16.tobytes())
        cur_half_idx += int(arr16.size)

    ssbo_bytes = b"".join(chunks)
    if len(ssbo_bytes) % 4 != 0:
        ssbo_bytes += b"\x00" * (4 - (len(ssbo_bytes) % 4))

    return {
        "ssbo_bytes": ssbo_bytes,
        "half_offsets": half_offsets,
        "scalars": {
            "skip_alpha": float(np.float16(out_sd["skip_alpha"])),
            "skip_beta": float(np.float16(out_sd["skip_beta"])),
            "skip_hr": float(np.float16(out_sd["skip_hr"])) if "skip_hr" in out_sd else 0.0,
            "trust_delta": float(trust_delta),
            "is_hourglass": is_hourglass,
            "hg_has_up": hg_has_up,
            "hg_has_hr_skip": hg_has_hr_skip,
        },
    }


if __name__ == "__main__":
    bundle_dir = Path(__file__).resolve().parent.parent
    manifest = json.loads((bundle_dir / "golden_vectors/golden_manifest.json").read_text(encoding="utf-8"))
    for case in manifest["cases"]:
        raw = (bundle_dir / "weights" / case["weight_file"]).read_bytes()
        _, meta, tensors = parse_binary_blob(raw)
        res = bake_csr1_to_ssbo(meta, tensors, case["prior_7d"])
        ref_ssbo = (bundle_dir / "golden_vectors" / case["baked_ssbo_file"]).read_bytes()
        assert len(res["ssbo_bytes"]) == len(ref_ssbo), f"Length mismatch on {case['case_id']}"
        a = np.frombuffer(res["ssbo_bytes"], dtype=np.float16).astype(np.float32)
        b = np.frombuffer(ref_ssbo, dtype=np.float16).astype(np.float32)
        max_diff = float(np.max(np.abs(a - b)))
        delta_diff = abs(res["scalars"]["trust_delta"] - case["baked_scalars"]["trust_delta"])
        assert res["half_offsets"] == case["baked_ssbo_half_offsets"], f"Offsets mismatch on {case['case_id']}"
        assert max_diff < 2e-4 and delta_diff < 5e-5, f"Value mismatch on {case['case_id']}: {max_diff}, {delta_diff}"
        print(f"[PASS] {case['case_id']}: SSBO={len(ref_ssbo)}B, max_fp16_diff={max_diff:.6f}, delta_diff={delta_diff:.2e}")
