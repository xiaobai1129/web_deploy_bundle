"""Client-L / S / P Edge-Deployable Super-Resolution Architectures (x2 and x1.5).

Supports:
- `arch_version="v1"`: Round-1 baseline (RepConv3x3 on S/P body only, 2D middle FiLM, bilinear skip, LeakyReLU)
- `arch_version="v2"`: Zero-deployment-overhead V2 upgrade (SeqECBRepConv2d, 5D prior baking, Mod-8 phase, bicubic+PReLU)
- `arch_version="v3"`: Final V3 master architecture (Idea #301-#320):
    * Strict-FP32 kernel folding + norm-decoupled sequential 1x1 expansion + full-layer STE-FP16 QAT
    * 7D RD-complexity-decoupled prior + full-stack (incl. tail) K=4 orthogonal dynamic bases (HyperBasisConv3x3)
    * SPAN-Lite parameter-free skip highway + Mod-16 macroblock/CTU phase bias + D4 phase equivariance
    * Client-L TMU-friendly 17x17x17 Seam3DLUT parallel deblocking branch
    * Client-P 2-scale stateless hourglass (x2 & x1.5) covering full 64x64 HEVC CTUs at ~24 GMACs
    * QP-adaptive soft trust-region anti-ringing envelope clamp at tail output
    * Dual-scale (x2 & x1.5) YGuidedChromaHead for full-color YUV420 evaluation
"""

from __future__ import annotations

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

V1_PRIOR_DIM = 2
V2_PRIOR_DIM = 5
V3_PRIOR_DIM = 7


def ste_fp16(x: torch.Tensor) -> torch.Tensor:
    """Straight-through estimator simulating IEEE-754 float16 rounding and range clamp."""
    return x + (x.to(torch.float16).clamp(-65504.0, 65504.0).to(x.dtype) - x).detach()


def apply_d4_tensor(x: torch.Tensor, op: int) -> torch.Tensor:
    """Apply D4 dihedral transform op in [0..7] on (..., H, W) tensor."""
    op_i = int(op) & 7
    if op_i & 1:
        x = x.flip(-1)
    if op_i & 2:
        x = x.flip(-2)
    if op_i & 4:
        x = x.transpose(-2, -1)
    return x


def apply_d4_flow(flow: torch.Tensor, op: int) -> torch.Tensor:
    """Apply D4 dihedral transform op in [0..7] to optical flow (..., 2, H, W) = (dx, dy)."""
    op_i = int(op) & 7
    fl = apply_d4_tensor(flow, op_i)
    u = fl.select(-3, 0)
    v = fl.select(-3, 1)
    if op_i & 1:
        u = -u
    if op_i & 2:
        v = -v
    if op_i & 4:
        u, v = v, u
    return torch.stack([u, v], dim=-3)


def _expand_grouped_kernel(w_g: torch.Tensor, in_ch: int, out_ch: int, groups: int) -> torch.Tensor:
    """Embed a grouped Conv2d kernel (out_ch, in_ch // groups, k, k) into dense (out_ch, in_ch, k, k)."""
    if groups <= 1:
        return w_g
    og = out_ch // groups
    ig = in_ch // groups
    k_h, k_w = w_g.shape[-2], w_g.shape[-1]
    w_full = w_g.new_zeros((out_ch, in_ch, k_h, k_w))
    for g in range(groups):
        w_full[g * og : (g + 1) * og, g * ig : (g + 1) * ig] = w_g[g * og : (g + 1) * og]
    return w_full


class RepConv3x3(nn.Module):
    """V1 multi-branch linear block (Conv3x3 + Conv1x1 + Identity + Fixed Laplacian)."""

    def __init__(self, channels: int, use_laplacian: bool = True, deploy: bool = False):
        super().__init__()
        self.channels = channels
        self.use_laplacian = use_laplacian
        self.deploy = deploy

        if deploy:
            self.reparam_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        else:
            self.conv3x3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
            self.conv1x1 = nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=True)
            self.use_identity = True
            if use_laplacian:
                lap = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
                self.register_buffer("lap_kernel", lap.view(1, 1, 3, 3))
                self.lap_scale = nn.Parameter(torch.zeros(channels, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            return self.reparam_conv(x)
        out = self.conv3x3(x) + self.conv1x1(x) + x
        if self.use_laplacian:
            w_lap = self.lap_kernel.expand(self.channels, 1, 3, 3) * self.lap_scale
            out = out + F.conv2d(x, w_lap, bias=None, padding=1, groups=self.channels)
        return out

    @torch.no_grad()
    def get_equivalent_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.deploy:
            return self.reparam_conv.weight.data.clone(), self.reparam_conv.bias.data.clone()

        w = self.conv3x3.weight.data.clone()
        b = self.conv3x3.bias.data.clone()
        w[:, :, 1:2, 1:2] += self.conv1x1.weight.data
        b += self.conv1x1.bias.data
        idx = torch.arange(self.channels, device=w.device)
        w[idx, idx, 1, 1] += 1.0
        if self.use_laplacian:
            w_lap = (self.lap_kernel.expand(self.channels, 1, 3, 3) * self.lap_scale).squeeze(1)
            w[idx, idx, :, :] += w_lap
        return w, b

    @torch.no_grad()
    def switch_to_deploy(self) -> None:
        if self.deploy:
            return
        w, b = self.get_equivalent_kernel_bias()
        self.reparam_conv = nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1, bias=True).to(w.device, w.dtype)
        self.reparam_conv.weight.data.copy_(w)
        self.reparam_conv.bias.data.copy_(b)
        for attr in ("conv3x3", "conv1x1", "lap_kernel", "lap_scale"):
            if hasattr(self, attr):
                delattr(self, attr)
        self.deploy = True


class SeqECBRepConv2d(nn.Module):
    """V2 Five-Branch Sequential-Parallel Reparameterized 3x3 Block (Idea #004).

    Branches during training (no nonlinearity inside):
      1. Standard 3x3 conv (in_ch -> out_ch)
      2. Sequential rank-expanded conv: 1x1(in_ch -> mid_ch, bias=False) -> 3x3(mid_ch -> out_ch)
      3. Cross-channel 1x1 conv (in_ch -> out_ch)
      4. Directional coding-boundary priors: 1x1(in_ch -> out_ch, bias=False) -> Fixed Depthwise Sobel-X / Sobel-Y
      5. Identity + Depthwise Laplacian residual (enabled when in_ch == out_ch)

    Algebraically folds with exact zero boundary error into one Conv2d(in_ch, out_ch, 3, padding=1, bias=True).
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        expand_ratio: int = 2,
        zero_init_out: bool = False,
        deploy: bool = False,
        norm_expand: bool = False,
    ):
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.has_identity = (self.in_ch == self.out_ch) and (not zero_init_out)
        self.deploy = bool(deploy)
        self.norm_expand = bool(norm_expand)

        if self.deploy:
            self.reparam_conv = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=3, padding=1, bias=True)
            return

        mid_ch = max(self.out_ch, int(round(max(self.in_ch, self.out_ch) * expand_ratio)))

        # Branch 1: standard 3x3
        self.conv3x3 = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=3, padding=1, bias=True)

        # Branch 2: sequential linear expansion (bias=False on inner 1x1 ensures 0 padding error)
        self.expand_1x1 = nn.Conv2d(self.in_ch, mid_ch, kernel_size=1, padding=0, bias=False)
        self.expand_3x3 = nn.Conv2d(mid_ch, self.out_ch, kernel_size=3, padding=1, bias=True)

        # Branch 3: 1x1 mix
        self.conv1x1 = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=1, padding=0, bias=True)

        # Branch 4: directional horizontal/vertical block-boundary Sobel branches (bias=False on 1x1)
        self.sobel_x_1x1 = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=1, padding=0, bias=False)
        self.sobel_y_1x1 = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=1, padding=0, bias=False)
        sx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32) * 0.25
        sy = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32) * 0.25
        self.register_buffer("sobel_x_kernel", sx.view(1, 1, 3, 3))
        self.register_buffer("sobel_y_kernel", sy.view(1, 1, 3, 3))

        # Branch 5: identity + learnable depthwise Laplacian when channels match
        if self.has_identity:
            lap = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
            self.register_buffer("lap_kernel", lap.view(1, 1, 3, 3))
            self.lap_scale = nn.Parameter(torch.zeros(self.out_ch, 1, 1, 1))

        if zero_init_out:
            nn.init.normal_(self.conv3x3.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.conv3x3.bias)
            nn.init.zeros_(self.expand_3x3.weight)
            nn.init.zeros_(self.expand_3x3.bias)
            nn.init.zeros_(self.conv1x1.weight)
            nn.init.zeros_(self.conv1x1.bias)
            nn.init.zeros_(self.sobel_x_1x1.weight)
            nn.init.zeros_(self.sobel_y_1x1.weight)

    def _compute_equivalent_weight_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.deploy:
            return self.reparam_conv.weight.float(), self.reparam_conv.bias.float()

        with torch.amp.autocast("cuda", enabled=False):
            # Strict-FP32 5-branch algebraic kernel folding (Idea #301)
            w = self.conv3x3.weight.float().clone()
            b = self.conv3x3.bias.float().clone()

            w_1x1 = self.expand_1x1.weight.float()[:, :, 0, 0]
            if self.norm_expand:
                w_1x1 = w_1x1 / (w_1x1.norm(dim=1, keepdim=True) + 1e-6)
            w_seq = torch.einsum("omhw,mi->oihw", self.expand_3x3.weight.float(), w_1x1)
            w = w + w_seq
            b = b + self.expand_3x3.bias.float()

            w_center = F.pad(self.conv1x1.weight.float(), (1, 1, 1, 1))
            w = w + w_center
            b = b + self.conv1x1.bias.float()

            w = w + self.sobel_x_1x1.weight.float() * self.sobel_x_kernel.float()
            w = w + self.sobel_y_1x1.weight.float() * self.sobel_y_kernel.float()

            if self.has_identity:
                idx = torch.arange(self.out_ch, device=w.device)
                eye = torch.zeros_like(w)
                eye[idx, idx, 1, 1] = 1.0
                lap_w = torch.zeros_like(w)
                lap_w[idx, idx, :, :] = (
                    self.lap_kernel.float().expand(self.out_ch, 1, 3, 3) * self.lap_scale.float()
                ).squeeze(1)
                w = w + eye + lap_w

            return w, b

    def equivalent_weight_l2(self) -> torch.Tensor:
        """Differentiable L2 norm on equivalent folded weights (Idea #019 stability regularizer)."""
        w, _ = self._compute_equivalent_weight_bias()
        return (w * w).mean()

    def forward(
        self,
        x: torch.Tensor,
        qat_fp16: bool = False,
        precomputed_wb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if precomputed_wb is not None:
            w, b = precomputed_wb
            w = w.to(dtype=x.dtype)
            b = b.to(dtype=x.dtype)
            if qat_fp16:
                w = ste_fp16(w)
                b = ste_fp16(b)
            return F.conv2d(x, w, b, padding=1)
        if self.deploy and not qat_fp16 and x.dtype == self.reparam_conv.weight.dtype:
            return self.reparam_conv(x)
        if self.deploy:
            w = self.reparam_conv.weight.to(dtype=x.dtype)
            b = self.reparam_conv.bias.to(dtype=x.dtype)
            if qat_fp16:
                w = ste_fp16(w)
                b = ste_fp16(b)
            return F.conv2d(x, w, b, padding=1)
        w, b = self._compute_equivalent_weight_bias()
        w = w.to(dtype=x.dtype)
        b = b.to(dtype=x.dtype)
        if qat_fp16:
            w = ste_fp16(w)
            b = ste_fp16(b)
        return F.conv2d(x, w, b, padding=1)

    @torch.no_grad()
    def get_equivalent_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        w, b = self._compute_equivalent_weight_bias()
        return w.detach().clone(), b.detach().clone()

    @torch.no_grad()
    def switch_to_deploy(self) -> None:
        if self.deploy:
            return
        w, b = self.get_equivalent_kernel_bias()
        self.reparam_conv = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=3, padding=1, bias=True).to(w.device, w.dtype)
        self.reparam_conv.weight.data.copy_(w)
        self.reparam_conv.bias.data.copy_(b)
        for attr in (
            "conv3x3",
            "expand_1x1",
            "expand_3x3",
            "conv1x1",
            "sobel_x_1x1",
            "sobel_y_1x1",
            "sobel_x_kernel",
            "sobel_y_kernel",
            "lap_kernel",
            "lap_scale",
        ):
            if hasattr(self, attr):
                delattr(self, attr)
        self.deploy = True


class FiLMConditioner(nn.Module):
    """V1 middle-stage FiLM mapper [segment_qp_norm, codec_id] (B, 2) -> (gamma, beta)."""

    def __init__(self, channels: int, prior_dim: int = V1_PRIOR_DIM):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(prior_dim, 16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(16, channels * 2),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.channels = channels

    def forward(self, feat: torch.Tensor, prior: torch.Tensor | None) -> torch.Tensor:
        if prior is None:
            return feat
        gb = self.mlp(prior).view(feat.shape[0], self.channels * 2, 1, 1)
        gamma, beta = gb[:, : self.channels], gb[:, self.channels :]
        return feat * (1.0 + gamma) + beta


class PreFoldablePriorModulator(nn.Module):
    """V2 Segment-Level All-Layer Prior Modulator (Idea #003).

    Predicts per-layer (gamma_l, beta_l) for head + all body blocks from 5D segment prior:
      prior = [qp_norm, codec_id, is_live_vbv, log_bpp_norm, has_prior].
    During deployment, these scalars are folded once per 2-second segment on CPU directly into
    each deployed Conv3x3 weight & bias (and head phase_bias), leaving 0.00 GPU FiLM MACs/frame.
    """

    def __init__(self, channels: int, num_layers: int, prior_dim: int = V2_PRIOR_DIM):
        super().__init__()
        self.channels = int(channels)
        self.num_layers = int(num_layers)
        self.prior_dim = int(prior_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.prior_dim, 24),
            nn.SiLU(inplace=True),
            nn.Linear(24, self.num_layers * self.channels * 2),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def predict_params(self, prior: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = prior.shape[0]
        raw = self.mlp(prior).view(b, self.num_layers, 2, self.channels)
        return raw[:, :, 0, :], raw[:, :, 1, :]


class HyperBasisConv3x3(nn.Module):
    """V3 Dynamic Spatial-Kernel Basis Block (Idea #303):
      W_eff(p) = (W_base + sum_{k=1..K} alpha_k(p) * expand_grouped(DeltaW_k)) * (1 + gamma(p))
      b_eff(p) = (b_base + sum_{k=1..K} alpha_k(p) * Deltab_k) * (1 + gamma(p)) + beta(p)

    At deploy/segment-switch time on CPU, `bake_with_params` folds everything into ONE static Conv2d(3x3).
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        num_bases: int = 4,
        basis_groups: int = 1,
        norm_expand: bool = True,
        zero_init_out: bool = False,
        deploy: bool = False,
    ):
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.num_bases = max(0, int(num_bases))
        g = max(1, int(basis_groups))
        while g > 1 and (self.in_ch % g != 0 or self.out_ch % g != 0):
            g -= 1
        self.basis_groups = g
        self.deploy = bool(deploy)

        self.base = SeqECBRepConv2d(
            self.in_ch,
            self.out_ch,
            expand_ratio=2,
            zero_init_out=zero_init_out,
            deploy=deploy,
            norm_expand=norm_expand,
        )
        if self.num_bases > 0:
            self.basis_weight = nn.Parameter(
                torch.zeros(self.num_bases, self.out_ch, self.in_ch // self.basis_groups, 3, 3)
            )
            self.basis_bias = nn.Parameter(torch.zeros(self.num_bases, self.out_ch))
            nn.init.normal_(self.basis_weight, mean=0.0, std=2e-3)

    def equivalent_weight_l2(
        self, precomputed_wb: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> torch.Tensor:
        if precomputed_wb is not None:
            w, _ = precomputed_wb
            reg = (w * w).mean()
        else:
            reg = self.base.equivalent_weight_l2()
        if self.num_bases > 0 and hasattr(self, "basis_weight"):
            reg = reg + 0.1 * (self.basis_weight * self.basis_weight).mean()
        return reg

    def basis_orthogonality_loss(self) -> torch.Tensor:
        if self.num_bases < 2 or not hasattr(self, "basis_weight"):
            return self.base.reparam_conv.weight.new_zeros(()) if self.base.deploy else self.base.conv3x3.weight.new_zeros(())
        flat = self.basis_weight.float().reshape(self.num_bases, -1)
        flat = F.normalize(flat, dim=1, eps=1e-6)
        sim = torch.matmul(flat, flat.t()).abs()
        off_diag = sim - torch.diag(torch.diag(sim))
        return off_diag.sum() / max(1, self.num_bases * (self.num_bases - 1))

    def forward(
        self,
        x: torch.Tensor,
        alphas: torch.Tensor | None = None,
        gamma: torch.Tensor | None = None,
        beta: torch.Tensor | None = None,
        qat_fp16: bool = False,
        precomputed_wb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        y = self.base(x, qat_fp16=qat_fp16, precomputed_wb=precomputed_wb)
        if self.num_bases > 0 and alphas is not None and hasattr(self, "basis_weight"):
            # Idea #356: unit-Frobenius spectral normalization per basis kernel removes the
            # bilinear scale deadlock between W_basis (std 2e-3) and the prior MLP (0.1*std 1e-2).
            bw = self.basis_weight.to(dtype=x.dtype)
            bw = bw / (bw.reshape(self.num_bases, -1).norm(dim=1).view(-1, 1, 1, 1, 1) + 1e-6)
            bb = self.basis_bias.to(dtype=x.dtype)
            al = alphas.to(dtype=x.dtype)
            if qat_fp16:
                bw = ste_fp16(bw)
                bb = ste_fp16(bb)
                al = ste_fp16(al)
            k_n = self.num_bases
            if x.shape[0] * self.out_ch * k_n * x.shape[-2] * x.shape[-1] < (1 << 30):
                g = self.basis_groups
                og = self.out_ch // g
                ig = self.in_ch // g
                bw_g = bw.view(k_n, g, og, ig, 3, 3).transpose(0, 1).reshape(g * k_n * og, ig, 3, 3)
                bb_g = bb.view(k_n, g, og).transpose(0, 1).reshape(g * k_n * og)
                y_all = F.conv2d(x, bw_g, bb_g, padding=1, groups=g)
                b_sz, _, h_out, w_out = y_all.shape
                y_k = (y_all.view(b_sz, g, k_n, og, h_out, w_out) * al.view(b_sz, 1, k_n, 1, 1, 1)).sum(dim=2)
                y = y + y_k.reshape(b_sz, self.out_ch, h_out, w_out)
            else:
                for k in range(k_n):
                    yk = F.conv2d(x, bw[k], bb[k], padding=1, groups=self.basis_groups)
                    y = y + al[:, k].view(-1, 1, 1, 1) * yk
        if gamma is not None and beta is not None:
            g = gamma.to(dtype=x.dtype)
            b = beta.to(dtype=x.dtype)
            if qat_fp16:
                g = ste_fp16(g)
                b = ste_fp16(b)
            y = y * (1.0 + g.unsqueeze(-1).unsqueeze(-1)) + b.unsqueeze(-1).unsqueeze(-1)
        return y

    @torch.no_grad()
    def switch_to_deploy(self) -> None:
        self.base.switch_to_deploy()
        self.deploy = True

    @torch.no_grad()
    def bake_with_params(
        self,
        alphas_1d: torch.Tensor | None,
        gamma_1d: torch.Tensor | None,
        beta_1d: torch.Tensor | None,
    ) -> None:
        """Fold base SeqECB + K bases + (gamma, beta) into one static Conv2d on CPU/GPU."""
        self.switch_to_deploy()
        w = self.base.reparam_conv.weight.data.float().clone()
        b = self.base.reparam_conv.bias.data.float().clone()
        if self.num_bases > 0 and alphas_1d is not None and hasattr(self, "basis_weight"):
            al = alphas_1d.float().to(w.device)
            bw = self.basis_weight.data.float().to(w.device)
            # Match training-time unit-Frobenius spectral normalization exactly (Idea #356)
            bw = bw / (bw.reshape(self.num_bases, -1).norm(dim=1).view(-1, 1, 1, 1, 1) + 1e-6)
            bb = self.basis_bias.data.float().to(w.device)
            for k in range(self.num_bases):
                w = w + al[k] * _expand_grouped_kernel(bw[k], self.in_ch, self.out_ch, self.basis_groups)
                b = b + al[k] * bb[k]
        if gamma_1d is not None and beta_1d is not None:
            g = gamma_1d.float().to(w.device)
            bt = beta_1d.float().to(w.device)
            w = w * (1.0 + g).view(-1, 1, 1, 1)
            b = b * (1.0 + g) + bt
        self.base.reparam_conv.weight.data.copy_(w.to(self.base.reparam_conv.weight.dtype))
        self.base.reparam_conv.bias.data.copy_(b.to(self.base.reparam_conv.bias.dtype))
        for attr in ("basis_weight", "basis_bias"):
            if hasattr(self, attr):
                delattr(self, attr)
        self.num_bases = 0


class V3HyperPriorModulator(nn.Module):
    """V3 Full-Stack (incl. tail) 7D Prior -> per-layer (K=4 basis coeffs, gamma_l, beta_l)."""

    def __init__(self, layer_channels: list[int], prior_dim: int = V3_PRIOR_DIM, num_bases: int = 4):
        super().__init__()
        self.layer_channels = [int(c) for c in layer_channels]
        self.prior_dim = int(prior_dim)
        self.num_bases = int(num_bases)
        self.total_out = sum(self.num_bases + 2 * c for c in self.layer_channels)
        hidden = 12 if self.total_out > 400 else 16
        self.mlp = nn.Sequential(
            nn.Linear(self.prior_dim, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, self.total_out),
        )
        # Small non-zero init breaks gradient saddle point while keeping initial perturbation tiny
        nn.init.normal_(self.mlp[-1].weight, mean=0.0, std=1e-2)
        nn.init.zeros_(self.mlp[-1].bias)
        # Idea #356: break bilinear saddle on alpha rows specifically while keeping FiLM (gamma, beta) tiny
        if self.num_bases > 0:
            with torch.no_grad():
                off = 0
                alpha_bias = torch.linspace(-0.08, 0.08, self.num_bases)
                for ch in self.layer_channels:
                    nn.init.normal_(self.mlp[-1].weight[off : off + self.num_bases], mean=0.0, std=0.12)
                    self.mlp[-1].bias[off : off + self.num_bases].copy_(alpha_bias)
                    off += self.num_bases + 2 * ch
        # Idea #358: extra light segment-wise Seam-3DLUT gain head, g_lut(p) = 2*sigmoid(raw)
        # Initialized to exactly zero -> g_lut = 1.0 (no-op), then learns high-QP deblock strength.
        self.lut_head = nn.Linear(self.prior_dim, 1)
        nn.init.zeros_(self.lut_head.weight)
        nn.init.zeros_(self.lut_head.bias)

    def predict_layer_params(
        self,
        prior: torch.Tensor,
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        mlp_dtype = self.mlp[0].weight.dtype
        raw = self.mlp(prior.to(dtype=mlp_dtype))
        out: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        off = 0
        for ch in self.layer_channels:
            # Idea #356: bounded strong-response tanh map alpha_k in [-0.20, +0.20]
            alphas = 0.20 * torch.tanh(raw[:, off : off + self.num_bases])
            off += self.num_bases
            # Keep FiLM (gamma, beta) on the conservative 0.1 scale for stable early training
            gamma = 0.1 * raw[:, off : off + ch]
            off += ch
            beta = 0.1 * raw[:, off : off + ch]
            off += ch
            out.append((alphas, gamma, beta))
        return out

    def predict_lut_gain(self, prior: torch.Tensor) -> torch.Tensor:
        """Idea #358: per-sample Seam-3DLUT gain g(p) in [0, 2], exactly 1.0 at init."""
        raw = self.lut_head(prior.to(self.lut_head.weight.dtype)).squeeze(-1)
        return 2.0 * torch.sigmoid(raw)


class Seam3DLUT(nn.Module):
    """TMU-friendly 17x17x17 trilinear 3D LUT deblocking branch for Client-L (Idea #306, 0 Conv MACs)."""

    def __init__(self, grid_size: int = 17, seam_period: int = 8):
        super().__init__()
        self.grid_size = int(grid_size)
        self.seam_period = int(seam_period)
        self.table = nn.Parameter(torch.zeros(1, 1, self.grid_size, self.grid_size, self.grid_size))
        self.seam_mask = nn.Parameter(torch.full((1, 1, self.seam_period, self.seam_period), 0.1))

    def _tiled_mask(self, h: int, w: int, d4_op: int | None = None) -> torch.Tensor:
        m = self.seam_mask
        if d4_op is not None:
            m = apply_d4_tensor(m, d4_op)
        p = self.seam_period
        rep_h = (h + p - 1) // p
        rep_w = (w + p - 1) // p
        return m.repeat(1, 1, rep_h, rep_w)[:, :, :h, :w]

    def forward(
        self,
        x: torch.Tensor,
        out_hw: tuple[int, int],
        d4_op: int | torch.Tensor | None = None,
        qat_fp16: bool = False,
        lut_gain: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, _, h, w = x.shape
        xp = F.pad(x, (1, 1, 1, 1), mode="replicate")
        dh_l = (0.5 * (xp[:, :, 1:-1, :-2] - x) + 0.5).clamp(0.0, 1.0)
        dh_r = (0.5 * (xp[:, :, 1:-1, 2:] - x) + 0.5).clamp(0.0, 1.0)
        dv_u = (0.5 * (xp[:, :, :-2, 1:-1] - x) + 0.5).clamp(0.0, 1.0)
        dv_d = (0.5 * (xp[:, :, 2:, 1:-1] - x) + 0.5).clamp(0.0, 1.0)

        tbl = self.table.to(dtype=x.dtype)
        sm = self.seam_mask.to(dtype=x.dtype)
        if qat_fp16:
            tbl = ste_fp16(tbl)
            sm = ste_fp16(sm)
        tbl_b = tbl.expand(b, -1, -1, -1, -1)

        grid_h = torch.stack([2.0 * dh_r - 1.0, 2.0 * dh_l - 1.0, 2.0 * x - 1.0], dim=-1).squeeze(1).unsqueeze(1).to(dtype=tbl_b.dtype)
        grid_v = torch.stack([2.0 * dv_d - 1.0, 2.0 * dv_u - 1.0, 2.0 * x - 1.0], dim=-1).squeeze(1).unsqueeze(1).to(dtype=tbl_b.dtype)
        grid_hv = torch.cat([grid_h, grid_v], dim=1)
        lut_hv = F.grid_sample(tbl_b, grid_hv, mode="bilinear", padding_mode="border", align_corners=True)
        lut_avg = 0.5 * (lut_hv[:, :, 0] + lut_hv[:, :, 1])

        if d4_op is None or isinstance(d4_op, int):
            p = self.seam_period
            m_src = apply_d4_tensor(sm, d4_op) if isinstance(d4_op, int) else sm
            rep_h, rep_w = (h + p - 1) // p, (w + p - 1) // p
            mask_full = m_src.repeat(1, 1, rep_h, rep_w)[:, :, :h, :w]
        else:
            p = self.seam_period
            rep_h, rep_w = (h + p - 1) // p, (w + p - 1) // p
            sm_all = torch.stack([apply_d4_tensor(sm.squeeze(0), op_id) for op_id in range(8)], dim=0)
            mask_full = sm_all[(d4_op.long() & 7)].repeat(1, 1, rep_h, rep_w)[:, :, :h, :w]

        delta_lr = lut_avg * mask_full
        if lut_gain is not None:
            # Idea #358: prior-gated strength; deployment bakes g(p) into seam_mask on CPU.
            delta_lr = delta_lr * lut_gain.to(dtype=delta_lr.dtype).view(-1, 1, 1, 1)
        return F.interpolate(delta_lr, size=out_hw, mode="bilinear", align_corners=False)


class QPAdaptiveTrustClamp(nn.Module):
    """QP-adaptive local neighborhood soft trust-region anti-ringing envelope clamp (Idea #304, #359)."""

    def __init__(self, use_prior: bool = True, prior_dim: int = V3_PRIOR_DIM):
        super().__init__()
        self.use_prior = bool(use_prior)
        # delta(p) ~ 0.012 at CRF22, ~0.006 on screen, ~0.042 at CRF38
        self.theta = nn.Parameter(torch.tensor([-5.5, -5.0, 1.0], dtype=torch.float32))
        # Idea #359.1: lightweight 7D prior -> delta(p) head in [0.004, 0.080], unlocked fast LR group.
        if self.use_prior:
            self.delta_head = nn.Linear(int(prior_dim), 1)
            nn.init.zeros_(self.delta_head.weight)
            with torch.no_grad():
                self.delta_head.bias.fill_(-2.2)  # sigmoid(-2.2) ~ 0.10 -> delta ~ 0.0116
        self.register_buffer("baked_delta", torch.tensor(-1.0, dtype=torch.float32))
        self.is_baked = False

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self.is_baked = bool(float(self.baked_delta.detach().cpu().item()) >= 0.0)

    def compute_delta(self, prior: torch.Tensor | None, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if self.is_baked:
            return self.baked_delta.to(device=device, dtype=dtype).view(1, 1, 1, 1)
        if self.use_prior and prior is not None and hasattr(self, "delta_head"):
            # Idea #359: full 7D (QP, codec, live/VOD, bpp, rd_complexity, screen, has_prior) delta map
            raw = self.delta_head(prior.to(device=device, dtype=self.delta_head.weight.dtype)).squeeze(-1).float()
            delta = 0.004 + 0.076 * torch.sigmoid(raw)
            return delta.to(dtype=dtype).view(-1, 1, 1, 1)
        th = self.theta.float().to(device)
        d0 = F.softplus(th[0])
        d1 = F.softplus(th[1])
        if self.use_prior and prior is not None:
            pr = prior.float()
            qp_norm = pr[:, 0].clamp(0.0, 1.0)
            is_screen = pr[:, 5].clamp(0.0, 1.0) if pr.shape[1] >= 6 else torch.zeros_like(qp_norm)
            has_p = pr[:, -1].clamp(0.0, 1.0)
            q_step = torch.pow(2.0, ((51.0 * qp_norm - 24.0) / 6.0).clamp(-4.0, 4.0))
            eff_q = torch.where(has_p > 0.5, q_step, torch.ones_like(q_step))
            scr_damp = 1.0 - 0.75 * torch.sigmoid(th[2]) * is_screen * has_p
            delta = (0.004 + d0 + d1 * eff_q) * scr_damp
            return delta.to(dtype=dtype).view(-1, 1, 1, 1)
        delta = (0.004 + d0 + d1).to(dtype=dtype)
        return delta.view(1, 1, 1, 1)

    @torch.no_grad()
    def bake_prior(self, prior_1d: torch.Tensor | None) -> None:
        if prior_1d is not None and prior_1d.ndim == 1:
            prior_1d = prior_1d.unsqueeze(0)
        d = self.compute_delta(prior_1d, torch.float32, self.theta.device)
        self.baked_delta.fill_(float(d.mean().item()))
        self.is_baked = True

    def forward(
        self,
        y_raw: torch.Tensor,
        x_lr: torch.Tensor,
        prior: torch.Tensor | None = None,
        qat_fp16: bool = False,
    ) -> torch.Tensor:
        x_pad = F.pad(x_lr, (1, 1, 1, 1), mode="replicate")
        y_max = F.max_pool2d(x_pad, kernel_size=3, stride=1)
        y_min = -F.max_pool2d(-x_pad, kernel_size=3, stride=1)
        out_hw = (y_raw.shape[-2], y_raw.shape[-1])
        lo_env = F.interpolate(y_min, size=out_hw, mode="nearest")
        hi_env = F.interpolate(y_max, size=out_hw, mode="nearest")
        delta = self.compute_delta(prior, y_raw.dtype, y_raw.device)
        if qat_fp16:
            delta = ste_fp16(delta)
        lo = lo_env - delta
        hi = hi_env + delta
        if self.training and not qat_fp16:
            temp = 60.0
            y_soft = lo + F.softplus(temp * (y_raw - lo)) / temp - F.softplus(temp * (y_raw - hi)) / temp
            # Leaky soft-clamp (0.02 slope outside [lo, hi]) prevents dead-gradient collapse
            # when high-fan-in tails (e.g., 720p Client-P 64->9) overshoot [lo, hi] early,
            # while eval / QAT / deploy remain exact hard clamp to [lo, hi] ∩ [0, 1].
            y_leaky = y_soft + 0.02 * (y_raw - y_soft)
            return y_leaky.clamp(-0.5, 1.5)
        return torch.minimum(torch.maximum(y_raw, lo), hi).clamp(0.0, 1.0)


class YGuidedChromaHead(nn.Module):
    """Dual-scale (x2 & x1.5) Luma-Guided Chroma Refinement Head for YUV420 (Idea #011 & #319).

    - 540p->1080p (scale=2.0): 480x270 UV aligns 1:1 with PixelUnshuffle(2) 270p Y features -> PixelShuffle(2) -> 960x540 UV
    - 720p->1080p (scale=1.5): 640x360 UV aligns 1:1 with PixelUnshuffle(2) 360p Y features -> PixelUnshuffle(2)->Conv->PixelShuffle(3) -> 960x540 UV
    """

    def __init__(self, y_feat_ch: int = 28, chroma_ch: int = 16, scale: float = 2.0):
        super().__init__()
        self.scale = float(scale)
        self.fuse = nn.Conv2d(2 + y_feat_ch, chroma_ch, kernel_size=1, padding=0, bias=True)
        self.dw = nn.Conv2d(chroma_ch, chroma_ch, kernel_size=3, padding=1, groups=chroma_ch, bias=True)
        self.pw = nn.Conv2d(chroma_ch, chroma_ch, kernel_size=1, padding=0, bias=True)
        self.act = nn.PReLU(chroma_ch, init=0.1)
        if abs(self.scale - 2.0) < 1e-4:
            self.out_conv = nn.Conv2d(chroma_ch, 2 * 4, kernel_size=3, padding=1, bias=True)
            self.unshuffle = nn.Identity()
            self.shuffle = nn.PixelShuffle(2)
        elif abs(self.scale - 1.5) < 1e-4:
            self.unshuffle = nn.PixelUnshuffle(2)
            self.out_conv = nn.Conv2d(chroma_ch * 4, 2 * 9, kernel_size=3, padding=1, bias=True)
            self.shuffle = nn.PixelShuffle(3)
        else:
            raise ValueError(f"Unsupported YGuidedChromaHead scale={scale}; expected 2.0 or 1.5")
        nn.init.normal_(self.out_conv.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, uv_lr: torch.Tensor, y_feat_lr: torch.Tensor) -> torch.Tensor:
        out_h = int(round(uv_lr.shape[-2] * self.scale))
        out_w = int(round(uv_lr.shape[-1] * self.scale))
        base = F.interpolate(uv_lr, size=(out_h, out_w), mode="bicubic", align_corners=False)
        h = self.act(self.fuse(torch.cat([uv_lr, y_feat_lr], dim=1)))
        h = self.act(self.pw(self.dw(h)))
        res = self.shuffle(self.out_conv(self.unshuffle(h)))
        return (base + res).clamp(0.0, 1.0)


class ClientSRNet(nn.Module):
    """Unified Client-L / Client-S / Client-P super-resolution network (v1, v2 & v3)."""

    def __init__(
        self,
        tier: str = "S",
        scale: float = 2.0,
        use_prior: bool = True,
        use_rep: bool = True,
        deploy: bool = False,
        arch_version: str = "v2",
    ):
        super().__init__()
        self.tier = tier.upper()
        self.scale = float(scale)
        self.use_prior = bool(use_prior)
        self.use_rep = bool(use_rep)
        self.deploy = bool(deploy)
        self.arch_version = str(arch_version).lower()
        if self.arch_version not in ("v1", "v2", "v3"):
            raise ValueError(f"Unsupported arch_version={arch_version}; expected 'v1', 'v2', or 'v3'")
        if self.arch_version == "v1":
            self.prior_dim = V1_PRIOR_DIM
        elif self.arch_version == "v2":
            self.prior_dim = V2_PRIOR_DIM
        else:
            self.prior_dim = V3_PRIOR_DIM

        self.is_hourglass = False

        if self.arch_version == "v3" and self.tier == "P":
            # === V3 Client-P 2-Scale Stateless Hourglass (Idea #305 & #317, ~23.9 GMACs, <420 KB) ===
            self.is_hourglass = True
            self.skip_mode = "bicubic"
            num_bases = 4 if self.use_prior else 0
            if abs(self.scale - 2.0) < 1e-4:
                self.unshuffle_factor = 1
                self.shuffle_factor = 2
                stem_ch, mid_ch, num_body, out_ch = 20, 48, 4, 4
                self.hg_has_up = True
                self.hg_has_hr_skip = False
            elif abs(self.scale - 1.5) < 1e-4:
                self.unshuffle_factor = 1
                self.shuffle_factor = 3
                # Idea #357: stem_ch=16 -> hg_unshuffle(s2) is exactly (B,64,H/2,W/2), 1:1 aligned
                # with the mid_ch=64 hourglass trunk so the 720p full-res shallow features can be
                # added back losslessly (0-parameter highway) instead of being thrown away.
                stem_ch, mid_ch, num_body, out_ch = 16, 64, 3, 9
                self.hg_has_up = False
                self.hg_has_hr_skip = True
                self.skip_hr = nn.Parameter(torch.tensor(0.25))
            else:
                raise ValueError(f"Unsupported scale {scale}; expected 2.0 or 1.5")

            self.channels = mid_ch
            self.num_body = num_body
            self.stem1 = HyperBasisConv3x3(1, stem_ch, num_bases=num_bases, basis_groups=1, norm_expand=True, deploy=deploy)
            self.act_stem1 = nn.PReLU(stem_ch, init=0.1)
            self.stem2 = HyperBasisConv3x3(stem_ch, stem_ch, num_bases=num_bases, basis_groups=stem_ch, norm_expand=True, deploy=deploy)
            self.act_stem2 = nn.PReLU(stem_ch, init=0.1)

            self.hg_unshuffle = nn.PixelUnshuffle(2)
            unshuf_ch = stem_ch * 4
            down_g = math.gcd(unshuf_ch, mid_ch)
            self.down = HyperBasisConv3x3(unshuf_ch, mid_ch, num_bases=num_bases, basis_groups=down_g, norm_expand=True, deploy=deploy)
            self.act_down = nn.PReLU(mid_ch, init=0.1)

            self.phase_period = 8  # Mod-16 alignment on 2x-unshuffled (270p / 360p) grid
            self.phase_bias = nn.Parameter(torch.zeros(1, mid_ch, self.phase_period, self.phase_period))

            self.body = nn.ModuleList(
                [
                    HyperBasisConv3x3(mid_ch, mid_ch, num_bases=num_bases, basis_groups=mid_ch, norm_expand=True, deploy=deploy)
                    for _ in range(num_body)
                ]
            )
            self.body_acts = nn.ModuleList([nn.PReLU(mid_ch, init=0.1) for _ in range(num_body)])
            self.skip_alpha = nn.Parameter(torch.tensor(0.25))
            self.skip_beta = nn.Parameter(torch.tensor(0.25))

            layer_ch = [stem_ch, stem_ch, mid_ch] + [mid_ch] * num_body
            if self.hg_has_up:
                self.up = HyperBasisConv3x3(mid_ch, unshuf_ch, num_bases=num_bases, basis_groups=down_g, norm_expand=True, deploy=deploy)
                self.act_up = nn.PReLU(unshuf_ch, init=0.1)
                self.hg_shuffle = nn.PixelShuffle(2)
                self.refine = HyperBasisConv3x3(stem_ch, stem_ch, num_bases=num_bases, basis_groups=stem_ch, norm_expand=True, deploy=deploy)
                self.act_refine = nn.PReLU(stem_ch, init=0.1)
                self.tail = HyperBasisConv3x3(stem_ch, out_ch, num_bases=num_bases, basis_groups=1, norm_expand=True, zero_init_out=True, deploy=deploy)
                layer_ch.extend([unshuf_ch, stem_ch, out_ch])
            else:
                self.tail = HyperBasisConv3x3(mid_ch, out_ch, num_bases=num_bases, basis_groups=1, norm_expand=True, zero_init_out=True, deploy=deploy)
                layer_ch.append(out_ch)

            self.shuffle = nn.PixelShuffle(self.shuffle_factor)
            self.prior_mod = V3HyperPriorModulator(layer_ch, prior_dim=V3_PRIOR_DIM, num_bases=4) if self.use_prior else None
            self.trust_clamp = QPAdaptiveTrustClamp(use_prior=self.use_prior, prior_dim=self.prior_dim)
            self.seam_lut = None
            return

        if abs(self.scale - 2.0) < 1e-4:
            if self.tier == "L":
                self.unshuffle_factor = 2
                self.shuffle_factor = 4
                channels = 16
                num_body = 4
                v1_rep = False
                v3_body_groups = 1
            elif self.tier == "S":
                self.unshuffle_factor = 2
                self.shuffle_factor = 4
                channels = 28
                num_body = 5
                v1_rep = use_rep
                v3_body_groups = 2
            elif self.tier == "P":
                self.unshuffle_factor = 1
                self.shuffle_factor = 2
                channels = 32
                num_body = 6
                v1_rep = use_rep
                v3_body_groups = 1
            else:
                raise ValueError(f"Unknown tier: {tier}")
        elif abs(self.scale - 1.5) < 1e-4:
            self.unshuffle_factor = 2
            self.shuffle_factor = 3
            if self.tier == "L":
                channels, num_body, v1_rep, v3_body_groups = 16, 4, False, 1
            elif self.tier == "S":
                channels, num_body, v1_rep, v3_body_groups = 24, 5, use_rep, 2
            elif self.tier == "P":
                channels, num_body, v1_rep, v3_body_groups = 32, 6, use_rep, 1
            else:
                raise ValueError(f"Unknown tier: {tier}")
        else:
            raise ValueError(f"Unsupported scale {scale}; expected 2.0 or 1.5")

        in_ch = self.unshuffle_factor * self.unshuffle_factor
        out_ch = self.shuffle_factor * self.shuffle_factor
        self.channels = channels
        self.num_body = num_body
        self.unshuffle = nn.PixelUnshuffle(self.unshuffle_factor) if self.unshuffle_factor > 1 else nn.Identity()
        self.shuffle = nn.PixelShuffle(self.shuffle_factor)

        if self.arch_version == "v1":
            self.head = nn.Conv2d(in_ch, channels, kernel_size=3, padding=1, bias=True)
            self.act = nn.LeakyReLU(0.1, inplace=True)
            blocks = [
                RepConv3x3(channels, use_laplacian=True, deploy=deploy)
                if v1_rep
                else nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
                for _ in range(num_body)
            ]
            self.body = nn.ModuleList(blocks)
            self.film = FiLMConditioner(channels, prior_dim=V1_PRIOR_DIM) if self.use_prior else None
            self.tail = nn.Conv2d(channels, out_ch, kernel_size=3, padding=1, bias=True)
            nn.init.normal_(self.tail.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.tail.bias)
            return

        if self.arch_version == "v2":
            # === V2 Architecture (Idea #003, #004, #005, #014) ===
            self.skip_mode = "bicubic"
            self.head = SeqECBRepConv2d(in_ch, channels, expand_ratio=2, zero_init_out=False, deploy=deploy)
            self.act_head = nn.PReLU(channels, init=0.1)

            self.phase_period = 8 // self.unshuffle_factor  # 4 for 270p (L/S), 8 for 540p (P)
            self.phase_bias = nn.Parameter(torch.zeros(1, channels, self.phase_period, self.phase_period))

            self.body = nn.ModuleList(
                [
                    SeqECBRepConv2d(channels, channels, expand_ratio=2, zero_init_out=False, deploy=deploy)
                    if use_rep
                    else nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
                    for _ in range(num_body)
                ]
            )
            self.body_acts = nn.ModuleList([nn.PReLU(channels, init=0.1) for _ in range(num_body)])
            self.prior_mod = PreFoldablePriorModulator(channels, 1 + num_body, prior_dim=V2_PRIOR_DIM) if self.use_prior else None
            self.tail = SeqECBRepConv2d(channels, out_ch, expand_ratio=2, zero_init_out=True, deploy=deploy)
            return

        # === V3 Architecture for Client-L & Client-S (Idea #301-#306, #316, #319) ===
        self.skip_mode = "bicubic"
        num_bases = 4 if self.use_prior else 0
        self.head = HyperBasisConv3x3(in_ch, channels, num_bases=num_bases, basis_groups=1, norm_expand=True, zero_init_out=False, deploy=deploy)
        self.act_head = nn.PReLU(channels, init=0.1)

        self.phase_period = 16 // self.unshuffle_factor  # 8 for 270p/360p (Mod-16 macroblock/CTU phase)
        self.phase_bias = nn.Parameter(torch.zeros(1, channels, self.phase_period, self.phase_period))

        self.body = nn.ModuleList(
            [
                HyperBasisConv3x3(channels, channels, num_bases=num_bases, basis_groups=v3_body_groups, norm_expand=True, zero_init_out=False, deploy=deploy)
                for _ in range(num_body)
            ]
        )
        self.body_acts = nn.ModuleList([nn.PReLU(channels, init=0.1) for _ in range(num_body)])
        self.skip_alpha = nn.Parameter(torch.tensor(0.25))
        self.skip_beta = nn.Parameter(torch.tensor(0.25))
        self.tail = HyperBasisConv3x3(channels, out_ch, num_bases=num_bases, basis_groups=1, norm_expand=True, zero_init_out=True, deploy=deploy)
        layer_ch = [channels] * (1 + num_body) + [out_ch]
        self.prior_mod = V3HyperPriorModulator(layer_ch, prior_dim=V3_PRIOR_DIM, num_bases=4) if self.use_prior else None
        self.seam_lut = Seam3DLUT(grid_size=17, seam_period=8) if self.tier in ("L", "S") else None
        self.trust_clamp = QPAdaptiveTrustClamp(use_prior=self.use_prior, prior_dim=self.prior_dim)

    def _tiled_phase_bias(self, h: int, w: int, d4_op: int | torch.Tensor | None = None, batch_size: int = 1, qat_fp16: bool = False) -> torch.Tensor:
        pb = ste_fp16(self.phase_bias) if qat_fp16 else self.phase_bias
        p = self.phase_period
        rep_h = (h + p - 1) // p
        rep_w = (w + p - 1) // p
        if d4_op is None or isinstance(d4_op, int):
            src = apply_d4_tensor(pb, d4_op) if isinstance(d4_op, int) else pb
            return src.repeat(1, 1, rep_h, rep_w)[:, :, :h, :w]
        pb_all = torch.stack([apply_d4_tensor(pb.squeeze(0), op_id) for op_id in range(8)], dim=0)
        return pb_all[(d4_op.long() & 7)].repeat(1, 1, rep_h, rep_w)[:, :, :h, :w]

    @staticmethod
    def _prelu_qat(act: nn.PReLU, x: torch.Tensor, qat_fp16: bool) -> torch.Tensor:
        if qat_fp16:
            y = F.prelu(x, ste_fp16(act.weight.to(dtype=x.dtype)))
            return ste_fp16(y)
        return act(x)

    def _all_v3_hyper_layers(self) -> list[HyperBasisConv3x3]:
        if self.is_hourglass:
            layers = [self.stem1, self.stem2, self.down, *list(self.body)]
            if self.hg_has_up:
                layers.extend([self.up, self.refine])
            layers.append(self.tail)
            return layers
        return [self.head, *list(self.body), self.tail]

    def precompute_step_wb(self) -> list[tuple[torch.Tensor, torch.Tensor] | None]:
        """Fold each V3 HyperBasisConv3x3 base SeqECB once per step for reuse across grad_accum chunks (#354)."""
        if self.arch_version != "v3":
            return []
        return [
            None if ly.base.deploy else ly.base._compute_equivalent_weight_bias()
            for ly in self._all_v3_hyper_layers()
        ]

    def forward(
        self,
        x: torch.Tensor,
        prior: torch.Tensor | None = None,
        qat_fp16: bool = False,
        d4_op: int | torch.Tensor | None = None,
        return_feat: bool = False,
        precomputed_base: torch.Tensor | None = None,
        precomputed_wbs: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        out_h = int(round(x.shape[-2] * self.scale))
        out_w = int(round(x.shape[-1] * self.scale))

        if self.arch_version == "v1":
            base = F.interpolate(x, size=(out_h, out_w), mode="bilinear", align_corners=False)
            feat = self.act(self.head(self.unshuffle(x)))
            for idx, blk in enumerate(self.body):
                feat = self.act(blk(feat))
                if self.film is not None and idx == len(self.body) // 2:
                    feat = self.film(feat, prior)
            res = self.shuffle(self.tail(feat))
            out = (base + res).clamp(0.0, 1.0)
            return (out, feat) if return_feat else out

        if self.arch_version == "v2":
            base = F.interpolate(x, size=(out_h, out_w), mode="bicubic", align_corners=False)
            gammas = betas = None
            if self.prior_mod is not None and prior is not None:
                gammas, betas = self.prior_mod.predict_params(prior)

            feat = self.head(self.unshuffle(x))
            feat = feat + self._tiled_phase_bias(feat.shape[-2], feat.shape[-1])
            if gammas is not None and betas is not None:
                feat = feat * (1.0 + gammas[:, 0].unsqueeze(-1).unsqueeze(-1)) + betas[:, 0].unsqueeze(-1).unsqueeze(-1)
            feat = self.act_head(feat)

            for idx, (blk, act) in enumerate(zip(self.body, self.body_acts)):
                feat = blk(feat)
                if gammas is not None and betas is not None:
                    l_idx = idx + 1
                    feat = feat * (1.0 + gammas[:, l_idx].unsqueeze(-1).unsqueeze(-1)) + betas[:, l_idx].unsqueeze(-1).unsqueeze(-1)
                feat = act(feat)

            res = self.shuffle(self.tail(feat))
            out = (base + res).clamp(0.0, 1.0)
            return (out, feat) if return_feat else out

        # === V3 Forward ===
        if precomputed_base is not None:
            base = precomputed_base.to(dtype=x.dtype)
        else:
            base = F.interpolate(x, size=(out_h, out_w), mode="bicubic", align_corners=False)
        layer_params = None
        if self.prior_mod is not None and prior is not None:
            layer_params = self.prior_mod.predict_layer_params(prior)

        def _wb(i: int) -> tuple[torch.Tensor, torch.Tensor] | None:
            if precomputed_wbs is not None and i < len(precomputed_wbs):
                return precomputed_wbs[i]
            return None

        if self.is_hourglass:
            def _lp(i: int):
                return layer_params[i] if layer_params is not None else (None, None, None)

            s1 = self._prelu_qat(self.act_stem1, self.stem1(x, *_lp(0), qat_fp16=qat_fp16, precomputed_wb=_wb(0)), qat_fp16)
            s2 = self._prelu_qat(self.act_stem2, self.stem2(s1, *_lp(1), qat_fp16=qat_fp16, precomputed_wb=_wb(1)), qat_fp16)
            f_down = self.down(self.hg_unshuffle(s2), *_lp(2), qat_fp16=qat_fp16, precomputed_wb=_wb(2))
            f_down = f_down + self._tiled_phase_bias(f_down.shape[-2], f_down.shape[-1], d4_op=d4_op, batch_size=x.shape[0], qat_fp16=qat_fp16)
            f_head = self._prelu_qat(self.act_down, f_down, qat_fp16)

            feat = f_head
            f_mid = f_head
            mid_idx = len(self.body) // 2
            for idx, (blk, act) in enumerate(zip(self.body, self.body_acts)):
                feat = self._prelu_qat(act, blk(feat, *_lp(3 + idx), qat_fp16=qat_fp16, precomputed_wb=_wb(3 + idx)), qat_fp16)
                if idx == mid_idx:
                    f_mid = feat

            sa = ste_fp16(self.skip_alpha) if qat_fp16 else self.skip_alpha
            sb = ste_fp16(self.skip_beta) if qat_fp16 else self.skip_beta
            feat = feat + sa.to(feat.dtype) * f_head + sb.to(feat.dtype) * f_mid
            if qat_fp16:
                feat = ste_fp16(feat)
            bottleneck_feat = feat

            p_off = 3 + len(self.body)
            if self.hg_has_up:
                f_up = self._prelu_qat(self.act_up, self.up(feat, *_lp(p_off), qat_fp16=qat_fp16, precomputed_wb=_wb(p_off)), qat_fp16)
                f_hi = self.hg_shuffle(f_up) + s2
                if qat_fp16:
                    f_hi = ste_fp16(f_hi)
                f_ref = self._prelu_qat(self.act_refine, self.refine(f_hi, *_lp(p_off + 1), qat_fp16=qat_fp16, precomputed_wb=_wb(p_off + 1)), qat_fp16)
                res = self.shuffle(self.tail(f_ref, *_lp(p_off + 2), qat_fp16=qat_fp16, precomputed_wb=_wb(p_off + 2)))
            else:
                if getattr(self, "hg_has_hr_skip", False):
                    # Idea #357: lossless 720p shallow-feature highway, reuses op_span_combine at deploy.
                    sh = ste_fp16(self.skip_hr) if qat_fp16 else self.skip_hr
                    feat = feat + sh.to(feat.dtype) * self.hg_unshuffle(s2)
                    if qat_fp16:
                        feat = ste_fp16(feat)
                res = self.shuffle(self.tail(feat, *_lp(p_off), qat_fp16=qat_fp16, precomputed_wb=_wb(p_off)))
        else:
            def _lp(i: int):
                return layer_params[i] if layer_params is not None else (None, None, None)

            f_h = self.head(self.unshuffle(x), *_lp(0), qat_fp16=qat_fp16, precomputed_wb=_wb(0))
            f_h = f_h + self._tiled_phase_bias(f_h.shape[-2], f_h.shape[-1], d4_op=d4_op, batch_size=x.shape[0], qat_fp16=qat_fp16)
            f_head = self._prelu_qat(self.act_head, f_h, qat_fp16)

            feat = f_head
            f_mid = f_head
            mid_idx = len(self.body) // 2
            for idx, (blk, act) in enumerate(zip(self.body, self.body_acts)):
                feat = self._prelu_qat(act, blk(feat, *_lp(1 + idx), qat_fp16=qat_fp16, precomputed_wb=_wb(1 + idx)), qat_fp16)
                if idx == mid_idx:
                    f_mid = feat

            sa = ste_fp16(self.skip_alpha) if qat_fp16 else self.skip_alpha
            sb = ste_fp16(self.skip_beta) if qat_fp16 else self.skip_beta
            feat = feat + sa.to(feat.dtype) * f_head + sb.to(feat.dtype) * f_mid
            if qat_fp16:
                feat = ste_fp16(feat)
            bottleneck_feat = feat
            res = self.shuffle(self.tail(feat, *_lp(1 + len(self.body)), qat_fp16=qat_fp16, precomputed_wb=_wb(1 + len(self.body))))
            if self.seam_lut is not None:
                lut_gain = (
                    self.prior_mod.predict_lut_gain(prior)
                    if (self.prior_mod is not None and prior is not None)
                    else None
                )
                res = res + self.seam_lut(x, (out_h, out_w), d4_op=d4_op, qat_fp16=qat_fp16, lut_gain=lut_gain)

        y_raw = base + res
        out = self.trust_clamp(y_raw, x, prior=prior, qat_fp16=qat_fp16)
        return (out, bottleneck_feat) if return_feat else out

    def equivalent_kernel_l2_reg(
        self, precomputed_wbs: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None
    ) -> torch.Tensor:
        """Compute L2 norm across equivalent deployable Conv2d kernels (Idea #019 / #301)."""
        if self.arch_version == "v1":
            return torch.zeros((), device=next(self.parameters()).device)
        if self.arch_version == "v2":
            regs = [self.head.equivalent_weight_l2(), self.tail.equivalent_weight_l2()]
            for blk in self.body:
                if isinstance(blk, SeqECBRepConv2d):
                    regs.append(blk.equivalent_weight_l2())
            return sum(regs) / max(1, len(regs))
        regs = [
            ly.equivalent_weight_l2(
                precomputed_wb=(precomputed_wbs[i] if precomputed_wbs is not None and i < len(precomputed_wbs) else None)
            )
            for i, ly in enumerate(self._all_v3_hyper_layers())
        ]
        return sum(regs) / max(1, len(regs))

    def basis_orthogonality_loss(self) -> torch.Tensor:
        """Cosine orthogonality regularizer across K=4 dynamic kernel bases (Idea #303)."""
        if self.arch_version != "v3" or not self.use_prior:
            return torch.zeros((), device=next(self.parameters()).device)
        losses = [ly.basis_orthogonality_loss() for ly in self._all_v3_hyper_layers() if ly.num_bases >= 2]
        if not losses:
            return torch.zeros((), device=next(self.parameters()).device)
        return sum(losses) / len(losses)

    def basis_diversity_loss(
        self, layer_params: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None
    ) -> torch.Tensor:
        """In-batch cross-prior variance floor on alpha_k(p) to stop basis collapse (Idea #356.3)."""
        if self.arch_version != "v3" or not self.use_prior or layer_params is None:
            return torch.zeros((), device=next(self.parameters()).device)
        losses = []
        for alphas, _, _ in layer_params:
            if alphas.shape[0] < 2 or alphas.shape[1] < 2:
                continue
            var = alphas.float().var(dim=0)
            losses.append(F.relu(0.004 - var).mean())
        if not losses:
            return torch.zeros((), device=next(self.parameters()).device)
        return sum(losses) / len(losses)

    @torch.no_grad()
    def switch_to_deploy(self) -> None:
        if self.arch_version == "v3":
            for ly in self._all_v3_hyper_layers():
                ly.switch_to_deploy()
            self.deploy = True
            return
        if isinstance(self.head, (RepConv3x3, SeqECBRepConv2d)):
            self.head.switch_to_deploy()
        for blk in self.body:
            if isinstance(blk, (RepConv3x3, SeqECBRepConv2d)):
                blk.switch_to_deploy()
        if hasattr(self, "tail") and isinstance(self.tail, (RepConv3x3, SeqECBRepConv2d)):
            self.tail.switch_to_deploy()
        self.deploy = True

    @torch.no_grad()
    def bake_segment_prior(self, prior_1d: torch.Tensor | None) -> "ClientSRNet":
        """Pre-fold a 2s-segment 5D/7D prior directly into deployed Conv3x3 weights on CPU.

        Returns a self-contained static deploy copy with `use_prior=False` (0.00 GPU FiLM/basis MACs/frame).
        """
        if self.arch_version not in ("v2", "v3"):
            raise NotImplementedError("bake_segment_prior is supported for arch_version in ('v2', 'v3') only")
        if prior_1d is not None and prior_1d.ndim == 1:
            prior_1d = prior_1d.unsqueeze(0)

        baked = copy.deepcopy(self)
        baked.eval()
        baked.switch_to_deploy()

        if self.arch_version == "v2":
            if baked.prior_mod is not None and prior_1d is not None:
                gammas, betas = baked.prior_mod.predict_params(prior_1d.to(baked.phase_bias.device, baked.phase_bias.dtype))
                g0 = gammas[0, 0]
                b0 = betas[0, 0]
                s0 = (1.0 + g0).view(-1, 1, 1, 1)
                baked.head.reparam_conv.weight.data.mul_(s0)
                baked.head.reparam_conv.bias.data.mul_(1.0 + g0).add_(b0)
                baked.phase_bias.data.mul_((1.0 + g0).view(1, -1, 1, 1))

                for idx, blk in enumerate(baked.body):
                    g_l = gammas[0, idx + 1]
                    b_l = betas[0, idx + 1]
                    s_l = (1.0 + g_l).view(-1, 1, 1, 1)
                    blk.reparam_conv.weight.data.mul_(s_l)
                    blk.reparam_conv.bias.data.mul_(1.0 + g_l).add_(b_l)

                baked.prior_mod = None
            baked.use_prior = False
            return baked

        # V3 CPU baking: collapse HyperBasisConv3x3 + V3HyperPriorModulator + QPAdaptiveTrustClamp
        layers = baked._all_v3_hyper_layers()
        if baked.prior_mod is not None and prior_1d is not None:
            dev = baked.phase_bias.device
            pr_in = prior_1d.to(device=dev, dtype=baked.phase_bias.dtype)
            layer_params = baked.prior_mod.predict_layer_params(pr_in)
            for ly, (al, ga, bt) in zip(layers, layer_params):
                ly.bake_with_params(al[0], ga[0], bt[0])
            # Idea #358: bake g_lut(p) into the static 8x8 seam mask (0 shader changes, +2 B blob)
            if baked.seam_lut is not None:
                g = baked.prior_mod.predict_lut_gain(pr_in)
                baked.seam_lut.seam_mask.data.mul_(g.mean().to(baked.seam_lut.seam_mask.dtype).view(1))
            baked.prior_mod = None
        else:
            for ly in layers:
                ly.bake_with_params(None, None, None)
        baked.trust_clamp.bake_prior(prior_1d)
        baked.use_prior = False
        return baked

    @torch.no_grad()
    def forward_baked_fp16(self, x: torch.Tensor, prior: torch.Tensor | None = None) -> torch.Tensor:
        """Exact mobile-shader simulation: fold + segment-bake + IEEE-754 float16 inference (Idea #319)."""
        out, _ = self.forward_baked_fp16_feat(x, prior)
        return out

    @torch.no_grad()
    def forward_baked_fp16_feat(
        self, x: torch.Tensor, prior: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Idea #364.2: baked-FP16 deploy-regime forward that ALSO returns the bottleneck luma feature."""
        if self.arch_version not in ("v2", "v3"):
            y = self(x.half(), prior.half() if prior is not None else None)
            return y.float(), torch.zeros((), device=x.device)
        if prior is None or not self.use_prior:
            baked = self.bake_segment_prior(None).half() if self.arch_version == "v3" else copy.deepcopy(self).eval()
            if self.arch_version == "v2":
                baked.switch_to_deploy()
                baked = baked.half()
            out, feat = baked(x.half(), None, return_feat=True)
            return out.float(), feat.float()

        b = x.shape[0]
        out = torch.empty(
            (b, 1, int(round(x.shape[-2] * self.scale)), int(round(x.shape[-1] * self.scale))),
            dtype=torch.float32,
            device=x.device,
        )
        feat_out: torch.Tensor | None = None
        pr_cpu = prior.detach().cpu()
        groups: dict[tuple[float, ...], list[int]] = {}
        for i in range(b):
            key = tuple(round(float(v), 5) for v in pr_cpu[i].tolist())
            groups.setdefault(key, []).append(i)
        for idxs in groups.values():
            baked = self.bake_segment_prior(prior[idxs[0]]).half()
            idx_t = torch.tensor(idxs, dtype=torch.long, device=x.device)
            o, f = baked(x[idx_t].half(), None, return_feat=True)
            out[idx_t] = o.float()
            if feat_out is None:
                feat_out = torch.empty(
                    (b, f.shape[1], f.shape[2], f.shape[3]), dtype=torch.float32, device=x.device
                )
            feat_out[idx_t] = f.float()
        return out, feat_out if feat_out is not None else torch.zeros((), device=x.device)


def build_client_model(
    tier: str = "S",
    scale: float = 2.0,
    use_prior: bool = True,
    arch_version: str = "v2",
) -> ClientSRNet:
    return ClientSRNet(
        tier=tier,
        scale=scale,
        use_prior=use_prior,
        use_rep=True,
        deploy=False,
        arch_version=arch_version,
    )


def count_params_and_macs(model: nn.Module, input_h: int = 540, input_w: int = 960) -> dict[str, float]:
    """Compute exact parameter count and single-frame GMACs across Conv2d layers."""
    total_params = sum(p.numel() for p in model.parameters())
    mac_model = model
    if isinstance(model, ClientSRNet) and not model.deploy and model.arch_version in ("v2", "v3"):
        mac_model = copy.deepcopy(model).eval()
        mac_model.switch_to_deploy()

    total_macs = 0
    hooks = []

    def _hook(mod: nn.Conv2d, inp: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
        nonlocal total_macs
        _, _, oh, ow = out.shape
        k_h, k_w = mod.kernel_size
        macs = oh * ow * mod.in_channels * mod.out_channels * k_h * k_w // mod.groups
        total_macs += macs

    for m in mac_model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(_hook))

    was_training = mac_model.training
    mac_model.eval()
    device = next(mac_model.parameters()).device
    dtype = next(mac_model.parameters()).dtype
    prior_dim = int(getattr(mac_model, "prior_dim", V1_PRIOR_DIM))
    dummy = torch.zeros(1, 1, input_h, input_w, device=device, dtype=dtype)
    with torch.no_grad():
        try:
            mac_model(dummy, torch.zeros(1, prior_dim, device=device, dtype=dtype))
        except TypeError:
            mac_model(dummy)

    for h in hooks:
        h.remove()
    if was_training:
        mac_model.train()

    return {
        "params": int(total_params),
        "params_k": round(total_params / 1e3, 2),
        "gmacs": round(total_macs / 1e9, 3),
    }
