# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels for mhc_pre and mhc_post on XPU.

Replaces the eager PyTorch implementation (218 kernel launches per mhc_pre call)
with fused Triton kernels (3 launches total for mhc_pre, 1 for mhc_post).

Algorithm for mhc_pre:
  1. GEMM: res_2d @ fn.t()  →  kept as torch.mm (1 launch)
  2. Triton fused kernel: sqrsum + rms_norm + sigmoid + sinkhorn + einsum (1 launch)

DSv4 constants: hc=4, H=4096, hc3=24, hcH=16384, sinkhorn_repeat=20
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _mhc_pre_fused_kernel(
    # Inputs
    res_ptr,         # [N, hc, H] fp32
    gemm_out_ptr,    # [N, hc3] fp32
    hc_scale_ptr,    # [3] fp32
    hc_base_ptr,     # [hc3] fp32
    # Outputs
    post_mix_ptr,    # [N, hc] fp32
    comb_mix_ptr,    # [N, hc*hc] fp32
    # Intermediate buffer for pre_mix scalars (for einsum kernel)
    pre_mix_buf_ptr, # [N, hc] fp32
    # Strides
    stride_res_n,    # = hc * H
    stride_gemm_n,   # = hc3
    stride_post_n,   # = hc
    stride_comb_n,   # = hc * hc
    # Scalar params
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    # Constexpr
    HC: tl.constexpr,           # 4
    HC3: tl.constexpr,          # 24
    HCH: tl.constexpr,         # hc * H = 16384
    SINKHORN_REPEAT: tl.constexpr,  # 20
    BLOCK_H: tl.constexpr,     # tile size for sqrsum reduction
):
    """Fused post-GEMM kernel for mhc_pre (sinkhorn part).

    Each program handles one token:
      - compute sqrsum (sum of squares over hcH elements)
      - rms normalize gemm_out
      - sigmoid + sinkhorn normalization
      - store post_mix, comb_mix, pre_mix
    In between: all scalar/small-matrix ops (rms, sigmoid, sinkhorn) in registers.
    """
    pid_n = tl.program_id(0)

    # --- Load constants ---
    scale0 = tl.load(hc_scale_ptr).to(tl.float32)
    scale1 = tl.load(hc_scale_ptr + 1).to(tl.float32)
    scale2 = tl.load(hc_scale_ptr + 2).to(tl.float32)

    # --- Pass 1: compute sqrsum = sum(res_2d^2) ---
    sqrsum = tl.zeros([], dtype=tl.float32)
    res_base = pid_n * stride_res_n
    for off in range(0, HCH, BLOCK_H):
        h_offs = tl.arange(0, BLOCK_H) + off
        mask = h_offs < HCH
        vals = tl.load(res_ptr + res_base + h_offs, mask=mask, other=0.0)
        sqrsum += tl.sum(vals * vals)

    # --- Compute RMS ---
    rms = tl.math.rsqrt(sqrsum / HCH + rms_eps)

    # --- Load gemm_out sections and apply rms ---
    pre_offs = tl.arange(0, 4)
    post_offs = tl.arange(0, 4) + HC
    cm_offs = tl.arange(0, 16) + 2 * HC

    pre_raw = tl.load(gemm_out_ptr + pid_n * stride_gemm_n + pre_offs) * rms
    post_raw = tl.load(gemm_out_ptr + pid_n * stride_gemm_n + post_offs) * rms
    cm_raw = tl.load(gemm_out_ptr + pid_n * stride_gemm_n + cm_offs) * rms

    # --- Sigmoid + scaling ---
    pre_base = tl.load(hc_base_ptr + pre_offs)
    pre_mix = tl.sigmoid(pre_raw * scale0 + pre_base) + hc_pre_eps  # [4]

    post_base = tl.load(hc_base_ptr + post_offs)
    post_mix = tl.sigmoid(post_raw * scale1 + post_base) * hc_post_mult_value  # [4]

    cm_base = tl.load(hc_base_ptr + cm_offs)
    cm = cm_raw * scale2 + cm_base  # [16]

    # --- Softmax along rows (dim=-1 in [4,4]) ---
    cm_2d = tl.reshape(cm, (HC, HC))
    row_max = tl.max(cm_2d, axis=1)  # [4]
    cm_2d = cm_2d - row_max[:, None]
    cm_2d = tl.exp(cm_2d)
    row_sum = tl.sum(cm_2d, axis=1)  # [4]
    cm_2d = cm_2d / row_sum[:, None]
    cm_2d = cm_2d + hc_sinkhorn_eps

    # --- Sinkhorn normalization ---
    col_sum = tl.sum(cm_2d, axis=0)  # [4]
    cm_2d = cm_2d / (col_sum[None, :] + hc_sinkhorn_eps)

    for _ in range(SINKHORN_REPEAT - 1):
        r_sum = tl.sum(cm_2d, axis=1)
        cm_2d = cm_2d / (r_sum[:, None] + hc_sinkhorn_eps)
        c_sum = tl.sum(cm_2d, axis=0)
        cm_2d = cm_2d / (c_sum[None, :] + hc_sinkhorn_eps)

    # --- Store post_mix and comb_mix ---
    tl.store(post_mix_ptr + pid_n * stride_post_n + tl.arange(0, 4), post_mix)
    comb_flat = tl.reshape(cm_2d, (HC * HC,))
    tl.store(comb_mix_ptr + pid_n * stride_comb_n + tl.arange(0, 16), comb_flat)

    # --- Store pre_mix to buffer for Pass 2 ---
    tl.store(pre_mix_buf_ptr + pid_n * HC + tl.arange(0, 4), pre_mix)


@triton.jit
def _mhc_pre_einsum_kernel(
    # Inputs
    res_ptr,         # [N, hc, H] fp32
    pre_mix_ptr,     # [N, hc] fp32
    # Output
    layer_input_ptr, # [N, H] bf16
    # Dims
    H: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Compute layer_input[n,h] = sum_j(pre_mix[n,j] * res[n,j,h]).

    Grid: (N, ceil(H/BLOCK_H))
    """
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    # Load pre_mix scalars for this token
    pm0 = tl.load(pre_mix_ptr + pid_n * HC + 0).to(tl.float32)
    pm1 = tl.load(pre_mix_ptr + pid_n * HC + 1).to(tl.float32)
    pm2 = tl.load(pre_mix_ptr + pid_n * HC + 2).to(tl.float32)
    pm3 = tl.load(pre_mix_ptr + pid_n * HC + 3).to(tl.float32)

    res_base = pid_n * HC * H

    # Accumulate: layer_input[h] = pm0*res[0,h] + pm1*res[1,h] + ...
    r0 = tl.load(res_ptr + res_base + 0 * H + h_offs, mask=h_mask, other=0.0)
    r1 = tl.load(res_ptr + res_base + 1 * H + h_offs, mask=h_mask, other=0.0)
    r2 = tl.load(res_ptr + res_base + 2 * H + h_offs, mask=h_mask, other=0.0)
    r3 = tl.load(res_ptr + res_base + 3 * H + h_offs, mask=h_mask, other=0.0)

    acc = pm0 * r0 + pm1 * r1 + pm2 * r2 + pm3 * r3

    tl.store(layer_input_ptr + pid_n * H + h_offs, acc.to(tl.bfloat16), mask=h_mask)


@triton.jit
def _mhc_post_fused_kernel(
    # Inputs
    x_ptr,           # [N, H] bf16 - layer output
    res_ptr,         # [N, hc, H] bf16 - residual
    post_mix_ptr,    # [N, hc] fp32
    comb_mix_ptr,    # [N, hc*hc] fp32
    # Output
    out_ptr,         # [N, hc, H] bf16
    # Strides
    stride_x_n,      # = H
    stride_res_n,    # = hc * H
    stride_post_n,   # = hc
    stride_comb_n,   # = hc * hc
    stride_out_n,    # = hc * H
    # Constexpr
    H: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Fused mhc_post kernel.

    out[n, hco, h] = post[n, hco] * x[n, h]
                   + sum_hci(comb[n, hci, hco] * residual[n, hci, h])

    Grid: (N, ceil(H/BLOCK_H))
    Each program handles one token and one H-tile for all hc output heads.
    HC=4 is unrolled explicitly to avoid tensor indexing issues.
    """
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    # Load post_mix scalars
    post_base = pid_n * stride_post_n
    p0 = tl.load(post_mix_ptr + post_base + 0).to(tl.float32)
    p1 = tl.load(post_mix_ptr + post_base + 1).to(tl.float32)
    p2 = tl.load(post_mix_ptr + post_base + 2).to(tl.float32)
    p3 = tl.load(post_mix_ptr + post_base + 3).to(tl.float32)

    # Load comb_mix[pid_n, :16] as flat array, then extract scalars
    # comb layout: [hci=0..3, hco=0..3] flattened row-major
    # comb[hci, hco] = comb_flat[hci * HC + hco]
    comb_base = pid_n * stride_comb_n
    c00 = tl.load(comb_mix_ptr + comb_base + 0).to(tl.float32)
    c01 = tl.load(comb_mix_ptr + comb_base + 1).to(tl.float32)
    c02 = tl.load(comb_mix_ptr + comb_base + 2).to(tl.float32)
    c03 = tl.load(comb_mix_ptr + comb_base + 3).to(tl.float32)
    c10 = tl.load(comb_mix_ptr + comb_base + 4).to(tl.float32)
    c11 = tl.load(comb_mix_ptr + comb_base + 5).to(tl.float32)
    c12 = tl.load(comb_mix_ptr + comb_base + 6).to(tl.float32)
    c13 = tl.load(comb_mix_ptr + comb_base + 7).to(tl.float32)
    c20 = tl.load(comb_mix_ptr + comb_base + 8).to(tl.float32)
    c21 = tl.load(comb_mix_ptr + comb_base + 9).to(tl.float32)
    c22 = tl.load(comb_mix_ptr + comb_base + 10).to(tl.float32)
    c23 = tl.load(comb_mix_ptr + comb_base + 11).to(tl.float32)
    c30 = tl.load(comb_mix_ptr + comb_base + 12).to(tl.float32)
    c31 = tl.load(comb_mix_ptr + comb_base + 13).to(tl.float32)
    c32 = tl.load(comb_mix_ptr + comb_base + 14).to(tl.float32)
    c33 = tl.load(comb_mix_ptr + comb_base + 15).to(tl.float32)

    # Load x[pid_n, h_offs]
    x_vals = tl.load(x_ptr + pid_n * stride_x_n + h_offs, mask=h_mask, other=0.0)
    x_f32 = x_vals.to(tl.float32)

    # Load residual[pid_n, j, h_offs] for j=0..3
    res_base = pid_n * stride_res_n
    r0 = tl.load(res_ptr + res_base + 0 * H + h_offs, mask=h_mask, other=0.0).to(tl.float32)
    r1 = tl.load(res_ptr + res_base + 1 * H + h_offs, mask=h_mask, other=0.0).to(tl.float32)
    r2 = tl.load(res_ptr + res_base + 2 * H + h_offs, mask=h_mask, other=0.0).to(tl.float32)
    r3 = tl.load(res_ptr + res_base + 3 * H + h_offs, mask=h_mask, other=0.0).to(tl.float32)

    # out[hco=0] = p0*x + c00*r0 + c10*r1 + c20*r2 + c30*r3
    out0 = p0 * x_f32 + c00 * r0 + c10 * r1 + c20 * r2 + c30 * r3
    # out[hco=1] = p1*x + c01*r0 + c11*r1 + c21*r2 + c31*r3
    out1 = p1 * x_f32 + c01 * r0 + c11 * r1 + c21 * r2 + c31 * r3
    # out[hco=2] = p2*x + c02*r0 + c12*r1 + c22*r2 + c32*r3
    out2 = p2 * x_f32 + c02 * r0 + c12 * r1 + c22 * r2 + c32 * r3
    # out[hco=3] = p3*x + c03*r0 + c13*r1 + c23*r2 + c33*r3
    out3 = p3 * x_f32 + c03 * r0 + c13 * r1 + c23 * r2 + c33 * r3

    # Store [hco, h_offs] as bf16
    out_base = pid_n * stride_out_n
    tl.store(out_ptr + out_base + 0 * H + h_offs, out0.to(tl.bfloat16), mask=h_mask)
    tl.store(out_ptr + out_base + 1 * H + h_offs, out1.to(tl.bfloat16), mask=h_mask)
    tl.store(out_ptr + out_base + 2 * H + h_offs, out2.to(tl.bfloat16), mask=h_mask)
    tl.store(out_ptr + out_base + 3 * H + h_offs, out3.to(tl.bfloat16), mask=h_mask)


# --- Python wrappers ---


def mhc_pre_xpu_triton(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Triton-fused mhc_pre: 2 kernel launches instead of 218 eager ops.

    Launch 1: torch.mm (GEMM)
    Launch 2: Triton fused (sqrsum + rms + sigmoid + sinkhorn + einsum)
    """
    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32

    hc = residual.shape[-2]
    H = residual.shape[-1]
    hc3 = hc * 2 + hc * hc
    hcH = hc * H

    outer_shape = residual.shape[:-2]
    # Avoid .item() sync - compute N from Python ints
    N = 1
    for s in outer_shape:
        N *= s

    # Cast to fp32 and reshape
    res_fp32 = residual.reshape(N, hc, H).to(torch.float32)
    res_2d = res_fp32.reshape(N, hcH)

    # Launch 1: GEMM  [N, hcH] × [hcH, hc3] → [N, hc3]
    gemm_out = res_2d @ fn.t()

    # Allocate outputs
    post_mix = torch.empty(N, hc, dtype=torch.float32, device=residual.device)
    comb_mix = torch.empty(N, hc * hc, dtype=torch.float32, device=residual.device)
    layer_input = torch.empty(N, H, dtype=torch.bfloat16, device=residual.device)
    pre_mix_buf = torch.empty(N, hc, dtype=torch.float32, device=residual.device)

    # Launch 2: fused sinkhorn kernel (sqrsum + rms + sigmoid + sinkhorn)
    BLOCK_H = 1024
    grid_sinkhorn = (N,)

    _mhc_pre_fused_kernel[grid_sinkhorn](
        res_fp32, gemm_out, hc_scale, hc_base,
        post_mix, comb_mix, pre_mix_buf,
        # Strides
        hc * H,   # stride_res_n
        hc3,      # stride_gemm_n
        hc,       # stride_post_n
        hc * hc,  # stride_comb_n
        # Scalar params
        rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
        # Constexpr
        HC=hc, HC3=hc3, HCH=hcH,
        SINKHORN_REPEAT=sinkhorn_repeat,
        BLOCK_H=BLOCK_H,
    )

    # Launch 3: einsum kernel (layer_input = sum_j pre_mix[j] * res[j,h])
    BLOCK_H_EINSUM = 1024
    grid_einsum = (N, triton.cdiv(H, BLOCK_H_EINSUM))

    _mhc_pre_einsum_kernel[grid_einsum](
        res_fp32, pre_mix_buf, layer_input,
        H=H, HC=hc, BLOCK_H=BLOCK_H_EINSUM,
    )

    return (
        post_mix.reshape(*outer_shape, hc, 1),
        comb_mix.reshape(*outer_shape, hc, hc),
        layer_input.reshape(*outer_shape, H),
    )


def mhc_post_xpu_triton(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    """Triton-fused mhc_post: 1 kernel launch instead of ~20 eager ops."""
    assert x.dtype == torch.bfloat16
    assert residual.dtype == torch.bfloat16

    hc = residual.shape[-2]
    H = residual.shape[-1]
    N_shape = residual.shape[:-2]
    N = 1
    for s in N_shape:
        N *= s

    x_flat = x.reshape(N, H)
    res_flat = residual.reshape(N, hc, H)
    post_flat = post_layer_mix.reshape(N, hc).to(torch.float32)
    comb_flat = comb_res_mix.reshape(N, hc * hc).to(torch.float32)

    out = torch.empty(N, hc, H, dtype=torch.bfloat16, device=residual.device)

    BLOCK_H = 1024
    grid = (N, triton.cdiv(H, BLOCK_H))

    _mhc_post_fused_kernel[grid](
        x_flat, res_flat, post_flat, comb_flat, out,
        # Strides
        H,        # stride_x_n
        hc * H,   # stride_res_n
        hc,       # stride_post_n
        hc * hc,  # stride_comb_n
        hc * H,   # stride_out_n
        # Constexpr
        H=H, HC=hc, BLOCK_H=BLOCK_H,
    )

    return out.reshape(*N_shape, hc, H)
