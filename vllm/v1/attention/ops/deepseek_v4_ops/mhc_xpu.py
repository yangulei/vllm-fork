# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XPU PyTorch implementations of mhc_pre and mhc_post.

Tilelang is CUDA-only. On XPU we run the same algorithm in pure PyTorch,
verified against an explicit per-token reference in
`dev-tools/triage/v4_xpu/test_phase_i_mhc.py`.
"""

from __future__ import annotations

import torch


def mhc_pre_xpu_torch(
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
    """Returns (post_mix [..., hc, 1] fp32, comb_mix [..., hc, hc] fp32,
    layer_input [..., H] bf16)."""
    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32

    hc = residual.shape[-2]
    H = residual.shape[-1]
    hc3 = hc * 2 + hc * hc
    hcH = hc * H
    assert fn.shape == (hc3, hcH)
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (hc3,)

    outer_shape = residual.shape[:-2]
    N = int(torch.tensor(outer_shape).prod().item()) if outer_shape else 1

    res_fp32 = residual.reshape(N, hc, H).to(torch.float32)
    res_2d = res_fp32.reshape(N, hcH)

    gemm_out = res_2d @ fn.t()
    sqrsum = (res_2d * res_2d).sum(dim=-1)
    rms = torch.rsqrt(sqrsum / hcH + rms_eps)
    mixes = gemm_out * rms.unsqueeze(-1)

    pre_raw = mixes[:, :hc]
    post_raw = mixes[:, hc : 2 * hc]
    cm_raw = mixes[:, 2 * hc :].reshape(N, hc, hc)

    pre_mix = torch.sigmoid(pre_raw * hc_scale[0] + hc_base[:hc]) + hc_pre_eps
    post_mix = (
        torch.sigmoid(post_raw * hc_scale[1] + hc_base[hc : 2 * hc])
        * hc_post_mult_value
    )
    cm = cm_raw * hc_scale[2] + hc_base[2 * hc :].reshape(hc, hc)

    cm = torch.softmax(cm, dim=-1) + hc_sinkhorn_eps
    cm = cm / (cm.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        cm = cm / (cm.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
        cm = cm / (cm.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)

    layer_input = torch.einsum("nj,njh->nh", pre_mix, res_fp32).to(torch.bfloat16)

    return (
        post_mix.reshape(*outer_shape, hc, 1),
        cm.reshape(*outer_shape, hc, hc),
        layer_input.reshape(*outer_shape, H),
    )


def mhc_post_xpu_torch(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    """out[i, hco, h] = post[i, hco] * x[i, h]
    + sum_hci comb[i, hci, hco] * residual[i, hci, h]."""
    assert x.dtype == torch.bfloat16
    assert residual.dtype == torch.bfloat16
    hc = residual.shape[-2]
    H = residual.shape[-1]
    N_shape = residual.shape[:-2]

    x_f = x.to(torch.float32).reshape(-1, H)
    r_f = residual.to(torch.float32).reshape(-1, hc, H)
    p = post_layer_mix.reshape(-1, hc, 1).to(torch.float32)
    c = comb_res_mix.reshape(-1, hc, hc).to(torch.float32)

    out = p * x_f.unsqueeze(1) + torch.einsum("nio,nih->noh", c, r_f)
    return out.to(torch.bfloat16).reshape(*N_shape, hc, H)
