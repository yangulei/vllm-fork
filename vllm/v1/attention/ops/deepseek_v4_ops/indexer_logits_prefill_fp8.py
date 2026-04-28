# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportPrivateImportUsage=false, reportCallIssue=false, reportUnusedVariable=false, reportGeneralTypeIssues=false, reportUnreachable=false

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 sparse-indexer prefill logits kernel for DeepSeek V4 on XPU.

Computes per-token logits over a variable contiguous K range and reduces across
query heads with per-token-per-head weights already carrying the Q-side scale.
The K-side per-token scale is applied inline.
"""

import torch

from vllm.triton_utils import tl, triton

HEAD_DIM = 128
BLOCK_D = 128
BLOCK_K = 64


@triton.jit
def _mask_invalid_logits_kernel(
    out_ptr,
    cu_seqlen_ks_ptr,
    cu_seqlen_ke_ptr,
    out_stride_t,
    K_MAX,
    BLOCK_K: tl.constexpr,
):
    t = tl.program_id(0)
    k_start = tl.program_id(1) * BLOCK_K
    k_off = k_start + tl.arange(0, BLOCK_K)

    ke = tl.load(cu_seqlen_ke_ptr + t).to(tl.int32)
    ks = tl.load(cu_seqlen_ks_ptr + t).to(tl.int32)
    k_len = ke - ks

    valid_store = k_off < K_MAX
    invalid = valid_store & (k_off >= k_len)
    tl.store(out_ptr + t * out_stride_t + k_off, float("-inf"), mask=invalid)


@triton.jit
def _indexer_logits_prefill_fp8_kernel(
    q_ptr,
    k_ptr,
    k_scale_ptr,
    weights_ptr,
    cu_seqlen_ks_ptr,
    cu_seqlen_ke_ptr,
    out_ptr,
    q_stride_t,
    q_stride_h,
    k_stride_k,
    weights_stride_t,
    out_stride_t,
    HEAD_DIM_C: tl.constexpr,
    BLOCK_D_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)

    d_off = tl.arange(0, BLOCK_D_C)
    q = tl.load(
        q_ptr + t * q_stride_t + h * q_stride_h + d_off,
        mask=d_off < HEAD_DIM_C,
        other=0.0,
    ).to(tl.float8e4nv)
    q = q.to(tl.float32)

    ks = tl.load(cu_seqlen_ks_ptr + t).to(tl.int32)
    ke = tl.load(cu_seqlen_ke_ptr + t).to(tl.int32)
    k_len = ke - ks
    weight = tl.load(weights_ptr + t * weights_stride_t + h).to(tl.float32)

    num_k_blocks = tl.cdiv(k_len, BLOCK_K)
    for k_block_idx in range(num_k_blocks):
        k_start = k_block_idx * BLOCK_K
        k_off = k_start + tl.arange(0, BLOCK_K)
        valid = k_off < k_len
        k_idx = ks + k_off
        k_block = tl.load(
            k_ptr + k_idx[:, None] * k_stride_k + d_off[None, :],
            mask=valid[:, None] & (d_off[None, :] < HEAD_DIM_C),
            other=0.0,
        ).to(tl.float8e4nv)
        k_block = k_block.to(tl.float32)
        dots = tl.sum(k_block * q[None, :], axis=1)
        k_scale = tl.load(k_scale_ptr + k_idx, mask=valid, other=0.0).to(tl.float32)
        logits = dots * k_scale * weight
        tl.atomic_add(out_ptr + t * out_stride_t + k_off, logits, mask=valid)


def _ref_indexer_logits_prefill_fp8(
    q_fp8: torch.Tensor,
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    t_tokens, num_heads, _ = q_fp8.shape
    k_lens = cu_seqlen_ke.to(torch.int64) - cu_seqlen_ks.to(torch.int64)
    k_max = int(k_lens.max().item()) if k_lens.numel() > 0 else 0
    out = torch.full((t_tokens, k_max), float("-inf"), device=q_fp8.device, dtype=torch.float32)
    for t in range(t_tokens):
        ks = int(cu_seqlen_ks[t].item())
        ke = int(cu_seqlen_ke[t].item())
        k_len = ke - ks
        if k_len <= 0:
            continue
        q_f = q_fp8[t].float()
        k_f = k_fp8[ks:ke].float()
        dots = q_f @ k_f.transpose(0, 1)
        scaled = dots * weights[t].float().unsqueeze(-1)
        scaled = scaled * k_scale[ks:ke].float().unsqueeze(0)
        out[t, :k_len] = scaled.sum(dim=0)
    return out


def indexer_logits_prefill_fp8(
    q_fp8: torch.Tensor,
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    """Compute sparse-indexer prefill logits.

    Args:
        q_fp8: [T, H, 128] torch.float8_e4m3fn.
        k_fp8: [K_TOTAL, 128] torch.float8_e4m3fn.
        k_scale: [K_TOTAL] fp32 per-key scale.
        weights: [T, H] fp32 folded weights.
        cu_seqlen_ks: [T + 1] or [T] int32 start indices; first T values used.
        cu_seqlen_ke: [T + 1] or [T] int32 end indices; first T values used.
    Returns:
        [T, K_MAX] fp32 logits with -inf padding beyond each row's valid range.
    """
    assert q_fp8.ndim == 3, f"expected q_fp8 [T, H, D], got {tuple(q_fp8.shape)}"
    assert k_fp8.ndim == 2, f"expected k_fp8 [K_TOTAL, D], got {tuple(k_fp8.shape)}"
    assert weights.ndim == 2, f"expected weights [T, H], got {tuple(weights.shape)}"
    assert q_fp8.shape[0] == weights.shape[0]
    assert q_fp8.shape[1] == weights.shape[1]
    assert q_fp8.shape[2] == HEAD_DIM, f"expected head_dim={HEAD_DIM}, got {q_fp8.shape[2]}"
    assert k_fp8.shape[1] == HEAD_DIM, f"expected head_dim={HEAD_DIM}, got {k_fp8.shape[1]}"
    assert q_fp8.dtype == torch.float8_e4m3fn
    assert k_fp8.dtype == torch.float8_e4m3fn
    assert k_scale.shape[0] == k_fp8.shape[0]

    t_tokens = q_fp8.shape[0]
    cu_seqlen_ks = cu_seqlen_ks[:t_tokens].to(dtype=torch.int32, device=q_fp8.device)
    cu_seqlen_ke = cu_seqlen_ke[:t_tokens].to(dtype=torch.int32, device=q_fp8.device)
    k_lens = cu_seqlen_ke.to(torch.int64) - cu_seqlen_ks.to(torch.int64)
    k_max = int(k_lens.max().item()) if k_lens.numel() > 0 else 0

    if t_tokens == 0:
        return torch.empty((0, k_max), device=q_fp8.device, dtype=torch.float32)
    if k_max == 0:
        return torch.empty((t_tokens, 0), device=q_fp8.device, dtype=torch.float32)

    if q_fp8.device.type != "xpu":
        return _ref_indexer_logits_prefill_fp8(
            q_fp8, k_fp8, k_scale, weights, cu_seqlen_ks, cu_seqlen_ke
        )

    out = torch.zeros((t_tokens, k_max), device=q_fp8.device, dtype=torch.float32)
    init_grid = (t_tokens, triton.cdiv(k_max, BLOCK_K))
    init_launcher = _mask_invalid_logits_kernel[init_grid]
    init_launcher(
        out,
        cu_seqlen_ks,
        cu_seqlen_ke,
        out.stride(0),
        k_max,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )

    grid = (t_tokens, q_fp8.shape[1])
    logits_launcher = _indexer_logits_prefill_fp8_kernel[grid]
    logits_launcher(
        q_fp8,
        k_fp8,
        k_scale,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        out,
        q_fp8.stride(0),
        q_fp8.stride(1),
        k_fp8.stride(0),
        weights.stride(0),
        out.stride(0),
        HEAD_DIM_C=HEAD_DIM,
        BLOCK_D_C=BLOCK_D,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )
    return out


__all__ = ["indexer_logits_prefill_fp8"]
