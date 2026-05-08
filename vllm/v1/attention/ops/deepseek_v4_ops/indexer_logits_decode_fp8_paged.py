# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportPrivateImportUsage=false, reportCallIssue=false, reportUnusedVariable=false, reportGeneralTypeIssues=false, reportUnreachable=false

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 sparse-indexer decode logits kernel for DeepSeek V4 on XPU."""

import torch

from vllm.triton_utils import tl, triton

HEAD_DIM = 128
BLOCK_D = 128
BLOCK_POS = 64


@triton.jit
def _mask_invalid_decode_logits_kernel(
    out_ptr,
    seq_lens_ptr,
    out_stride_t,
    MAX_SEQ,
    BLOCK_POS_C: tl.constexpr,
):
    t = tl.program_id(0)
    pos_start = tl.program_id(1) * BLOCK_POS_C
    pos_off = pos_start + tl.arange(0, BLOCK_POS_C)

    seq_len = tl.load(seq_lens_ptr + t).to(tl.int32)
    valid_store = pos_off < MAX_SEQ
    invalid = valid_store & (pos_off >= seq_len)
    tl.store(out_ptr + t * out_stride_t + pos_off, float("-inf"), mask=invalid)


@triton.jit
def _indexer_logits_decode_fp8_paged_kernel(
    q_ptr,
    k_cache_ptr,
    k_scale_ptr,
    weights_ptr,
    seq_lens_ptr,
    block_table_ptr,
    out_ptr,
    q_stride_t,
    q_stride_h,
    q_stride_d,
    k_stride_block,
    k_stride_slot,
    k_stride_d,
    k_scale_stride_block,
    k_scale_stride_slot,
    weights_stride_t,
    weights_stride_h,
    seq_lens_stride_t,
    block_table_stride_b,
    block_table_stride_blk,
    out_stride_t,
    out_stride_pos,
    NEXT_N,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM_C: tl.constexpr,
    BLOCK_D_C: tl.constexpr,
    BLOCK_POS_C: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)

    d_off = tl.arange(0, BLOCK_D_C)
    q = tl.load(
        q_ptr + t * q_stride_t + h * q_stride_h + d_off * q_stride_d,
        mask=d_off < HEAD_DIM_C,
        other=0.0,
    ).to(tl.float8e4nv)
    q = q.to(tl.float32)

    seq_len = tl.load(seq_lens_ptr + t * seq_lens_stride_t).to(tl.int32)
    weight = tl.load(weights_ptr + t * weights_stride_t + h * weights_stride_h).to(tl.float32)
    batch_idx = t // NEXT_N

    for pos_block_idx in range(tl.cdiv(seq_len, BLOCK_POS_C)):
        pos_start = pos_block_idx * BLOCK_POS_C
        pos_off = pos_start + tl.arange(0, BLOCK_POS_C)
        valid = pos_off < seq_len

        block_rank = pos_off // BLOCK_SIZE
        slot_idx = pos_off % BLOCK_SIZE
        block_idx = tl.load(
            block_table_ptr + batch_idx * block_table_stride_b + block_rank * block_table_stride_blk,
            mask=valid,
            other=0,
        ).to(tl.int32)

        k = tl.load(
            k_cache_ptr
            + block_idx[:, None] * k_stride_block
            + slot_idx[:, None] * k_stride_slot
            + d_off[None, :] * k_stride_d,
            mask=valid[:, None] & (d_off[None, :] < HEAD_DIM_C),
            other=0.0,
        ).to(tl.float8e4nv)
        k = k.to(tl.float32)

        dots = tl.sum(k * q[None, :], axis=1)
        k_scale = tl.load(
            k_scale_ptr
            + block_idx * k_scale_stride_block
            + slot_idx * k_scale_stride_slot,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        logits = dots * k_scale * weight
        tl.atomic_add(
            out_ptr + t * out_stride_t + pos_off * out_stride_pos,
            logits,
            mask=valid,
        )


def _ref_indexer_logits_decode_fp8_paged(
    q_fp8: torch.Tensor,
    k_cache: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
) -> torch.Tensor:
    batch_size, next_n, num_heads, _ = q_fp8.shape
    block_size = k_cache.shape[1]
    t_tokens = batch_size * next_n
    q_flat = q_fp8.reshape(t_tokens, num_heads, HEAD_DIM)
    seq_flat = seq_lens.reshape(t_tokens)
    max_seq_len = int(seq_flat.max().item()) if seq_flat.numel() > 0 else 0
    out = torch.full((t_tokens, max_seq_len), float("-inf"), device=q_fp8.device, dtype=torch.float32)
    for t in range(t_tokens):
        seq_len = int(seq_flat[t].item())
        if seq_len <= 0:
            continue
        out[t, :seq_len] = 0.0
        batch_idx = t // next_n
        for pos in range(seq_len):
            block_idx = int(block_table[batch_idx, pos // block_size].item())
            slot_idx = pos % block_size
            k_vec = k_cache[block_idx, slot_idx].float()
            k_s = float(k_scale[block_idx, slot_idx].item())
            for h in range(num_heads):
                q_vec = q_flat[t, h].float()
                dot = float((q_vec * k_vec).sum().item())
                out[t, pos] += dot * k_s * float(weights[t, h].item())
    return out


def indexer_logits_decode_fp8_paged(
    q_fp8: torch.Tensor,
    k_cache: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    """Compute sparse-indexer decode logits over a paged FP8 K cache."""
    assert q_fp8.ndim == 4, f"expected q_fp8 [B, next_n, H, D], got {tuple(q_fp8.shape)}"
    assert k_cache.ndim == 3, f"expected k_cache [num_blocks, block_size, D], got {tuple(k_cache.shape)}"
    assert k_scale.ndim == 2, f"expected k_scale [num_blocks, block_size], got {tuple(k_scale.shape)}"
    assert weights.ndim == 2, f"expected weights [B*next_n, H], got {tuple(weights.shape)}"
    assert seq_lens.ndim == 2, f"expected seq_lens [B, next_n], got {tuple(seq_lens.shape)}"
    assert block_table.ndim == 2, f"expected block_table [B, max_blocks], got {tuple(block_table.shape)}"
    assert q_fp8.shape[3] == HEAD_DIM, f"expected head_dim={HEAD_DIM}, got {q_fp8.shape[3]}"
    assert k_cache.shape[2] == HEAD_DIM, f"expected head_dim={HEAD_DIM}, got {k_cache.shape[2]}"
    assert q_fp8.dtype == torch.float8_e4m3fn
    assert k_cache.dtype == torch.float8_e4m3fn
    assert k_scale.shape == k_cache.shape[:2]

    batch_size, next_n, num_heads, _ = q_fp8.shape
    t_tokens = batch_size * next_n
    assert weights.shape == (t_tokens, num_heads)
    assert seq_lens.shape == (batch_size, next_n)
    assert block_table.shape[0] == batch_size

    q_flat = q_fp8.reshape(t_tokens, num_heads, HEAD_DIM).contiguous()
    seq_flat = seq_lens.reshape(t_tokens).to(dtype=torch.int32, device=q_fp8.device).contiguous()
    weights = weights.to(dtype=torch.float32, device=q_fp8.device).contiguous()
    block_table = block_table.to(dtype=torch.int32, device=q_fp8.device).contiguous()
    k_scale = k_scale.to(dtype=torch.float32, device=q_fp8.device).contiguous()

    # Use block_table shape as upper bound for max_seq_len to avoid
    # .item() which forces a CPU-GPU sync on every call.
    block_size = k_cache.shape[1]
    max_seq_len = block_table.shape[1] * block_size

    if t_tokens == 0:
        return torch.empty((0, max_seq_len), device=q_fp8.device, dtype=torch.float32)
    if max_seq_len == 0:
        return torch.empty((t_tokens, 0), device=q_fp8.device, dtype=torch.float32)

    if q_fp8.device.type != "xpu":
        return _ref_indexer_logits_decode_fp8_paged(
            q_fp8, k_cache, k_scale, weights, seq_lens, block_table
        )

    out = torch.zeros((t_tokens, max_seq_len), device=q_fp8.device, dtype=torch.float32)
    mask_grid = (t_tokens, triton.cdiv(max_seq_len, BLOCK_POS))
    _mask_invalid_decode_logits_kernel[mask_grid](
        out,
        seq_flat,
        out.stride(0),
        max_seq_len,
        BLOCK_POS_C=BLOCK_POS,
        num_warps=4,
        num_stages=2,
    )

    grid = (t_tokens, num_heads)
    _indexer_logits_decode_fp8_paged_kernel[grid](
        q_flat,
        k_cache,
        k_scale,
        weights,
        seq_flat,
        block_table,
        out,
        q_flat.stride(0),
        q_flat.stride(1),
        q_flat.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_scale.stride(0),
        k_scale.stride(1),
        weights.stride(0),
        weights.stride(1),
        seq_flat.stride(0),
        block_table.stride(0),
        block_table.stride(1),
        out.stride(0),
        out.stride(1),
        next_n,
        BLOCK_SIZE=block_size,
        HEAD_DIM_C=HEAD_DIM,
        BLOCK_D_C=BLOCK_D,
        BLOCK_POS_C=BLOCK_POS,
        num_warps=4,
        num_stages=2,
    )
    return out


__all__ = ["indexer_logits_decode_fp8_paged"]
