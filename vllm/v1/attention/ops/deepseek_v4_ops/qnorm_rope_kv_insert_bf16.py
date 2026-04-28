# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""bf16 Triton port of fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert.

Drop-in replacement for the CUDA op
``torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert`` on the
XPU bf16-KV path. UE8M0 FP8 quant is intentionally dropped: the SWA cache
on XPU is plain bf16 of shape ``[num_blocks, block_size, 512]``.
"""

import torch

from vllm.triton_utils import tl, triton

HEAD_DIM = 512
ROPE_DIM = 64
NOPE_DIM = HEAD_DIM - ROPE_DIM
HALF_ROPE = ROPE_DIM // 2


@triton.jit
def _xpu_fused_v4_qnorm_rope_kv_insert_bf16_kernel(
    q_ptr,
    kv_ptr,
    cache_ptr,
    slot_mapping_ptr,
    position_ids_ptr,
    cos_sin_cache_ptr,
    eps,
    num_tokens_insert,
    num_heads_q,
    block_size,
    q_token_stride,
    q_head_stride,
    kv_token_stride,
    cache_block_stride,
    cache_token_stride,
    cos_sin_token_stride,
    HEAD_DIM_C: tl.constexpr,
    NOPE_DIM_C: tl.constexpr,
    HALF_ROPE_C: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    slot_idx = tl.program_id(1)
    is_kv = slot_idx == num_heads_q

    if is_kv & (token_idx >= num_tokens_insert):
        return

    offsets = tl.arange(0, HEAD_DIM_C)

    if is_kv:
        row_base = kv_ptr + token_idx * kv_token_stride
    else:
        slot_i64 = slot_idx.to(tl.int64)
        row_base = q_ptr + token_idx * q_token_stride + slot_i64 * q_head_stride

    x = tl.load(row_base + offsets).to(tl.float32)

    rrms = tl.full((), 1.0, tl.float32)
    if not is_kv:
        # Per-head RMSNorm with no learnable weight, fp32 throughout.
        variance = tl.sum(x * x, axis=0) / HEAD_DIM_C
        rrms = tl.rsqrt(variance + eps)
        x = x * rrms

    # GPT-J interleaved RoPE on dims [NOPE_DIM, HEAD_DIM). Partner-load trick:
    # each lane fetches x[i^1] (its pair-mate) via a second masked load and
    # uses an even/odd select to pick the +sin vs -sin branch. Same pattern
    # as fused_inv_rope_fp8_quant but in the forward direction (sin sign
    # flipped relative to the inverse). RMSNorm scales every element by the
    # same scalar rrms, so partner_post_norm = partner_raw * rrms.
    is_rope = offsets >= NOPE_DIM_C
    rope_local = offsets - NOPE_DIM_C
    cs_idx = tl.maximum(rope_local >> 1, 0)
    pos = tl.load(position_ids_ptr + token_idx)
    cache_base = cos_sin_cache_ptr + pos * cos_sin_token_stride
    cos_v = tl.load(cache_base + cs_idx, mask=is_rope, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE_C + cs_idx, mask=is_rope, other=0.0)

    partner_raw = tl.load(row_base + (offsets ^ 1), mask=is_rope, other=0.0).to(
        tl.float32
    )
    partner = partner_raw * rrms

    # Forward GPT-J RoPE on pair (even, odd):
    #   new_even = even * cos - odd * sin
    #   new_odd  = even * sin + odd * cos
    # Per-lane self/partner identification:
    #   even lane: out = self*cos - partner*sin
    #   odd  lane: out = self*cos + partner*sin
    is_even = (rope_local & 1) == 0
    rotated = tl.where(
        is_even,
        x * cos_v - partner * sin_v,
        x * cos_v + partner * sin_v,
    )
    x = tl.where(is_rope, rotated, x)

    if is_kv:
        slot_id = tl.load(slot_mapping_ptr + token_idx)
        if slot_id < 0:
            return
        block_idx = (slot_id // block_size).to(tl.int64)
        pos_in_block = (slot_id % block_size).to(tl.int64)
        dst = (
            cache_ptr
            + block_idx * cache_block_stride
            + pos_in_block * cache_token_stride
            + offsets
        )
        tl.store(dst, x.to(cache_ptr.dtype.element_ty))
    else:
        tl.store(row_base + offsets, x.to(q_ptr.dtype.element_ty))


def xpu_fused_v4_qnorm_rope_kv_insert_bf16(
    q: torch.Tensor,
    kv: torch.Tensor,
    swa_kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    position_ids: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
) -> None:
    assert q.dim() == 3 and q.size(2) == HEAD_DIM
    assert kv.dim() == 2 and kv.size(1) == HEAD_DIM
    assert swa_kv_cache.dim() == 3 and swa_kv_cache.size(2) == HEAD_DIM
    assert q.dtype == kv.dtype == swa_kv_cache.dtype == torch.bfloat16
    assert slot_mapping.dtype == torch.int64
    assert position_ids.dtype == torch.int64
    assert cos_sin_cache.dtype == torch.float32
    assert cos_sin_cache.dim() == 2 and cos_sin_cache.size(1) == ROPE_DIM
    assert q.is_contiguous() and kv.is_contiguous()
    assert q.size(0) == kv.size(0) == position_ids.size(0)
    assert slot_mapping.size(0) <= q.size(0)

    num_tokens_full = q.size(0)
    num_tokens_insert = slot_mapping.size(0)
    num_heads_q = q.size(1)
    block_size = swa_kv_cache.size(1)

    if num_tokens_full == 0:
        return

    grid = (num_tokens_full, num_heads_q + 1)
    _xpu_fused_v4_qnorm_rope_kv_insert_bf16_kernel[grid](
        q,
        kv,
        swa_kv_cache,
        slot_mapping,
        position_ids,
        cos_sin_cache,
        eps,
        num_tokens_insert,
        num_heads_q,
        block_size,
        q.stride(0),
        q.stride(1),
        kv.stride(0),
        swa_kv_cache.stride(0),
        swa_kv_cache.stride(1),
        cos_sin_cache.stride(0),
        HEAD_DIM_C=HEAD_DIM,
        NOPE_DIM_C=NOPE_DIM,
        HALF_ROPE_C=HALF_ROPE,
    )
