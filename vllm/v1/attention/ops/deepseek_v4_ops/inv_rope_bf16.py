# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Inverse RoPE kernel for DeepseekV4 attention output, bf16 path (XPU).

Mirrors the partner-load GPT-J RoPE in fused_inv_rope_fp8_quant.py but
emits bf16 instead of FP8+block-scales. Used by the XPU O-projection
pipeline: o (bf16) -> inv_rope -> o_bf16 -> torch.einsum with bf16 wo_a.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _xpu_fused_inv_rope_bf16_per_head(
    o_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    out_ptr,
    num_tokens,
    heads_per_group: tl.constexpr,
    o_stride_token,
    o_stride_head,
    cache_stride_pos,
    out_stride_token,
    out_stride_group,
    HEAD_DIM: tl.constexpr,
    ROPE_START: tl.constexpr,
    HALF_ROPE: tl.constexpr,
):
    pid_token = tl.program_id(0).to(tl.int64)
    pid_gh = tl.program_id(1).to(tl.int64)

    if pid_token >= num_tokens:
        return

    g = pid_gh // heads_per_group
    head_in_group = pid_gh % heads_per_group
    global_head = pid_gh

    input_base = o_ptr + pid_token * o_stride_token + global_head * o_stride_head
    offsets = tl.arange(0, HEAD_DIM)
    x = tl.load(input_base + offsets).to(tl.float32)

    pos = tl.load(positions_ptr + pid_token)
    cache_base = cos_sin_cache_ptr + pos * cache_stride_pos
    is_rope = offsets >= ROPE_START
    rope_local = offsets - ROPE_START

    x_partner = tl.load(input_base + (offsets ^ 1), mask=is_rope, other=0.0).to(
        tl.float32
    )
    cs_idx = tl.maximum(rope_local >> 1, 0)
    cos_v = tl.load(cache_base + cs_idx, mask=is_rope, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope, other=0.0)
    x_add = x * cos_v + x_partner * sin_v
    x_sub = x * cos_v - x_partner * sin_v
    is_even = (rope_local & 1) == 0
    rotated = tl.where(is_even, x_add, x_sub)
    x = tl.where(is_rope, rotated, x)

    out_base = (
        out_ptr
        + pid_token * out_stride_token
        + g * out_stride_group
        + head_in_group * HEAD_DIM
    )
    tl.store(out_base + offsets, x.to(tl.bfloat16))


def xpu_fused_inv_rope_bf16(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int = 448,
    rope_dim: int = 64,
) -> torch.Tensor:
    """Inverse RoPE, bf16 in/out. cos_sin_cache layout: cos[:half] || sin[half:]
    (halves, not pairs). Output is R-contiguous to match fp8_einsum's `bhr` layout.
    """
    num_tokens, num_heads, head_dim = o.shape
    assert num_heads == n_groups * heads_per_group
    assert head_dim == nope_dim + rope_dim
    assert rope_dim % 2 == 0
    assert cos_sin_cache.shape[-1] == rope_dim
    assert cos_sin_cache.dtype == torch.float32
    assert o.dtype == torch.bfloat16

    out = torch.empty(
        (num_tokens, n_groups, heads_per_group * head_dim),
        dtype=torch.bfloat16,
        device=o.device,
    )

    grid = (num_tokens, n_groups * heads_per_group)
    _xpu_fused_inv_rope_bf16_per_head[grid](
        o,
        positions,
        cos_sin_cache,
        out,
        num_tokens,
        heads_per_group=heads_per_group,
        o_stride_token=o.stride(0),
        o_stride_head=o.stride(1),
        cache_stride_pos=cos_sin_cache.stride(0),
        out_stride_token=out.stride(0),
        out_stride_group=out.stride(1),
        HEAD_DIM=head_dim,
        ROPE_START=nope_dim,
        HALF_ROPE=rope_dim // 2,
        num_warps=1,
    )
    return out
