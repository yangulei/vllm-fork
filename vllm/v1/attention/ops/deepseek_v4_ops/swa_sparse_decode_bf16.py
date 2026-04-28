# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""bf16 Triton port of FlashMLA SWA-only sparse decode for DeepseekV4.

Drop-in replacement for ``flash_mla_with_kvcache(..., is_fp8_kvcache=True,
extra_k_cache=None, extra_indices_in_kvcache=None, extra_topk_length=None)``
on the XPU bf16-KV path. Only handles ``swa_only=True`` (compress_ratio<=1).

Inputs
------
q             : [T, H, 512]               bfloat16 (post qnorm+rope)
swa_kv_cache  : [num_blocks, 64, 512]     bfloat16 paged cache
                Caller passes a flat view ``[num_blocks*64, 512]``.
swa_indices   : [T, W]                    int32, slot ids (=block*64+offset)
                ``-1`` marks padding; valid count per token is ``swa_lens[t]``.
swa_lens      : [T]                       int32 (#valid keys, <= W)
attn_sink     : [H]                       fp32  (one extra sink logit per head)
softmax_scale : float32

Output (in-place)
-----------------
out           : [T, H, 512]               bfloat16

Semantics (per-head FlashAttention with one virtual "sink" key whose
*value* is zero and whose *logit* is the corresponding scalar in
``attn_sink``):

    s_i = softmax_scale * <q, K[swa_indices[i]]>     (i < swa_lens, else -inf)
    s_sink = attn_sink[h]
    m = max(max(s), s_sink)
    num = sum_i exp(s_i - m) * V[swa_indices[i]]   (V==K for MLA, dim_v=512)
    den = sum_i exp(s_i - m) + exp(s_sink - m)
    out = num / den
"""

import torch

from vllm.triton_utils import tl, triton

HEAD_DIM = 512


@triton.jit
def _xpu_v4_swa_sparse_decode_bf16_kernel(
    q_ptr,
    kv_flat_ptr,
    indices_ptr,
    lens_ptr,
    sink_ptr,
    out_ptr,
    softmax_scale,
    q_stride_t,
    q_stride_h,
    kv_stride_block,
    kv_stride_s,
    idx_stride_t,
    out_stride_t,
    out_stride_h,
    W: tl.constexpr,
    D: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)

    swa_len = tl.load(lens_ptr + t)

    d_off = tl.arange(0, D)
    q = tl.load(q_ptr + t * q_stride_t + h * q_stride_h + d_off).to(tl.float32)

    # FlashAttention online-softmax accumulators
    m_i = tl.full((), float("-inf"), tl.float32)
    l_i = tl.zeros((), tl.float32)
    acc = tl.zeros((D,), tl.float32)

    for w_start in tl.range(0, W, BLOCK_W):
        w_off = w_start + tl.arange(0, BLOCK_W)
        valid = w_off < swa_len
        slot = tl.load(indices_ptr + t * idx_stride_t + w_off, mask=valid, other=0).to(
            tl.int32
        )
        # Defensive mask: caller may pad with -1 within swa_len.
        valid = valid & (slot >= 0)

        k = tl.load(
            kv_flat_ptr
            + (slot // BLOCK_SIZE)[:, None] * kv_stride_block
            + (slot % BLOCK_SIZE)[:, None] * kv_stride_s
            + d_off[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)

        s = tl.sum(k * q[None, :], axis=1) * softmax_scale
        s = tl.where(valid, s, float("-inf"))

        # Online softmax update.
        m_new = tl.maximum(m_i, tl.max(s, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new)

        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * k, axis=0)
        m_i = m_new

    # Fold attention sink as one extra logit with V==0 (contributes only
    # to the denominator). attn_sink is per-head fp32.
    s_sink = tl.load(sink_ptr + h).to(tl.float32)
    m_new = tl.maximum(m_i, s_sink)
    alpha = tl.exp(m_i - m_new)
    p_sink = tl.exp(s_sink - m_new)
    l_i = l_i * alpha + p_sink
    acc = acc * alpha
    m_i = m_new

    # Guard against all-masked tokens (swa_len==0 and sink==-inf): produce 0.
    out = tl.where(l_i > 0, acc / l_i, tl.zeros_like(acc))
    tl.store(out_ptr + t * out_stride_t + h * out_stride_h + d_off, out.to(tl.bfloat16))


def xpu_v4_swa_sparse_decode_bf16(
    q: torch.Tensor,
    swa_kv_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    out: torch.Tensor,
) -> None:
    assert q.dtype == torch.bfloat16
    assert swa_kv_cache.dtype == torch.bfloat16
    assert out.dtype == torch.bfloat16
    assert swa_indices.dtype == torch.int32
    assert swa_lens.dtype == torch.int32
    assert q.shape[-1] == HEAD_DIM
    assert swa_kv_cache.shape[-1] == HEAD_DIM
    assert out.shape == q.shape

    T, H, D = q.shape
    W = swa_indices.shape[-1]

    assert swa_kv_cache.dim() == 3, (
        f"expected paged KV cache shape [num_blocks, block_size, head_dim], "
        f"got {tuple(swa_kv_cache.shape)}"
    )
    block_size = swa_kv_cache.shape[1]
    idx2d = swa_indices.view(T, W)

    BLOCK_W = 64 if W >= 64 else triton.next_power_of_2(W)
    grid = (T, H)
    _xpu_v4_swa_sparse_decode_bf16_kernel[grid](
        q,
        swa_kv_cache,
        idx2d,
        swa_lens,
        attn_sink,
        out,
        float(softmax_scale),
        q.stride(0),
        q.stride(1),
        swa_kv_cache.stride(0),
        swa_kv_cache.stride(1),
        idx2d.stride(0),
        out.stride(0),
        out.stride(1),
        W=W,
        D=D,
        BLOCK_W=BLOCK_W,
        BLOCK_SIZE=block_size,
        num_warps=4,
        num_stages=2,
    )
