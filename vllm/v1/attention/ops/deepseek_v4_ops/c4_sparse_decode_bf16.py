# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# pyright: reportUnknownVariableType=none, reportUnknownMemberType=none, reportUnknownArgumentType=none, reportUnknownParameterType=none, reportMissingParameterType=none, reportCallIssue=none, reportUnreachable=none, reportPrivateImportUsage=none, reportUnusedCallResult=none
"""bf16 Triton C4 sparse decode over topk ∪ SWA with attention sink.

This kernel targets the DeepSeek V4 main MLA sparse decode path on XPU.
It fuses top-k sparse keys and SWA sparse keys into one online-softmax
accumulation and folds an attention sink as an extra zero-value logit.

The public entry point also provides a CPU fallback reference implementation,
which is used by the phase parity tests.
"""

import math

import torch

from vllm.triton_utils import tl, triton

HEAD_DIM = 512
BLOCK_K = 64


@triton.jit
def _c4_sparse_decode_bf16_kernel(
    q_ptr,
    kv_ptr,
    swa_kv_ptr,
    topk_indices_ptr,
    topk_lens_ptr,
    swa_indices_ptr,
    swa_lens_ptr,
    sink_ptr,
    out_ptr,
    softmax_scale,
    q_stride_t,
    q_stride_h,
    kv_stride_s,
    swa_kv_stride_s,
    topk_stride_t,
    swa_stride_t,
    out_stride_t,
    out_stride_h,
    TOPK_MAX: tl.constexpr,
    SWA_W: tl.constexpr,
    D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)

    d_off = tl.arange(0, D)
    q = tl.load(q_ptr + t * q_stride_t + h * q_stride_h + d_off).to(tl.float32)

    topk_len = tl.load(topk_lens_ptr + t)
    swa_len = tl.load(swa_lens_ptr + t)

    m_i = tl.full((), float("-inf"), tl.float32)
    d_i = tl.zeros((), tl.float32)
    num_i = tl.zeros((D,), tl.float32)

    for k_start in tl.range(0, TOPK_MAX, BLOCK_K):
        k_off = k_start + tl.arange(0, BLOCK_K)
        valid = k_off < topk_len
        slot = tl.load(
            topk_indices_ptr + t * topk_stride_t + k_off,
            mask=valid,
            other=-1,
        ).to(tl.int32)
        valid = valid & (slot >= 0)
        valid_count = tl.sum(valid.to(tl.int32), axis=0)
        block_has_values = valid_count > 0

        k = tl.load(
            kv_ptr + slot[:, None] * kv_stride_s + d_off[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)
        logits = tl.sum(k * q[None, :], axis=1) * softmax_scale
        logits = tl.where(valid, logits, float("-inf"))

        m_block = tl.max(logits, axis=0)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(logits - m_new)
        p = tl.where(valid, p, 0.0)

        d_next = d_i * alpha + tl.sum(p, axis=0)
        num_next = num_i * alpha + tl.sum(p[:, None] * k, axis=0)
        d_i = tl.where(block_has_values, d_next, d_i)
        num_i = tl.where(block_has_values, num_next, num_i)
        m_i = tl.where(block_has_values, m_new, m_i)

    for k_start in tl.range(0, SWA_W, BLOCK_K):
        k_off = k_start + tl.arange(0, BLOCK_K)
        valid = k_off < swa_len
        slot = tl.load(
            swa_indices_ptr + t * swa_stride_t + k_off,
            mask=valid,
            other=-1,
        ).to(tl.int32)
        valid = valid & (slot >= 0)

        dedup = tl.zeros((BLOCK_K,), tl.int32)
        for topk_start in tl.range(0, TOPK_MAX, BLOCK_K):
            topk_off = topk_start + tl.arange(0, BLOCK_K)
            topk_valid = topk_off < topk_len
            topk_slot = tl.load(
                topk_indices_ptr + t * topk_stride_t + topk_off,
                mask=topk_valid,
                other=-1,
            ).to(tl.int32)
            matches = (slot[:, None] == topk_slot[None, :]) & topk_valid[None, :]
            dedup = dedup | (tl.sum(matches.to(tl.int32), axis=1) > 0).to(tl.int32)

        valid = valid & (dedup == 0)
        valid_count = tl.sum(valid.to(tl.int32), axis=0)
        block_has_values = valid_count > 0

        k = tl.load(
            swa_kv_ptr + slot[:, None] * swa_kv_stride_s + d_off[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)
        logits = tl.sum(k * q[None, :], axis=1) * softmax_scale
        logits = tl.where(valid, logits, float("-inf"))

        m_block = tl.max(logits, axis=0)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(logits - m_new)
        p = tl.where(valid, p, 0.0)

        d_next = d_i * alpha + tl.sum(p, axis=0)
        num_next = num_i * alpha + tl.sum(p[:, None] * k, axis=0)
        d_i = tl.where(block_has_values, d_next, d_i)
        num_i = tl.where(block_has_values, num_next, num_i)
        m_i = tl.where(block_has_values, m_new, m_i)

    sink = tl.load(sink_ptr + h).to(tl.float32)
    sink_is_finite = (sink == sink) & (tl.abs(sink) != float("inf"))
    m_new = tl.maximum(m_i, sink)
    alpha = tl.exp(m_i - m_new)
    sink_p = tl.exp(sink - m_new)
    d_next = d_i * alpha + sink_p
    num_next = num_i * alpha
    d_i = tl.where(sink_is_finite, d_next, d_i)
    num_i = tl.where(sink_is_finite, num_next, num_i)
    m_i = tl.where(sink_is_finite, m_new, m_i)

    out = tl.where(d_i > 0, num_i / d_i, tl.zeros_like(num_i))
    tl.store(out_ptr + t * out_stride_t + h * out_stride_h + d_off, out.to(tl.bfloat16))


def _ref_c4_sparse_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_lens: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    swa_kv_cache: torch.Tensor | None = None,
) -> torch.Tensor:
    T, H, D = q.shape
    out = torch.zeros((T, H, D), dtype=torch.float32, device=q.device)
    kv_f32 = kv_cache.to(torch.float32)
    swa_kv_f32 = (swa_kv_cache if swa_kv_cache is not None else kv_cache).to(torch.float32)
    q_f32 = q.to(torch.float32)
    sink_f32 = attn_sink.to(torch.float32)

    for t in range(T):
        topk_valid = {
            int(topk_indices[t, i].item())
            for i in range(int(topk_lens[t].item()))
            if int(topk_indices[t, i].item()) >= 0
        }
        for h in range(H):
            q_vec = q_f32[t, h]
            logits: list[float] = []
            vals: list[torch.Tensor] = []

            for i in range(int(topk_lens[t].item())):
                slot = int(topk_indices[t, i].item())
                if slot < 0:
                    continue
                kv_vec = kv_f32[slot]
                logits.append(float(softmax_scale * torch.dot(q_vec, kv_vec).item()))
                vals.append(kv_vec)

            for i in range(int(swa_lens[t].item())):
                slot = int(swa_indices[t, i].item())
                if slot < 0 or slot in topk_valid:
                    continue
                kv_vec = swa_kv_f32[slot]
                logits.append(float(softmax_scale * torch.dot(q_vec, kv_vec).item()))
                vals.append(kv_vec)

            sink_logit = float(sink_f32[h].item())
            all_logits = logits + [sink_logit]
            m = max(all_logits) if all_logits else float("-inf")
            if not math.isfinite(m):
                continue

            exps = [math.exp(logit - m) for logit in all_logits]
            denom = sum(exps)
            if denom == 0.0:
                continue

            num = torch.zeros((D,), dtype=torch.float32, device=q.device)
            for weight, value in zip(exps[:-1], vals):
                num = num + weight * value
            out[t, h] = num / denom

    return out.to(torch.bfloat16)


def c4_sparse_decode_bf16(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_lens: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    out: torch.Tensor,
    swa_kv_cache: torch.Tensor | None = None,
) -> None:
    assert q.dtype == torch.bfloat16
    assert kv_cache.dtype == torch.bfloat16
    assert out.dtype == torch.bfloat16
    assert topk_indices.dtype == torch.int32
    assert topk_lens.dtype == torch.int32
    assert swa_indices.dtype == torch.int32
    assert swa_lens.dtype == torch.int32
    assert attn_sink.dtype == torch.float32
    assert q.shape[-1] == HEAD_DIM
    assert kv_cache.dim() == 2 and kv_cache.shape[-1] == HEAD_DIM
    assert out.shape == q.shape
    assert topk_indices.dim() == 2
    assert swa_indices.dim() == 2
    assert topk_indices.shape[0] == q.shape[0]
    assert swa_indices.shape[0] == q.shape[0]
    assert topk_lens.shape == (q.shape[0],)
    assert swa_lens.shape == (q.shape[0],)
    assert attn_sink.shape == (q.shape[1],)

    if q.device.type != "xpu":
        _ = out.copy_(
            _ref_c4_sparse_decode(
                q,
                kv_cache,
                topk_indices,
                topk_lens,
                swa_indices,
                swa_lens,
                attn_sink,
                softmax_scale,
            )
        )
        return

    T, H, D = q.shape
    topk_max = topk_indices.shape[1]
    swa_w = swa_indices.shape[1]

    # Use separate SWA KV cache if provided (XPU: caches are separate tensors)
    swa_kv = swa_kv_cache if swa_kv_cache is not None else kv_cache

    grid = (T, H)
    _c4_sparse_decode_bf16_kernel[grid](
        q,
        kv_cache,
        swa_kv,
        topk_indices,
        topk_lens,
        swa_indices,
        swa_lens,
        attn_sink,
        out,
        float(softmax_scale),
        q.stride(0),
        q.stride(1),
        kv_cache.stride(0),
        swa_kv.stride(0),
        topk_indices.stride(0),
        swa_indices.stride(0),
        out.stride(0),
        out.stride(1),
        TOPK_MAX=topk_max,
        SWA_W=swa_w,
        D=D,
        BLOCK_K=BLOCK_K,
        num_warps=4,
    )
