# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Triton sparse decode for DeepSeek V4 on XPU.

Strategy:
  1. Fused Triton kernel gathers topk + SWA KV slots from two separate
     caches into a compacted flat workspace — single kernel launch,
     no intermediate index tensors.
  2. Call c4_sparse_prefill_bf16 (per-head Triton kernel optimized for
     D=512 MQA with num_warps=16).

This replaces the previous CUTLASS paged decode which used a Python
for-loop gather + flash_attn_varlen_func.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_dual_gather_kernel(
    # Output workspace: [B * K_total, D]  (row-major)
    ws_ptr,
    # Source caches (flat): [N_topk, D] and [N_swa, D]
    kv_ptr,
    swa_kv_ptr,
    # Index tensors: [B, max_topk] and [B, max_swa]
    topk_idx_ptr,
    swa_idx_ptr,
    # Length tensors: [B]
    topk_lens_ptr,
    swa_lens_ptr,
    # Strides
    topk_idx_stride_b: tl.constexpr,  # stride of topk_idx along batch dim
    swa_idx_stride_b: tl.constexpr,   # stride of swa_idx along batch dim
    # Constants
    K_total: tl.constexpr,
    max_topk: tl.constexpr,
    max_swa: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    N_kv: tl.int64,   # kv_flat total rows (for clamping)
    N_swa: tl.int64,  # swa_kv_flat total rows (for clamping)
):
    """Fused gather kernel: for each (batch, slot) position, read from
    topk cache or SWA cache based on per-request lengths, and write
    compacted into the workspace.

    Grid: (B, K_total) — one program per (batch, slot) pair.
    Each program copies D elements.
    """
    bid = tl.program_id(0)  # batch index
    sid = tl.program_id(1)  # slot index within K_total

    nt = tl.load(topk_lens_ptr + bid).to(tl.int32)
    ns = tl.load(swa_lens_ptr + bid).to(tl.int32)

    # Output row in workspace
    ws_row = bid * K_total + sid
    d_offs = tl.arange(0, BLOCK_D)
    ws_base = ws_ptr + ws_row * D

    if sid < nt:
        # Topk region: read from kv_flat
        src_idx = tl.load(topk_idx_ptr + bid * topk_idx_stride_b + sid).to(
            tl.int64)
        # Clamp to valid range
        src_idx = tl.where(src_idx < 0, tl.zeros_like(src_idx), src_idx)
        src_idx = tl.where(src_idx >= N_kv, N_kv - 1, src_idx)
        src_base = kv_ptr + src_idx * D
        for d_start in tl.static_range(0, D, BLOCK_D):
            d_idx = d_start + d_offs
            mask = d_idx < D
            vals = tl.load(src_base + d_idx, mask=mask)
            tl.store(ws_base + d_idx, vals, mask=mask)
    elif sid < nt + ns:
        # SWA region: read from swa_kv_flat
        swa_pos = sid - nt
        src_idx = tl.load(
            swa_idx_ptr + bid * swa_idx_stride_b + swa_pos).to(tl.int64)
        src_idx = tl.where(src_idx < 0, tl.zeros_like(src_idx), src_idx)
        src_idx = tl.where(src_idx >= N_swa, N_swa - 1, src_idx)
        src_base = swa_kv_ptr + src_idx * D
        for d_start in tl.static_range(0, D, BLOCK_D):
            d_idx = d_start + d_offs
            mask = d_idx < D
            vals = tl.load(src_base + d_idx, mask=mask)
            tl.store(ws_base + d_idx, vals, mask=mask)
    else:
        # Padding region: write zeros
        zero = tl.zeros([BLOCK_D], dtype=tl.bfloat16)
        for d_start in tl.static_range(0, D, BLOCK_D):
            d_idx = d_start + d_offs
            mask = d_idx < D
            tl.store(ws_base + d_idx, zero, mask=mask)


def fused_dual_gather(
    ws: torch.Tensor,           # [B, K_total, D] output
    topk_idx: torch.Tensor,     # [B, max_topk] int64, clamped
    swa_idx: torch.Tensor,      # [B, max_swa] int64, clamped
    topk_lens: torch.Tensor,    # [B] int32
    swa_lens: torch.Tensor,     # [B] int32
    max_topk: int,
    max_swa: int,
    kv_flat: torch.Tensor,      # [N_topk, D] bf16
    swa_kv_flat: torch.Tensor,  # [N_swa, D] bf16
) -> None:
    B, K_total, D = ws.shape
    assert K_total == max_topk + max_swa

    # Choose BLOCK_D: must be power of 2, >= D
    BLOCK_D = triton.next_power_of_2(D)

    grid = (B, K_total)
    _fused_dual_gather_kernel[grid](
        ws.view(-1, D),
        kv_flat,
        swa_kv_flat,
        topk_idx,
        swa_idx,
        topk_lens,
        swa_lens,
        topk_idx.stride(0),
        swa_idx.stride(0),
        K_total=K_total,
        max_topk=max_topk,
        max_swa=max_swa,
        D=D,
        BLOCK_D=BLOCK_D,
        N_kv=kv_flat.shape[0],
        N_swa=swa_kv_flat.shape[0],
    )


class CutlassSparseDecodeState:
    """Pre-allocated workspace for Triton sparse decode.

    Instantiate once per attention layer. All buffers are sized for the
    worst-case and reused every decode step — zero per-call allocations
    on the hot path.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        max_model_len: int,
        max_batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self._max_batch = max_batch_size
        self._device = device
        self._dtype = dtype

        # Lazily initialized on first call when we know max_topk + max_swa.
        self._K_total: int = 0
        self._kv_workspace: torch.Tensor | None = None
        self._ws_indices: torch.Tensor | None = None

    def _ensure_buffers(self, B: int, max_topk: int, max_swa: int) -> None:
        """Allocate/resize workspace buffers when shapes change."""
        K_total = max_topk + max_swa
        need_init = (
            self._kv_workspace is None
            or K_total != self._K_total
            or B > self._kv_workspace.shape[0] // max(self._K_total, 1)
        )
        if not need_init:
            return

        self._K_total = K_total
        max_B = max(B, self._max_batch)

        # Flat workspace: [max_B * K_total, head_dim]
        self._kv_workspace = torch.zeros(
            max_B * K_total, self.head_dim,
            dtype=self._dtype, device=self._device,
        )

        # Sequential indices: [max_B, K_total]
        # Row i = [i*K_total, i*K_total+1, ..., i*K_total+K_total-1]
        base = torch.arange(
            0, max_B * K_total, K_total,
            dtype=torch.int32, device=self._device,
        )
        offsets = torch.arange(
            K_total, dtype=torch.int32, device=self._device,
        )
        self._ws_indices = base.unsqueeze(1) + offsets.unsqueeze(0)

    def __call__(
        self,
        q: torch.Tensor,         # [B, num_heads, head_dim]
        kv_cache: torch.Tensor,  # [total_slots, head_dim] or [slots, 1, hd]
        swa_kv_cache: torch.Tensor,
        topk_indices: torch.Tensor,  # [B, max_topk]
        topk_lens: torch.Tensor,     # [B]
        swa_indices: torch.Tensor,   # [B, max_swa]
        swa_lens: torch.Tensor,      # [B]
        softmax_scale: float,
        out: torch.Tensor,           # [B, num_heads, head_dim]
    ) -> None:
        from vllm.v1.attention.ops.deepseek_v4_ops.c4_sparse_prefill_bf16 import (  # noqa: E501
            c4_sparse_prefill_bf16,
        )

        B = q.shape[0]
        head_dim = self.head_dim
        max_topk = topk_indices.shape[1]
        max_swa = swa_indices.shape[1]
        K_total = max_topk + max_swa

        self._ensure_buffers(B, max_topk, max_swa)

        # Flatten KV caches to [total_slots, head_dim]
        kv_flat = kv_cache.view(-1, head_dim)
        swa_kv_flat = swa_kv_cache.view(-1, head_dim)

        # Clamp indices to valid range (padding slots may be -1)
        topk_idx = topk_indices.long().clamp(0, kv_flat.shape[0] - 1)
        swa_idx = swa_indices.long().clamp(0, swa_kv_flat.shape[0] - 1)

        # Fused Triton gather: build compacted layout and gather KV data
        # from two source caches in a single kernel launch.
        ws = self._kv_workspace[:B * K_total].view(B, K_total, head_dim)
        fused_dual_gather(
            ws, topk_idx, swa_idx,
            topk_lens.to(torch.int32), swa_lens.to(torch.int32),
            max_topk, max_swa,
            kv_flat, swa_kv_flat,
        )

        # Combined lens: topk_lens + swa_lens
        combined_lens = (topk_lens + swa_lens).to(torch.int32)

        # Call Triton sparse prefill kernel (optimized for D=512 MQA)
        c4_sparse_prefill_bf16(
            q=q,
            kv_workspace=ws.view(-1, head_dim),
            topk_indices=self._ws_indices[:B, :K_total],
            topk_lens=combined_lens,
            softmax_scale=softmax_scale,
            out=out,
        )
