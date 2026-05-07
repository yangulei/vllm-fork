# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
CUTLASS paged decode for DeepSeek V4 sparse attention on XPU.

Strategy:
  1. Gather topk + SWA KV slots into a pre-allocated paged buffer.
  2. Split Q heads into GQA groups of 16 (kernel's max ratio).
  3. Call flash_attn_varlen_func with pre-built block_table + seqused_k.

Performance: all metadata tensors (cu_seqlens_q, block_table) are pre-allocated
once and reused across decode steps. Only the KV gather (index into cache) and
seqused_k fill happen per call — zero torch.cat/zeros/arange allocations.
"""

import torch


# Block size for the temporary paged KV buffer.
# 16 is the smallest supported by the CUTLASS paged decode kernel.
_PAGE_SIZE = 16

# Max GQA ratio supported by the CUTLASS kernel
_MAX_GQA_RATIO = 16


class CutlassSparseDecodeState:
    """Pre-allocated workspace for CUTLASS sparse decode.

    Instantiate once per attention layer. All buffers are sized for the
    worst-case (max_model_len, max_batch_size) and reused every decode step.
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
        # GQA split params
        assert num_heads % _MAX_GQA_RATIO == 0 or num_heads <= _MAX_GQA_RATIO
        if num_heads <= _MAX_GQA_RATIO:
            self.gqa_groups = 1
            self.q_per_group = num_heads
        else:
            self.gqa_groups = num_heads // _MAX_GQA_RATIO
            self.q_per_group = _MAX_GQA_RATIO

        self.num_heads = num_heads
        self.head_dim = head_dim

        # Max KV slots per request: use max_model_len as a safe upper bound
        # (topk_indices can be up to index_topk=512, swa up to window_size,
        # but we don't know index_topk here, so just use max_model_len).
        max_kv_total = max_model_len
        max_kv_padded = ((max_kv_total + _PAGE_SIZE - 1) // _PAGE_SIZE
                         * _PAGE_SIZE)
        self.max_blocks_per_seq = max_kv_padded // _PAGE_SIZE
        self.max_kv_padded = max_kv_padded

        max_B = max_batch_size
        total_blocks = max_B * self.max_blocks_per_seq
        eff_batch = max_B * self.gqa_groups

        # Pre-allocate paged KV buffer (filled each call via index gather)
        self._kv_buf = torch.zeros(
            total_blocks, _PAGE_SIZE, 1, head_dim,
            dtype=dtype, device=device,
        )

        # cu_seqlens_q: [0, 1, 2, ..., eff_batch] — fixed for decode
        self._cu_seqlens_q = torch.arange(
            0, eff_batch + 1, dtype=torch.int32, device=device,
        )

        # Block table: sequential blocks, repeated for GQA groups
        # Shape: [eff_batch, max_blocks_per_seq]
        # For B requests: request i uses blocks [i*mpbs, (i+1)*mpbs)
        seq_blocks = torch.arange(
            total_blocks, dtype=torch.int32, device=device,
        ).view(max_B, self.max_blocks_per_seq)
        # Repeat each row gqa_groups times: [max_B*gqa, max_blocks_per_seq]
        self._block_table_full = seq_blocks.repeat_interleave(
            self.gqa_groups, dim=0
        )

        # seqused_k: [eff_batch] — filled each call
        self._seqused_k = torch.zeros(
            eff_batch, dtype=torch.int32, device=device,
        )

        self._max_batch = max_B
        self._device = device

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
        from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func

        B = q.shape[0]
        head_dim = self.head_dim
        gqa_groups = self.gqa_groups
        eff_batch = B * gqa_groups

        # Flatten KV caches
        kv_flat = kv_cache.view(-1, head_dim)
        swa_kv_flat = swa_kv_cache.view(-1, head_dim)

        # Actual slot counts this call
        max_topk = topk_indices.shape[1]
        max_swa = swa_indices.shape[1]

        # Always use pre-allocated max layout. The kernel reads only up to
        # seqused_k tokens, so extra blocks in the buffer are harmless.
        blocks_per_seq = self.max_blocks_per_seq
        total_blocks = B * blocks_per_seq

        # Clamp indices to valid range (padding slots may be -1)
        topk_idx = topk_indices.long().clamp(0, kv_flat.shape[0] - 1)
        swa_idx = swa_indices.long().clamp(0, swa_kv_flat.shape[0] - 1)

        # Gather into pre-allocated buffer with COMPACTED layout.
        # The kernel reads positions [0, seqused_k) sequentially, so valid
        # topk tokens must be followed immediately by valid SWA tokens with
        # NO gap. We write per-request to place SWA at offset topk_lens[i].
        kv_buf = self._kv_buf[:total_blocks].view(
            B, blocks_per_seq * _PAGE_SIZE, head_dim)

        for i in range(B):
            n_topk = topk_lens[i].item()
            n_swa = swa_lens[i].item()
            # Valid topk at [0, n_topk)
            kv_buf[i, :n_topk] = kv_flat[topk_idx[i, :n_topk]]
            # Valid SWA immediately after at [n_topk, n_topk+n_swa)
            kv_buf[i, n_topk:n_topk + n_swa] = (
                swa_kv_flat[swa_idx[i, :n_swa]])

        # Reshape back to paged format for kernel
        gathered_kv = self._kv_buf[:total_blocks]

        # Fill seqused_k: actual KV length per GQA-expanded sequence
        all_lens = (topk_lens + swa_lens).to(torch.int32)
        # Expand [B] -> [B*gqa_groups] without allocation
        self._seqused_k[:eff_batch].view(B, gqa_groups)[:] = (
            all_lens.unsqueeze(1).expand(B, gqa_groups))
        seqused_k = self._seqused_k[:eff_batch]

        # Block table: always use pre-allocated sequential layout
        block_table = self._block_table_full[:eff_batch]

        # Q reshape: [B, 128, hd] -> [B*gqa, 16, hd] (contiguous view)
        q_grouped = q.view(B * gqa_groups, self.q_per_group, head_dim)

        # cu_seqlens_q slice
        cu_seqlens_q = self._cu_seqlens_q[:eff_batch + 1]

        # Call CUTLASS paged decode
        fa_out = flash_attn_varlen_func(
            q=q_grouped,
            k=gathered_kv,
            v=gathered_kv,  # MLA: K and V share the same latent
            max_seqlen_q=1,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_k=self.max_kv_padded,
            seqused_k=seqused_k,
            softmax_scale=softmax_scale,
            causal=False,
            block_table=block_table,
            out=None,
        )

        # fa_out: [B*gqa, q_per_group, head_dim] -> [B, num_heads, head_dim]
        out.copy_(fa_out.view(B, self.num_heads, head_dim))
