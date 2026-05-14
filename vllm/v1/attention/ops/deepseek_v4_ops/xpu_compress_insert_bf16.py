# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportPrivateImportUsage=false, reportCallIssue=false, reportUnusedVariable=false, reportGeneralTypeIssues=false, reportUnreachable=false

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""XPU bf16 fused compress + RMSNorm + RoPE + KV cache insert kernel.

Fused kernel for DeepSeek-V4 compressor on XPU: gathers state cache entries,
computes softmax-weighted compression, applies RMSNorm and RoPE, then stores
the resulting 512-dim vector into a bf16 paged KV cache.
Used for all MLA compressor layers (compress_ratio=4 or 128, head_dim=512).
"""

import torch

from vllm.triton_utils import tl, triton

HEAD_SIZE = 512
TRITON_BLOCK_SIZE = 512


@triton.jit
def _xpu_compress_insert_bf16_kernel(
    state_cache_ptr,
    state_cache_stride0,
    state_cache_stride1,
    state_width,
    token_to_req_indices_ptr,
    positions_ptr,
    slot_mapping_ptr,
    block_table_ptr,
    block_table_stride0,
    state_cache_block_size,
    rms_norm_weight_ptr,
    rms_norm_eps,
    cos_sin_cache_ptr,
    cos_sin_stride0,
    k_cache_ptr,
    kv_slot_mapping_ptr,
    kv_cache_block_size,
    k_cache_stride0,
    k_cache_stride1,
    HEAD_SIZE_C: tl.constexpr,
    TRITON_BLOCK_SIZE_C: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    OVERLAP: tl.constexpr,
    ROPE_HEAD_DIM: tl.constexpr,
):
    token_idx = tl.program_id(0)

    slot_id = tl.load(slot_mapping_ptr + token_idx)
    position = tl.load(positions_ptr + token_idx)
    kv_slot_idx = tl.load(kv_slot_mapping_ptr + token_idx)

    # Guard: only store for valid tokens at compression boundaries
    should_store = (slot_id >= 0) & ((position + 1) % COMPRESS_RATIO == 0) & (kv_slot_idx >= 0)

    # Use safe defaults for conditional computation
    req_idx = tl.load(token_to_req_indices_ptr + token_idx)

    N_GATHER: tl.constexpr = (1 + OVERLAP) * COMPRESS_RATIO
    NOPE_HEAD_DIM: tl.constexpr = HEAD_SIZE_C - ROPE_HEAD_DIM
    HALF_ROPE: tl.constexpr = ROPE_HEAD_DIM // 2
    NUM_PAIRS: tl.constexpr = TRITON_BLOCK_SIZE_C // 2
    NOPE_PAIRS: tl.constexpr = NOPE_HEAD_DIM // 2

    gather_offsets = tl.arange(0, N_GATHER)
    start = position - N_GATHER + 1
    gather_pos = start + gather_offsets
    valid_pos = gather_pos >= 0
    safe_pos = tl.maximum(gather_pos, 0)
    head_offset = (gather_offsets >= COMPRESS_RATIO).to(tl.int32) * HEAD_SIZE_C

    logical_block_idx = safe_pos // state_cache_block_size
    state_block_idx = tl.load(
        block_table_ptr + req_idx * block_table_stride0 + logical_block_idx,
        mask=valid_pos,
        other=0,
    )
    pos_in_block = safe_pos % state_cache_block_size

    offsets = tl.arange(0, TRITON_BLOCK_SIZE_C)
    valid_offsets = offsets < HEAD_SIZE_C
    state_block_idx_i64 = state_block_idx.to(tl.int64)
    row_base = (
        state_cache_ptr
        + state_block_idx_i64 * state_cache_stride0
        + pos_in_block * state_cache_stride1
    )
    load_mask = valid_pos[:, None] & valid_offsets[None, :]

    scores = tl.load(
        row_base[:, None] + state_width + head_offset[:, None] + offsets[None, :],
        mask=load_mask,
        other=float("-inf"),
    )
    weights = tl.softmax(scores, dim=0)

    kv = tl.load(
        row_base[:, None] + head_offset[:, None] + offsets[None, :],
        mask=load_mask,
        other=0.0,
    )
    compressed_kv = tl.sum(kv * weights, axis=0)

    rms_w = tl.load(rms_norm_weight_ptr + offsets, mask=valid_offsets, other=0.0)
    variance = tl.sum(compressed_kv * compressed_kv, axis=0) / HEAD_SIZE_C
    normed = compressed_kv * tl.rsqrt(variance + rms_norm_eps) * rms_w

    pair_2d = tl.reshape(normed, (NUM_PAIRS, 2))
    even, odd = tl.split(pair_2d)
    pair_idx = tl.arange(0, NUM_PAIRS)
    rope_pair_local = pair_idx - NOPE_PAIRS
    is_rope_pair = rope_pair_local >= 0
    cs_idx = tl.maximum(rope_pair_local, 0)
    compressed_pos = (position // COMPRESS_RATIO) * COMPRESS_RATIO
    rope_base = cos_sin_cache_ptr + compressed_pos * cos_sin_stride0
    cos_v = tl.load(rope_base + cs_idx, mask=is_rope_pair, other=1.0)
    sin_v = tl.load(rope_base + HALF_ROPE + cs_idx, mask=is_rope_pair, other=0.0)

    rotated_even = even * cos_v - odd * sin_v
    rotated_odd = odd * cos_v + even * sin_v
    even = tl.where(is_rope_pair, rotated_even, even)
    odd = tl.where(is_rope_pair, rotated_odd, odd)
    result = tl.interleave(even, odd)

    # Only store if this token should produce a compressed entry
    kv_block_idx = (kv_slot_idx // kv_cache_block_size).to(tl.int64)
    kv_pos_in_block = (kv_slot_idx % kv_cache_block_size).to(tl.int64)
    # When should_store is False, kv_slot_idx=-1 → kv_block_idx/pos are garbage,
    # but the store mask prevents any write.
    dst_ptr = (
        k_cache_ptr
        + kv_block_idx * k_cache_stride0
        + kv_pos_in_block * k_cache_stride1
        + offsets
    )
    tl.store(dst_ptr, result.to(k_cache_ptr.dtype.element_ty),
             mask=valid_offsets & should_store)


def _ref_compress_insert_bf16(
    state_cache: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    cos_sin_cache: torch.Tensor,
    k_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    kv_cache_block_size: int,
    compress_ratio: int,
    overlap: int,
    rope_head_dim: int,
) -> None:
    state_width = state_cache.shape[-1] // 2
    head_size = state_width // (1 + overlap)
    state_block_size = state_cache.shape[1]
    n_gather = (1 + overlap) * compress_ratio
    nope_head_dim = head_size - rope_head_dim
    half_rope = rope_head_dim // 2

    for token_idx in range(positions.numel()):
        if int(slot_mapping[token_idx].item()) < 0:
            continue

        position = int(positions[token_idx].item())
        if (position + 1) % compress_ratio != 0:
            continue

        req_idx = int(token_to_req_indices[token_idx].item())
        start = position - n_gather + 1
        kv_rows = []
        score_rows = []
        for gather_idx in range(n_gather):
            gather_pos = start + gather_idx
            if gather_pos < 0:
                kv_rows.append(torch.zeros(head_size, dtype=torch.float32, device=state_cache.device))
                score_rows.append(
                    torch.full(
                        (head_size,),
                        float("-inf"),
                        dtype=torch.float32,
                        device=state_cache.device,
                    )
                )
                continue

            logical_block_idx = gather_pos // state_block_size
            state_block_idx = int(block_table[req_idx, logical_block_idx].item())
            pos_in_block = gather_pos % state_block_size
            row = state_cache[state_block_idx, pos_in_block].float()
            head_offset = head_size if gather_idx >= compress_ratio else 0
            kv_rows.append(row[head_offset : head_offset + head_size])
            score_offset = state_width + head_offset
            score_rows.append(row[score_offset : score_offset + head_size])

        kv = torch.stack(kv_rows, dim=0)
        scores = torch.stack(score_rows, dim=0)
        weights = torch.softmax(scores, dim=0)
        compressed = (kv * weights).sum(dim=0)

        variance = compressed.pow(2).mean()
        normed = compressed * torch.rsqrt(variance + rms_norm_eps)
        normed = normed * rms_norm_weight.float()

        if rope_head_dim > 0:
            compressed_pos = (position // compress_ratio) * compress_ratio
            cos = cos_sin_cache[compressed_pos, :half_rope].float()
            sin = cos_sin_cache[compressed_pos, half_rope:].float()
            rope = normed[nope_head_dim:].reshape(half_rope, 2)
            even = rope[:, 0]
            odd = rope[:, 1]
            rotated = torch.empty_like(rope)
            rotated[:, 0] = even * cos - odd * sin
            rotated[:, 1] = odd * cos + even * sin
            normed = normed.clone()
            normed[nope_head_dim:] = rotated.reshape(rope_head_dim)

        kv_slot = int(kv_slot_mapping[token_idx].item())
        if kv_slot < 0:
            continue
        kv_block_idx = kv_slot // kv_cache_block_size
        kv_pos_in_block = kv_slot % kv_cache_block_size
        k_cache[kv_block_idx, kv_pos_in_block] = normed.to(torch.bfloat16)


def xpu_compress_insert_bf16(
    state_cache: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    cos_sin_cache: torch.Tensor,
    k_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    kv_cache_block_size: int,
    compress_ratio: int,
    overlap: int,
    rope_head_dim: int,
) -> None:
    assert state_cache.dim() == 3
    state_width = state_cache.shape[-1] // 2
    assert state_cache.shape[-1] % 2 == 0
    assert state_width == (1 + overlap) * HEAD_SIZE
    assert state_cache.dtype in (torch.bfloat16, torch.float32)
    assert token_to_req_indices.dtype == torch.int32
    assert positions.dtype == torch.int64
    assert slot_mapping.dtype == torch.int64
    assert block_table.dtype == torch.int32
    assert rms_norm_weight.shape == (HEAD_SIZE,)
    assert rms_norm_weight.dtype == torch.float32
    assert cos_sin_cache.dim() == 2 and cos_sin_cache.shape[1] == rope_head_dim
    assert cos_sin_cache.dtype == torch.float32
    assert k_cache.dim() == 3 and k_cache.shape[-1] == HEAD_SIZE
    assert k_cache.dtype == torch.bfloat16
    assert kv_slot_mapping.dtype == torch.int64
    assert token_to_req_indices.numel() == positions.numel() == slot_mapping.numel()
    assert kv_slot_mapping.numel() == positions.numel()
    assert compress_ratio > 0
    assert overlap >= 0
    assert rope_head_dim % 2 == 0
    assert rope_head_dim <= HEAD_SIZE
    assert state_cache.shape[1] > 0
    assert kv_cache_block_size == k_cache.shape[1]

    if positions.numel() == 0:
        return

    if state_cache.device.type != "xpu":
        _ref_compress_insert_bf16(
            state_cache=state_cache,
            token_to_req_indices=token_to_req_indices,
            positions=positions,
            slot_mapping=slot_mapping,
            block_table=block_table,
            rms_norm_weight=rms_norm_weight,
            rms_norm_eps=rms_norm_eps,
            cos_sin_cache=cos_sin_cache,
            k_cache=k_cache,
            kv_slot_mapping=kv_slot_mapping,
            kv_cache_block_size=kv_cache_block_size,
            compress_ratio=compress_ratio,
            overlap=overlap,
            rope_head_dim=rope_head_dim,
        )
        return

    num_tokens = positions.numel()
    state_width = state_cache.shape[-1] // 2
    _xpu_compress_insert_bf16_kernel[(num_tokens,)](
        state_cache,
        state_cache.stride(0),
        state_cache.stride(1),
        state_width,
        token_to_req_indices,
        positions,
        slot_mapping,
        block_table,
        block_table.stride(0),
        state_cache.shape[1],  # state_cache_block_size
        rms_norm_weight,
        rms_norm_eps,
        cos_sin_cache,
        cos_sin_cache.stride(0),
        k_cache,
        kv_slot_mapping,
        kv_cache_block_size,
        k_cache.stride(0),
        k_cache.stride(1),
        HEAD_SIZE_C=HEAD_SIZE,
        TRITON_BLOCK_SIZE_C=TRITON_BLOCK_SIZE,
        COMPRESS_RATIO=compress_ratio,
        OVERLAP=overlap,
        ROPE_HEAD_DIM=rope_head_dim,
    )


__all__ = ["xpu_compress_insert_bf16"]
