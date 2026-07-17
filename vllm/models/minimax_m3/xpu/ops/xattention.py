# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dispatch MiniMax-M3 sparse attention (MSA) to the xattention SYCL kernels.

The lightning indexer (block score + top-k) and block-sparse GQA attend for
MiniMax-M3 have hand-tuned SYCL/SYCL-TLA implementations in the ``xattention``
extension (``intel-innersource/...xattention``), exposed as pybind functions on
the ``xattention._C`` module:

  * ``minimax_m3_index_score`` / ``minimax_m3_index_topk`` -- prefill indexer.
  * ``minimax_m3_index_decode``                           -- decode indexer.
  * ``minimax_m3_sparse_attn`` / ``minimax_m3_sparse_attn_decode`` -- attend.

Their signatures mirror the Triton XPU wrappers in ``index_topk.py`` /
``sparse_attn.py`` one-for-one, so these thin wrappers simply forward the call.
The extension supports the production MiniMax-M3 shapes (index/head dim 128,
GQA group size 16, top-k <= 64); when it is not built the caller falls back to
the Triton kernels (see ``xpu/ops/__init__.py``).
"""

from functools import cache

import torch

# The xattention pybind extension module name (see setup.py ``xattention._C``).
_XATTENTION_MODULE = "xattention._C"

# The MSA ops the extension must export for this dispatch path to be usable.
_REQUIRED_OPS = (
    "minimax_m3_index_score",
    "minimax_m3_index_topk",
    "minimax_m3_index_decode",
    "minimax_m3_sparse_attn",
    "minimax_m3_sparse_attn_decode",
)


@cache
def _load_xattention():
    """Import ``xattention._C`` once, returning the module or ``None``.

    Returns ``None`` if the extension is not installed or was built without the
    MSA kernels (``XATTENTION_ENABLED_KERNELS_MSA``), so callers can fall back to
    the Triton XPU kernels. Cached so the import is attempted at most once.
    """
    try:
        import xattention._C as _msa
    except ImportError:
        return None
    if not all(hasattr(_msa, op) for op in _REQUIRED_OPS):
        return None
    return _msa


def has_xattention_msa() -> bool:
    """True when the xattention MSA kernels are importable on this system."""
    return _load_xattention() is not None


# ---------------------------------------------------------------------------
# Indexer
# ---------------------------------------------------------------------------
@torch.no_grad()
def minimax_m3_index_score(
    idx_q: torch.Tensor,  # [total_q, num_idx_heads, head_dim]
    index_kv_cache: torch.Tensor,  # [num_blocks, 128, head_dim]
    block_table: torch.Tensor,  # [batch, max_blocks]
    cu_seqlens_q: torch.Tensor,  # [batch+1] int32
    seq_lens: torch.Tensor,  # [batch] int32
    prefix_lens: torch.Tensor,  # [batch] int32
    max_query_len: int,
    max_seq_len: int,
    num_kv_heads: int,
    sm_scale: float,
) -> torch.Tensor:
    """Prefill lightning-indexer block scores (xattention SYCL kernel)."""
    return _load_xattention().minimax_m3_index_score(
        idx_q,
        index_kv_cache,
        block_table,
        cu_seqlens_q,
        seq_lens,
        prefix_lens,
        max_query_len,
        max_seq_len,
        num_kv_heads,
        sm_scale,
    )


@torch.no_grad()
def minimax_m3_index_topk(
    score: torch.Tensor,  # [num_idx_heads, total_q, max_block]
    cu_seqlens_q: torch.Tensor,  # [batch+1] int32
    prefix_lens: torch.Tensor,  # [batch] int32
    max_query_len: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
) -> torch.Tensor:
    """Prefill indexer top-k block selection (xattention SYCL kernel)."""
    return _load_xattention().minimax_m3_index_topk(
        score,
        cu_seqlens_q,
        prefix_lens,
        max_query_len,
        topk,
        init_blocks,
        local_blocks,
    )


@torch.no_grad()
def minimax_m3_index_decode(
    idx_q: torch.Tensor,  # [total_q, num_idx_heads, head_dim]
    index_kv_cache: torch.Tensor,  # [num_blocks, 128, head_dim]
    block_table: torch.Tensor,  # [num_reqs, max_blocks]
    seq_lens: torch.Tensor,  # [num_reqs] int32
    max_seq_len: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    num_kv_heads: int,
    sm_scale: float,
    decode_query_len: int,
) -> torch.Tensor:
    """Decode lightning indexer, fused score + top-k (xattention SYCL kernel)."""
    return _load_xattention().minimax_m3_index_decode(
        idx_q,
        index_kv_cache,
        block_table,
        seq_lens,
        max_seq_len,
        topk,
        init_blocks,
        local_blocks,
        num_kv_heads,
        sm_scale,
        decode_query_len,
    )


# ---------------------------------------------------------------------------
# Block-sparse attend
# ---------------------------------------------------------------------------
@torch.no_grad()
def minimax_m3_sparse_attn(
    q: torch.Tensor,  # [total_q, num_heads, head_dim]
    kv_cache: torch.Tensor,  # [num_blocks, 2, 128, num_kv_heads, head_dim]
    topk_idx: torch.Tensor,  # [num_kv_heads, total_q, topk]
    block_table: torch.Tensor,  # [batch, max_blocks]
    cu_seqlens_q: torch.Tensor,  # [batch+1] int32
    seq_lens: torch.Tensor,  # [batch] int32
    prefix_lens: torch.Tensor,  # [batch] int32
    max_query_len: int,
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,  # [total_q, num_heads, head_dim]
) -> None:
    """Prefill block-sparse GQA attend (xattention SYCL kernel)."""
    _load_xattention().minimax_m3_sparse_attn(
        q,
        kv_cache,
        topk_idx,
        block_table,
        cu_seqlens_q,
        seq_lens,
        prefix_lens,
        max_query_len,
        num_kv_heads,
        sm_scale,
        output,
    )


@torch.no_grad()
def minimax_m3_sparse_attn_decode(
    q: torch.Tensor,  # [total_q, num_heads, head_dim]
    kv_cache: torch.Tensor,  # [num_blocks, 2, 128, num_kv_heads, head_dim]
    topk_idx: torch.Tensor,  # [num_kv_heads, total_q, topk]
    block_table: torch.Tensor,  # [num_reqs, max_blocks]
    seq_lens: torch.Tensor,  # [num_reqs] int32
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,  # [total_q, num_heads, head_dim]
    decode_query_len: int,
) -> None:
    """Decode block-sparse GQA attend (xattention SYCL kernel)."""
    _load_xattention().minimax_m3_sparse_attn_decode(
        q,
        kv_cache,
        topk_idx,
        block_table,
        seq_lens,
        num_kv_heads,
        sm_scale,
        output,
        decode_query_len,
    )
