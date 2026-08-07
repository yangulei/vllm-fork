# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dispatch MiniMax-M3 sparse attention (MSA) to the xattention SYCL kernels.

The lightning indexer (block score + top-k) and block-sparse GQA attend for
MiniMax-M3 have hand-tuned SYCL/SYCL-TLA implementations in the ``xattention``
extension (``intel-innersource/...deepklox``).  It exposes them through one
framework-neutral Python interface (``xattention/msa_interface.py``) that vLLM and
SGLang share:

  * ``msa_index_score`` / ``msa_index_topk`` -- prefill indexer, stages 1a / 1b.
  * ``msa_index_decode``                     -- decode indexer (fused).
  * ``msa_sparse_attn`` / ``msa_sparse_attn_decode`` -- block-sparse attend.

Those entry points are keyword-only, default ``sm_scale``, return their output,
and accept either paged-addressing convention.  vLLM's cross-platform modules
instead resolve the ``minimax_m3_*`` names as module globals, interchangeably with
the Triton fallback, so the wrappers below keep that signature and adapt it.

The extension supports the production MiniMax-M3 shapes (index/head dim 128, GQA
group size 16, top-k <= 64); when it is not built the caller falls back to the
Triton kernels (see ``xpu/ops/__init__.py``).
"""

from functools import cache

import torch

# The MSA entry points the extension must export for this dispatch path to be
# usable.  A build without ``XATTENTION_ENABLED_KERNELS=msa`` imports fine but
# lacks these.
_REQUIRED_OPS = (
    "msa_index_score",
    "msa_index_topk",
    "msa_index_decode",
    "msa_sparse_attn",
    "msa_sparse_attn_decode",
)


@cache
def _load_xattention():
    """Import ``xattention`` once, returning the module or ``None``.

    Returns ``None`` if the extension is not installed or was built without the
    MSA kernels, so callers can fall back to the Triton XPU kernels. Cached so the
    import is attempted at most once.
    """
    try:
        import xattention
    except ImportError:
        return None
    if not all(hasattr(xattention, op) for op in _REQUIRED_OPS):
        return None
    return xattention


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
    return _load_xattention().msa_index_score(
        idx_q,
        index_kv_cache=index_kv_cache,
        seq_lens=seq_lens,
        block_table=block_table,
        cu_seqlens=cu_seqlens_q,
        prefix_lens=prefix_lens,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_seq_len,
        sm_scale=sm_scale,
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
    return _load_xattention().msa_index_topk(
        score,
        topk=topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        cu_seqlens=cu_seqlens_q,
        prefix_lens=prefix_lens,
        max_seqlen_q=max_query_len,
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
    return _load_xattention().msa_index_decode(
        idx_q,
        index_kv_cache=index_kv_cache,
        seq_lens=seq_lens,
        topk=topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        block_table=block_table,
        max_seqlen=max_seq_len,
        decode_query_len=decode_query_len,
        sm_scale=sm_scale,
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
    _load_xattention().msa_sparse_attn(
        q,
        topk_idx=topk_idx,
        seq_lens=seq_lens,
        kv_cache=kv_cache,
        block_table=block_table,
        cu_seqlens=cu_seqlens_q,
        prefix_lens=prefix_lens,
        max_seqlen_q=max_query_len,
        sm_scale=sm_scale,
        out=output,
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
    _load_xattention().msa_sparse_attn_decode(
        q,
        topk_idx=topk_idx,
        seq_lens=seq_lens,
        kv_cache=kv_cache,
        block_table=block_table,
        decode_query_len=decode_query_len,
        sm_scale=sm_scale,
        out=output,
    )
