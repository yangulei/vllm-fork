# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Intel XPU fused ops for MiniMax-M3."""

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.models.minimax_m3.xpu.ops import xattention as _xattn
from vllm.models.minimax_m3.xpu.ops.gemma_rmsnorm import (
    gemma_fused_add_rmsnorm,
    gemma_rmsnorm,
)
from vllm.models.minimax_m3.xpu.ops.index_topk import (
    minimax_m3_index_decode,
    minimax_m3_index_score,
    minimax_m3_index_topk,
)
from vllm.models.minimax_m3.xpu.ops.sparse_attn import (
    minimax_m3_sparse_attn,
    minimax_m3_sparse_attn_decode,
)

logger = init_logger(__name__)


def _use_xattention_msa() -> bool:
    """Prefer the xattention SYCL MSA kernels when enabled and importable."""
    return envs.VLLM_XPU_USE_XATTENTION_MSA and _xattn.has_xattention_msa()


def install_xpu_sparse_attn() -> None:
    """Route MiniMax-M3 block-sparse attention to the fastest available XPU impl.

    Prefers the xattention SYCL kernels (``flash_attn_2_xpu``) when the extension
    is built and ``VLLM_XPU_USE_XATTENTION_MSA`` is set (default); otherwise falls
    back to the XPU-tuned Triton kernels (``xpu/ops/sparse_attn.py``).

    The cross-platform impl (``common.sparse_attention.MiniMaxM3SparseTritonImpl``)
    resolves ``minimax_m3_sparse_attn`` / ``minimax_m3_sparse_attn_decode`` as
    module globals (bound at import via ``from ...common.ops.sparse_attn import
    ...``). Rebind those names to the selected XPU copy so the specialised kernels
    are used without editing any cross-platform module. Mirrors the
    ``_install_xpu_rmsnorm`` monkeypatch pattern in ``xpu/model.py``. Idempotent.
    """
    from vllm.models.minimax_m3.common import sparse_attention as _common_attn

    if _use_xattention_msa():
        logger.info_once(
            "MiniMax-M3: dispatching block-sparse attention to xattention "
            "(flash_attn_2_xpu) SYCL kernels."
        )
        _common_attn.minimax_m3_sparse_attn = _xattn.minimax_m3_sparse_attn
        _common_attn.minimax_m3_sparse_attn_decode = (
            _xattn.minimax_m3_sparse_attn_decode
        )
        return

    _common_attn.minimax_m3_sparse_attn = minimax_m3_sparse_attn
    _common_attn.minimax_m3_sparse_attn_decode = minimax_m3_sparse_attn_decode


def install_xpu_index_topk() -> None:
    """Route the MiniMax-M3 lightning indexer to the fastest available XPU impl.

    Prefers the xattention SYCL kernels (``flash_attn_2_xpu``) when the extension
    is built and ``VLLM_XPU_USE_XATTENTION_MSA`` is set (default); otherwise falls
    back to the XPU-tuned Triton kernels (``xpu/ops/index_topk.py``).

    ``common.indexer`` resolves ``minimax_m3_index_score`` /
    ``minimax_m3_index_topk`` / ``minimax_m3_index_decode`` as module globals
    (bound at import via ``from ...common.ops.index_topk import ...``). Rebind
    those names to the selected XPU copy so the specialised indexer kernels are
    used without editing any cross-platform module. Mirrors
    ``install_xpu_sparse_attn``. Idempotent.
    """
    from vllm.models.minimax_m3.common import indexer as _common_indexer

    if _use_xattention_msa():
        logger.info_once(
            "MiniMax-M3: dispatching lightning indexer to xattention "
            "(flash_attn_2_xpu) SYCL kernels."
        )
        _common_indexer.minimax_m3_index_score = _xattn.minimax_m3_index_score
        _common_indexer.minimax_m3_index_topk = _xattn.minimax_m3_index_topk
        _common_indexer.minimax_m3_index_decode = _xattn.minimax_m3_index_decode
        return

    _common_indexer.minimax_m3_index_score = minimax_m3_index_score
    _common_indexer.minimax_m3_index_topk = minimax_m3_index_topk
    _common_indexer.minimax_m3_index_decode = minimax_m3_index_decode


__all__ = [
    "gemma_rmsnorm",
    "gemma_fused_add_rmsnorm",
    "minimax_m3_sparse_attn",
    "minimax_m3_sparse_attn_decode",
    "minimax_m3_index_score",
    "minimax_m3_index_topk",
    "minimax_m3_index_decode",
    "install_xpu_sparse_attn",
    "install_xpu_index_topk",
]
