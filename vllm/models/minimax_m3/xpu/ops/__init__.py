# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Intel XPU fused ops for MiniMax-M3."""

from vllm.models.minimax_m3.xpu.ops.gemma_rmsnorm import (
    gemma_fused_add_rmsnorm,
    gemma_rmsnorm,
)
from vllm.models.minimax_m3.xpu.ops.sparse_attn import (
    minimax_m3_sparse_attn,
    minimax_m3_sparse_attn_decode,
)


def install_xpu_sparse_attn() -> None:
    """Route MiniMax-M3 block-sparse attention to the XPU-tuned Triton kernels.

    The cross-platform impl (``common.sparse_attention.MiniMaxM3SparseTritonImpl``)
    resolves ``minimax_m3_sparse_attn`` / ``minimax_m3_sparse_attn_decode`` as
    module globals (bound at import via ``from ...common.ops.sparse_attn import
    ...``). Rebind those names to the XPU copy so the XPU-specialised kernels are
    used without editing any cross-platform module. Mirrors the
    ``_install_xpu_rmsnorm`` monkeypatch pattern in ``xpu/model.py``. Idempotent.
    """
    from vllm.models.minimax_m3.common import sparse_attention as _common_attn

    _common_attn.minimax_m3_sparse_attn = minimax_m3_sparse_attn
    _common_attn.minimax_m3_sparse_attn_decode = minimax_m3_sparse_attn_decode


__all__ = [
    "gemma_rmsnorm",
    "gemma_fused_add_rmsnorm",
    "minimax_m3_sparse_attn",
    "minimax_m3_sparse_attn_decode",
    "install_xpu_sparse_attn",
]
