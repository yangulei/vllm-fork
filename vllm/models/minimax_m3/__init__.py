# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax M3 model — hardware-isolated entry point.

The implementation lives under ``nvidia/``, ``amd/`` and ``xpu/``; this module
picks the right one for the current platform and re-exports the public classes
used by the model registry. (Mirrors ``vllm.models.deepseek_v4``.) The ``xpu``
package reuses the NVIDIA classes and overrides only the Gemma RMSNorm.
"""

from typing import TYPE_CHECKING

from vllm.platforms import current_platform

# The NVIDIA branch is the static default that type-checkers see; the ROCm and
# XPU branches override it at runtime (kept type-compatible via type: ignore).
if TYPE_CHECKING or not (
    current_platform.is_rocm() or current_platform.is_xpu()
):
    from .nvidia.model import (
        MiniMaxM3SparseForCausalLM,
        MiniMaxM3SparseForConditionalGeneration,
    )
    from .nvidia.mtp import MiniMaxM3MTP
elif current_platform.is_xpu():
    from .xpu.model import (  # type: ignore[assignment]
        MiniMaxM3SparseForCausalLM,
        MiniMaxM3SparseForConditionalGeneration,
    )
    from .xpu.mtp import MiniMaxM3MTP  # type: ignore[assignment]
    from .xpu.ops import install_xpu_index_topk, install_xpu_sparse_attn

    # Route the cross-platform Triton block-sparse attend + lightning indexer to
    # the XPU-tuned kernels (xpu/ops/{sparse_attn,index_topk}.py) without editing
    # the common modules.
    install_xpu_sparse_attn()
    install_xpu_index_topk()
else:
    from .amd.model import (  # type: ignore[assignment]
        MiniMaxM3SparseForCausalLM,
        MiniMaxM3SparseForConditionalGeneration,
    )
    from .amd.mtp import MiniMaxM3MTP  # type: ignore[assignment]

__all__ = [
    "MiniMaxM3MTP",
    "MiniMaxM3SparseForCausalLM",
    "MiniMaxM3SparseForConditionalGeneration",
]
