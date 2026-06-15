# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax M3 model — hardware-isolated entry point.

The implementation lives under ``nvidia/`` and ``amd/``; this module picks the
right one for the current platform and re-exports the public classes used by
the model registry. (Mirrors ``vllm.models.deepseek_v4``.)

# XPU: the public class names are resolved lazily (PEP 562 ``__getattr__``) so
# that importing this package does not eagerly pull in the heavy NVIDIA/AMD
# model + multimodal stack. XPU runs the MSA path through the platform-neutral
# ``common`` package (Triton indexer + block-sparse attend, see
# ``common.indexer.select_indexer_impl_cls`` and
# ``common.sparse_attention.select_main_impl_cls``), which lets the standalone
# MSA ops be imported and unit-tested on XPU without the full model framework.
# The lazy import keeps this file API-compatible with upstream so the change
# rebases cleanly once vllm-project/vllm#45381 lands.
"""

from typing import TYPE_CHECKING

from vllm.platforms import current_platform

if TYPE_CHECKING:
    from .nvidia.model import (
        MiniMaxM3SparseForCausalLM,
        MiniMaxM3SparseForConditionalGeneration,
    )
    from .nvidia.mtp import MiniMaxM3MTP

__all__ = [
    "MiniMaxM3MTP",
    "MiniMaxM3SparseForCausalLM",
    "MiniMaxM3SparseForConditionalGeneration",
]


def _load(name: str):
    # The NVIDIA branch is the static default; the ROCm branch overrides it at
    # runtime. Both ultimately select the Triton MSA impl on non-Blackwell /
    # non-CUDA platforms (XPU included).
    if not current_platform.is_rocm():
        from . import nvidia as _impl_pkg
    else:
        from . import amd as _impl_pkg  # type: ignore[no-redef]

    if name == "MiniMaxM3MTP":
        from importlib import import_module

        return getattr(import_module(f"{_impl_pkg.__name__}.mtp"), "MiniMaxM3MTP")
    from importlib import import_module

    return getattr(import_module(f"{_impl_pkg.__name__}.model"), name)


def __getattr__(name: str):  # PEP 562
    if name in __all__:
        return _load(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
