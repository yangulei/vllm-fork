# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dispatch tests for the MiniMax-M3 sparse-attention (MSA) XPU ops.

These verify the *routing* logic in ``vllm.models.minimax_m3.xpu.ops`` — that
the lightning indexer and block-sparse attend are dispatched to the xattention
SYCL kernels (the ``xattention`` package) when that extension is available and
``VLLM_XPU_USE_XATTENTION_MSA`` is set, and fall back to the Triton XPU kernels
otherwise. The routing is exercised with a stub extension so the test runs on
CPU without XPU hardware or a built extension.
"""

import sys
import types

import pytest

# The framework-neutral entry points the extension must export. The
# shim rejects a build that lacks them (e.g. one built without
# ``XATTENTION_ENABLED_KERNELS=msa``) and falls back to Triton.
_MSA_OPS = (
    "msa_index_score",
    "msa_index_topk",
    "msa_index_decode",
    "msa_sparse_attn",
    "msa_sparse_attn_decode",
)


def _make_stub_extension(ops: tuple[str, ...] = _MSA_OPS) -> types.ModuleType:
    stub = types.ModuleType("xattention")
    stub.__path__ = []  # mark as a package
    for op in ops:
        setattr(stub, op, lambda *a, **k: None)
    return stub


def _install_stub_extension(monkeypatch, ext: types.ModuleType | None) -> None:
    """Install a stub ``xattention`` package (or force its import to fail).

    Keeps the real (XPU-only) package off the import path. Passing ``ext=None``
    registers a ``None`` entry in ``sys.modules``, which makes
    ``import xattention`` raise ``ImportError``.
    """
    monkeypatch.setitem(sys.modules, "xattention", ext)


@pytest.fixture
def _clear_xattn_cache():
    """Reset the cached ``xattention`` import between cases."""
    from vllm.models.minimax_m3.xpu.ops import xattention as xattn

    xattn._load_xattention.cache_clear()
    yield
    sys.modules.pop("xattention", None)
    xattn._load_xattention.cache_clear()


def test_dispatch_prefers_xattention_when_available(monkeypatch, _clear_xattn_cache):
    monkeypatch.setenv("VLLM_XPU_USE_XATTENTION_MSA", "1")
    _install_stub_extension(monkeypatch, _make_stub_extension())

    from vllm.models.minimax_m3.common import indexer as ix
    from vllm.models.minimax_m3.common import sparse_attention as sa
    from vllm.models.minimax_m3.xpu import ops

    assert ops._use_xattention_msa() is True
    ops.install_xpu_sparse_attn()
    ops.install_xpu_index_topk()

    xattn_mod = "vllm.models.minimax_m3.xpu.ops.xattention"
    assert sa.minimax_m3_sparse_attn.__module__ == xattn_mod
    assert sa.minimax_m3_sparse_attn_decode.__module__ == xattn_mod
    assert ix.minimax_m3_index_score.__module__ == xattn_mod
    assert ix.minimax_m3_index_topk.__module__ == xattn_mod
    assert ix.minimax_m3_index_decode.__module__ == xattn_mod


def test_dispatch_falls_back_to_triton_without_extension(
    monkeypatch, _clear_xattn_cache
):
    monkeypatch.setenv("VLLM_XPU_USE_XATTENTION_MSA", "1")
    # Force the import to fail even when the extension is actually installed:
    # a ``None`` entry in ``sys.modules`` makes ``import xattention._C``
    # raise ``ImportError``, faithfully simulating a system without the build.
    _install_stub_extension(monkeypatch, None)

    from vllm.models.minimax_m3.common import indexer as ix
    from vllm.models.minimax_m3.common import sparse_attention as sa
    from vllm.models.minimax_m3.xpu import ops

    assert ops._use_xattention_msa() is False
    ops.install_xpu_sparse_attn()
    ops.install_xpu_index_topk()

    assert sa.minimax_m3_sparse_attn.__module__.endswith("xpu.ops.sparse_attn")
    assert ix.minimax_m3_index_score.__module__.endswith("xpu.ops.index_topk")


def test_dispatch_respects_disable_env(monkeypatch, _clear_xattn_cache):
    """With the extension present but the toggle off, use the Triton kernels."""
    monkeypatch.setenv("VLLM_XPU_USE_XATTENTION_MSA", "0")
    _install_stub_extension(monkeypatch, _make_stub_extension())

    from vllm.models.minimax_m3.common import sparse_attention as sa
    from vllm.models.minimax_m3.xpu import ops

    assert ops._use_xattention_msa() is False
    ops.install_xpu_sparse_attn()
    assert sa.minimax_m3_sparse_attn.__module__.endswith("xpu.ops.sparse_attn")


def test_dispatch_falls_back_when_msa_kernels_not_built(
    monkeypatch, _clear_xattn_cache
):
    """An xattention build without the MSA kernels must not be selected."""
    monkeypatch.setenv("VLLM_XPU_USE_XATTENTION_MSA", "1")
    _install_stub_extension(monkeypatch, _make_stub_extension(ops=("msa_index_score",)))

    from vllm.models.minimax_m3.common import sparse_attention as sa
    from vllm.models.minimax_m3.xpu import ops

    assert ops._use_xattention_msa() is False
    ops.install_xpu_sparse_attn()
    assert sa.minimax_m3_sparse_attn.__module__.endswith("xpu.ops.sparse_attn")
