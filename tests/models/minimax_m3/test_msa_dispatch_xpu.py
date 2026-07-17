# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dispatch tests for the MiniMax-M3 sparse-attention (MSA) XPU ops.

These verify the *routing* logic in ``vllm.models.minimax_m3.xpu.ops`` — that
the lightning indexer and block-sparse attend are dispatched to the xattention
SYCL kernels (``xattention._C``) when that extension is available and
``VLLM_XPU_USE_XATTENTION_MSA`` is set, and fall back to the Triton XPU kernels
otherwise. The routing is exercised with a stub extension so the test runs on
CPU without XPU hardware or a built extension.
"""

import sys
import types

import pytest

_MSA_OPS = (
    "minimax_m3_index_score",
    "minimax_m3_index_topk",
    "minimax_m3_index_decode",
    "minimax_m3_sparse_attn",
    "minimax_m3_sparse_attn_decode",
)


def _make_stub_extension() -> types.ModuleType:
    stub = types.ModuleType("xattention._C")
    for op in _MSA_OPS:
        setattr(stub, op, lambda *a, **k: None)
    return stub


def _install_stub_extension(monkeypatch, ext: types.ModuleType | None) -> None:
    """Install a stub ``xattention._C`` (or force its import to fail).

    ``import xattention._C`` first imports the ``xattention`` parent package, so
    a lightweight stub parent is registered to keep the real (XPU-only) package
    off the import path. Passing ``ext=None`` registers a ``None`` submodule
    entry, which makes ``import xattention._C`` raise ``ImportError``.
    """
    parent = types.ModuleType("xattention")
    parent.__path__ = []  # mark as a package
    monkeypatch.setitem(sys.modules, "xattention", parent)
    monkeypatch.setitem(sys.modules, "xattention._C", ext)
    if ext is not None:
        parent._C = ext


@pytest.fixture
def _clear_xattn_cache():
    """Reset the cached ``xattention._C`` import between cases."""
    from vllm.models.minimax_m3.xpu.ops import xattention as xattn

    xattn._load_xattention.cache_clear()
    yield
    sys.modules.pop("xattention._C", None)
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
