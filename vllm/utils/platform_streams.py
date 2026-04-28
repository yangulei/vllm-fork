# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Platform-aware stream/event helpers.

Mirrors the small surface of `torch.cuda.{Event,Stream,stream}` that vLLM
multi-stream code paths need, but dispatches on `current_platform` so XPU
(and other accelerators that expose `torch.<device>.{Event,Stream,stream}`)
can construct the right primitives.

CUDA call sites that hardcode `torch.cuda.Event()` / `torch.cuda.Stream()`
crash at import/construction time on XPU because `torch.cuda.is_available()`
is False and the constructors raise. Use `make_event()` / `make_stream()`
instead.

The `stream_context()` ctx manager additionally accepts `None` (no-op),
so callers can pass an optional stream and use a single `with` block.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import torch

from vllm.platforms import current_platform


def make_event() -> torch.cuda.Event | torch.xpu.Event:
    if current_platform.is_xpu():
        return torch.xpu.Event()
    return torch.cuda.Event()


def make_stream() -> torch.cuda.Stream | torch.xpu.Stream:
    if current_platform.is_xpu():
        return torch.xpu.Stream()
    return torch.cuda.Stream()


@contextlib.contextmanager
def stream_context(
    stream: torch.cuda.Stream | torch.xpu.Stream | None,
) -> Iterator[None]:
    """Activate `stream`, or no-op if `None`."""
    if stream is None:
        yield
        return
    if isinstance(stream, torch.xpu.Stream):
        with torch.xpu.stream(stream):
            yield
    else:
        with torch.cuda.stream(stream):
            yield
