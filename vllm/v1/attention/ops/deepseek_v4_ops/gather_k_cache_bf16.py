# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""bf16 paged-cache gather for DeepseekV4 SWA prefill on XPU.

Drop-in replacement for ``dequantize_and_gather_k_cache`` on the XPU bf16-KV
path. The FP8 dequant kernel hardcodes the
``[0:448 fp8 + 448:576 bf16 + UE8M0 scales]`` layout; on XPU the SWA cache
is plain ``[num_blocks, 64, 512]`` bfloat16, so this kernel is a pure paged
row-gather.

Writes the last ``gather_len`` tokens of each request into
``out[batch, offset:offset+gather_len, :512]``.
"""

import torch

from vllm.triton_utils import tl, triton

HEAD_DIM = 512


@triton.jit
def _xpu_v4_gather_k_cache_bf16_kernel(
    out_ptr,
    out_stride0,
    out_stride1,
    k_cache_ptr,
    seq_lens_ptr,
    block_table_ptr,
    gather_lens_ptr,
    offset,
    cache_block_stride0,
    cache_block_stride1,
    max_blocks_per_seq: tl.constexpr,
    cache_block_size: tl.constexpr,
    D: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    worker_id = tl.program_id(1)
    num_workers = tl.num_programs(1)

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    if gather_lens_ptr is not None:  # noqa: SIM108
        gather_len = tl.load(gather_lens_ptr + batch_idx)
    else:
        gather_len = seq_len
    start_pos = seq_len - gather_len

    d_off = tl.arange(0, D)
    block_table_row_ptr = block_table_ptr + batch_idx * max_blocks_per_seq

    for i in range(worker_id, gather_len, num_workers):
        pos = start_pos + i
        block_in_seq = pos // cache_block_size
        pos_in_block = pos % cache_block_size

        physical_block_idx = tl.load(block_table_row_ptr + block_in_seq).to(tl.int64)
        row_ptr = (
            k_cache_ptr
            + physical_block_idx * cache_block_stride0
            + pos_in_block * cache_block_stride1
        )

        vals = tl.load(row_ptr + d_off)
        tl.store(
            out_ptr + batch_idx * out_stride0 + (offset + i) * out_stride1 + d_off,
            vals,
        )


def xpu_v4_gather_k_cache_bf16(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor | None,
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
) -> None:
    assert out.dtype == torch.bfloat16
    assert k_cache.dtype == torch.bfloat16
    assert k_cache.dim() == 3 and k_cache.shape[1] == block_size
    assert k_cache.shape[2] == HEAD_DIM
    num_reqs = seq_lens.shape[0]
    NUM_WORKERS = 128
    _xpu_v4_gather_k_cache_bf16_kernel[(num_reqs, NUM_WORKERS)](
        out,
        out.stride(0),
        out.stride(1),
        k_cache,
        seq_lens,
        block_table,
        gather_lens,
        offset,
        k_cache.stride(0),
        k_cache.stride(1),
        max_blocks_per_seq=block_table.shape[-1],
        cache_block_size=block_size,
        D=HEAD_DIM,
    )
