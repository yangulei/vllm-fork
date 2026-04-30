# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
128x128 block-scaled FP8 -> bf16 weight dequant for DeepseekV4 wo_a (XPU).

wo_a layout (vLLM Fp8LinearMethod with is_bmm=True, bmm_batch_size=G):
  weight:           [G, O, R] float8_e4m3fn
  weight_scale_inv: [G, ceil(O/128), ceil(R/128)] float32
Output:
  wo_a_bf16:        [G, O, R] bfloat16
                    bf16[g,o,r] = fp8[g,o,r] * scale[g, o//128, r//128]
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _dequant_fp8_block128_to_bf16_kernel(
    fp8_ptr,
    scale_ptr,
    out_ptr,
    O_dim,
    R_dim,
    fp8_stride_g,
    fp8_stride_o,
    scale_stride_g,
    scale_stride_o,
    out_stride_g,
    out_stride_o,
    BLOCK_O: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid_g = tl.program_id(0).to(tl.int64)
    pid_o = tl.program_id(1).to(tl.int64)
    pid_r = tl.program_id(2).to(tl.int64)

    o_base = pid_o * BLOCK_O
    r_base = pid_r * BLOCK_R

    o_offs = o_base + tl.arange(0, BLOCK_O)
    r_offs = r_base + tl.arange(0, BLOCK_R)
    o_mask = o_offs < O_dim
    r_mask = r_offs < R_dim
    mask = o_mask[:, None] & r_mask[None, :]

    fp8_ptrs = (
        fp8_ptr
        + pid_g * fp8_stride_g
        + o_offs[:, None] * fp8_stride_o
        + r_offs[None, :]
    )
    fp8_vals = tl.load(fp8_ptrs, mask=mask, other=0.0).to(tl.float32)

    scale = tl.load(scale_ptr + pid_g * scale_stride_g + pid_o * scale_stride_o + pid_r)
    bf16_vals = (fp8_vals * scale).to(tl.bfloat16)

    out_ptrs = (
        out_ptr
        + pid_g * out_stride_g
        + o_offs[:, None] * out_stride_o
        + r_offs[None, :]
    )
    tl.store(out_ptrs, bf16_vals, mask=mask)


def dequant_fp8_block128_to_bf16(
    fp8_weight: torch.Tensor,
    scale: torch.Tensor,
    block: int = 128,
) -> torch.Tensor:
    assert fp8_weight.dtype == torch.float8_e4m3fn
    if scale.dtype == torch.float8_e8m0fnu:
        # E8M0 stores raw exponent byte; value = 2^(byte - 127).
        # .to(float32) performs semantic conversion, so use .view(uint8) first.
        scale = torch.exp2(scale.view(torch.uint8).to(torch.float32) - 127.0)
    assert scale.dtype == torch.float32
    assert fp8_weight.ndim == 3
    G, O_dim, R_dim = fp8_weight.shape
    assert scale.shape[0] == G
    assert scale.shape[1] == (O_dim + block - 1) // block
    assert scale.shape[2] == (R_dim + block - 1) // block
    assert fp8_weight.is_contiguous()
    assert scale.is_contiguous()

    out = torch.empty((G, O_dim, R_dim), dtype=torch.bfloat16, device=fp8_weight.device)

    grid = (G, scale.shape[1], scale.shape[2])
    _dequant_fp8_block128_to_bf16_kernel[grid](
        fp8_weight,
        scale,
        out,
        O_dim,
        R_dim,
        fp8_weight.stride(0),
        fp8_weight.stride(1),
        scale.stride(0),
        scale.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_O=block,
        BLOCK_R=block,
        num_warps=4,
    )
    return out
