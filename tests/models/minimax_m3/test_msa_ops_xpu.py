# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XPU correctness tests for the MiniMax-M3 MSA (sparse-attention) Triton ops.

The MiniMax-M3 sparse-attention path (lightning indexer + block-sparse GQA
attend) runs on the platform-neutral Triton kernels in
``vllm.models.minimax_m3.common.ops`` (the NVIDIA ``fmha_sm100`` MSA path is
Blackwell-only and is *not* used on XPU -- ``select_main_impl_cls`` /
``select_indexer_impl_cls`` fall back to the Triton impls on XPU). These tests
verify those Triton kernels both compile on Intel Triton and are numerically
correct, without needing to load the full (very large) M3 model.

Strategy: when the selected top-k blocks cover *every* block of a request's
sequence, block-sparse attention is mathematically identical to dense causal
attention, so we compare the kernel output against a dense PyTorch reference.
The lightning indexer is checked by verifying it selects the same top-k blocks
as a reference ranking of its own block scores.
"""

import pytest
import torch

from vllm.models.minimax_m3.common.ops.index_topk import (
    SPARSE_BLOCK_SIZE,
    minimax_m3_index_score,
    minimax_m3_index_topk,
)
from vllm.models.minimax_m3.common.ops.sparse_attn import (
    minimax_m3_sparse_attn,
    minimax_m3_sparse_attn_decode,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_xpu(), reason="MiniMax-M3 MSA XPU op tests require XPU"
)

DEVICE = "xpu"
B = SPARSE_BLOCK_SIZE  # 128: one sparse block == one KV page


def _make_block_table(num_reqs: int, max_blocks: int) -> torch.Tensor:
    bt = torch.empty(num_reqs, max_blocks, dtype=torch.int32, device=DEVICE)
    for r in range(num_reqs):
        for b in range(max_blocks):
            bt[r, b] = r * max_blocks + b
    return bt


def _dense_attn(q_th, K, V, sm_scale):
    # q_th: [head_dim], K/V: [s, head_dim]  ->  [head_dim]
    scores = (K @ q_th) * sm_scale
    p = torch.softmax(scores, dim=-1)
    return p @ V


@pytest.mark.parametrize("seq_lens_list", [[200, 128], [128, 200, 130]])
@pytest.mark.parametrize("gqa", [1, 4])
def test_sparse_decode_equals_dense(seq_lens_list, gqa):
    """Block-sparse decode == dense causal attention when top-k covers all blocks."""
    torch.manual_seed(0)
    head_dim = 128
    num_kv_heads = 1
    num_heads = num_kv_heads * gqa
    num_reqs = len(seq_lens_list)
    decode_query_len = 1
    total_q = num_reqs * decode_query_len
    max_blocks = (max(seq_lens_list) + B - 1) // B + 1
    num_blocks = num_reqs * max_blocks
    sm_scale = 1.0 / (head_dim**0.5)
    dtype = torch.bfloat16

    q = torch.randn(total_q, num_heads, head_dim, device=DEVICE, dtype=dtype)
    kv_cache = torch.randn(
        num_blocks, 2, B, num_kv_heads, head_dim, device=DEVICE, dtype=dtype
    )
    block_table = _make_block_table(num_reqs, max_blocks)
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=DEVICE)

    nblk = [(s + B - 1) // B for s in seq_lens_list]
    topk = max(nblk)
    # topk_idx holds *logical* block indices (the kernel maps them to physical
    # pages via block_table); -1 pads unused slots.
    topk_idx = torch.full(
        (num_kv_heads, total_q, topk), -1, dtype=torch.int32, device=DEVICE
    )
    for r in range(num_reqs):
        for j in range(nblk[r]):
            topk_idx[0, r, j] = j

    output = torch.empty_like(q)
    minimax_m3_sparse_attn_decode(
        q, kv_cache, topk_idx, block_table, seq_lens,
        num_kv_heads, sm_scale, output, decode_query_len,
    )
    torch.xpu.synchronize()
    assert torch.isfinite(output).all()

    got = output.float()
    for r in range(num_reqs):
        s = seq_lens_list[r]
        K = torch.cat([kv_cache[int(block_table[r, b]), 0, :, 0, :]
                       for b in range(nblk[r])], 0)[:s].float()
        V = torch.cat([kv_cache[int(block_table[r, b]), 1, :, 0, :]
                       for b in range(nblk[r])], 0)[:s].float()
        for h in range(num_heads):
            ref = _dense_attn(q[r, h].float(), K, V, sm_scale)
            torch.testing.assert_close(got[r, h], ref, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("seq_len", [128, 200, 384])
@pytest.mark.parametrize("gqa", [1, 4])
def test_sparse_prefill_equals_dense_causal(seq_len, gqa):
    """Block-sparse prefill == dense causal attention when top-k covers all blocks."""
    torch.manual_seed(1)
    head_dim = 128
    num_kv_heads = 1
    num_heads = num_kv_heads * gqa
    max_blocks = (seq_len + B - 1) // B
    sm_scale = 1.0 / (head_dim**0.5)
    dtype = torch.bfloat16
    total_q = seq_len

    q = torch.randn(total_q, num_heads, head_dim, device=DEVICE, dtype=dtype)
    kv_cache = torch.randn(
        max_blocks, 2, B, num_kv_heads, head_dim, device=DEVICE, dtype=dtype
    )
    block_table = torch.arange(
        max_blocks, dtype=torch.int32, device=DEVICE
    ).view(1, max_blocks)
    cu_seqlens_q = torch.tensor([0, seq_len], dtype=torch.int32, device=DEVICE)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=DEVICE)
    prefix_lens = torch.tensor([0], dtype=torch.int32, device=DEVICE)

    nblk = (seq_len + B - 1) // B
    topk = nblk
    topk_idx = torch.full(
        (num_kv_heads, total_q, topk), -1, dtype=torch.int32, device=DEVICE
    )
    for t in range(total_q):
        for j in range(nblk):
            topk_idx[0, t, j] = j  # kernel applies the causal mask itself

    output = torch.empty_like(q)
    minimax_m3_sparse_attn(
        q, kv_cache, topk_idx, block_table, cu_seqlens_q, seq_lens,
        prefix_lens, seq_len, num_kv_heads, sm_scale, output,
    )
    torch.xpu.synchronize()
    assert torch.isfinite(output).all()

    K = torch.cat([kv_cache[j, 0, :, 0, :] for j in range(nblk)], 0)[:seq_len].float()
    V = torch.cat([kv_cache[j, 1, :, 0, :] for j in range(nblk)], 0)[:seq_len].float()
    got = output.float()
    # spot-check a spread of query positions (full O(L*H) compare is slow)
    for t in range(0, total_q, max(1, total_q // 16)):
        for h in range(num_heads):
            ref = _dense_attn(q[t, h].float(), K[: t + 1], V[: t + 1], sm_scale)
            torch.testing.assert_close(got[t, h], ref, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("seq_len", [200, 384])
def test_indexer_score_topk_selection(seq_len):
    """Lightning indexer selects the same top-k blocks as a reference ranking."""
    torch.manual_seed(2)
    head_dim = 64
    num_kv_heads = 1
    max_blocks = (seq_len + B - 1) // B
    sm_scale = 1.0 / (head_dim**0.5)
    dtype = torch.bfloat16
    total_q = seq_len
    topk = 1  # force a selective choice for any token spanning >1 block

    idx_q = torch.randn(total_q, num_kv_heads, head_dim, device=DEVICE, dtype=dtype)
    index_kv_cache = torch.randn(max_blocks, B, head_dim, device=DEVICE, dtype=dtype)
    block_table = torch.arange(
        max_blocks, dtype=torch.int32, device=DEVICE
    ).view(1, max_blocks)
    cu_seqlens_q = torch.tensor([0, seq_len], dtype=torch.int32, device=DEVICE)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=DEVICE)
    prefix_lens = torch.tensor([0], dtype=torch.int32, device=DEVICE)

    score = minimax_m3_index_score(
        idx_q, index_kv_cache, block_table, cu_seqlens_q, seq_lens,
        prefix_lens, seq_len, seq_len, num_kv_heads, sm_scale,
    )
    topk_idx = minimax_m3_index_topk(
        score, cu_seqlens_q, prefix_lens, seq_len, topk,
        init_blocks=0, local_blocks=0,
    )
    torch.xpu.synchronize()

    nblk_total = (seq_len + B - 1) // B
    # For query tokens whose causal window spans more than `topk` blocks, the
    # indexer must pick exactly the highest-scoring blocks (per its own scores).
    checked = 0
    for t in range(B + 1, total_q, max(1, total_q // 16)):
        nb_t = (t // B) + 1
        if nb_t <= topk:
            continue
        ref_top = set(torch.topk(score[0, t, :nb_t], topk).indices.tolist())
        sel = {x for x in topk_idx[0, t, :].tolist() if x >= 0}
        assert sel, f"no blocks selected for token {t}"
        assert sel.issubset(set(range(nb_t)))
        assert sel == ref_top, f"token {t}: selected {sorted(sel)} != ref {sorted(ref_top)}"
        checked += 1
    assert checked > 0, "no multi-block query positions were exercised"
    assert torch.isfinite(score[0, :, :nblk_total]).any()
