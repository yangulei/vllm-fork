# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only DeepseekV2/DeepseekV3 model."""

import os
import typing
from collections.abc import Callable, Iterable
from itertools import islice
from typing import Any

import torch
from torch import nn
from transformers import DeepseekV2Config, DeepseekV3Config

from vllm.attention import Attention
from vllm.attention.backends.abstract import AttentionBackend
from vllm.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ParallelConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.fused_moe import SharedFusedMoE
from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
    is_rocm_aiter_fusion_shared_expert_enabled,
    is_rocm_aiter_moe_enabled,
)
from vllm.model_executor.layers.layernorm import LayerNorm, RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mla import MLAModules, MultiHeadLatentAttentionWrapper
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.deep_gemm import fp8_mqa_logits, fp8_paged_mqa_logits
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerBackend,
    DeepseekV32IndexerMetadata,
)
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec

from .interfaces import MixtureOfExperts, SupportsLoRA, SupportsPP
from .utils import (
    PPMissingLayer,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

if current_platform.is_cuda_alike():
    from vllm import _custom_ops as ops
elif current_platform.is_xpu():
    from vllm._ipex_ops import ipex_ops as ops

logger = init_logger(__name__)


def per_token_group_quant_fp8_pytorch(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype | None = None,
    column_major_scales: bool = False,
    out_q: torch.Tensor | None = None,
    use_ue8m0: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch-native implementation of per-token-group quantization for FP8.
    """
    if use_ue8m0 is None:
        # Default fallback - could import is_deep_gemm_e8m0_used if needed
        use_ue8m0 = False

    if dtype is None:
        dtype = current_platform.fp8_dtype()

    # Validate inputs
    assert x.shape[-1] % group_size == 0, (
        f"Last dimension {x.shape[-1]} must be divisible by group_size {group_size}"
    )
    assert x.stride(-1) == 1, "Input tensor groups must be contiguous"

    # Get FP8 range
    # finfo = torch.finfo(dtype)
    finfo = torch.finfo(torch.float8_e4m3fnuz)  # HACK for gaudi2
    fp8_min = finfo.min
    fp8_max = finfo.max

    # Prepare output tensor
    if out_q is None:
        x_q = torch.empty_like(x, dtype=dtype)
    else:
        assert out_q.shape == x.shape
        x_q = out_q

    # Reshape input for group processing
    # Original shape: (..., last_dim)
    # Target shape: (..., num_groups, group_size)
    original_shape = x.shape
    num_groups = original_shape[-1] // group_size

    # Reshape to separate groups
    group_shape = original_shape[:-1] + (num_groups, group_size)
    x_grouped = x.view(group_shape)

    # Compute per-group absolute maximum values
    # Shape: (..., num_groups)
    abs_max = torch.amax(torch.abs(x_grouped), dim=-1, keepdim=False)
    abs_max = torch.maximum(abs_max, torch.tensor(eps, device=x.device, dtype=x.dtype))

    # Compute scales
    scale_raw = abs_max / fp8_max

    if use_ue8m0:
        # For UE8M0 format, scales must be powers of 2
        scales = torch.pow(2.0, torch.ceil(torch.log2(scale_raw)))
    else:
        scales = scale_raw

    # Expand scales for broadcasting with grouped data
    # Shape: (..., num_groups, 1)
    scales_expanded = scales.unsqueeze(-1)

    # Quantize the grouped data
    x_scaled = x_grouped / scales_expanded
    x_clamped = torch.clamp(x_scaled, fp8_min, fp8_max)
    x_quantized = x_clamped.to(dtype)

    # Reshape back to original shape
    x_q.copy_(x_quantized.view(original_shape))

    # Prepare scales tensor in requested format
    if column_major_scales:
        # Column-major: (num_groups,) + batch_dims
        # Transpose the scales to put group dimension first
        scales_shape = (num_groups,) + original_shape[:-1]
        x_s = scales.permute(-1, *range(len(original_shape) - 1))
        x_s = x_s.contiguous().view(scales_shape)
    else:
        # Row-major: batch_dims + (num_groups,)
        x_s = scales.contiguous()

    # Ensure scales are float32
    x_s = x_s.to(torch.float32)

    return x_q, x_s


class DeepseekV2MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        is_sequence_parallel=False,
        prefix: str = "",
    ) -> None:
        super().__init__()

        # If is_sequence_parallel, the input and output tensors are sharded
        # across the ranks within the tp_group. In this case the weights are
        # replicated and no collective ops are needed.
        # Otherwise we use standard TP with an allreduce at the end.
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class DeepseekV2MoE(nn.Module):
    def __init__(
        self,
        config: DeepseekV2Config | DeepseekV3Config,
        parallel_config: ParallelConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        self.routed_scaling_factor = config.routed_scaling_factor

        self.ep_group = get_ep_group().device_group
        self.ep_rank = self.ep_group.rank()
        self.ep_size = self.ep_group.size()
        self.n_routed_experts: int = config.n_routed_experts
        self.n_shared_experts: int = config.n_shared_experts

        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.n_routed_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )
        if config.topk_method == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float32)
            )
        else:
            self.gate.e_score_correction_bias = None

        # Load balancing settings.
        eplb_config = parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb

        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_logical_experts = self.n_routed_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        if (
            config.n_shared_experts is None
            or is_rocm_aiter_fusion_shared_expert_enabled()
        ):
            self.shared_experts = None
        else:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts

            self.shared_experts = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                is_sequence_parallel=self.is_sequence_parallel,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )

        self.experts = SharedFusedMoE(
            shared_experts=self.shared_experts,
            gate=self.gate,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            prefix=f"{prefix}.experts",
            scoring_func=config.scoring_func,
            # we do scaling outside, set factor to 1.0 to avoid double mul
            # aiter applies routed_scaling_factor internally
            routed_scaling_factor=1.0
            if not is_rocm_aiter_moe_enabled()
            else self.routed_scaling_factor,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
            n_shared_experts=config.n_shared_experts
            if is_rocm_aiter_fusion_shared_expert_enabled()
            else None,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # Chunk the hidden states so they aren't replicated across TP ranks.
        # This avoids duplicate computation in self.experts.
        # TODO: We can replace the all_reduce at the end of attn with a
        # reduce_scatter instead of chunking here.
        if self.is_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        if self.experts.is_internal_router:
            # In this case, the gate/router runs inside the FusedMoE class
            fused_moe_out = self.experts(
                hidden_states=hidden_states, router_logits=hidden_states
            )
        else:
            # router_logits: (num_tokens, n_experts)
            router_logits, _ = self.gate(hidden_states)
            fused_moe_out = self.experts(
                hidden_states=hidden_states, router_logits=router_logits
            )

        shared_output, final_hidden_states = fused_moe_out
        if self.shared_experts is None:
            assert shared_output is None

        # Fix FP16 overflow
        # See DeepseekV2DecoderLayer for more details.
        if hidden_states.dtype != torch.float16:
            if not is_rocm_aiter_moe_enabled():
                final_hidden_states *= self.routed_scaling_factor
        elif self.shared_experts is not None:
            assert shared_output is not None
            shared_output *= 1.0 / self.routed_scaling_factor

        if self.shared_experts is not None:
            assert shared_output is not None
            final_hidden_states += shared_output

        if self.is_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, 0
            )
            final_hidden_states = final_hidden_states[:num_tokens]
        elif self.tp_size > 1:
            final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(
                final_hidden_states
            )

        return final_hidden_states.view(num_tokens, hidden_dim)


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    import math

    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class DeepseekV2Attention(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        rope_theta: float = 10000,
        rope_scaling: dict[str, Any] | None = None,
        max_position_embeddings: int = 8192,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        topk_indices_buffer: torch.Tensor | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        self.num_local_heads = num_heads // tp_size
        self.scaling = self.qk_head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        assert topk_indices_buffer is None, (
            "topk_indices_buffer is not \
        supported for DeepseekV2Attention"
        )

        if self.q_lora_rank is not None:
            self.q_a_proj = ReplicatedLinear(
                self.hidden_size,
                self.q_lora_rank,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_a_proj",
            )
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_b_proj",
            )
        else:
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj",
            )

        self.kv_a_proj_with_mqa = ReplicatedLinear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_a_proj_with_mqa",
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        # O projection.
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        if rope_scaling:
            rope_scaling["rope_type"] = "deepseek_yarn"

        self.rotary_emb = get_rope(
            qk_rope_head_dim,
            rotary_dim=qk_rope_head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=False,
        )

        if rope_scaling:
            mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
            scaling_factor = rope_scaling["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale

        self.attn = Attention(
            self.num_local_heads,
            self.qk_head_dim,
            self.scaling,
            num_kv_heads=self.num_local_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if self.q_lora_rank is not None:
            q = self.q_a_proj(hidden_states)[0]
            q = self.q_a_layernorm(q)
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
        kv_a, _ = latent_cache.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        latent_cache = latent_cache.unsqueeze(1)
        kv_a = self.kv_a_layernorm(kv_a)
        kv = self.kv_b_proj(kv_a)[0]
        kv = kv.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_pe = latent_cache[:, :, self.kv_lora_rank :]

        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)

        q[..., self.qk_nope_head_dim :] = q_pe
        k = torch.empty_like(q)
        k[..., : self.qk_nope_head_dim] = k_nope
        k[..., self.qk_nope_head_dim :] = k_pe
        # padding value to qk_head_dim for alignment
        v = torch.nn.functional.pad(
            v, [0, self.qk_head_dim - self.v_head_dim], value=0
        ).view(-1, self.num_local_heads * self.qk_head_dim)
        attn_output = self.attn(q, k, v)
        attn_output = attn_output.view(-1, self.num_local_heads, self.qk_head_dim)[
            ..., : self.v_head_dim
        ].reshape(-1, self.num_local_heads * self.v_head_dim)
        output, _ = self.o_proj(attn_output)
        return output


class DeepseekV32IndexerCache(torch.nn.Module, AttentionLayerBase):
    def __init__(
        self, head_dim: int, dtype: torch.dtype, prefix: str, cache_config: CacheConfig
    ):
        super().__init__()
        self.kv_cache = [torch.tensor([])]
        self.head_dim = head_dim
        self.prefix = prefix
        self.cache_config = cache_config
        self.dtype = dtype
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return MLAAttentionSpec(  # Only has one vector instead of K + V
            block_size=self.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
        )

    def forward(self): ...

    def get_attn_backend(self) -> AttentionBackend:
        return DeepseekV32IndexerBackend


def sparse_attn_indexer_pytorch(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
) -> torch.Tensor:
    """
    Pure PyTorch implementation of sparse_attn_indexer.

    This function performs sparse attention indexing by:
    1. Quantizing and caching K tensors
    2. Computing attention logits using FP8 operations
    3. Finding top-k attention indices for sparse attention
    """
    # Handle dummy run case
    attn_metadata = get_forward_context().attn_metadata
    # if not isinstance(attn_metadata, dict):
    #     return sparse_attn_indexer_fake(
    #         hidden_states,
    #         k_cache_prefix,
    #         kv_cache,
    #         q_fp8,
    #         k,
    #         weights,
    #         quant_block_size,
    #         scale_fmt,
    #         topk_tokens,
    #         head_dim,
    #         max_model_len,
    #         total_seq_lens,
    #         topk_indices_buffer,
    #     )

    # attn_metadata = attn_metadata[k_cache_prefix]
    # assert isinstance(attn_metadata, DeepseekV32IndexerMetadata)
    slot_mapping = attn_metadata.slot_mapping
    is_prompt = attn_metadata.is_prompt

    # Initialize topk_indices_buffer
    topk_indices_buffer[: hidden_states.shape[0]] = -1

    k_fp8, k_scale = _pytorch_indexer_k_quant_and_cache(
        k, kv_cache, slot_mapping, quant_block_size, scale_fmt
    )
    if is_prompt:
        logits = _pytorch_fp8_mqa_logits(
            q_fp8,
            k_fp8,
            k_scale,
            weights,
        )
        topk_indices = _pytorch_topk_with_bounds(logits, topk_tokens)
        topk_indices = topk_indices.view(-1, topk_indices.shape[-1])
        topk_indices_buffer[: topk_indices.shape[0], : topk_indices.shape[-1]] = (
            topk_indices.to(dtype=torch.int32)
        )
    else:
        # PyTorch implementation of fp8_paged_mqa_logits
        logits = _pytorch_fp8_paged_mqa_logits(
            q_fp8,
            kv_cache,
            weights,
            attn_metadata,
            max_model_len,
        )

        # Apply position masking and get top-k indices
        # q_fp8: [padded_token_num, num_heads, head_dim]
        padded_token_num = q_fp8.size(0)
        topk_indices = _pytorch_decode_topk_with_masking(
            logits, attn_metadata, topk_tokens, padded_token_num, max_model_len
        )
        topk_indices_buffer[: topk_indices.shape[0], : topk_indices.shape[-1]] = (
            topk_indices
        )

    return topk_indices_buffer


def _pytorch_indexer_k_quant_and_cache(
    k: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
) -> None:
    """PyTorch implementation of indexer_k_quant_and_cache kernel."""
    head_dim = k.shape[-1]
    k = k.view(-1, head_dim)  # [total_tokens, head_dim]

    k_fp8, k_scale = per_token_group_quant_fp8_pytorch(
        k,
        group_size=quant_block_size,
        column_major_scales=False,
        use_ue8m0=(scale_fmt == "ue8m0"),
    )

    k_fp8_bytes = k_fp8.view(-1, head_dim).view(torch.uint8)
    scale_bytes = k_scale.view(torch.uint8).view(-1, 4)
    fp8_bytes = torch.cat(
        [k_fp8_bytes, scale_bytes], dim=-1
    )  # [total_tokens, head_dim + 4]

    kv_cache = kv_cache.view(-1, head_dim + 4)
    slot_mapping = slot_mapping.view(-1)

    # kv_cache[slot_mapping] = fp8_bytes
    kv_cache.index_copy_(0, slot_mapping, fp8_bytes)

    return k_fp8, k_scale


def _pytorch_fp8_mqa_logits(
    q_fp8: torch.Tensor,
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """PyTorch implementation of fp8_mqa_logits.

    Optimized with vectorized operations where possible.
    """

    # Convert k_scale from uint8 to float32 and dequantize k_fp8
    k_scale_f32 = k_scale.view(torch.float32)
    k_dequant = k_fp8.float() * k_scale_f32  # [total_k_len, head_dim]

    q_float = q_fp8.float()  # [batch_size, num_heads, head_dim]
    weighted_q = weights[..., None] * q_float
    weights_sum = torch.sum(weighted_q, dim=-2)
    logits = torch.matmul(weights_sum, k_dequant.squeeze(0).T)
    logits.relu_()
    return logits


def _pytorch_topk_with_bounds(
    logits: torch.Tensor,
    topk_tokens: int,
) -> torch.Tensor:
    """PyTorch implementation of bounded top-k selection."""
    seq_len = logits.shape[0]

    mask = torch.triu(
        torch.ones(logits.shape, device=logits.device, dtype=torch.bool), diagonal=1
    )

    # Apply bounds masking before topk
    logits = logits.masked_fill(mask, float("-inf"))

    # Get top-k indices
    topk_indices = logits.topk(min(topk_tokens, seq_len), dim=-1)[1]
    topk_indices = topk_indices.masked_fill(mask[:, : min(topk_tokens, seq_len)], -1)

    return topk_indices


def _pytorch_fp8_paged_mqa_logits(
    q_fp8: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    attn_metadata,
    max_model_len: int,
) -> torch.Tensor:
    """PyTorch implementation of fp8_paged_mqa_logits.

    Args:
        q_fp8: Query tensor of shape [B, H, D]. Casted to `torch.float8_e4m3fn` by caller.
        kv_cache_fp8: Paged KV-cache in packed FP8+scale layout with shape
            [num_blocks, block_size, D+4], dtype `torch.uint8`. The last
            4 bytes per (block,pos) store the `float` dequant scale.
        weights: Tensor of shape [B, H], dtype `torch.float32`.
        max_model_len: Maximum sequence length used to size the logits output.

    Returns:
        Logits tensor of shape [B, max_model_len], dtype
        `torch.float32`.
    """
    padded_token_num, q_heads, head_dim = q_fp8.shape
    # kv_cache: [num_blocks, block_size, D+4] -> [num_blocks, block_size, 1, D+4]
    kv_cache = kv_cache.unsqueeze(-2)  # Add head dimension
    kv_heads = kv_cache.shape[-2]

    block_groups = attn_metadata.block_groups
    block_list = attn_metadata.block_list
    block_mapping = attn_metadata.block_mapping

    from vllm_gaudi.extension.ops import batch2block
    from vllm_gaudi.extension.utils import VLLMKVCache

    q_float = q_fp8.float() * weights[..., None]
    q_float = batch2block(q_float, block_mapping.float()).unsqueeze(-2)
    # q_float: [padded_block_num, q_head, 1, head_dim] [:, 64, 1, 128]

    indexer_cache_k = VLLMKVCache()
    fetch_from_cache = indexer_cache_k.fetch_from_cache
    key = fetch_from_cache(kv_cache, block_list)
    # key: [padded_block_num, block_size, kv_head, head_dim+4] [:, 128, 1, 132]

    k_fp8_val = key[..., :head_dim].view(torch.float8_e4m3fn)
    k_scale_val = key[..., head_dim:].view(torch.float32)
    k_dequant = k_fp8_val.float() * k_scale_val
    k_dequant = k_dequant.transpose(1, 2)

    if kv_heads != q_heads:
        assert q_heads % kv_heads == 0
        # q_fp8: [padded_block_num, q_head, 1, head_dim] [:, 64, 1, 128]
        # key: [padded_block_num, kv_head, block_size, head_dim] [:, 1, 128, 128]
        k_dequant = k_dequant.repeat_interleave(int(q_heads / kv_heads), dim=1)

    attn = torch.matmul(q_float, k_dequant.transpose(-2, -1))
    # attn: [padded_block_num, q_head, 1, block_size] [:, 64, 1, 128]
    attn = attn.relu().sum(dim=1).squeeze(-2)

    bs = padded_token_num
    num_blocks = block_groups.shape[0]
    block_size = attn.shape[-1]
    device = q_fp8.device
    # Per-sequence block mask: [bs, num_blocks] (True where block belongs to the sequence)
    seq_ids = torch.arange(bs, device=device, dtype=block_groups.dtype)  # [bs]
    mask_blocks = block_groups[None, :] == seq_ids[:, None]  # [bs, num_blocks] bool

    # Previous column of mask (right-shift with zero at column 0), all static-shape ops
    prev_mask = torch.cat(
        [
            torch.zeros(bs, 1, dtype=mask_blocks.dtype, device=device),
            mask_blocks[:, :-1],
        ],
        dim=1,
    )

    # Rising edge one-hot for start of the contiguous run: [bs, num_blocks] bool
    start_onehot = mask_blocks & (~prev_mask)

    # ---- Use MATMUL (GEMM) to get start index (no cumsum) ----
    # Dot with column indices to obtain the start position per sequence.
    col_idx = torch.arange(
        num_blocks, device=device, dtype=torch.float32
    )  # [num_blocks]
    start_idx = (start_onehot.to(torch.float32) @ col_idx).to(torch.int64)  # [bs] long

    # Block indices and positions (zeros for non-selected blocks)
    block_idx = torch.arange(num_blocks, device=device, dtype=torch.int64)[
        None, :
    ]  # [1, num_blocks]
    pos_blocks = (block_idx - start_idx[:, None]) * mask_blocks.to(
        torch.int64
    )  # [bs, num_blocks]

    # Optional: respect max_model_len capacity (drop overflow blocks)
    max_blocks_per_seq = max_model_len // block_size
    cap_mask = (pos_blocks < max_blocks_per_seq).to(attn.dtype)  # [bs, num_blocks]

    # Token indices inside each block: [1,1,block_size], long
    idx_in_block = torch.arange(block_size, device=device, dtype=torch.int64)[
        None, None, :
    ]

    # Target token positions in static buffer [bs, num_blocks * block_size]
    # Clamp negative (non-selected) pos to 0; src is zero there anyway.
    target_idx = (
        pos_blocks[:, :, None].clamp(min=0) * block_size
    ) + idx_in_block  # [bs, num_blocks, block_size]

    # Values to scatter (zero out non-selected and overflow)
    values = (
        attn[None, :, :].expand(bs, num_blocks, block_size)
        * mask_blocks[:, :, None].to(attn.dtype)
        * cap_mask[:, :, None]
    )

    # Flatten for scatter_add_ (static shapes)
    target_idx_flat = target_idx.reshape(bs, num_blocks * block_size)
    values_flat = values.reshape(bs, num_blocks * block_size)

    # Output flat buffer (static size); compact to front via scatter_add_
    out_flat = torch.zeros(bs, num_blocks * block_size, dtype=attn.dtype, device=device)
    out_flat.scatter_add_(dim=1, index=target_idx_flat, src=values_flat)

    # Final logits [bs, max_model_len]
    logits = out_flat[:, :max_model_len].contiguous()

    return logits


def _pytorch_decode_topk_with_masking(
    logits: torch.Tensor,
    attn_metadata,
    topk_tokens: int,
    padded_token_num: int,
    max_model_len: int,
) -> torch.Tensor:
    """PyTorch implementation of decode top-k with position masking."""
    current_device = logits.device

    # Create position mask
    positions = (
        torch.arange(max_model_len, device=current_device)
        .unsqueeze(0)
        .expand(padded_token_num, -1)
    )
    index_end_pos = attn_metadata.input_positions

    # Apply mask and get top-k
    mask = positions <= index_end_pos
    logits = logits.masked_fill(~mask, float("-inf"))
    topk_indices = logits.topk(topk_tokens, dim=-1)[1].to(torch.int32)

    # Clamp out-of-range indices
    topk_indices[topk_indices > index_end_pos] = -1

    return topk_indices


def sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
) -> torch.Tensor:
    # Check if PyTorch implementation should be used
    use_pytorch_impl = (
        os.environ.get("VLLM_USE_PYTORCH_SPARSE_INDEXER", "false").lower() == "true"
    )

    if use_pytorch_impl:
        return sparse_attn_indexer_pytorch(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            q_fp8,
            k,
            weights,
            quant_block_size,
            scale_fmt,
            topk_tokens,
            head_dim,
            max_model_len,
            total_seq_lens,
            topk_indices_buffer,
        )

    # Original implementation (unchanged for compatibility)
    # careful! this will be None in dummy run
    attn_metadata = get_forward_context().attn_metadata
    if not isinstance(attn_metadata, dict):
        return sparse_attn_indexer_fake(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            q_fp8,
            k,
            weights,
            quant_block_size,
            scale_fmt,
            topk_tokens,
            head_dim,
            max_model_len,
            total_seq_lens,
            topk_indices_buffer,
        )
    attn_metadata = attn_metadata[k_cache_prefix]
    assert isinstance(attn_metadata, DeepseekV32IndexerMetadata)
    slot_mapping = attn_metadata.slot_mapping
    has_decode = attn_metadata.num_decodes > 0
    has_prefill = attn_metadata.num_prefills > 0
    num_decode_tokens = attn_metadata.num_decode_tokens

    ops.indexer_k_quant_and_cache(
        k,
        kv_cache,
        slot_mapping,
        quant_block_size,
        scale_fmt,
    )

    topk_indices_buffer[: hidden_states.shape[0]] = -1
    if has_prefill:
        prefill_metadata = attn_metadata.prefill
        for chunk in prefill_metadata.chunks:
            k_fp8 = torch.empty(
                [chunk.total_seq_lens, head_dim],
                device=k.device,
                dtype=torch.float8_e4m3fn,
            )
            k_scale = torch.empty(
                [chunk.total_seq_lens, 4],
                device=k.device,
                dtype=torch.uint8,
            )
            ops.cp_gather_indexer_k_quant_cache(
                kv_cache,
                k_fp8,
                k_scale,
                chunk.block_table,
                chunk.cu_seq_lens,
            )
            logits = fp8_mqa_logits(
                q_fp8[chunk.token_start : chunk.token_end],
                (k_fp8, k_scale.view(torch.float32)),
                weights[chunk.token_start : chunk.token_end],
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
            )
            num_rows = logits.shape[0]
            assert topk_tokens == 2048, "top_k_per_row assumes size 2048"
            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]
            torch.ops._C.top_k_per_row(
                logits,
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                topk_indices,
                num_rows,
                logits.stride(0),
                logits.stride(1),
            )

    if has_decode:
        decode_metadata = attn_metadata.decode
        # kv_cache size requirement [num_block, block_size, n_head, head_dim],
        # we only have [num_block, block_size, head_dim],
        kv_cache = kv_cache.unsqueeze(-2)
        decode_lens = decode_metadata.decode_lens
        if decode_metadata.requires_padding:
            # pad in edge case where we have short chunked prefill length <
            # decode_threshold since we unstrictly split
            # prefill and decode by decode_threshold
            # (currently set to 1 + speculative tokens)
            padded_q_fp8_decode_tokens = pack_seq_triton(
                q_fp8[:num_decode_tokens], decode_lens
            )
        else:
            padded_q_fp8_decode_tokens = q_fp8[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_fp8.shape[1:]
            )
        # TODO: move and optimize below logic with triton kernels
        batch_size = padded_q_fp8_decode_tokens.shape[0]
        next_n = padded_q_fp8_decode_tokens.shape[1]
        assert batch_size == decode_metadata.seq_lens.shape[0]
        num_padded_tokens = batch_size * next_n
        logits = fp8_paged_mqa_logits(
            padded_q_fp8_decode_tokens,
            kv_cache,
            weights[:num_padded_tokens],
            decode_metadata.seq_lens,
            decode_metadata.block_table,
            decode_metadata.schedule_metadata,
            max_model_len=max_model_len,
        )
        num_rows = logits.shape[0]
        assert topk_tokens == 2048, "top_k_per_row assumes size 2048"
        topk_indices = topk_indices_buffer[:num_decode_tokens, :topk_tokens]

        torch.ops._C.top_k_per_row_decode(
            logits,
            next_n,
            decode_metadata.seq_lens,
            topk_indices,
            num_rows,
            logits.stride(0),
            logits.stride(1),
        )
        if decode_metadata.requires_padding:
            # if padded, we need to unpack
            # the topk indices removing padded tokens
            topk_indices = unpack_seq_triton(
                topk_indices.reshape(batch_size, -1, topk_indices.shape[-1]),
                decode_lens,
            )
            topk_indices_buffer[:num_decode_tokens, : topk_indices.shape[-1]] = (
                topk_indices
            )

    return topk_indices_buffer


def sparse_attn_indexer_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
) -> torch.Tensor:
    # profile run
    # NOTE(Chen): create the max possible flattened_kv. So that
    # profile_run can get correct memory usage.
    _flattened_kv = torch.empty(
        [total_seq_lens, head_dim + 4], device=k.device, dtype=torch.uint8
    )
    _k_fp8 = _flattened_kv[..., :head_dim].view(torch.float8_e4m3fn).contiguous()
    _k_scale = _flattened_kv[..., head_dim:].view(torch.float32).contiguous()
    return topk_indices_buffer


direct_register_custom_op(
    op_name="sparse_attn_indexer",
    op_func=sparse_attn_indexer,
    mutates_args=["topk_indices_buffer"],
    fake_impl=sparse_attn_indexer_fake,
    dispatch_key=current_platform.dispatch_key,
)


class Indexer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config,
        hidden_size: int,
        q_lora_rank: int,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        topk_indices_buffer: torch.Tensor | None,
        prefix: str = "",
    ):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = config
        # self.indexer_cfg = config.attn_module_list_cfg[0]["attn_index"]
        self.topk_tokens = config.index_topk
        self.n_head = config.index_n_heads  # 64
        self.head_dim = config.index_head_dim  # 128
        self.rope_dim = config.qk_rope_head_dim  # 64
        self.q_lora_rank = q_lora_rank  # 1536
        # no tensor parallel, just replicated
        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.head_dim * self.n_head,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        self.wk = ReplicatedLinear(
            hidden_size,
            self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wk",
        )
        self.k_norm = LayerNorm(self.head_dim, eps=1e-6)
        self.weights_proj = ReplicatedLinear(
            hidden_size, self.n_head, quant_config=None, prefix=f"{prefix}.weights_proj"
        )
        self.softmax_scale = self.head_dim**-0.5

        self.scale_fmt = "ue8m0"
        self.quant_block_size = 128  # TODO: get from config
        self.topk_indices_buffer = topk_indices_buffer

        # NOTE: (zyongye) we use fp8 naive cache,
        #       where we store value in fp8 and scale in fp32
        #       per self.quant_block_size element
        self.k_cache = DeepseekV32IndexerCache(
            head_dim=self.head_dim + self.head_dim // self.quant_block_size * 4,
            dtype=torch.uint8,
            prefix=f"{prefix}.k_cache",
            cache_config=cache_config,
        )
        self.max_model_len = vllm_config.model_config.max_model_len
        self.prefix = prefix
        from vllm.v1.attention.backends.mla.indexer import get_max_prefill_buffer_size

        self.max_total_seq_len = get_max_prefill_buffer_size(vllm_config)

    def forward(
        self, hidden_states: torch.Tensor, qr: torch.Tensor, positions, rotary_emb
    ) -> torch.Tensor:
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        q, _ = self.wq_b(qr)
        q = q.view(-1, self.n_head, self.head_dim)
        q_pe, q_nope = torch.split(
            q, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )

        k, _ = self.wk(hidden_states)
        k = self.k_norm(k)
        k_pe, k_nope = torch.split(
            k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )

        q_pe, k_pe = rotary_emb(positions, q_pe, k_pe.unsqueeze(1))
        q = torch.cat([q_pe, q_nope], dim=-1)
        k = torch.cat([k_pe.squeeze(1), k_nope], dim=-1)

        # we only quant q here since k quant is fused with cache insertion
        q = q.view(-1, self.head_dim)

        # Use PyTorch implementation if environment variable is set
        use_pytorch_quant = (
            typing.cast(
                str, os.environ.get("VLLM_USE_PYTORCH_FP8_QUANT", "false")
            ).lower()
            == "true"
        )

        if use_pytorch_quant:
            q_fp8, q_scale = per_token_group_quant_fp8_pytorch(
                q,
                self.quant_block_size,
                column_major_scales=False,
                use_ue8m0=self.scale_fmt is not None,
            )
        else:
            q_fp8, q_scale = per_token_group_quant_fp8(
                q,
                self.quant_block_size,
                column_major_scales=False,
                use_ue8m0=self.scale_fmt is not None,
            )
        q_fp8 = q_fp8.view(
            -1, self.n_head, self.head_dim
        )  # [padded_token_num, n_head, head_dim] n_head: 64
        q_scale = q_scale.view(-1, self.n_head, 1)  # [padded_token_num, n_head, 1]

        weights, _ = self.weights_proj(hidden_states)  # [padded_token_num, n_head]
        weights = (
            weights.unsqueeze(-1) * q_scale * self.softmax_scale * self.n_head**-0.5
        )
        weights = weights.squeeze(-1)
        return torch.ops.vllm.sparse_attn_indexer(
            hidden_states,
            self.k_cache.prefix,
            self.k_cache.kv_cache[0],
            q_fp8,
            k,
            weights,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
        )


class DeepseekV2MLAAttention(nn.Module):
    """
    Main reference: DeepseekV2 paper, and FlashInfer Implementation
    (https://arxiv.org/abs/2405.04434 and https://github.com/flashinfer-ai/flashinfer/pull/551).

        For more info see MLACommonImpl in:
        vllm/v1/attention/backends/mla/utils.py
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        rope_theta: float = 10000,
        rope_scaling: dict[str, Any] | None = None,
        max_position_embeddings: int = 8192,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank

        self.num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        self.num_local_heads = num_heads // tp_size

        self.scaling = self.qk_head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        if self.q_lora_rank is not None:
            self.fused_qkv_a_proj = MergedColumnParallelLinear(
                self.hidden_size,
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.fused_qkv_a_proj",
                disable_tp=True,
            )
        else:
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_a_proj_with_mqa",
            )

        if self.q_lora_rank is not None:
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                self.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_b_proj",
            )
        else:
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj",
            )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        if rope_scaling:
            rope_scaling["rope_type"] = "deepseek_yarn"
        self.rotary_emb = get_rope(
            qk_rope_head_dim,
            rotary_dim=qk_rope_head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=False,
        )
        if rope_scaling:
            mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
            scaling_factor = rope_scaling["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale

        self.is_v32 = hasattr(config, "index_topk")

        if self.is_v32:
            self.indexer = Indexer(
                vllm_config,
                config,
                hidden_size,
                q_lora_rank,
                quant_config,
                cache_config,
                topk_indices_buffer,
                f"{prefix}.indexer",
            )
        else:
            self.indexer = None

        mla_modules = MLAModules(
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            rotary_emb=self.rotary_emb,
            o_proj=self.o_proj,
            fused_qkv_a_proj=self.fused_qkv_a_proj
            if self.q_lora_rank is not None
            else None,
            kv_a_proj_with_mqa=self.kv_a_proj_with_mqa
            if self.q_lora_rank is None
            else None,
            q_a_layernorm=self.q_a_layernorm if self.q_lora_rank is not None else None,
            q_b_proj=self.q_b_proj if self.q_lora_rank is not None else None,
            q_proj=self.q_proj if self.q_lora_rank is None else None,
            indexer=self.indexer,
            is_sparse=self.is_v32,
            topk_indices_buffer=topk_indices_buffer,
        )

        self.mla_attn = MultiHeadLatentAttentionWrapper(
            self.hidden_size,
            self.num_local_heads,
            self.scaling,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
            self.q_lora_rank,
            self.kv_lora_rank,
            mla_modules,
            cache_config,
            quant_config,
            prefix,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.mla_attn(positions, hidden_states)


class DeepseekV2DecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        config: DeepseekV2Config | None = None,
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        super().__init__()

        if config is None:
            config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        # DecoderLayers are created with `make_layers` which passes the prefix
        # with the layer's index.
        layer_idx = int(prefix.split(sep=".")[-1])
        self.layer_idx = layer_idx
        if model_config.use_mla:
            attn_cls = DeepseekV2MLAAttention
        else:
            attn_cls = DeepseekV2Attention
        self.self_attn = attn_cls(
            vllm_config=vllm_config,
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank if hasattr(config, "q_lora_rank") else None,
            kv_lora_rank=config.kv_lora_rank,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            topk_indices_buffer=topk_indices_buffer,
        )

        if (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        ):
            self.mlp = DeepseekV2MoE(
                config=config,
                parallel_config=parallel_config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.routed_scaling_factor = config.routed_scaling_factor

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> torch.Tensor:
        # Self Attention
        if residual is None:
            residual = hidden_states.clone()
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        if hidden_states.dtype == torch.float16:
            # Fix FP16 overflow
            # We scale both hidden_states and residual before
            # rmsnorm, and rmsnorm result would not affect by scale.
            hidden_states *= 1.0 / self.routed_scaling_factor
            if self.layer_idx == 0:
                # The residual is shared by all layers, we only scale it on
                # first layer.
                residual *= 1.0 / self.routed_scaling_factor

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        if isinstance(self.mlp, DeepseekV2MLP) and hidden_states.dtype == torch.float16:
            # Fix FP16 overflow
            # Scaling the DeepseekV2MLP output, it is the input of
            # input_layernorm of next decoder layer.
            # The scaling of DeepseekV2MOE output would be done in the forward
            # of DeepseekV2MOE
            hidden_states *= 1.0 / self.routed_scaling_factor

        return hidden_states, residual


@support_torch_compile
class DeepseekV2Model(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.device = current_platform.device_type

        self.vocab_size = config.vocab_size
        self.is_v32 = hasattr(config, "index_topk")
        if self.is_v32:
            topk_tokens = config.index_topk
            topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                topk_tokens,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            topk_indices_buffer = None

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: DeepseekV2DecoderLayer(
                vllm_config, prefix, topk_indices_buffer=topk_indices_buffer
            ),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class DeepseekV2ForCausalLM(nn.Module, SupportsPP, MixtureOfExperts, SupportsLoRA):
    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        # `packed_modules_mapping` needs to be modified before
        # initializing DeepseekV2Model, as it is passed inplace to
        # quantization config init and may be used to select the
        # quant_method for relevant layers during initialization.
        self.fuse_qkv_a_proj = (
            hasattr(config, "q_lora_rank") and config.q_lora_rank is not None
        )
        if self.fuse_qkv_a_proj:
            self.packed_modules_mapping["fused_qkv_a_proj"] = [
                "q_a_proj",
                "kv_a_proj_with_mqa",
            ]

        self.model = DeepseekV2Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        self.expert_weights = []

        # Set MoE hyperparameters
        self.num_moe_layers = config.num_hidden_layers - config.first_k_dense_replace
        self.num_expert_groups = config.n_group

        self.moe_layers: list[SharedFusedMoE] = []
        example_moe = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue

            assert isinstance(layer, DeepseekV2DecoderLayer)
            if isinstance(layer.mlp, DeepseekV2MoE):
                # Pick last one layer since the first ones may be dense layers.
                example_moe = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_moe is None:
            raise RuntimeError("No DeepseekV2MoE layer found in model.layers.")

        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_shared_experts = example_moe.n_shared_experts
        self.num_redundant_experts = example_moe.n_redundant_experts

    def set_eplb_state(
        self,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> None:
        for layer_idx, layer in enumerate(self.moe_layers):
            # Register the expert weights.
            self.expert_weights.append(layer.get_expert_weights())
            layer.set_eplb_state(
                moe_layer_idx=layer_idx,
                expert_load_view=expert_load_view,
                logical_to_physical_map=logical_to_physical_map,
                logical_replica_count=logical_replica_count,
            )

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.model.layers:
            if isinstance(layer.mlp, DeepseekV2MoE):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return SharedFusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
            num_redundant_experts=0,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            ("fused_qkv_a_proj", "q_a_proj", 0),
            ("fused_qkv_a_proj", "kv_a_proj_with_mqa", 1),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = SharedFusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts
            + (
                self.config.n_shared_experts
                if is_rocm_aiter_fusion_shared_expert_enabled()
                else 0
            ),
            num_redundant_experts=self.num_redundant_experts,
        )

        num_hidden_layers = self.config.num_hidden_layers
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            # support load less layers for test
            if name.startswith("model.layers."):
                import re

                layer_num = re.search(r"model\.layers\.(\d+)\.", name, re.IGNORECASE)
                if layer_num:
                    curr_layer_num = int(layer_num.group(1))
                    if curr_layer_num >= num_hidden_layers:
                        # print(f"Skipping loading weight: {name} not in model.")
                        continue
            if "rotary_emb.inv_freq" in name:
                continue

            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is not None:
                continue  # skip spec decode layers for main model

            is_fuse_shared_experts_layer = (
                is_rocm_aiter_fusion_shared_expert_enabled()
                and ("mlp.shared_experts" in name)
            )

            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                if is_fuse_shared_experts_layer:
                    continue
                name_mapped = name.replace(weight_name, param_name)

                # QKV fusion is optional, fall back to normal
                # weight loading if it's not enabled
                # if go with fusion option, then update name
                if (
                    param_name == "fused_qkv_a_proj"
                ) and name_mapped not in params_dict:
                    continue
                else:
                    name = name_mapped
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                is_expert_weight = False

                # Special handling: when AITER fusion_shared_experts is enabled,
                # checkpoints may provide a single widened shared_experts tensor
                # without explicit expert indices
                # (e.g. ...mlp.shared_experts.gate_proj.weight).
                # For models with multiple shared experts, split that tensor
                # evenly into per-shared-expert slices and load them into
                # appended expert slots mlp.experts.{n_routed_experts + j}.*
                # accordingly.
                num_chunks = 1
                if is_fuse_shared_experts_layer:
                    num_chunks = getattr(self.config, "n_shared_experts", 1) or 1
                    # Determine split axis based on op type
                    # gate/up: ColumnParallel → split along dim 0
                    # down: RowParallel → split along dim 1
                    split_dim = 1 if "down_proj.weight" in name else 0
                    total = loaded_weight.shape[split_dim]
                    assert total % num_chunks == 0, (
                        f"Shared expert weight dim {total} "
                        f"not divisible by num_chunks {num_chunks}"
                    )
                    chunk_size = total // num_chunks

                for j in range(num_chunks):
                    chunk_name = name
                    weight_to_load = loaded_weight

                    if is_fuse_shared_experts_layer:
                        if split_dim == 0:
                            weight_to_load = loaded_weight[
                                j * chunk_size : (j + 1) * chunk_size, :
                            ]
                        else:
                            weight_to_load = loaded_weight[
                                :, j * chunk_size : (j + 1) * chunk_size
                            ]
                        # Synthesize an expert-style name so expert mapping
                        # can route it
                        chunk_name = name.replace(
                            "mlp.shared_experts",
                            f"mlp.experts.{self.config.n_routed_experts + j}",
                        )

                    # Use expert_params_mapping to locate the destination
                    # param and delegate to its expert-aware weight_loader
                    # with expert_id.
                    for mapping in expert_params_mapping:
                        param_name, weight_name, expert_id, shard_id = mapping
                        if weight_name not in chunk_name:
                            continue

                        # Anyway, this is an expert weight and should not be
                        # attempted to load as other weights later
                        is_expert_weight = True

                        # Do not modify `name` since the loop may continue here
                        # Instead, create a new variable
                        name_mapped = chunk_name.replace(weight_name, param_name)

                        if is_pp_missing_parameter(name_mapped, self):
                            continue

                        param = params_dict[name_mapped]
                        # We should ask the weight loader to return success or
                        # not here since otherwise we may skip experts with
                        # other available replicas.
                        weight_loader = typing.cast(
                            Callable[..., bool], param.weight_loader
                        )
                        success = weight_loader(
                            param,
                            weight_to_load,
                            name_mapped,
                            shard_id=shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                        if success:
                            if not is_fuse_shared_experts_layer:
                                name = name_mapped
                            else:
                                loaded_params.add(name_mapped)
                            break
                    else:
                        if is_expert_weight:
                            # We've checked that this is an expert weight
                            # However it's not mapped locally to this rank
                            # So we simply skip it
                            continue

                        # Skip loading extra bias for GPTQ models.
                        if name.endswith(".bias") and name not in params_dict:
                            continue

                        # Remapping the name of FP8 kv-scale.
                        name = maybe_remap_kv_scale_name(name, params_dict)
                        if name is None:
                            continue

                        if is_pp_missing_parameter(name, self):
                            continue

                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
            if not is_fuse_shared_experts_layer:
                loaded_params.add(name)

        return loaded_params


class DeepseekV3ForCausalLM(DeepseekV2ForCausalLM):
    pass


# Compatibility with
# https://huggingface.co/deepseek-ai/DeepSeek-V3-Base/blob/main/configuration_deepseek.py
def get_spec_layer_idx_from_weight_name(
    config: DeepseekV2Config | DeepseekV3Config, weight_name: str
) -> int | None:
    if (
        hasattr(config, "num_nextn_predict_layers")
        and config.num_nextn_predict_layers > 0
    ):
        layer_idx = config.num_hidden_layers
        for i in range(config.num_nextn_predict_layers):
            if weight_name.startswith(f"model.layers.{layer_idx + i}."):
                return layer_idx + i
    return None
