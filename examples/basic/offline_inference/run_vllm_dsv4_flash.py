#!/usr/bin/env python
"""Run DeepSeek-V4-Flash on Intel XPU (BMG Arc Pro B60, ~24GB) offline and print output.

Python port of run_vllm_dsv4_flash.sh — uses vllm.LLM.generate() directly so you
can observe the model's output inline (no HTTP server, no client).

Strategy: Override num_hidden_layers via hf_overrides so vLLM builds a smaller
model. The HF safetensors checkpoint (43 layers) loads only the layers the model
instantiates; extra layers in the checkpoint are simply unused by vLLM's loader.
compress_ratios MUST be truncated to match num_hidden_layers (the model indexes
compress_ratios[layer_id] directly — see vllm/model_executor/models/deepseek_v4.py:845).

Layer budget: per-layer ~3.69GB (FP4 MoE dominates). With 24GB HBM and ~2GB
overhead, ~5 layers fit. Default is 4 layers using compress_ratios=[0,0,4,0]:
  layer 2 → C4 (exercises M5/M6/M7 kernels), layer 3 → SWA-only
  (C128 not yet XPU-ported; once ported, use [0,0,4,128]).

Usage:
  python run_vllm_dsv4_flash.py
  N_LAYERS=2 python run_vllm_dsv4_flash.py
  PROMPT="Your question here" python run_vllm_dsv4_flash.py
"""

from __future__ import annotations

import json
import os
import sys

# Env vars must be set BEFORE importing vllm/torch.
os.environ.setdefault("ZE_AFFINITY_MASK", "0")
os.environ.setdefault("ONEAPI_DEVICE_SELECTOR", "level_zero:0")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")

MODEL = os.environ.get("MODEL", "deepseek-ai/DeepSeek-V4-Flash")
N_LAYERS = int(os.environ.get("N_LAYERS", "4"))

# compress_ratios MUST have length == num_hidden_layers; the model indexes
# compress_ratios[layer_id] directly (deepseek_v4.py:845). Original full-model
# array is [0, 0, 4, 128, 4, 128, ...] length 44 (43 layers + 1 MTP).
_compress_env = os.environ.get("COMPRESS_RATIOS")
if _compress_env:
    COMPRESS_RATIOS = json.loads(_compress_env)
elif N_LAYERS == 4:
    # layer 2 → C4 (M5/M6/M7 kernels); layer 3 → SWA-only (C128 not yet XPU-ported).
    COMPRESS_RATIOS = [0, 0, 4, 0]
else:
    COMPRESS_RATIOS = [0] * N_LAYERS

assert len(COMPRESS_RATIOS) == N_LAYERS, (
    f"len(COMPRESS_RATIOS)={len(COMPRESS_RATIOS)} must equal N_LAYERS={N_LAYERS}"
)

HF_OVERRIDES = {
    "num_hidden_layers": N_LAYERS,
    "compress_ratios": COMPRESS_RATIOS,
    "num_nextn_predict_layers": 0,
}

DEFAULT_PROMPT = "What is 12 * 13? Think step by step, then give the final answer."
PROMPT = os.environ.get("PROMPT", DEFAULT_PROMPT)

print("=== DeepSeek-V4-Flash XPU launch (offline) ===")
print(f"Model:         {MODEL}")
print(f"n_layers:      {N_LAYERS}  (full model = 43)")
print(f"hf-overrides:  {json.dumps(HF_OVERRIDES)}")
print(f"prompt:        {PROMPT!r}")
print("==============================================")
sys.stdout.flush()

from vllm import LLM, SamplingParams  # noqa: E402


def main() -> None:
    # DeepSeek-V4 ships fp8 weights + fp4 expert weights, but XPU MLA Sparse
    # backend only supports fp16/bf16 KV cache (xpu_mla_sparse.py). bf16 compute
    # + auto KV dtype is the only working combo on BMG.
    llm = LLM(
        model=MODEL,
        hf_overrides=HF_OVERRIDES,
        trust_remote_code=True,
        dtype="bfloat16",
        kv_cache_dtype="auto",
        max_model_len=4096,
        max_num_seqs=1,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        enable_prefix_caching=False,
        distributed_executor_backend="mp",
        tensor_parallel_size=1,
    )

    sampling = SamplingParams(
        temperature=float(os.environ.get("TEMPERATURE", "0.7")),
        top_p=float(os.environ.get("TOP_P", "0.95")),
        max_tokens=int(os.environ.get("MAX_TOKENS", "512")),
    )

    # Try to apply the model's chat template if available; fall back to raw prompt.
    try:
        tokenizer = llm.get_tokenizer()
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPT}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[warn] chat template unavailable ({e}); using raw prompt.")
        formatted = PROMPT

    print("\n--- generating ---")
    outputs = llm.generate([formatted], sampling)

    print("\n=== MODEL OUTPUT ===")
    for out in outputs:
        for completion in out.outputs:
            print(completion.text)
    print("====================")


if __name__ == "__main__":
    main()
