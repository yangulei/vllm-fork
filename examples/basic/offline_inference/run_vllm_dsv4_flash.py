#!/usr/bin/env python
"""Run DeepSeek-V4-Flash on Intel XPU offline and print output.

Two modes:

1. Truncated harness (default, N_LAYERS=4, fits ~24GB BMG):
   Builds a 4-layer cut-down model via hf_overrides. Output is gibberish by
   construction (the bulk of language modeling lives in layers 4..42), but all
   XPU sparse kernel paths are exercised. Default compress_ratios=[0,0,4,128]
   hits both C4 (Indexer top-k) and C128 (heavily-compressed) paths.

2. Full model (FULL=1 or N_LAYERS=43): keeps the checkpoint's native
   num_hidden_layers=43 and compress_ratios; produces real output. Requires
   sufficient HBM (multi-GPU or larger device) — per-layer ~3.69GB.

Prompt: at compress_ratio=128, seq_len >= 128 is needed to attend to even one
compressed entry. The default prompt comfortably exceeds this.

Usage:
  python run_vllm_dsv4_flash.py                 # 4-layer harness, [0,0,4,128]
  FULL=1 python run_vllm_dsv4_flash.py          # full 43-layer model
  FULL=1 TP=8 python run_vllm_dsv4_flash.py     # full model with explicit TP
  N_LAYERS=43 python run_vllm_dsv4_flash.py     # same as FULL=1
  N_LAYERS=8 python run_vllm_dsv4_flash.py      # custom truncation
  COMPRESS_RATIOS='[0,0,4,128]' N_LAYERS=4 python run_vllm_dsv4_flash.py
  PROMPT="Your question here" python run_vllm_dsv4_flash.py
"""

from __future__ import annotations

import json
import os
import sys

# Env vars must be set BEFORE importing vllm/torch.
# os.environ.setdefault("ZE_AFFINITY_MASK", "4,5,6,7")
# os.environ.setdefault("ONEAPI_DEVICE_SELECTOR", "level_zero:0")
# os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
# os.environ.setdefault("VLLM_LOGGING_LEVEL", "DEBUG")

MODEL = os.environ.get("MODEL", "deepseek-ai/DeepSeek-V4-Flash")

# Native checkpoint topology (DeepSeek-V4-Flash config.json):
#   num_hidden_layers=43, num_nextn_predict_layers=1 (MTP),
#   compress_ratios length = 44 (= 43 + 1 MTP).
FULL_NUM_HIDDEN_LAYERS = 43
FULL_NUM_NEXTN_PREDICT_LAYERS = 1
FULL_COMPRESS_RATIOS = [
    0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
    4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
    4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0,
]
assert len(FULL_COMPRESS_RATIOS) == FULL_NUM_HIDDEN_LAYERS + FULL_NUM_NEXTN_PREDICT_LAYERS

FULL = os.environ.get("FULL", "0") == "1"
N_LAYERS = int(os.environ.get("N_LAYERS", str(FULL_NUM_HIDDEN_LAYERS if FULL else 4)))
FULL = FULL or N_LAYERS == FULL_NUM_HIDDEN_LAYERS

# compress_ratios MUST have length == num_hidden_layers + num_nextn_predict_layers;
# the model indexes compress_ratios[layer_id] directly (deepseek_v4.py).
# In FULL mode we leave it unset so vLLM picks it up from the checkpoint config.
_compress_env = os.environ.get("COMPRESS_RATIOS")
COMPRESS_RATIOS: list[int] | None
if _compress_env:
    COMPRESS_RATIOS = json.loads(_compress_env)
    NUM_NEXTN = int(os.environ.get("NUM_NEXTN", "0"))
elif FULL:
    # Use checkpoint's default compress_ratios, just disable MTP.
    COMPRESS_RATIOS = None
    NUM_NEXTN = 0
elif N_LAYERS == 4:
    # layer 2 → C4 (Indexer top-k); layer 3 → C128 (heavily-compressed attention).
    COMPRESS_RATIOS = [0, 0, 4, 128]
    NUM_NEXTN = 0
else:
    COMPRESS_RATIOS = FULL_COMPRESS_RATIOS[0:N_LAYERS]
    NUM_NEXTN = 0

if COMPRESS_RATIOS is not None:
    assert len(COMPRESS_RATIOS) == N_LAYERS + NUM_NEXTN, (
        f"len(COMPRESS_RATIOS)={len(COMPRESS_RATIOS)} must equal "
        f"N_LAYERS + NUM_NEXTN = {N_LAYERS} + {NUM_NEXTN} = {N_LAYERS + NUM_NEXTN}"
    )

HF_OVERRIDES: dict[str, object] = {}
if COMPRESS_RATIOS is not None:
    HF_OVERRIDES["compress_ratios"] = COMPRESS_RATIOS
if not FULL:
    # Truncated harness: shrink the model to fit a single small device.
    HF_OVERRIDES["num_hidden_layers"] = N_LAYERS
HF_OVERRIDES["num_nextn_predict_layers"] = NUM_NEXTN

DEFAULT_PROMPT = (
    "You are a careful math tutor. A student asks the following question and "
    "you must answer it with a clear, step-by-step explanation that a high "
    "school student can follow. Show every intermediate step, name the "
    "arithmetic property used at each step, double-check the result with a "
    "different method, and finally state the answer on its own line prefixed "
    "with 'Answer:'.\n\n"
    "Question: Compute 12 * 13 by hand. First decompose 13 into 10 + 3 and "
    "use the distributive property of multiplication over addition. Then "
    "verify the result by computing 12 * 13 a second way: decompose 12 into "
    "6 + 6 and use distribution again. Compare the two results, note that "
    "they must agree because multiplication of integers is commutative and "
    "associative, and explain in one sentence why these properties guarantee "
    "the agreement. Finally, sanity-check by estimating: 12 is close to 10 "
    "and 13 is close to 10, so the answer should be near 100 but a bit "
    "larger; confirm your computed value falls in that range."
)
PROMPT = os.environ.get("PROMPT", DEFAULT_PROMPT)

MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "4096"))
GPU_MEM_UTIL = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.85"))
tp_env = os.environ.get("TP")
TP_SIZE = int(tp_env) if tp_env is not None else (8 if FULL else 1)

print("=== DeepSeek-V4-Flash XPU launch (offline) ===")
print(f"Model:         {MODEL}")
print(f"mode:          {'FULL (43 layers)' if FULL else f'truncated harness ({N_LAYERS} layers)'}")
print(f"hf-overrides:  {json.dumps(HF_OVERRIDES)}")
print(f"max_model_len: {MAX_MODEL_LEN}, tp_size: {TP_SIZE}, gpu_mem_util: {GPU_MEM_UTIL}")
print(f"prompt:        {PROMPT!r}")
print("==============================================")
sys.stdout.flush()

from vllm import LLM, SamplingParams  # noqa: E402

# --- Benchmark mode ---
# BENCH=1 BATCH=16 INPUT_LEN=2048 python run_vllm_dsv4_flash.py
BENCH = os.environ.get("BENCH", "0") == "1"
BATCH_SIZE = int(os.environ.get("BATCH", "1"))
INPUT_LEN = int(os.environ.get("INPUT_LEN", "512"))


def main() -> None:
    max_num_seqs = BATCH_SIZE if BENCH else 1

    llm = LLM(
        model=MODEL,
        hf_overrides=HF_OVERRIDES,
        trust_remote_code=True,
        dtype="bfloat16",
        kv_cache_dtype="auto",
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=GPU_MEM_UTIL,
        enforce_eager=False,
        enable_prefix_caching=False,
        distributed_executor_backend="mp",
        tensor_parallel_size=TP_SIZE,
    )

    sampling = SamplingParams(
        temperature=float(os.environ.get("TEMPERATURE", "0")),
        top_p=float(os.environ.get("TOP_P", "1")),
        max_tokens=int(os.environ.get("MAX_TOKENS", "512" if not BENCH else "20")),
    )

    if BENCH:
        # Performance mode: random token IDs as input
        import random
        random.seed(42)
        tokenizer = llm.get_tokenizer()
        vocab_size = tokenizer.vocab_size or 100000
        prompts = [
            {"prompt_token_ids": [random.randint(1, vocab_size - 1)
                                  for _ in range(INPUT_LEN)]}
            for _ in range(BATCH_SIZE)
        ]
        print(f"\n--- benchmark: batch={BATCH_SIZE}, input_len={INPUT_LEN}, "
              f"max_tokens={sampling.max_tokens} ---")
        outputs = llm.generate(
            prompts=prompts,
            sampling_params=sampling,
        )
        # Print throughput summary
        total_input = BATCH_SIZE * INPUT_LEN
        total_output = sum(
            len(c.token_ids) for o in outputs for c in o.outputs
        )
        print(f"\n=== BENCHMARK RESULT ===")
        print(f"Requests: {BATCH_SIZE}")
        print(f"Total input tokens: {total_input}")
        print(f"Total output tokens: {total_output}")
        print("========================")
    else:
        # Correctness mode: single chat query
        messages = [{"role": "user", "content": PROMPT}]
        print("\n--- generating ---")
        outputs = llm.chat(messages, sampling_params=sampling)
        print("\n=== MODEL OUTPUT ===")
        for out in outputs:
            for completion in out.outputs:
                print(completion.text)
        print("====================")


if __name__ == "__main__":
    main()
