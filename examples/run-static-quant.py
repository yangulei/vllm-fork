from vllm import LLM, SamplingParams
import os

os.environ['VLLM_SKIP_WARMUP'] = 'true'
os.environ['VLLM_GRAPH_RESERVED_MEM'] = '0.2'
os.environ['VLLM_MOE_N_SLICE'] = '32'

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

if __name__ == "__main__":
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.0)

    # Create an LLM.
    # llm = LLM(model="facebook/opt-125m", dtype="bfloat16")

    # llm = LLM(model="/models/DeepSeek-R1-G2-FP8/",
    # llm = LLM(model="/models/DeepSeek-R1-BF16-layer5-w8afp8-static-no-ste-G2",
    # llm = LLM(model="/models/DeepSeek-R1-BF16-layer5-w8afp8-dynamic-no-ste-G2",
    llm = LLM(
            # model="/models/DeepSeek-R1-BF16-w8afp8-static-no-ste-G2",
            # model="/models/DeepSeek-R1-BF16-w8afp8-dynamic-no-ste-G2",
            # model="/models/DeepSeek-R1-BF16-layer5-w8afp8-static-no-ste-G2",
            model="/models/DeepSeek-R1-BF16-layer5-w8afp8-dynamic-no-ste-G2",
            trust_remote_code=True,
            #enforce_eager=True,
            dtype="bfloat16",
            max_num_seqs=32,
            use_v2_block_manager=True,
            tensor_parallel_size=1,
            # block_size=128,
            # max_num_batched_tokens=8192,
            max_model_len=4096,
            # use_padding_aware_scheduling=True,
            num_scheduler_steps=1,
            #   distributed_executor_backend='mp',
            gpu_memory_utilization=0.95,
            seed=2024)

    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

