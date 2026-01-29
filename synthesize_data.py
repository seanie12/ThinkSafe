from vllm import LLM, SamplingParams
from datasets import load_dataset, Dataset
import json
from utils.guard_model import LlamaGuard, WildGuard
import gc
import torch
import argparse
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run(model_type="qwen", size="0.6B", debug=False, temperature=0.6, top_p=0.9, use_wildguard=False):
    # Load dataset
    dataset = load_dataset("UWNSL/SafeChain", split="train")
    if debug:
        dataset = dataset.select(range(500))

    # Separate harmful and benign prompts
    harmful_prompts = []
    harmful_labels = []
    benign_prompts = []
    benign_labels = []

    for x in dataset:
        label = x["label"]
        if "harmful" in label:
            harmful_prompts.append(x["instruction"])
            harmful_labels.append(label)
        else:
            benign_prompts.append(x["instruction"])
            benign_labels.append(label)

    print(f"Loaded {len(harmful_prompts)} harmful prompts and {len(benign_prompts)} benign prompts")

    # Model selection based on type
    if model_type == "qwen":
        model_id = f"Qwen/Qwen3-{size}"
    
    elif model_type == "r1-distill":
        if size == "1.5B":
            model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        elif size == "7B":
            model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        elif size == "8B":
            model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print(f"Loading model: {model_id}")

    num_gpus = torch.cuda.device_count()
    llm = LLM(model_id, gpu_memory_utilization=0.95, tensor_parallel_size=num_gpus)

    # Set top_k based on model type (Qwen uses 20, others use 0)
    top_k = 20 if model_type == "qwen" else 0
    sampling_params = SamplingParams(
        temperature=temperature, top_p=top_p, max_tokens=16384, n=1, top_k=top_k)

    refusal_prompt = "The following prompt is harmful. You should refuse to answer the prompt.\n\nPrompt:"

    # Generate responses for harmful prompts
    print("Generating responses for harmful prompts...")
    messages_list = [
        [{"role": "user", "content": f"{refusal_prompt}\n\n{prompt}"}]
        for prompt in harmful_prompts
    ]

    harmful_outputs = llm.chat(messages_list, sampling_params)

    # Process harmful responses
    results = []
    for i, completion in enumerate(harmful_outputs):
        prompt = harmful_prompts[i]
        label = harmful_labels[i]
        response = completion.outputs[0].text
        # For r1-distill models, prepend <think>\n if not present
        if model_type == "r1-distill" and not response.startswith("<think>"):
            response = "<think>\n" + response

        # Only include responses with complete think blocks
        if "</think>" in response:
            results.append({
                "instruction": prompt,
                "response": response,
                "prompt_label": label
            })

    print(f"Added {len(results)} harmful prompts with generated refusals")

    # Step 5: Generate responses for benign prompts
    print("\n" + "="*80)
    print("STEP 5: Generating responses for benign prompts")
    print("="*80)

    conversations = [[{"role":"user", "content":prompt}] for prompt in benign_prompts]
    benign_outputs = llm.chat(conversations, sampling_params)

    # For some models (e.g. r1-distill) the model prepends <think> in generation.
    # To keep response format consistent, prepend a <think> tag to benign responses
    for i, completion in enumerate(benign_outputs):
        prompt = benign_prompts[i]
        label = benign_labels[i]
        response = completion.outputs[0].text
        # skip examples without the ending think
        if "</think>" not in response:
            continue 
        # For r1-distill models, prepend <think>\n if not present
        if model_type == "r1-distill" and not response.startswith("<think>"):
            response = "<think>\n" + response

        results.append({
            "instruction": prompt,
            "response": response,
            "prompt_label": label
        })

    print(f"Added {len(benign_prompts)} benign prompt-response pairs")

    # Clean up before filtering
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    # Step 6: Filter with guard model
    print("\n" + "="*80)
    guard_name = "WildGuard" if use_wildguard else "LlamaGuard"
    print(f"STEP 6: Filtering with {guard_name}")
    print("="*80)

    if use_wildguard:
        guard = WildGuard()
    else:
        guard = LlamaGuard(pbar=True)

    prompts = [x["instruction"] for x in results]
    responses = [x["response"] for x in results]
    predictions = guard.predict(prompts, responses)  # 0: safe, 1: unsafe

    # Filter out harmful responses (predictions == 1)
    filtered_results = []
    for i in range(len(results)):
        if predictions[i].item() == 0:  # Keep only safe responses
            results[i]["response_label"] = "safe"
            filtered_results.append(results[i])

    # Calculate statistics
    total_before = len(results)
    total_after = len(filtered_results)
    num_filtered = total_before - total_after
    safe_ratio = total_after / total_before if total_before > 0 else 0

    print(f"\nFiltering Statistics:")
    print(f"Total examples before filtering: {total_before}")
    print(f"Total examples after filtering: {total_after}")
    print(f"Examples removed (harmful): {num_filtered}")
    print(f"Safe ratio: {safe_ratio:.2%}")

    # Print a sample entry
    if filtered_results:
        print("\nSample JSON entry:")
        print(json.dumps(filtered_results[0], indent=2))

    # Save to JSON file
    suffix = "_wildguard" if use_wildguard else ""
    output_filename = f"./dataset/ThinkSafe-{model_type}-{size}{suffix}.json"
    if debug:
        output_filename = f"./dataset/ThinkSafe-{model_type}-{size}{suffix}_debug.json"
    with open(output_filename, 'w') as f:
        json.dump(filtered_results, f, indent=2)    

    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Harmful prompts with generated refusals: {len(harmful_prompts)}")
    print(f"Benign prompts: {len(benign_prompts)}")
    print(f"Total examples after {guard_name} filtering: {len(filtered_results)}")
    print(f"Saved to: {output_filename}")
    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True,
                       choices=["qwen", "r1-distill"],
                       help="Model type to use")
    parser.add_argument("--size", type=str, required=True,
                       help="Model size (0.6B, 1.7B, 4B, 8B, 14B, 32B for Qwen; 1.5B, 7B, 8B for R1-Distill)")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--use-wildguard", action="store_true",
                       help="Use WildGuard instead of LlamaGuard for filtering")
    args = parser.parse_args()

    # Validate size based on model type
    if args.model_type == "qwen":
        valid_sizes = ["0.6B", "1.7B", "4B", "8B", "14B", "32B"]
    elif args.model_type == "r1-distill":
        valid_sizes = ["1.5B", "7B", "8B"]

    if args.size not in valid_sizes:
        parser.error(f"Invalid size {args.size} for model type {args.model_type}. Valid sizes: {valid_sizes}")

    run(model_type=args.model_type, size=args.size, debug=args.debug,
        temperature=args.temperature, top_p=args.top_p, use_wildguard=args.use_wildguard)
