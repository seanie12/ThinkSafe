#!/usr/bin/env python3
import argparse
import requests
import os
from typing import List

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from peft import LoraConfig, get_peft_model

from trl import GRPOConfig, GRPOTrainer
from dotenv import load_dotenv
import wandb

DS_NAME = {
    # ThinkSafe: Qwen3 + LlamaGuard
    "thinksafe-0.6B": "icml2026-7516/ThinkSafe-Qwen3-0.6B",
    "thinksafe-1.7B": "icml2026-7516/ThinkSafe-Qwen3-1.7B",
    "thinksafe-4B": "icml2026-7516/ThinkSafe-Qwen3-4B",
    "thinksafe-8B": "icml2026-7516/ThinkSafe-Qwen3-8B",

    # ThinkSafe: R1-Distill + LlamaGuard
    "thinksafe-r1-1.5B": "icml2026-7516/ThinkSafe-R1-Distill-1.5B",
    "thinksafe-r1-7B": "icml2026-7516/ThinkSafe-R1-Distill-7B",
    "thinksafe-r1-8B": "icml2026-7516/ThinkSafe-R1-Distill-8B",

    # ThinkSafe: Qwen3 + WildGuard
    "thinksafe-0.6B-wildguard": "icml2026-7516/ThinkSafe-Qwen3-0.6B-WildGuard",
    "thinksafe-1.7B-wildguard": "icml2026-7516/ThinkSafe-Qwen3-1.7B-WildGuard",
    "thinksafe-4B-wildguard": "icml2026-7516/ThinkSafe-Qwen3-4B-WildGuard",
    "thinksafe-8B-wildguard": "icml2026-7516/ThinkSafe-Qwen3-8B-WildGuard",

    # ThinkSafe: R1-Distill + WildGuard
    "thinksafe-r1-1.5B-wildguard": "icml2026-7516/ThinkSafe-R1-Distill-1.5B-WildGuard",
    "thinksafe-r1-7B-wildguard": "icml2026-7516/ThinkSafe-R1-Distill-7B-WildGuard",
    "thinksafe-r1-8B-wildguard": "icml2026-7516/ThinkSafe-R1-Distill-8B-WildGuard",
}

GUARD_MODEL = "meta-llama/Llama-Guard-3-8B"
GUARD_TOKENIZER = AutoTokenizer.from_pretrained(GUARD_MODEL)
SAFE_TOKEN = GUARD_TOKENIZER.tokenize("safe")[0]
UNSAFE_TOKEN = GUARD_TOKENIZER.tokenize("unsafe")[0]


def _llama_guard_reward_factory(args):
    """
    LlamaGuard reward factory
    Send request to LlamaGuard vLLM server and return the logprob of 'safe' token as reward.
    """
    GUARD_URL = f"http://127.0.0.1:{args.guard_server_port}/v1/completions"

    def reward_fn(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        # 1. Format prompts for LlamaGuard vLLM server
        formatted_prompts = []
        for p, c in zip(prompts, completions):
            messages = [
                {"role": "user", "content": p},
                {"role": "assistant", "content": c}
            ]
            formatted = GUARD_TOKENIZER.apply_chat_template(
                messages,
                tokenize=False,
            )
            formatted_prompts.append(formatted)
        chats = [chat + "\n\n" for chat in formatted_prompts]
        assert len(chats) == len(prompts) and len(chats) == len(completions), "Length mismatch between prompts, completions, and formatted prompts"

        # 2. Send batch request to LlamaGuard vLLM server
        payload = {
            "model": GUARD_MODEL,
            "prompt": chats,
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": 2,
        }

        # 3. Get rewards from LlamaGuard vLLM server
        rewards = [0.0] * len(prompts)
        try:
            response = requests.post(GUARD_URL, json=payload).json()
            choices = response.get("choices", [])
            for i, choice in enumerate(choices):
                top_logprobs = choice["logprobs"]["top_logprobs"][0]

                # Extract logprobs of 'safe' and 'unsafe' tokens
                lp_safe = torch.tensor(top_logprobs.get(SAFE_TOKEN, -1e4))
                lp_unsafe = torch.tensor(top_logprobs.get(UNSAFE_TOKEN, -1e4))

                max_lp = max(lp_safe, lp_unsafe)
                exp_safe = torch.exp(lp_safe - max_lp)
                exp_unsafe = torch.exp(lp_unsafe - max_lp)

                rewards[i] = (exp_safe / (exp_safe + exp_unsafe)).item()

        except Exception as e:
            print(f"LlamaGuard Batch Reward Error: {e}")

        return rewards

    return reward_fn



def _format_reward_fn(args):
    """
    Check if the <think> ... </think> format is correct.
    """
    def reward_fn(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        rewards = []

        # Check if the <think> ... </think> pattern appears only once.
        for completion in completions:
            # 1. Check if the <think> ... </think> pattern appears only once.
            start_count = completion.count("<think>")
            end_count = completion.count("</think>")

            # R1-Distill: check if </think> appears exactly once with no <think>
            if "R1-Distill" in args.model_name:
                if start_count == 0 and end_count == 1:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            else:
                # 2. Check if the <think> ... </think> pattern is correctly formatted.
                if start_count == 1 and end_count == 1:
                    start_index = completion.find("<think>")
                    end_index = completion.find("</think>")

                    if start_index < end_index:
                        rewards.append(1.0) # Correct format
                    else:
                        rewards.append(0.0) # Incorrect format
                else:
                    rewards.append(0.0) # No <think> or </think> tags or too many tags

        return rewards

    return reward_fn


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO training with ThinkSafe + LlamaGuard reward.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--output_dir", type=str, default="grpo_outputs")
    parser.add_argument("--wandb_project", type=str, default="safety-reasoning")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--effective_batch_size", type=int, default=64)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)

    parser.add_argument("--use_lora", action="store_true", help="Enable training with LoRA.")
    parser.add_argument("--lora_r", type=int, default=32, help="The rank of the LoRA matrices.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="The dropout probability for LoRA layers.")

    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_completion_length", type=int, default=16384)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--kl_coeff", type=float, default=0.04)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--vllm_server_port", type=int, default=8000)
    parser.add_argument("--guard_server_port", type=int, default=8001)
    parser.add_argument("--vllm_group_port", type=int, default=51216)

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if accelerator.is_main_process and wandb is None:
        raise RuntimeError("wandb is required for logging but is not installed.")

    if accelerator.is_main_process and wandb is not None:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    if world_size > 1:
        torch.cuda.set_device(local_rank)
    device_map = {"": local_rank} if world_size > 1 else "auto"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if args.use_lora:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_r // 2,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        print("LoRA training enabled.")
        model = get_peft_model(model, peft_config)
        if accelerator.is_main_process:
            model.print_trainable_parameters()
        
        # enable input require grads
        model.enable_input_require_grads()

        # This ensures that the LoRA parameters maintain their gradient computation
        for _, param in model.named_parameters():
            if param.requires_grad:
                param.retain_grad()

    dataset_name = DS_NAME[args.dataset]
    dataset = load_dataset(dataset_name, split=args.dataset_split)
    dataset = dataset.map(lambda row: {"prompt": tokenizer.apply_chat_template([{
        "role": "user",
        "content": row["instruction"]
    }], tokenize=False, add_generation_prompt=True, enable_thinking=True)})

    if args.gradient_accumulation_steps is None:
        denom = args.per_device_train_batch_size * max(world_size, 1)
        args.gradient_accumulation_steps = max(args.effective_batch_size // denom, 1)

    # Monkey-patch VLLMClient to use custom group_port to avoid conflicts
    # when running multiple instances
    from trl.extras.vllm_client import VLLMClient
    original_init = VLLMClient.__init__
    def patched_init(self, base_url=None, host="0.0.0.0", server_port=8000,
                     group_port=None, connection_timeout=0.0):
        # Use custom group_port from args if provided, otherwise use the passed value
        if group_port is None:
            group_port = args.vllm_group_port
        original_init(self, base_url, host, server_port, group_port, connection_timeout)
    VLLMClient.__init__ = patched_init
    print(f"Using custom vLLM group port: {args.vllm_group_port}")

    training_args = GRPOConfig(
        beta=args.kl_coeff,
        temperature=args.temperature,
        epsilon=args.epsilon,
        loss_type="grpo",
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        logging_steps=args.logging_steps,
        max_grad_norm=1.0,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_generations=args.num_generations,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_train_epochs=args.epochs,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to="wandb",
        output_dir=args.output_dir,
        seed=args.seed,
        ddp_backend="nccl" if world_size > 1 else None,
        ddp_find_unused_parameters=False,
        run_name=args.wandb_run_name,
        use_vllm=True,
        vllm_mode='server',
        vllm_server_host='127.0.0.1',
        vllm_server_port=args.vllm_server_port,
    )

    guard_reward_fn = _llama_guard_reward_factory(args)
    format_reward_fn = _format_reward_fn(args)
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[guard_reward_fn, format_reward_fn],
        args=training_args,
        train_dataset=dataset,
        callbacks=[],
    )

    trainer.train()
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        final_dir = os.path.join(args.output_dir, "final")
        trainer.save_model(final_dir)


if __name__ == "__main__":
    main()
