from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.custom_dataset import get_dataset
from utils.data_collator import DataCollatorForKLTraining
from peft import LoraConfig, get_peft_model

from trl import SFTConfig, SFTTrainer
import torch
import argparse
import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs
from trainers.kl_trainer import ForwardKLMaskedSFTTrainer

DS_NAME = {
    # Baselines
    "safechain": "UWNSL/SafeChain",
    "direct-refusal": "TianshengHuang/DirectRefusal",

    # ThinkSafe: Qwen3 + LlamaGuard
    "thinksafe-0.6B": "icml2026-7516/ThinkSafe-Qwen3-0.6B   ",
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

def main(args):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        log_with="wandb",
        kwargs_handlers=[ddp_kwargs]
    )

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.student_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if "R1-Distill" in args.student_model:
        original_template = tokenizer.chat_template
        modified_template = original_template.replace(
            "{% set content = message['content'] %}{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}",
            "{% set content = message['content'] %}"
        )
        tokenizer.chat_template = modified_template

    dataset_name = DS_NAME[args.safety_dataset]

    # Only main process prepares the dataset
    if accelerator.is_main_process:
        print(f"Main process: Preparing dataset {dataset_name}")
        dataset = get_dataset(
            dataset_name, tokenizer, args.max_length,
            prompt_masking=args.prompt_masking, use_cache=True)

    # Synchronize all processes
    accelerator.wait_for_everyone()

    # Now all processes load the cached dataset
    if not accelerator.is_main_process:
        print(f"Process {accelerator.process_index}: Loading cached dataset")
        dataset = get_dataset(
            dataset_name, tokenizer, args.max_length,
            prompt_masking=args.prompt_masking, use_cache=True)

    world_size = accelerator.num_processes
    args.gradient_accumulation_steps = args.effective_batch_size // (args.per_device_batch_size * world_size)
    args.gradient_accumulation_steps = max(1, args.gradient_accumulation_steps)

    if args.use_lora:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        print("LoRA training enabled.")
        model = get_peft_model(model, peft_config)

        if accelerator.is_main_process:
            model.print_trainable_parameters()

    model.gradient_checkpointing_enable()
    if args.use_lora:
            # This ensures that the LoRA parameters maintain their gradient computation
            for _, param in model.named_parameters():
                if param.requires_grad:
                    param.retain_grad()

    training_arguments = SFTConfig(
        output_dir=f"{args.output_dir}/checkpoint",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim="adamw_torch",
        learning_rate=args.lr,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        logging_strategy="steps",
        save_strategy="epoch",
        remove_unused_columns=False,
        save_only_model=True,
        report_to="wandb",
        max_grad_norm=1.0,
        bf16=True,
        ddp_find_unused_parameters=False,
        packing=False,
        max_length=None,
        gradient_checkpointing=True,
        seed=args.seed
    )
    if accelerator.is_main_process:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args)
        )

    # Trainer
    if args.use_kl:
        print("use forward KL")
        data_collator = DataCollatorForKLTraining(
            tokenizer=tokenizer,
            mlm=False
        )
        trainer = ForwardKLMaskedSFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_arguments,
        processing_class=tokenizer,
        data_collator=data_collator,
        use_kl=True
        )
    else:
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            args=training_arguments,
            processing_class=tokenizer,
        )

    if accelerator.is_main_process:
        print("Sampled Text:\n")
        text = tokenizer.decode(trainer.train_dataset[0:10]["input_ids"][5])
        print(text)
        print("\n\n\n\n\n\n")

    trainer.train()
    if accelerator.is_main_process:
        final_output_dir = f"{args.output_dir}/{args.wandb_run_name}"
        trainer.save_model(final_output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--student_model', type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument('--output_dir', type=str, default="output_model")
    parser.add_argument("--wandb_project", type=str, default="safety-reasoning")
    parser.add_argument("--wandb_run_name", type=str)
    parser.add_argument("--effective_batch_size", type=int, default=8)
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--safety_dataset", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=16384)
    parser.add_argument("--prompt_masking", action="store_true", help="Enable prompt masking during training.")

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--use_kl", action="store_true")

    parser.add_argument("--use_lora", action="store_true", help="Enable training with LoRA.")
    parser.add_argument("--lora_r", type=int, default=32, help="The rank of the LoRA matrices.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="The dropout probability for LoRA layers.")

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    args.lora_alpha = args.lora_r // 2

    main(args)