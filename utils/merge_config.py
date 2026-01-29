from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse
import gc
import os
import torch
import json

def merge_lora_and_save_temp(checkpoint_path):
    """Check if LoRA adapter exists, merge with base model, and save to temporary directory."""
    lora_adapter_path = os.path.join(checkpoint_path, "adapter_config.json")
    
    if not os.path.exists(lora_adapter_path):
        print("No LoRA adapter found, using checkpoint directly")
        return
    
    print("LoRA adapter found, merging with base model...")
    
    with open(os.path.join(checkpoint_path, "adapter_config.json"), 'r') as f:
        adapter_config = json.load(f)
    
    base_model_name = adapter_config.get("base_model_name_or_path")
    if not base_model_name:
        raise ValueError("base_model_name_or_path not found in adapter_config.json")
    
    # Load and merge
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16, 
        device_map="cpu", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    merged_model = model.merge_and_unload()
    
    # Save to temp directory
    temp_dir = os.path.join(checkpoint_path, "tmp")
    merged_model.save_pretrained(temp_dir, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    tokenizer.save_pretrained(temp_dir)
    print("Model merging completed")

    # Clean up models and GPU memory
    del base_model, model, merged_model
    gc.collect()
    torch.cuda.empty_cache()
    print("GPU memory cleared")


def main():
    parser = argparse.ArgumentParser(description='Merge LoRA adapter with base model')
    parser.add_argument('--checkpoint_path', help='Path to checkpoint directory', type=str, required=True)
    args = parser.parse_args()
    
    print(f"Processing: {args.checkpoint_path}")
    merge_lora_and_save_temp(args.checkpoint_path)

if __name__ == "__main__":
    main()