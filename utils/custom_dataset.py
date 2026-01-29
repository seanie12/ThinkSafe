from datasets import load_dataset, load_from_disk
import os
import re
import warnings

FORWARD_KL_PROMPT_LABELS = ["adversarial_benign", "vanilla_benign"]

def extract_prompt_response(text):
    """Extract both question and answer from text using regex"""
    # Pattern to capture question and answer
    pattern = r"Question:\s*(.+?)\s*Answer:\s*(.+)"
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    think = "<think>\nI should not answer this question!\n</think>\n\n"
    if match:
        return {
            "prompt": match.group(1).strip(),
            "response": think + match.group(2).strip()
        }
    return None

def get_tokenizer_family(tokenizer):
    """
    Extract the tokenizer family from a tokenizer's name_or_path.
    This function normalizes model names to their base family while preserving version differences.
    
    Args:
        tokenizer: The tokenizer object with name_or_path attribute
        
    Returns:
        str: The tokenizer family identifier
    """
    model_name_or_path = tokenizer.name_or_path
    
    # Convert to lowercase for consistent matching
    model_name = model_name_or_path.lower()
    
    # Define patterns for common model families
    # These patterns preserve version numbers where important
    family_patterns = {
        # Qwen family - preserve version numbers
        'qwen3': r'qwen-?3|qwen3',
        'qwen2.5': r'qwen2\.5|qwen-?2\.5',
        'qwen2': r'qwen2(?!\.5)|qwen-?2(?!\.5)', 
        'qwen': r'qwen(?![23])',  # Matches Qwen but not Qwen2 or Qwen3
        
        # MobileLLM family - R1 series share same tokenizer
        # R2, R3 etc would be different if they exist
        'mobilellm_r1': r'mobilellm-r1|mobilellm-1b',
        'mobilellm': r'mobilellm(?!-r)',  # Generic MobileLLM without R series
        
        # Llama family - preserve major versions
        'llama3.2': r'llama-?3\.2',
        'llama3.1': r'llama-?3\.1',
        'llama3': r'llama-?3(?!\.)',
        'llama2': r'llama-?2',
        'llama': r'llama(?![23])',
        
        # Other common families
        'mistral': r'mistral',
        'mixtral': r'mixtral',
        'gpt4': r'gpt-?4',
        'gpt3': r'gpt-?3',
        'gpt2': r'gpt-?2',
        'gpt': r'gpt(?![234])',
        'bert': r'bert',
        'roberta': r'roberta',
        'phi3': r'phi-?3',
        'phi2': r'phi-?2',
        'phi': r'phi(?![23])',
        'gemma2': r'gemma-?2',
        'gemma': r'gemma(?!2)',
        'vicuna': r'vicuna',
        'alpaca': r'alpaca',
        'falcon': r'falcon',
        'pythia': r'pythia',
        'opt': r'opt-',
        'galactica': r'galactica',
        'bloom': r'bloom',
        't5': r't5',
        'flan': r'flan',
        'deepseek': r'deepseek',
        'yi': r'\byi\b',
        'baichuan2': r'baichuan-?2',
        'baichuan': r'baichuan(?!2)',
        'internlm2': r'internlm2',
        'internlm': r'internlm(?!2)',
    }
    
    # Check each pattern to find the family (order matters - check versioned patterns first)
    for family_name, pattern in family_patterns.items():
        if re.search(pattern, model_name):
            return family_name
    
    # If no pattern matches, use a fallback strategy
    if '/' in model_name_or_path:
        org, model = model_name_or_path.split('/', 1)
        # Remove size indicators and version suffixes
        base_model = re.split(r'[-_](\d+[bBmMkK]|\d+m|\d+b)', model.lower())[0]
        # Remove R1, R2, etc. suffixes
        base_model = re.split(r'[-_]r\d+', base_model)[0]
        # Remove -chat, -instruct, etc. suffixes
        base_model = re.split(r'[-_](chat|instruct|base)', base_model)[0]
        
        return base_model.lower().replace('-', '_')
    
    # Fallback: use the model name up to the first size indicator
    base_name = re.split(r'[-_](\d+[bBmMkK]|\d+m|\d+b)', model_name_or_path.lower())[0]
    return base_name.replace('/', '_').replace('-', '_')


def format_chat_template_batch(batch, tokenizer, max_length, prompt_masking=True):
    if "refusal" in batch:
        # Direct Refusal: no thinking
        formatted_chats = []
        prompt_texts = []

        for prompt_response in batch["refusal"]:

            item = extract_prompt_response(prompt_response)
            prompt = item["prompt"]
            response = item["response"]

            # Tokenize prompt only to get its length
            prompt_messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            prompt_texts.append(formatted_prompt)

            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
            formatted_chat = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            formatted_chats.append(formatted_chat)
        
        # Tokenize prompts to get their lengths
        prompt_tokenized = tokenizer(
            prompt_texts,
            truncation=True,
            padding=False,
            max_length=max_length,
            add_special_tokens=False
        )

        # Tokenize full conversations
        tokenized_output = tokenizer(
            formatted_chats,
            truncation=True,
            padding=False,
            max_length=max_length,
            add_special_tokens=False  # Special tokens already added by apply_chat_template
        )

        # Create labels with prompt tokens masked as -100 (if prompt_masking is True)
        labels = []
        for input_ids, prompt_ids in zip(tokenized_output["input_ids"], prompt_tokenized["input_ids"]):
            if prompt_masking:
                # Sanity check: verify prompt tokens match the beginning of full conversation
                if input_ids[:len(prompt_ids)] != prompt_ids:
                    warnings.warn(
                        "Mismatch between tokenized prompt and the start of tokenized prompt+completion. "
                        "This may be due to unexpected tokenizer behavior, whitespace issues, or special "
                        "token handling. Verify that the tokenizer is processing text consistently."
                    )
                label = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]
            else:
                # Use full input_ids as labels (no masking)
                label = input_ids
            labels.append(label)
        
        return {"input_ids": tokenized_output["input_ids"], "labels": labels}

    
    elif "response" in batch:
        # safechain / thinksafe / star-1 dataset 
        responses = batch["response"]
        
        if "instruction" in batch:
            instructions = batch["instruction"]
        elif "question" in batch:
            instructions = batch["question"]
        else:
            raise ValueError("Input batch does not contain 'instruction'/'question'/'response' keys.")
        prompt_labels = batch.get("prompt_label")

        formatted_chats = []
        prompt_texts = []
        forward_kl_masks = []

        for idx, (instruction, response) in enumerate(zip(instructions, responses)):
            # Skip examples where response has <think> but no </think>
            if "<think>" in response and "</think>" not in response:
                print("skip example with unclosed <think> tag")
                continue

            if "<think>" not in response:
                response = "<think>\n" + response

            # Normalize: if there's only one newline after </think>, add another one
            response = re.sub(r'</think>\n(?!\n)', '</think>\n\n', response)

            # Tokenize prompt only to get its length
            prompt_messages = [{"role": "user", "content": instruction}]
            formatted_prompt = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            prompt_texts.append(formatted_prompt)

            messages = [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response}
            ]
            formatted_chat = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            formatted_chats.append(formatted_chat)
            prompt_label = prompt_labels[idx] if prompt_labels is not None else None
            forward_kl_masks.append(1 if prompt_label in FORWARD_KL_PROMPT_LABELS else 0)

        # Tokenize prompts to get their lengths
        prompt_tokenized = tokenizer(
            prompt_texts,
            truncation=True,
            padding=False,
            max_length=max_length,
            add_special_tokens=False
        )

        # Tokenize full conversations
        tokenized_output = tokenizer(
            formatted_chats,
            truncation=True,
            padding=False,
            max_length=max_length,
            add_special_tokens=False  # Special tokens already added by apply_chat_template
        )

        # Create labels with prompt tokens masked as -100 (if prompt_masking is True)
        labels = []
        for input_ids, prompt_ids in zip(tokenized_output["input_ids"], prompt_tokenized["input_ids"]):
            if prompt_masking:
                # Sanity check: verify prompt tokens match the beginning of full conversation
                if input_ids[:len(prompt_ids)] != prompt_ids:
                    warnings.warn(
                        "Mismatch between tokenized prompt and the start of tokenized prompt+completion. "
                        "This may be due to unexpected tokenizer behavior, whitespace issues, or special "
                        "token handling. Verify that the tokenizer is processing text consistently."
                    )
                label = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]
            else:
                # Use full input_ids as labels (no masking)
                label = input_ids
            labels.append(label)

        return {"input_ids": tokenized_output["input_ids"], "labels": labels, "forward_kl_mask": forward_kl_masks}



    else:
        raise ValueError("Input batch does not contain 'instruction'/'response' or 'refusal' keys.")

def get_dataset(dataset_name, tokenizer, max_length, domain=None,
                cache_dir="./processed_data", length_filter=False, num_samples=-1, prompt_masking=True, use_cache=False):
    """
    Loads a dataset. If a pre-processed version exists on disk for the tokenizer family, it loads that.
    Otherwise, it processes the dataset from the Hugging Face Hub, saves it, and returns it.

    The tokenizer family is automatically determined from tokenizer.name_or_path.

    Args:
        dataset_name (str): The name of the dataset on the Hugging Face Hub.
        tokenizer (PreTrainedTokenizer): The tokenizer to use for processing.
        max_length (int): Maximum sequence length. Sequences longer than this are filtered out.
        domain (str): Optional domain filter.
        cache_dir (str): The directory to save/load the processed dataset.
        length_filter (bool): Whether to filter sequences by length.
        num_samples (int): Number of samples to randomly select from filtered data. -1 means use all.
        prompt_masking (bool): Whether to mask prompt tokens with -100 in labels. Default is True.
        use_cache (bool): Whether to load from/save to disk cache. Default is False.

    Returns:
        Dataset: The processed and tokenized dataset.
    """
    # Get the tokenizer family from the tokenizer itself
    tokenizer_family = get_tokenizer_family(tokenizer)

    # Create a specific path for the processed dataset based on tokenizer family
    dataset_safe_name = dataset_name.replace('/', '_').replace('.', '_')

    # Include prompt_masking in cache path to differentiate between masked and unmasked datasets
    masking_suffix = "masked" if prompt_masking else "unmasked"

    if domain is not None:
        processed_path = os.path.join(
            cache_dir,
            f"{dataset_safe_name}_{tokenizer_family}_{max_length}_{domain}_{masking_suffix}"
        )
    else:
        processed_path = os.path.join(
            cache_dir,
            f"{dataset_safe_name}_{tokenizer_family}_{max_length}_{masking_suffix}"
        )

    if use_cache and os.path.exists(processed_path):
        print(f"✅ Loading pre-processed dataset from disk: {processed_path}")
        print(f"   Tokenizer family: {tokenizer_family} (from {tokenizer.name_or_path})")
        return load_from_disk(processed_path)
    else:
        if use_cache:
            print(f"⚙️ Processed dataset not found for tokenizer family '{tokenizer_family}'.")
            print(f"   Model: {tokenizer.name_or_path}")
            print(f"   Tokenizing '{dataset_name}' and saving to {processed_path}...")
        else:
            print(f"⚙️ Cache disabled. Processing dataset '{dataset_name}' without caching...")
            print(f"   Model: {tokenizer.name_or_path}")
            print(f"   Tokenizer family: {tokenizer_family}")
        
        # Load the raw dataset from the Hub
        if "json" in dataset_name:
            raw_dataset = load_dataset("json", data_files=[dataset_name], split="train")
        else:
            raw_dataset = load_dataset(dataset_name, split="train")
        
        # Filter by domain if specified
        if domain is not None:
            raw_dataset = raw_dataset.filter(lambda x: x["domain"] == domain)
            print(f"Domain: {domain}, {len(raw_dataset)} samples")
        
        # Get the original column names to remove them after mapping
        original_columns = raw_dataset.column_names
        
        # Run the efficient, batched tokenization
        print(f"🔄 Tokenizing dataset with {tokenizer_family} tokenizer... with prompt-masking: {prompt_masking}" )
        tokenized_dataset = raw_dataset.map(
            format_chat_template_batch,
            fn_kwargs={"tokenizer": tokenizer, "max_length": max_length, "prompt_masking": prompt_masking},
            batched=True,
            batch_size=1000,
            num_proc=8,  # Use half of the available CPU cores
            remove_columns=original_columns,
            load_from_cache_file=False  # Force re-processing to avoid stale cache
        )
        
        if length_filter:
            # Filter out sequences that are too long
            print(f"🔍 Filtering sequences by length (<= {max_length})...")
            filtered_dataset = tokenized_dataset.filter(
                lambda x: len(x["input_ids"]) <= max_length,
                num_proc=8
            )
            print(f"📊 Filtered from {len(tokenized_dataset)} to {len(filtered_dataset)} samples")
        else:
            filtered_dataset = tokenized_dataset
            
        # Apply random sampling after filtering if needed
        if num_samples > 0 and len(filtered_dataset) > num_samples:
            print(f"🎲 Randomly sampling {num_samples} from {len(filtered_dataset)} filtered samples...")
            filtered_dataset = filtered_dataset.shuffle(seed=42).select(range(num_samples))

        if use_cache:
            print(f"💾 Saving processed dataset to disk for tokenizer family: {tokenizer_family}")
            filtered_dataset.save_to_disk(processed_path)
        else:
            print(f"✅ Dataset processed (not saved to disk)")

        return filtered_dataset

# Example usage:
if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    # Examples showing how different versions get different caches
    print("=" * 60)
    print("Testing tokenizer family detection:")
    print("=" * 60)
    
    # Test models focusing on Qwen3 and Qwen2 series
    test_models = [
        # Qwen3 series - all share same tokenizer
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-1.7B",
        "Qwen/Qwen3-4B",    
        "Qwen/Qwen3-8B",
        
        # For comparison - Qwen2 (different from Qwen3)
        "Qwen/Qwen2-0.5B",
        "Qwen/Qwen2-1.5B",
    ]
    
    for model in test_models:
        tokenizer = AutoTokenizer.from_pretrained(model)
        family = get_tokenizer_family(tokenizer)
        print(f"{model:35} -> {family}")
    
    print("\n" + "=" * 60)
    print("Cache paths that would be created:")
    print("=" * 60)
    
    # Show example cache paths
    for model in test_models:
        tokenizer = AutoTokenizer.from_pretrained(model)
        family = get_tokenizer_family(tokenizer)
        cache_path = f"./processed_data/my_dataset_{family}_2048"
        print(f"{model:35} -> {cache_path}")
    
    print("\n" + "=" * 60)
    print("Models sharing the same cache:")
    print("=" * 60)
    
    # Group by family
    families = {}
    for model in test_models:
        tokenizer = AutoTokenizer.from_pretrained(model)
        family = get_tokenizer_family(tokenizer)
        if family not in families:
            families[family] = []
        families[family].append(model)
    
    for family, models in families.items():
        print(f"\n{family}:")
        for model in models:
            print(f"  - {model}")