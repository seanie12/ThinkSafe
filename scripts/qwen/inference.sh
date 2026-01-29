#!/bin/bash

export TOKENIZERS_PARALLELISM=false

device=${1:-0}
size=${2:-0.6B}
task=${3:-aime24}
DS=${4:-thinksafe}
prompt_masking=${5:-true}
rank=${6:-32}
use_kl=${7:-false}
epochs=${8:-1}


export CUDA_VISIBLE_DEVICES=$device
gpu_count=$(python -c "import torch; print(torch.cuda.device_count())")
echo "Detected $gpu_count GPUs"
# Construct wandb_run_name
if [[ "$prompt_masking" == "true" ]]; then
    wandb_name="qwen3-${size}-${DS}-${rank}-pm"
else
    wandb_name="qwen3-${size}-${DS}-${rank}"
fi

# Add kl suffix if use_kl is true
if [[ "$use_kl" == "true" ]]; then
    wandb_name="${wandb_name}-kl"
fi

# Add epoch suffix only if epochs != 1
if [[ "$epochs" != "1" ]]; then
    wandb_name="${wandb_name}-${epochs}ep"
fi

# Construct model path based on wandb_name
model_path="output_model/${wandb_name}"
result_dir="results/${wandb_name}"

echo "Processing $task with prompt_masking $prompt_masking..."
    
# Step 1: Merge config (only if tmp directory doesn't exist)
if [ ! -d "${model_path}/tmp" ]; then
    echo "Directory ${model_path}/tmp does not exist. Running merge_config.py..."
    PYTHONPATH=./ python utils/merge_config.py --checkpoint_path "$model_path"
else
    echo "Directory ${model_path}/tmp already exists. Skipping merge_config.py..."
fi

# Step 2: Run evaluation (only if tmp directory exists)
if [ -d "${model_path}/tmp" ]; then
    echo "Running evaluation..."
    skythought evaluate \
        --model "${model_path}/tmp" \
        --task "$task" \
        --backend vllm \
        --backend-args tensor_parallel_size=1,gpu_memory_utilization=0.95 \
        --sampling-params temperature=0.6,top_p=0.95,top_k=20,max_tokens=32768 \
        --n 8 \
        --result-dir "$result_dir"
else
    echo "Error: ${model_path}/tmp does not exist after merge_config. Cannot run evaluation."
    exit 1
fi
