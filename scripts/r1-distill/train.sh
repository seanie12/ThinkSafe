#!/bin/bash

# Get GPU count directly from PyTorch
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")

# Set other parameters
size=${1:-1.5B}
DS=${2:-thinksafe-r1-1.5B}
prompt_masking=${3:-false}
rank=${4:-32}
epochs=${5:-1}
use_kl=${6:-false}

# Set model family based on size
if [ "$size" == "8B" ]; then
    model_family="Llama"
else
    model_family="Qwen"
fi

# Construct wandb_run_name
if [[ "$prompt_masking" == "true" ]]; then
    wandb_name="R1-${size}-${DS}-${rank}-pm"
else
    wandb_name="R1-${size}-${DS}-${rank}"
fi

# Add kl suffix if use_kl is true
if [[ "$use_kl" == "true" ]]; then
    wandb_name="${wandb_name}-kl"
fi

# Add epoch suffix
wandb_name="${wandb_name}-${epochs}ep"

echo "--- Starting training on ${NUM_GPUS} GPUs with ${DS} ---"
echo "--- WandB run name: ${wandb_name} ---"
echo "--- Prompt Masking: ${prompt_masking} ---"
echo "--- Use KL: ${use_kl} ---"
echo "--- Rank: ${rank} ---"

# Build the command with conditional prompt_masking flag
cmd="PYTHONPATH=./ accelerate launch \
--config_file ./default_config.yaml \
--num_processes $NUM_GPUS \
train.py \
--student_model \"deepseek-ai/DeepSeek-R1-Distill-${model_family}-${size}\" \
--wandb_run_name \"${wandb_name}\" \
--per_device_batch_size 1 \
--use_lora \
--safety_dataset $DS \
--epochs $epochs"

# Add prompt_masking flag if set to true
if [[ "$prompt_masking" == "true" ]]; then
    cmd="$cmd --prompt_masking"
fi

# Add use_kl flag if set to true
if [[ "$use_kl" == "true" ]]; then
    cmd="$cmd --use_kl"
fi

# Execute the command
eval $cmd