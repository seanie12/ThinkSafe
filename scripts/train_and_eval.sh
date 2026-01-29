#!/bin/bash

# Train and Evaluate Script
# Usage: ./scripts/train_and_eval.sh [devices] [model_type] [size] [dataset] [use_kl] [prompt_masking] [rank] [epochs]
# Example: ./scripts/train_and_eval.sh 0,1,2,3 qwen 0.6B thinksafe false true 32 1

set -e  # Exit on error

# Parse arguments
devices=${1:-0,1,2,3}
model_type=${2:-qwen}  # qwen or r1-distill
size=${3:-0.6B}
DS=${4:-thinksafe}
use_kl=${5:-false}
prompt_masking=${6:-true}
rank=${7:-32}
epochs=${8:-3}


# Parse devices into array
IFS=',' read -ra device_array <<< "$devices"
num_devices=${#device_array[@]}

# Validation
if [[ "$model_type" != "qwen" && "$model_type" != "r1-distill" ]]; then
    echo "Error: model_type must be 'qwen' or 'r1-distill'"
    exit 1
fi

if [ $num_devices -lt 1 ]; then
    echo "Error: At least 1 device must be specified"
    exit 1
fi

echo "=========================================="
echo "Starting Training and Evaluation Pipeline"
echo "=========================================="
echo "Devices: $devices (${num_devices} GPUs)"
echo "Model Type: $model_type"
echo "Size: $size"
echo "Dataset: $DS"
echo "Prompt Masking: $prompt_masking"
echo "Epochs: $epochs"
echo "Rank: $rank"
echo "Use KL: $use_kl"
echo "=========================================="

# Step 1: Run Training
echo ""
echo "Step 1: Training model on GPUs: $devices"
echo "Command: CUDA_VISIBLE_DEVICES=$devices scripts/${model_type}/train.sh $size $DS $prompt_masking $rank $epochs $use_kl"
CUDA_VISIBLE_DEVICES=$devices bash scripts/${model_type}/train.sh "$size" "$DS" "$prompt_masking" "$rank" "$epochs" "$use_kl"

if [ $? -ne 0 ]; then
    echo "Training failed. Exiting."
    exit 1
fi

echo ""
echo "Training completed successfully!"
echo ""

# Wait for all training processes to fully terminate and release GPU memory
echo "Waiting for GPU memory to be released..."
sleep 10
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
echo "GPU memory cleared."
echo ""

echo "=========================================="
echo "Starting Evaluation Pipeline"
echo "=========================================="
echo "Devices: $devices (${num_devices} GPUs)"
echo "Model Type: $model_type"
echo "Size: $size"
echo "Dataset: $DS"
echo "Prompt Masking: $prompt_masking"
echo "Use KL: $use_kl"
echo "=========================================="

# Run Parallel Evaluations
echo ""
echo "Running parallel evaluations..."
echo "=========================================="

# Define evaluation tasks
math_tasks=(aime24 gpqa_diamond math500 gsm8k)
safety_modes=(harmbench strongreject wildjailbreak xstest)

# Arrays to store PIDs and task info
math_pids=()
safety_pids=()
log_dir="logs/eval_${model_type}-${size}-${DS}-${rank}-${epochs}ep"
mkdir -p "$log_dir"

echo "Starting math evaluations with GPU queuing..."
echo "Log directory: $log_dir"
echo ""

# Track which task is running on which GPU (indexed by device)
declare -A gpu_pids
declare -A gpu_tasks

# Initialize GPU tracking
for device in "${device_array[@]}"; do
    gpu_pids[$device]=""
    gpu_tasks[$device]=""
done

# Launch math evaluations with proper GPU queuing
task_idx=0
while [ $task_idx -lt ${#math_tasks[@]} ]; do
    # Find an available GPU
    for device in "${device_array[@]}"; do
        # Check if this GPU is idle
        if [ -z "${gpu_pids[$device]}" ] || ! kill -0 ${gpu_pids[$device]} 2>/dev/null; then
            # GPU is available, assign next task
            if [ $task_idx -lt ${#math_tasks[@]} ]; then
                task=${math_tasks[$task_idx]}
                log_file="${log_dir}/${task}.log"

                echo "Launching $task on GPU $device (log: $log_file)"

                # Run inference.sh in background
                bash scripts/${model_type}/inference.sh "$device" "$size" "$task" "$DS" "$prompt_masking" "$rank" "$use_kl" "$epochs" > "$log_file" 2>&1 &
                pid=$!

                gpu_pids[$device]=$pid
                gpu_tasks[$device]=$task
                math_pids+=($pid)

                task_idx=$((task_idx + 1))
            fi
        fi
    done

    # Brief sleep to avoid busy waiting
    sleep 1
done

echo ""
echo "Math evaluations launched. Waiting for all to complete before starting safety evaluations..."
echo ""

# Wait for all math tasks to complete first
math_failed=0
for i in ${!math_pids[@]}; do
    pid=${math_pids[$i]}
    task=${math_tasks[$i]}

    echo "Waiting for $task (PID: $pid)..."
    if wait $pid; then
        echo "✓ $task completed successfully"
    else
        echo "✗ $task failed (exit code: $?)"
        math_failed=$((math_failed + 1))
    fi
done

echo ""
echo "All math tasks completed. Starting safety evaluations..."
echo ""

# Reset GPU tracking for safety evaluations
for device in "${device_array[@]}"; do
    gpu_pids[$device]=""
    gpu_tasks[$device]=""
done

# Launch safety evaluations with proper GPU queuing
task_idx=0
while [ $task_idx -lt ${#safety_modes[@]} ]; do
    # Find an available GPU
    for device in "${device_array[@]}"; do
        # Check if this GPU is idle
        if [ -z "${gpu_pids[$device]}" ] || ! kill -0 ${gpu_pids[$device]} 2>/dev/null; then
            # GPU is available, assign next task
            if [ $task_idx -lt ${#safety_modes[@]} ]; then
                mode=${safety_modes[$task_idx]}
                log_file="${log_dir}/safety_${mode}.log"

                echo "Launching safety evaluation ($mode) on GPU $device (log: $log_file)"

                # Run inference_safety.sh in background
                bash scripts/${model_type}/inference_safety.sh "$device" "$size" "$DS" "$mode" "$prompt_masking" "$rank" "$use_kl" "$epochs" > "$log_file" 2>&1 &
                pid=$!

                gpu_pids[$device]=$pid
                gpu_tasks[$device]=$mode
                safety_pids+=($pid)

                task_idx=$((task_idx + 1))
            fi
        fi
    done

    # Brief sleep to avoid busy waiting
    sleep 1
done

echo ""
echo "Safety evaluations launched. Waiting for completion..."
echo ""

# Wait for all safety tasks and track their status
safety_failed=0
for i in ${!safety_pids[@]}; do
    pid=${safety_pids[$i]}
    mode=${safety_modes[$i]}

    echo "Waiting for safety evaluation ($mode) (PID: $pid)..."
    if wait $pid; then
        echo "✓ Safety evaluation ($mode) completed successfully"
    else
        echo "✗ Safety evaluation ($mode) failed (exit code: $?)"
        safety_failed=$((safety_failed + 1))
    fi
done

failed=$((math_failed + safety_failed))
total_tasks=$((${#math_tasks[@]} + ${#safety_modes[@]}))

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo "Evaluations: $((total_tasks - failed))/${total_tasks} successful"
echo "  - Math: $((${#math_tasks[@]} - math_failed))/${#math_tasks[@]} successful"
echo "  - Safety: $((${#safety_modes[@]} - safety_failed))/${#safety_modes[@]} successful"

if [ $failed -gt 0 ]; then
    echo ""
    echo "Warning: $failed evaluation(s) failed. Check logs in $log_dir"
    exit 1
fi

echo ""
echo "All evaluations completed successfully!"
echo "Results saved in: results/${model_type}-${size}-${DS}-${rank}-${epochs}ep/"
echo "Logs saved in: $log_dir/"
echo "=========================================="
