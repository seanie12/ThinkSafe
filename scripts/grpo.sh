#!/bin/bash

##############################
#   Parse Input Parameters
##############################
if [ $# -lt 3 ]; then
    echo "Usage: $0 <BASE_MODEL> <DATASET_NAME> <GPU_IDS>"
    echo "Example: $0 deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B thinksafe-r1-1.5B-ablation 0,1,2,3"
    echo "GPU_IDS format: First GPU for VLLM, second for Guard, rest for GRPO training"
    exit 1
fi

BASE_MODEL="$1"
DATASET_NAME="$2"
GPU_IDS="$3"

# Parse GPU IDs
IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
if [ ${#GPU_ARRAY[@]} -lt 3 ]; then
    echo "Error: At least 3 GPU IDs required (VLLM, Guard, and at least 1 for GRPO)"
    exit 1
fi

VLLM_GPU_ID="${GPU_ARRAY[0]}"
GUARD_GPU_ID="${GPU_ARRAY[1]}"
GRPO_GPU_ID=$(IFS=','; echo "${GPU_ARRAY[*]:2}")

echo "BASE_MODEL: ${BASE_MODEL}"
echo "DATASET_NAME: ${DATASET_NAME}"
echo "VLLM_GPU_ID: ${VLLM_GPU_ID}"
echo "GUARD_GPU_ID: ${GUARD_GPU_ID}"
echo "GRPO_GPU_ID: ${GRPO_GPU_ID}"
echo ""

##############################
#    Random Port Selection
##############################
# Function to check if a port is available by actually trying to bind to it
is_port_available() {
    local port=$1
    # Try to bind to the port using Python (most reliable method)
    python3 -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(('127.0.0.1', $port)); s.close()" 2>/dev/null
    return $?
}

# Generate three unique available random ports between 1000 and 8000
# Third port is for vLLM distributed communication (group_port)
PORTS=()
MAX_ATTEMPTS=1000
attempts=0
while [ ${#PORTS[@]} -lt 3 ] && [ $attempts -lt $MAX_ATTEMPTS ]; do
    PORT=$((1000 + RANDOM % 7001))
    attempts=$((attempts + 1))
    # Check if port is not already in array and is available
    if [[ ! " ${PORTS[@]} " =~ " ${PORT} " ]] && is_port_available $PORT; then
        PORTS+=($PORT)
        echo "Found available port: ${PORT}"
        attempts=0  # Reset attempts counter after finding a port
    fi
done

# Check if we found enough ports
if [ ${#PORTS[@]} -lt 3 ]; then
    echo "Error: Could not find 3 available ports after $MAX_ATTEMPTS attempts"
    exit 1
fi

VLLM_PORT=${PORTS[0]}
GUARD_PORT=${PORTS[1]}
VLLM_GROUP_PORT=${PORTS[2]}

echo "Selected ports: VLLM=${VLLM_PORT}, Guard=${GUARD_PORT}, VLLM_Group=${VLLM_GROUP_PORT}"
echo ""

##############################
#       1. VLLM Server
##############################
GPU_MEMORY_UTILIZATION=0.95
MAX_MODEL_LEN=16384

echo "Starting Base model server on ${VLLM_GPU_ID} GPUs with ${BASE_MODEL}..."
echo "Base model server port: ${VLLM_PORT}"
CUDA_VISIBLE_DEVICES=${VLLM_GPU_ID} trl vllm-serve \
    --model ${BASE_MODEL} \
    --gpu_memory_utilization ${GPU_MEMORY_UTILIZATION} \
    --host 127.0.0.1 \
    --port ${VLLM_PORT} \
    --enable_prefix_caching True \
    --max_model_len ${MAX_MODEL_LEN} & # Background execution
VLLM_PID=$!

##############################
#    2. Guard Model Server
##############################
GUARD_MODEL="meta-llama/Llama-Guard-3-8B"

echo "Starting Guard model server on ${GUARD_GPU_ID} GPUs with ${GUARD_MODEL}..."
echo "Guard model server port: ${GUARD_PORT}"
CUDA_VISIBLE_DEVICES=${GUARD_GPU_ID} python -m vllm.entrypoints.openai.api_server \
    --model ${GUARD_MODEL} \
    --port ${GUARD_PORT} \
    --gpu-memory-utilization 0.95 \
    --enable_prefix_caching \
    --dtype bfloat16 &  # Background execution
GUARD_PID=$!

echo "Waiting for vLLM and Guard servers to initialize..."
sleep 30

##############################
#     3. GRPO Training
##############################
NUM_GPUS=$(echo $GRPO_GPU_ID | tr ',' '\n' | wc -l) # Automatically calculate GPU count

LORA_R=32
LORA_DROPOUT=0.05
EFFECTIVE_BATCH_SIZE=64
NUM_GENERATIONS=8
WANDB_RUN_NAME="${DATASET_NAME}_R${LORA_R}_BZ${EFFECTIVE_BATCH_SIZE}_Gen${NUM_GENERATIONS}"

echo "Starting training on ${NUM_GPUS} GPUs with ${DATASET_NAME}..."
echo "WandB run name: ${WANDB_RUN_NAME}"
echo "Output directory: ${OUTPUT_DIR}"
echo "VLLM server port: ${VLLM_PORT}"
echo "Guard server port: ${GUARD_PORT}"
echo "VLLM group port: ${VLLM_GROUP_PORT}"
PYTHONPATH=./ CUDA_VISIBLE_DEVICES=${GRPO_GPU_ID} accelerate launch \
  --config_file ./default_config.yaml \
  --num_processes "${NUM_GPUS}" \
  grpo_train.py \
  --model_name "${BASE_MODEL}" \
  --dataset "${DATASET_NAME}" \
  --wandb_run_name "${WANDB_RUN_NAME}" \
  --vllm_server_port "${VLLM_PORT}" \
  --guard_server_port "${GUARD_PORT}" \
  --vllm_group_port "${VLLM_GROUP_PORT}" \
  --use_lora \
  --lora_r ${LORA_R} \
  --lora_dropout ${LORA_DROPOUT} \
  --effective_batch_size ${EFFECTIVE_BATCH_SIZE} \
  --num_generations ${NUM_GENERATIONS} \
  --max_completion_length ${MAX_MODEL_LEN}

# After training, kill the background servers
kill $VLLM_PID $GUARD_PID
