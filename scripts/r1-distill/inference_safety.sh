# Set default values for parameters
device=${1:-0}
size=${2:-1.5B}
DS=${3:-thinksafe}
mode=${4:-harmbench}
prompt_masking=${5:-true}
rank=${6:-32}
use_kl=${7:-false}
epochs=${8:-1}

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

# Add epoch suffix only if epochs != 1
if [[ "$epochs" != "1" ]]; then
    wandb_name="${wandb_name}-${epochs}ep"
fi

# Construct model path based on wandb_name
model_path="output_model/${wandb_name}"
output_dir="safety_results/${wandb_name}"

# Step 1: Merge config (only if tmp directory doesn't exist)
if [ ! -d "${model_path}/tmp" ]; then
    echo "Directory ${model_path}/tmp does not exist. Running merge_config.py..."
    PYTHONPATH=./ python utils/merge_config.py --checkpoint_path "$model_path"
else
    echo "Directory ${model_path}/tmp already exists. Skipping merge_config.py..."
fi

# Step 2: Generate responses
CUDA_VISIBLE_DEVICES=$device python gen_response.py \
    --mode $mode \
    --model_path "${model_path}/tmp" \
    --output_dir "$output_dir" \
    --eval_path dataset/eval/harmbench.json \
    --top_k 0 \
    --top_p 0.95 \
    --temperature 0.6

# Step 3: Evaluate safety
echo "Evaluating safety..."

# Retry logic for HuggingFace rate limiting in eval_safety.py
max_retries=10
retry_count=0
base_delay=10

while [ $retry_count -lt $max_retries ]; do
    echo "Attempt $((retry_count + 1))/$max_retries: Starting safety evaluation..."

    if CUDA_VISIBLE_DEVICES=$device python eval_safety.py \
        --input_file "${output_dir}/${mode}.json" \
        --output_dir "$output_dir" \
        --mode $mode; then
        echo "✓ Safety evaluation completed successfully on attempt $((retry_count + 1))"
        exit 0
    else
        exit_code=$?
        echo "✗ Safety evaluation failed on attempt $((retry_count + 1)) with exit code: $exit_code"

        # Check if it's likely a rate limiting issue (exit code might vary)
        if [ $retry_count -lt $((max_retries - 1)) ]; then
            delay=$((base_delay * (2 ** retry_count) + RANDOM % 10))
            echo "Waiting $delay seconds before retry (exponential backoff + jitter)..."
            sleep $delay
        fi

        retry_count=$((retry_count + 1))
    fi
done

echo "✗ Safety evaluation failed after $max_retries attempts"
exit 1