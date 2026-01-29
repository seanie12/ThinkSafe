cd /lustre/fsw/portfolios/nvr/users/dachengl/runs/LLaMA-Factory
source /lustre/fsw/portfolios/nvr/users/dachengl/anaconda3/bin/activate
conda activate llamafactory

export HF_HOME=/lustre/fsw/portfolios/nvr/users/dachengl/.cache

echo "NNODES=$NNODES NODE_RANK=$NODE_RANK MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"

torchrun --nnodes $NNODES  --master_addr $MASTER_ADDR --master_port $MASTER_PORT --node_rank $NODE_RANK --nproc_per_node 8  /lustre/fsw/portfolios/nvr/users/dachengl/runs/LLaMA-Factory/src/train.py   --deepspeed /lustre/fsw/portfolios/nvr/users/dachengl/runs/LLaMA-Factory/examples/deepspeed/ds_z3_config.json  --stage sft  --template qwen25  --model_name_or_path saves/qwen-qwq/full/sft --do_train  --dataset qwq --finetuning_type full  --output_dir saves/qwen-qwq/full/sft_warm_competition_interview --overwrite_cache  --per_device_train_batch_size 1  --gradient_accumulation_steps 1  --lr_scheduler_type cosine --logging_steps 1  --save_steps 200  --warmup_ratio 0.1  --learning_rate 1e-5  --num_train_epochs 3.0  --plot_loss  --bf16 --flash_attn  --preprocessing_num_workers 16 --cutoff_len 8192   --report_to wandb --run_name qwen_qwq_warm_competition_interview
