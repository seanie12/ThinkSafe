#!/bin/bash
#SBATCH --job-name=llmservice
#SBATCH --account=llmservice_nlp_fm
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1                   # Number of tasks per node
#SBATCH --gres=gpu:8             # Number of GPUs per node
#SBATCH --time=4:00:00          # Maximum runtime
#SBATCH --partition=interactive      # Adjust to your cluster's partition
#SBATCH --exclusive
#SBATCH --dependency=singleton
#SBATCH --output=logs/%x_%j.out  # Standard output log
#SBATCH --error=logs/%x_%j.err   # Error log

# Environment setup (adjust paths as necessary)

echo "Running on node: $(hostname)"
export GPUS_PER_NODE=8
export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export WORLD_SIZE=$(($SLURM_NNODES * $GPUS_PER_NODE))

echo "WORLD_SIZE="$WORLD_SIZE
echo "MASTER_ADDR="$MASTER_ADDR
echo "MASTER_PORT="$MASTER_PORT

export HF_HOME=/lustre/fsw/portfolios/nvr/users/dachengl/.cache

srun --ntasks=3 --ntasks-per-node=1 --export=ALL,MASTER_PORT=$MASTER_PORT,MASTER_ADDR=$MASTER_ADDR,NODE_RANK=$SLURM_NODEID,WORLD_SIZE=$WORLD_SIZE,NNODES=$SLURM_JOB_NUM_NODES /bin/bash /lustre/fsw/portfolios/nvr/users/dachengl/runs/LLaMA-Factory/single_node.sh
