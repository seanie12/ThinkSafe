# ThinkSafe: Self-Generated Safety Alignment for Reasoning Models

## 🤗 Datasets & Checkpoints
All **ThinkSafe** datasets and trained model checkpoints can be found at our [Hugging Face repository](https://huggingface.co/Seanie-lee/collections).

## 🚀 Setup
### 1. Logging
Before starting the training process, ensure you are logged into wandb to monitor your experiments.
```bash
wandb login
```
### 2. Environment and Dependencies
It is recommended to use a python environment. Install [skythought](https://github.com/NovaSky-AI/SkyThought) then update vllm with the latest version as below. (Other dependencies: peft, datasets, wandb, flash-attention, trl)
```bash
# conda create -n thinksafe python=3.10 -y && \
conda activate thinksafe && \
cd SkyThought && \
pip install uv && \
uv pip install -e . && \
uv pip install vllm==0.10.0 --torch-backend=auto && \
pip install trl==0.21.0 && \
pip install ray==2.48.0 && \
pip install peft && \
pip install wandb && \
pip install flash-attn --no-cache-dir --no-build-isolation && \
pip install flashinfer-python==0.5.3
```

### 3. HuggingFace Dataset
GPQA dataset and Llama-Guard-3 require HF login.
```bash
huggingface-cli login
```

## 🛠 Usage Instructions
### 1. Integrated Training and Evaluation (train_and_eval.sh)
```/scripts/train_and_eval.sh``` script handles both the fine-tuning and safety evaluation of your models in a single run.

```bash
./scripts/train_and_eval.sh [devices] [model_type] [size] [dataset] [use_kl] [prompt_masking] [rank] [epochs]
```

#### Parameter Descriptions:
- **devices**: List of GPU IDs to use (e.g., ```0,1,2,3```).
- **model_type**: The architecture family of the model (```r1-distill``` or ```qwen```).
- **size**: The parameter size of the model (e.g., ```8B```, ```1.5B```).
- **dataset**: The name or path of the training dataset.
- **use_kl**: A boolean flag (true/false) to enable or disable the use of KL Divergence loss for benign prompts.
- **prompt_masking**: A boolean flag to determine whether to mask prompt tokens during loss calculation.
- **rank**: The rank used for LoRA.

### 2. Training with GRPO
We provide training scripts for Group Relative Policy Optimization (GRPO) to compare with our **ThinkSafe**.

```bash
./scripts/grpo.sh <BASE_MODEL> <DATASET_NAME> <GPU_IDS>
```


### 3. Citation

```bibtex
@article{
thinksafe,
title={{THINKSAFE}: Self-Generated Safety Alignment for Reasoning Models},
author={Seanie Lee and Sangwoo Park and Yumin Choi and Gyeongman Kim and Minki Kang and Jihun Yun and Dongmin Park and Jongho Park and Sung Ju Hwang},
journal={arXiv preprint arXiv:2601.23143},
year={2026}
}
```
