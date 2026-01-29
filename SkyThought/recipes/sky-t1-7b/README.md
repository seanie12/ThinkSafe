# Sky-T1-7B
For a detailed training recipes and technical details, refer to the [blog](https://novasky-ai.github.io/posts/sky-t1-7b/)

## SFT: Step 1 and Step 3 SFT
### Distillation Data Mixture
Make sure you have installed the `skythought` package as outlined in the [README.md](/README.md#usage). All the data curation commands are provided from the root directory of the repo.

For the distillation performed in step 1 and step 3, we use the following script for data generation. Replace the `$MODEL_NAME` with the model to be distilled from.
For each subset (`numina_math`, `numina_olympaids`, etc), the `score` command requires the output directory from the previous `generate` command. 

```shell
skythought generate \
    --task numina_math \
    --model $MODEL_NAME \
    --backend vllm \
    --backend-args tensor_parallel_size=4 \
    --sampling-params max_tokens=16384 \
    --result-dir ./data

skythought score \
    --task numina_math \
    --run-dir <path to output folder from generate>
```

```shell
skythought generate  \
    --task numina_olympiads \
    --model $MODEL_NAME \
    --backend vllm \
    --backend-args tensor_parallel_size=4 \
    --sampling-params max_tokens=16384 \
    --end 40000 \
    --result-dir ./data 

skythought score \
    --task numina_olympiads \
    --run-dir <path to output folder from generate>
```

```shell
skythought generate  \
    --task numina_amc_aime \
    --model $MODEL_NAME \
    --backend vllm \
    --backend-args tensor_parallel_size=4 \
    --sampling-params max_tokens=16384 \
    --result-dir ./data 

skythought score \
    --task numina_math \
    --run-dir <path to output folder from generate>
```

For step 1 and step 3 SFT, follow the instructions in `skythought/train`.

## RL: Step 2 and Step 4 RL
For RL training, install our modified fork of [VeRL](https://github.com/volcengine/verl) under `skythought/skythought-rl` and follow the instructions there, we also incorporate the math and coding testing utils from the [PRIME](https://github.com/PRIME-RL/PRIME) repo.

## Evaluation
For evaluation, we use the [script](https://github.com/QwenLM/Qwen2.5-Math/blob/main/evaluation/sh/eval.sh) from the [Qwen math eval suite](https://github.com/QwenLM/Qwen2.5-Math/tree/main/evaluation). We use vLLM version `0.6.2` for all evaluations. For AMC and AIME, we use temp=0.6, top_p=0.95 and n_sample=8. After the generation, we calculate the pass@1 using this [script](https://github.com/NovaSky-AI/SkyThought/tree/main/scripts/qwen_eval_bon.py). For MATH500 and OlympiadBench, we use greedy decoding. We use the [skythought system prompt](https://github.com/NovaSky-AI/SkyThought/blob/main/skythought/evals/models/model_configs.yaml) when evaluating all the models trained by us except for Sky-T1-mini which is evaluated [without a system prompt](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B#usage-recommendations).