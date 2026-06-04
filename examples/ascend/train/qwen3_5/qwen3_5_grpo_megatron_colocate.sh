#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-/data/model/Qwen3.5-2B}
DATASET=${DATASET:-/data/dataset/NuminaMath-TIR/data/train-00000-of-00001.parquet#1000}
OUTPUT_DIR=${OUTPUT_DIR:-output/grpo/qwen3_5_2b_megatron_colocate}
TRAIN_ITERS=${TRAIN_ITERS:-1}
PLUGIN=${PLUGIN:-examples/ascend/train/qwen3_5/manual_reward_plugin.py}

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1}
export NPROC_PER_NODE=${NPROC_PER_NODE:-2}
export USE_MCORE_GDN=${USE_MCORE_GDN:-0}
export VLLM_ASCEND_ENABLE_NZ=${VLLM_ASCEND_ENABLE_NZ:-0}

megatron rlhf \
    --rlhf_type grpo \
    --model "$MODEL" \
    --model_type qwen3_5 \
    --template qwen3_5 \
    --external_plugins "$PLUGIN" \
    --save_safetensors true \
    --dataset "$DATASET" \
    --train_iters "$TRAIN_ITERS" \
    --global_batch_size 4 \
    --micro_batch_size 1 \
    --steps_per_generation 1 \
    --num_generations 2 \
    --reward_funcs manual_alternating \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.35 \
    --vllm_tensor_parallel_size 2 \
    --vllm_max_model_len 256 \
    --vllm_enable_prefix_caching false \
    --vllm_enforce_eager true \
    --vllm_engine_kwargs '{"max_num_batched_tokens":128,"data_parallel_size":1}' \
    --max_length 128 \
    --max_completion_length 8 \
    --tuner_type full \
    --lr 1e-6 \
    --bf16 true \
    --beta 0.0 \
    --importance_sampling_level sequence \
    --epsilon 3e-4 \
    --epsilon_high 4e-4 \
    --dynamic_sample false \
    --overlong_filter false \
    --loss_type grpo \
    --sleep_level 2 \
    --offload_model true \
    --offload_bridge false \
    --offload_optimizer false \
    --optimizer_cpu_offload false \
    --use_precision_aware_optimizer false \
    --logging_steps 1 \
    --recompute_granularity selective \
    --recompute_modules core_attn \
    --finetune true \
    --dataloader_num_workers 0 \
    --dataset_num_proc 1 \
    --no_save_optim true \
    --no_save_rng true \
    --attention_backend flash \
    --temperature 1.0 \
    --padding_free true \
    --sequence_parallel true \
    --log_completions true \
    --report_to tensorboard \
    --save_steps 1000 \
    --eval_steps 1000 \
    --output_dir "$OUTPUT_DIR"
