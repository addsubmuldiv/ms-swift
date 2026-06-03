#!/usr/bin/env bash
set -euo pipefail

cd /data/zyh/tmp/worktrees/ms-swift/moonlight-shared-expert-lora-npu

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export TP="${TP:-1}"
export PP="${PP:-1}"
export EP="${EP:-2}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2}"
export MAX_LENGTH="${MAX_LENGTH:-1024}"
export TRAIN_ITERS="${TRAIN_ITERS:-10}"
export RUN_TAG="${RUN_TAG:-tp${TP}_pp${PP}_ep${EP}_n${NPROC_PER_NODE}}"

export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export USE_MCORE_GDN="${USE_MCORE_GDN:-1}"
export MEGATRON_LM_PATH=/data/zyh/code/Megatron-LM
export PYTHONPATH=/data/zyh/tmp/worktrees/mcore-bridge/moonlight-shared-expert-lora-npu/src:/data/zyh/code/Megatron-LM:/data/zyh/tmp/worktrees/ms-swift/moonlight-shared-expert-lora-npu:${PYTHONPATH:-}

/data/zyh/miniconda3/envs/swift_dev_v18_swiftpatch/bin/megatron sft \
    --model /data2/model/DeepSeek-R1-BF16 \
    --save_safetensors false \
    --merge_lora false \
    --dataset /data/dataset/self-cognition/self_cognition.jsonl \
    --loss_scale ignore_empty_think \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --split_dataset_ratio 0.01 \
    --tensor_model_parallel_size "${TP}" \
    --pipeline_model_parallel_size "${PP}" \
    --expert_model_parallel_size "${EP}" \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-3 \
    --micro_batch_size "${MICRO_BATCH_SIZE}" \
    --global_batch_size "${GLOBAL_BATCH_SIZE}" \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --train_iters "${TRAIN_ITERS}" \
    --finetune true \
    --gradient_accumulation_fusion false \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --output_dir "/data/zyh/tmp/deepseek_r1_lora_smoke/${RUN_TAG}" \
    --save_steps 1000 \
    --eval_steps 1000 \
    --max_length "${MAX_LENGTH}" \
    --dataloader_num_workers 1 \
    --dataset_num_proc 1 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --attention_backend flash
