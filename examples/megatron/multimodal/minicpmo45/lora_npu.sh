#!/usr/bin/env bash
set -euo pipefail

# Activate the NPU environment and source CANN before running this script.
# NPROC_PER_NODE=1 ASCEND_RT_VISIBLE_DEVICES=0 bash lora_npu.sh
megatron sft \
    --model /data/models/MiniCPM-4_5 \
    --model_type minicpmo \
    --template minicpmo4_5 \
    --dataset /data/dataset/librispeech_train_clean_100/train.jsonl \
    --split_dataset_ratio 0 \
    --tuner_type lora \
    --lora_rank 8 --lora_alpha 16 \
    --target_modules linear_qkv linear_proj linear_fc1 linear_fc2 \
    --freeze_llm false --freeze_vit true --freeze_aligner true \
    --tensor_model_parallel_size 1 --pipeline_model_parallel_size 1 \
    --sequence_parallel false --packing false --padding_free false \
    --micro_batch_size 1 --global_batch_size 1 \
    --train_iters 200 --lr 1e-4 --min_lr 1e-4 --lr_warmup_fraction 0 \
    --max_length 2048 --save_steps 100 --logging_steps 1 --eval_steps 1000 \
    --output_dir /data/zyh/experiments/minicpmo45/lora-tp1 --add_version false \
    --bf16 true --attention_backend flash --vit_attn_impl eager \
    --cross_entropy_loss_fusion false --gradient_accumulation_fusion false \
    --masked_softmax_fusion false --bias_dropout_fusion false \
    --recompute_granularity full --recompute_method uniform --recompute_num_layers 1 \
    --dataset_num_proc 1 --dataloader_num_workers 0 --report_to tensorboard \
    --save_safetensors false "$@"
