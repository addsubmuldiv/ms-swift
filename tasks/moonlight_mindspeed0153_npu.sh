#!/usr/bin/env bash
set -euo pipefail

cd /data/zyh/code/ms-swift

export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export ASCEND_RT_VISIBLE_DEVICES=6,7
export NPROC_PER_NODE=2
export MEGATRON_LM_PATH=/data/zyh/code/Megatron-LM
export PYTHONPATH=/data/zyh/code/Megatron-LM:/data/zyh/code/ms-swift:${PYTHONPATH:-}

/data/zyh/miniconda3/envs/swift_dev_v18_swiftpatch/bin/megatron sft \
    --model '/data/model/Moonlight-16B-A3B-Instruct' \
    --save_safetensors true \
    --merge_lora true \
    --dataset '/data/dataset/self-cognition/self_cognition.jsonl' \
    --loss_scale ignore_empty_think \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --freeze_parameters_ratio 1 \
    --target_regex '.*.mlp.shared_experts.*.' \
    --split_dataset_ratio 0.01 \
    --expert_model_parallel_size 1 \
    --pipeline_model_parallel_size 2 \
    --decoder_last_pipeline_num_layers 13 \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-3 \
    --micro_batch_size 8 \
    --global_batch_size 16 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --gradient_accumulation_fusion false \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --output_dir /data/zyh/code/ms-swift/megatron_output/moonlight_mindspeed0153_npu \
    --num_train_epochs 3 \
    --logging_steps 2 \
    --max_length 2048 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --attention_backend flash
