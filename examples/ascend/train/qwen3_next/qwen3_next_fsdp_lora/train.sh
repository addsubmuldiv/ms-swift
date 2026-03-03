nproc_per_node=8

ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch --config_file "./examples/ascend/train/qwen3_next/qwen3_next_fsdp_lora/fsdp.json" \
    swift/cli/sft.py \
    --model Qwen/Qwen3-Next-80B-A3B-Instruct \
    --tuner_type lora \
    --dataset 'swift/self-cognition#1000' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --gradient_checkpointing false \
    --weight_decay 0.1 \
    --target_modules q_proj k_proj v_proj o_proj \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --template qwen3_nothinking \
    --max_length 1024 \
    --output_dir output \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_author swift \
    --model_name swift-robot