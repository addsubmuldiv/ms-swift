# MiniCPM-o 4.5 音频 SFT

MiniCPM-o 4.5 的 Megatron 适配提供音频输入、文本输出的 SFT 路径，使用 `model_type=minicpmo`、`template=minicpmo4_5`。语音生成联合 loss、流式/全双工训练不在本阶段范围内；MiniCPM-o 2.6 也不属于本适配。

## 环境与数据

当前真实验证环境：Ascend 910B3、CANN 9.1.1、Python 3.12、PyTorch 2.10.0、torch_npu 2.10.0.post4、Transformers 5.5.4、Megatron-Core 0.16.0、MindSpeed `core_r0.16.0`。ms-swift 和 mcore-bridge 均从包含本适配的源码安装。GPU 的 H20-3e 单卡 LoRA 已验证（Python 3.11、PyTorch 2.9.0+cu130、Transformer Engine 2.9.0、FlashAttention 2.8.1）；GPU 的 Full 和其他并行配置尚未验证。

准备环境时参考 [Ascend 文档](./Ascend.md)，补充 `soundfile`、`librosa==0.11.0` 和 `timm`。音频 SFT 不需要安装 TTS/vocoder 依赖。请勿直接修改下载目录中的官方模型代码。

正式训练使用 ModelScope 的 `openslr/librispeech_asr`：完整 `train.clean.100` 包含 28,539 条语音，另下载独立的 `test.clean`（2,620 条）。脚本直连下载约 6.78 GB 的 15 个 Parquet 分片，逐个核对固定 SHA256，并输出 `train.jsonl`、`test.jsonl` 及音频校验清单到 `/data/dataset/librispeech_train_clean_100`。

```shell
python examples/megatron/multimodal/minicpmo45/prepare_librispeech.py
```

可选的小型功能验证集使用 `hf-internal-testing/librispeech_asr_dummy`：73 条真实英文语音及转写、16 kHz 单声道、约 9 MB 下载量，最长 29.4 秒。以下脚本先检查 ModelScope，没有相同的小型数据集时再从 HF 镜像直连下载；每次请求均禁用代理并校验固定 SHA256。

```shell
python examples/megatron/multimodal/minicpmo45/prepare_data.py
```

数据保存在 `/data/dataset/librispeech_asr_dummy`，包括原始 Parquet、FLAC、来源清单及确定性生成的 `train.jsonl`（64条）、`validation.jsonl`（9条）、`overfit8.jsonl`（训练集中最短8条）和 `mixed2.jsonl`（训练集中最短与最长各一条）。这些子集用于功能、过拟合和恢复验证，不用于报告泛化性能。

数据格式如下，`audios` 使用实际存在的本地绝对路径，转写保留原文：

```json
{"messages": [{"role": "user", "content": "<audio>Please transcribe this audio."}, {"role": "assistant", "content": "THE ORIGINAL TRANSCRIPT"}], "audios": ["/data/dataset/librispeech_asr_dummy/audio/sample.flac"]}
```

## 训练

将 ModelScope 的 `OpenBMB/MiniCPM-o-4_5` 下载到 `/data/models/MiniCPM-4_5`。激活 NPU 环境并加载 CANN 后，从仓库根目录运行：

```shell
NPROC_PER_NODE=1 ASCEND_RT_VISIBLE_DEVICES=0 \
bash examples/megatron/multimodal/minicpmo45/lora_npu.sh \
    --output_dir /data/experiments/minicpmo45-lora

# TP=2 + SP，混合长度音频，micro batch size=2。
NPROC_PER_NODE=2 ASCEND_RT_VISIBLE_DEVICES=0,1 \
bash examples/megatron/multimodal/minicpmo45/lora_npu.sh \
    --tensor_model_parallel_size 2 --sequence_parallel true \
    --micro_batch_size 2 --global_batch_size 2 \
    --dataset /data/dataset/librispeech_train_clean_100/train.jsonl \
    --output_dir /data/experiments/minicpmo45-tp2
```

默认训练 LLM LoRA，冻结音频编码器、视觉编码器和对齐模块。`freeze_vit` 控制视觉及音频编码器，`freeze_aligner` 控制视觉 resampler 及音频投影层。TTS 不参与训练，其原始权重在导出时保留。

全参数训练时可以训练 LLM、音频编码器和音频投影，单独冻结未使用的视觉组件：

```shell
NPROC_PER_NODE=4 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
bash examples/megatron/multimodal/minicpmo45/lora_npu.sh \
    --tuner_type full --tensor_model_parallel_size 4 --sequence_parallel true \
    --freeze_vit false --freeze_aligner true \
    --freeze_parameters visual.vpm visual.resampler \
    --trainable_parameters visual.audio_projection_layer \
    --lr 1e-6 --min_lr 1e-6 \
    --output_dir /data/experiments/minicpmo45-full
```

保持 `packing=false`、`padding_free=false`。已完成独立的 TP=1、TP=2+SP、PP=2 LoRA 训练/保存验证，以及 TP=4+SP 全参数训练/恢复验证；未验证这些并行方式的任意组合，也未验证 CP。

## 恢复与导出

脚本默认保留 optimizer 和 RNG 状态。恢复 LoRA 使用 `--mcore_adapter checkpoint目录 --finetune false`；恢复全参数训练使用 `--mcore_model checkpoint目录 --finetune false`。保持相同的训练步数上限、数据顺序和并行配置。

导出时保留原 HF 模型目录，并设置 `save_missing_weights=true`。TTS 不在 Megatron 训练状态中，必须从原 HF checkpoint 补回，不能只导出 LLM 和音频模块。

```shell
NPROC_PER_NODE=1 ASCEND_RT_VISIBLE_DEVICES=0 \
megatron export \
    --model /data/models/MiniCPM-4_5 --model_type minicpmo --template minicpmo4_5 \
    --mcore_adapter /data/experiments/minicpmo45-lora/checkpoint-200 \
    --to_hf true --merge_lora true --save_missing_weights true \
    --torch_dtype bfloat16 --vit_attn_impl eager \
    --output_dir /data/experiments/minicpmo45-merged-hf
```

全参数导出把 `mcore_adapter` 替换为 `mcore_model`，去掉 `merge_lora`，并按原 checkpoint 的 TP/PP 配置启动相应数量的进程。

## GPU/NPU 200 步 loss 对照

两端均使用 Megatron 和完整 LibriSpeech 训练集，固定相同的模型、训练参数、seed/data_seed、数据顺序和初始 LoRA。比较配置关闭 LoRA dropout。审计插件导出标准 PEFT `initial-adapter`，并在训练 JSONL 同级目录保存 `comparison_features/`，用于消除两台 CPU 做音频 FFT 时的末位差异。

先在 NPU 运行：

```shell
NPROC_PER_NODE=1 ASCEND_RT_VISIBLE_DEVICES=0 \
bash examples/megatron/multimodal/minicpmo45/lora_npu.sh \
    --train_iters 200 --save_steps 200 --lora_dropout 0 --seed 42 --data_seed 42 \
    --external_plugins tests/megatron/minicpmo45_cross_device_audit.py \
    --callbacks minicpmo45_cross_device_audit \
    --output_dir /data/experiments/minicpmo45-npu
```

将原模型、完整数据集及 `comparison_features/` 同步到 GPU 的相同路径，将初始 `initial-adapter/` 复制到 GPU。使用相同版本的本次源码运行同一脚本，保留以上训练参数，加 `--adapters 初始适配器目录`，设置 `CUDA_VISIBLE_DEVICES` 并将输出改为独立 GPU 目录。GPU 应加载全部 200 个共享特征，不生成新的特征缓存。模型 shard、原音频、训练 JSONL 均需校验一致。

收集两端运行目录后执行：

```shell
python tests/megatron/minicpmo45_compare_runs.py \
    --npu /data/experiments/minicpmo45-npu \
    --gpu /data/experiments/minicpmo45-gpu \
    --output-dir /data/experiments/minicpmo45-comparison
```

脚本核对训练参数、全部初始可训练张量以及 200 步的 token、标签、音频特征与边界，要求 loss/梯度有限，且完整 200 步 loss 的 MAE 严格小于 0.02；不裁掉预热步骤。输出 `loss.csv`、`comparison.json` 和 `training-config.json`。

已验证的 H20-3e / Ascend 910B3 单卡对照中，完整 200 步 loss MAE 为 0.00282515；最大单步绝对差为 0.05955768（验收阈值针对平均误差）。288 个初始可训练张量和 200 个输入批次完全一致。该结果仅覆盖上述 BF16 LoRA 配置，不代表 ASR 的 WER 评测。

## 验证与限制

验证包含1414个权重张量的精确往返、音频 embedding 对齐、真实音频训练、冻结参数哈希和确定性恢复。实际混合长短音频的独立编码检查入口是 `tests/megatron/minicpmo45_audio_batch.py`，训练审计插件为 `tests/megatron/minicpmo45_audit.py`。

可选的跨后端诊断入口是 `tests/megatron/minicpmo45_hf_reference.py` 和 `tests/megatron/minicpmo45_parity.py`。FP32 对齐需在生成参考与 Megatron 比较时均显式关闭混合精度。比较脚本的 `--dtype float32 --attention-backend local` 使用原生注意力参考路径，并检查 logits、loss；梯度参考由 HF 脚本的 `--backward` 生成。

跨后端逐元素 logits 相等不作为本阶段的验收条件。诊断中 BF16 logits 未通过 `atol=0.05, rtol=0.02`；FP32 单卡前后向通过 `atol=1e-4, rtol=1e-4`，双卡仍有个别元素超出该容差。脚本保留原断言，未降低阈值；这些结果不表示逐元素完全对齐。确定性恢复需在两次进程中使用一致的确定性配置；审计插件会为本次验证进程开启 PyTorch 确定性算法。

Transformers 5 的兼容处理限于模型加载、Whisper 注意力返回接口和非流式 Qwen3 文本生成，不代表已支持 TTS 或流式缓存。
