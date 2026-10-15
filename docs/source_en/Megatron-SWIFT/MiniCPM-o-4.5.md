# MiniCPM-o 4.5 Audio SFT

The Megatron adapter supports audio-input, text-output SFT with `model_type=minicpmo` and `template=minicpmo4_5`. Joint speech-output losses, streaming/full-duplex training, and MiniCPM-o 2.6 are outside this stage.

## Environment and data

The verified environment uses Ascend 910B3, CANN 9.1.1, Python 3.12, PyTorch 2.10.0, torch_npu 2.10.0.post4, Transformers 5.5.4, Megatron-Core 0.16.0, and MindSpeed `core_r0.16.0`. Install both ms-swift and mcore-bridge from source containing this adapter. Single-GPU H20-3e LoRA is validated with Python 3.11, PyTorch 2.9.0+cu130, Transformer Engine 2.9.0, and FlashAttention 2.8.1. GPU full tuning and other GPU parallel configurations remain unvalidated.

Follow the [Ascend guide](./Ascend.md) and install `soundfile`, `librosa==0.11.0`, and `timm`. Audio-to-text SFT does not require TTS/vocoder dependencies. Keep the downloaded official model code unchanged.

Use ModelScope's `openslr/librispeech_asr` for training: the complete `train.clean.100` contains 28,539 recordings, with an independent `test.clean` split of 2,620 recordings. The script downloads about 6.78 GB across 15 Parquet shards directly, verifies each pinned SHA256, and writes `train.jsonl`, `test.jsonl`, and audio checksums under `/data/dataset/librispeech_train_clean_100`.

```shell
python examples/megatron/multimodal/minicpmo45/prepare_librispeech.py
```

The optional small integration fixture is `hf-internal-testing/librispeech_asr_dummy`: 73 real English recordings with transcripts, 16 kHz mono audio, about 9 MB to download, and a maximum duration of 29.4 seconds. The preparation script checks ModelScope first and falls back to the HF mirror when that same small fixture is unavailable. All requests bypass proxies and the pinned SHA256 is verified.

```shell
python examples/megatron/multimodal/minicpmo45/prepare_data.py
```

The output directory `/data/dataset/librispeech_asr_dummy` contains the original Parquet, extracted FLAC recordings, provenance manifest, and deterministic `train.jsonl` (64 samples), `validation.jsonl` (9 samples), `overfit8.jsonl` (the eight shortest training recordings), and `mixed2.jsonl` (the shortest and longest training recordings). These subsets test functionality, overfitting, and resume behavior; they do not measure generalization.

Use real absolute audio paths and preserve the original transcripts:

```json
{"messages": [{"role": "user", "content": "<audio>Please transcribe this audio."}, {"role": "assistant", "content": "THE ORIGINAL TRANSCRIPT"}], "audios": ["/data/dataset/librispeech_asr_dummy/audio/sample.flac"]}
```

## Training

Download `OpenBMB/MiniCPM-o-4_5` from ModelScope into `/data/models/MiniCPM-4_5`. Activate the NPU environment, source CANN, and run from the repository root:

```shell
NPROC_PER_NODE=1 ASCEND_RT_VISIBLE_DEVICES=0 \
bash examples/megatron/multimodal/minicpmo45/lora_npu.sh \
    --output_dir /data/experiments/minicpmo45-lora

# TP=2 + SP, mixed audio lengths, micro batch size=2.
NPROC_PER_NODE=2 ASCEND_RT_VISIBLE_DEVICES=0,1 \
bash examples/megatron/multimodal/minicpmo45/lora_npu.sh \
    --tensor_model_parallel_size 2 --sequence_parallel true \
    --micro_batch_size 2 --global_batch_size 2 \
    --dataset /data/dataset/librispeech_train_clean_100/train.jsonl \
    --output_dir /data/experiments/minicpmo45-tp2
```

The default trains LLM LoRA while freezing the audio/vision encoders and aligners. `freeze_vit` controls both encoders; `freeze_aligner` controls the vision resampler and audio projector. TTS remains untrained and its original weights are preserved on export.

For full tuning of the LLM, audio encoder, and audio projector, freeze unused vision components separately:

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

Keep `packing=false` and `padding_free=false`. Separate LoRA training/save checks cover TP=1, TP=2+SP, and PP=2. Full tuning/resume covers TP=4+SP. Arbitrary combinations of these configurations and CP have not been validated.

## Resume and export

The script retains optimizer and RNG state. Resume LoRA with `--mcore_adapter checkpoint_directory --finetune false`, or full tuning with `--mcore_model checkpoint_directory --finetune false`. Keep the same training-step limit, data ordering, and parallel configuration.

Keep the original HF directory available and use `save_missing_weights=true` for export. TTS is absent from the Megatron training state and must be restored from the original checkpoint.

```shell
NPROC_PER_NODE=1 ASCEND_RT_VISIBLE_DEVICES=0 \
megatron export \
    --model /data/models/MiniCPM-4_5 --model_type minicpmo --template minicpmo4_5 \
    --mcore_adapter /data/experiments/minicpmo45-lora/checkpoint-200 \
    --to_hf true --merge_lora true --save_missing_weights true \
    --torch_dtype bfloat16 --vit_attn_impl eager \
    --output_dir /data/experiments/minicpmo45-merged-hf
```

For full tuning, replace `mcore_adapter` with `mcore_model`, omit `merge_lora`, and launch the number of processes required by the checkpoint's TP/PP configuration.

## GPU/NPU 200-step loss comparison

Use Megatron and the complete LibriSpeech training split on both devices, with identical model weights, hyperparameters, seeds, data ordering, and initial LoRA weights. Disable LoRA dropout in this comparison. The audit plugin exports a standard PEFT `initial-adapter` and writes `comparison_features/` beside the training JSONL to eliminate differences in CPU audio FFT rounding.

Run NPU first:

```shell
NPROC_PER_NODE=1 ASCEND_RT_VISIBLE_DEVICES=0 \
bash examples/megatron/multimodal/minicpmo45/lora_npu.sh \
    --train_iters 200 --save_steps 200 --lora_dropout 0 --seed 42 --data_seed 42 \
    --external_plugins tests/megatron/minicpmo45_cross_device_audit.py \
    --callbacks minicpmo45_cross_device_audit \
    --output_dir /data/experiments/minicpmo45-npu
```

Copy the original model, complete dataset, and `comparison_features/` to the same paths on GPU, and copy the initial `initial-adapter/` to GPU. Run the same script from the same source versions, keep all training arguments above, add `--adapters initial_adapter_directory`, set `CUDA_VISIBLE_DEVICES`, and use a separate GPU output directory. GPU must load all 200 shared features without creating new cache entries. Verify matching checksums for model shards, original audio, and training JSONL.

Collect both run directories and execute:

```shell
python tests/megatron/minicpmo45_compare_runs.py \
    --npu /data/experiments/minicpmo45-npu \
    --gpu /data/experiments/minicpmo45-gpu \
    --output-dir /data/experiments/minicpmo45-comparison
```

The script checks training arguments, every initial trainable tensor, and all 200 token/label/audio-feature/boundary batches. Losses and gradient norms must be finite, and loss MAE across all 200 steps must be strictly below 0.02, without discarding warmup steps. Outputs are `loss.csv`, `comparison.json`, and `training-config.json`.

The verified single-device H20-3e / Ascend 910B3 comparison has loss MAE 0.00282515 over all 200 steps. The maximum single-step absolute difference is 0.05955768; the acceptance threshold applies to the mean error. All 288 initial trainable tensors and 200 input batches match exactly. This result covers the BF16 LoRA configuration above and does not measure ASR word error rate.

## Validation and limitations

Checks cover exact round-trip conversion of 1,414 weight tensors, audio embeddings, real audio training, frozen-parameter hashes, and deterministic resume. Use `tests/megatron/minicpmo45_audio_batch.py` to compare real short/long audio batches against independent encodes. The optional training audit plugin is `tests/megatron/minicpmo45_audit.py`.

Optional cross-backend diagnostic entry points are `tests/megatron/minicpmo45_hf_reference.py` and `tests/megatron/minicpmo45_parity.py`. FP32 comparisons explicitly disable mixed precision in both implementations. The comparison script uses `--dtype float32 --attention-backend local` for native attention and checks logits and loss. Generate the HF gradient reference with `--backward`.

Cross-backend elementwise logits parity is outside this stage's acceptance criteria. Diagnostic BF16 logits fail `atol=0.05, rtol=0.02`. Single-device FP32 forward/backward checks pass `atol=1e-4, rtol=1e-4`, while a few two-device elements exceed that tolerance. The scripts retain their original assertions without relaxing thresholds; these results do not establish complete elementwise parity. Deterministic resume requires matching determinism settings in both runs; the audit plugin enables PyTorch deterministic algorithms for its validation process.

Transformers 5 compatibility covers loading, the Whisper attention return interface, and non-streaming Qwen3 text generation. It does not establish support for TTS or streaming caches.
