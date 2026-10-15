# Copyright (c) ModelScope Contributors. All rights reserved.
"""Compare all corresponding steps of audited GPU/NPU training runs."""
import argparse
import csv
import json
import math
from pathlib import Path


def read_run(root, steps):
    events = [json.loads(line) for line in (root / 'cross-device.jsonl').read_text().splitlines()]
    losses, batches = {}, {}
    for event in events:
        destination = losses if event['event'] == 'loss' else batches
        step = event['step']
        assert step not in destination, f'Duplicate {event["event"]} at step {step}: {root}'
        destination[step] = event
    expected = set(range(1, steps + 1))
    assert set(losses) == set(batches) == expected, f'Incomplete steps: {root}'
    for event in losses.values():
        assert math.isfinite(event['loss']) and math.isfinite(event['grad_norm']), event
    initial = json.loads((root / 'initial-trainable.json').read_text())
    assert initial, f'Missing initial trainable parameters: {root}'
    return losses, batches, initial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=Path, required=True)
    parser.add_argument('--npu', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--mae-limit', type=float, default=0.02)
    options = parser.parse_args()
    gpu, gpu_batches, gpu_initial = read_run(options.gpu, options.steps)
    npu, npu_batches, npu_initial = read_run(options.npu, options.steps)
    keys = (
        'seed',
        'data_seed',
        'train_iters',
        'micro_batch_size',
        'global_batch_size',
        'tensor_model_parallel_size',
        'pipeline_model_parallel_size',
        'sequence_parallel',
        'tuner_type',
        'lora_rank',
        'lora_alpha',
        'lora_dropout',
        'target_modules',
        'freeze_llm',
        'freeze_vit',
        'freeze_aligner',
        'lr',
        'min_lr',
        'lr_warmup_fraction',
        'weight_decay',
        'adam_beta1',
        'adam_beta2',
        'adam_eps',
        'clip_grad',
        'max_length',
        'bf16',
        'fp16',
        'packing',
        'padding_free',
        'attention_backend',
        'vit_attn_impl',
        'model_type',
        'template',
        'dataset',
        'split_dataset_ratio',
        'dataset_shuffle',
        'streaming',
        'lazy_tokenize',
        'lora_dtype',
        'lora_bias',
        'use_rslora',
        'optimizer',
        'use_distributed_optimizer',
        'lr_decay_style',
        'lr_decay_iters',
        'lr_warmup_iters',
        'cross_entropy_loss_fusion',
        'gradient_accumulation_fusion',
        'masked_softmax_fusion',
        'bias_dropout_fusion',
        'recompute_granularity',
        'recompute_method',
        'recompute_num_layers',
        'dataset_num_proc',
        'dataloader_num_workers',
        'freeze_parameters',
        'trainable_parameters',
        'freeze_parameters_regex',
        'trainable_parameters_regex',
    )
    gpu_args = json.loads((options.gpu / 'args.json').read_text())
    npu_args = json.loads((options.npu / 'args.json').read_text())
    differences = {key for key in gpu_args.keys() | npu_args.keys() if gpu_args.get(key) != npu_args.get(key)}
    allowed = {'adapters', 'output_dir', 'tensorboard_dir', 'use_flash_attn'}
    assert differences <= allowed, f'Unexpected argument differences: {differences - allowed}'
    if 'use_flash_attn' in differences:
        backend = gpu_args['attention_backend']
        assert backend == npu_args['attention_backend']
        if backend.startswith('<AttnBackend.'):
            backend = backend.partition('.')[2].partition(':')[0]
        assert backend == 'flash'
        assert gpu_args.get('use_flash_attn') is None and npu_args['use_flash_attn'] is True
    assert {
        key: gpu_args[key]
        for key in keys
    } == {
        key: npu_args[key]
        for key in keys
    }, 'Training configurations differ.'
    assert gpu_initial == npu_initial, 'GPU/NPU initial trainable weights differ.'
    assert gpu_batches == npu_batches, 'GPU/NPU input tokens, labels, or audio batches differ.'
    gpu_cache = json.loads((options.gpu / 'feature-cache.json').read_text())
    assert gpu_cache == {'hits': options.steps, 'misses': 0}, 'GPU must reuse every shared audio feature tensor.'
    rows = []
    for step in range(1, options.steps + 1):
        difference = abs(gpu[step]['loss'] - npu[step]['loss'])
        rows.append((step, gpu[step]['loss'], npu[step]['loss'], difference))
    mae = sum(row[3] for row in rows) / options.steps
    result = {
        'steps': options.steps,
        'mae': mae,
        'mae_limit': options.mae_limit,
        'max_absolute_error': max(row[3] for row in rows),
        'initial_trainable_tensors_equal': len(gpu_initial),
        'input_batches_equal': len(gpu_batches),
        'gpu_shared_feature_hits': gpu_cache['hits'],
        'matching_arguments': len(gpu_args.keys() | npu_args.keys()) - len(differences),
        'expected_argument_differences': sorted(differences),
        'passed': mae < options.mae_limit,
    }
    options.output_dir.mkdir(parents=True, exist_ok=True)
    (options.output_dir / 'training-config.json').write_text(json.dumps({key: gpu_args[key] for key in keys}, indent=2))
    with (options.output_dir / 'loss.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(('step', 'gpu_loss', 'npu_loss', 'absolute_error'))
        writer.writerows(rows)
    (options.output_dir / 'comparison.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    assert result['passed'], f'Loss MAE {mae} must be < {options.mae_limit}.'


if __name__ == '__main__':
    main()
