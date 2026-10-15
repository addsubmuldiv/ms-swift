# Copyright (c) ModelScope Contributors. All rights reserved.
"""Audit matched TP=1 GPU/NPU runs with shared preprocessed audio features.

Pass --external_plugins tests/megatron/minicpmo45_cross_device_audit.py
     --callbacks minicpmo45_cross_device_audit --lora_dropout 0.
The first run exports initial-adapter; pass that directory with --adapters
to the second run so device-specific RNGs cannot change LoRA initialization.
The first run writes comparison_features beside the dataset JSONL. Copy that
directory to the second machine before its run to eliminate CPU FFT differences.
"""
import hashlib
import json
import torch
from pathlib import Path

from swift.megatron.callbacks import megatron_callbacks_map
from swift.megatron.callbacks.base import MegatronCallback


def tensor_signature(value):
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return {
            'dtype': str(tensor.dtype),
            'shape': list(tensor.shape),
            'sha256': hashlib.sha256(memoryview(tensor.view(torch.uint8).numpy())).hexdigest(),
        }
    if isinstance(value, (list, tuple)):
        return [tensor_signature(item) for item in value]
    return value


class MiniCPMOCrossDeviceAudit(MegatronCallback):

    def on_train_begin(self):
        assert self.args.tensor_model_parallel_size == self.args.pipeline_model_parallel_size == 1
        assert self.args.global_batch_size == self.args.micro_batch_size == 1
        assert self.args.lora_dropout == 0, 'Use matching dropout-free comparison configurations.'
        root = Path(self.args.output_dir)
        self.destination = root / 'cross-device.jsonl'
        self.destination.write_text('')
        self.feature_cache = Path(self.args.dataset[0]).parent / 'comparison_features'
        self.feature_cache.mkdir(exist_ok=True)
        self.cache_hits = self.cache_misses = 0
        parameters = {}
        for model in self.trainer.unwrapped_models:
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    parameters[name] = tensor_signature(parameter)
        (root / 'initial-trainable.json').write_text(json.dumps(parameters, indent=2, sort_keys=True))
        self.trainer.bridge.save_weights(
            self.trainer.unwrapped_models,
            str(root / 'initial-adapter'),
            peft_format=True,
            args=self.args,
            processor=self.trainer.template.processor)
        original_get_batch = self.trainer.get_batch

        def get_batch(*args, **kwargs):
            batch = original_get_batch(*args, **kwargs)
            keys = ('input_ids', 'labels', 'audio_features', 'audio_feature_lens', 'audio_bounds')
            metadata = {key: tensor_signature(batch[key]) for key in keys if key in batch and key != 'audio_features'}
            identity = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
            filename = self.feature_cache / f'{identity}.pt'
            features = batch['audio_features']
            if filename.exists():
                self.cache_hits += 1
                cached = torch.load(filename, map_location='cpu', weights_only=True)
                assert cached['metadata'] == metadata, 'Cached audio token/label metadata differs.'
                assert cached['features'].dtype == features.dtype and cached['features'].shape == features.shape
                batch['audio_features'] = cached['features'].to(features.device)
            else:
                self.cache_misses += 1
                torch.save({'metadata': metadata, 'features': features.detach().cpu()}, filename)
            self.write({
                'event': 'batch',
                'step': self.state.iteration + 1,
                'inputs': {
                    key: tensor_signature(batch[key])
                    for key in keys if key in batch
                },
            })
            return batch

        self.trainer.get_batch = get_batch

    def write(self, record):
        with self.destination.open('a') as stream:
            stream.write(json.dumps(record) + '\n')

    def on_log(self, logs):
        if 'loss' in logs:
            self.write({
                'event': 'loss',
                'step': self.state.iteration,
                'loss': logs['loss'],
                'grad_norm': logs.get('grad_norm')
            })

    def on_train_end(self):
        (Path(self.args.output_dir) / 'feature-cache.json').write_text(
            json.dumps({
                'hits': self.cache_hits,
                'misses': self.cache_misses,
            }, indent=2))


megatron_callbacks_map['minicpmo45_cross_device_audit'] = MiniCPMOCrossDeviceAudit
