# Copyright (c) ModelScope Contributors. All rights reserved.
"""Optional integration-test callback for real training; never enabled by default.

Pass --external_plugins tests/megatron/minicpmo45_audit.py --callbacks minicpmo45_audit.
"""
import hashlib
import json
import torch
import torch.distributed as dist
from pathlib import Path

from swift.megatron.callbacks import megatron_callbacks_map
from swift.megatron.callbacks.base import MegatronCallback

torch.use_deterministic_algorithms(True)


def digest(parameter):
    data = parameter.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(data)).hexdigest()


class MiniCPMOAudit(MegatronCallback):

    def parameters(self):
        for index, model in enumerate(self.trainer.unwrapped_models):
            for name, parameter in model.named_parameters():
                yield f'{index}.{name}', parameter

    def on_train_begin(self):
        self.frozen = {name: digest(p) for name, p in self.parameters() if not p.requires_grad}
        self.trainable = {name: digest(p) for name, p in self.parameters() if p.requires_grad}
        self.gradient_steps = 0

    def on_step_end(self):
        nonzero = False
        for _, parameter in self.parameters():
            if not parameter.requires_grad:
                continue
            gradient = getattr(parameter, 'main_grad', parameter.grad)
            if gradient is not None:
                assert torch.isfinite(gradient).all(), 'Non-finite trainable gradient.'
                nonzero |= bool(gradient.count_nonzero())
        assert nonzero, 'No trainable parameter received a nonzero gradient.'
        self.gradient_steps += 1

    def on_train_end(self):
        changed = []
        for name, parameter in self.parameters():
            value = digest(parameter)
            if name in self.frozen:
                assert value == self.frozen[name], f'Frozen parameter changed: {name}'
            elif value != self.trainable[name]:
                changed.append(name)
        assert changed and self.gradient_steps
        result = {
            'frozen_tensors_unchanged': len(self.frozen),
            'trainable_tensors_changed': changed,
            'finite_nonzero_gradient_steps': self.gradient_steps,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        }
        destination = Path(self.args.output_dir) / f'audit-rank{dist.get_rank()}.json'
        destination.write_text(json.dumps(result, indent=2))
        print(f'AUDIT PASS: {destination}', flush=True)


megatron_callbacks_map['minicpmo45_audit'] = MiniCPMOAudit
