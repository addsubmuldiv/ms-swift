# Copyright (c) ModelScope Contributors. All rights reserved.
"""Real-checkpoint NPU parity test; launch with torchrun (one or two ranks).

python -m torch.distributed.run --standalone --nproc_per_node=1 \
    tests/megatron/minicpmo45_parity.py --reference /path/to/hf-reference.pt
"""
import argparse
import json
import os
import torch
import torch.distributed as dist
from contextlib import nullcontext
from dataclasses import asdict
from safetensors import safe_open

from swift.megatron import MegatronExportArguments
from swift.megatron.model import get_mcore_model
from swift.model import get_model_processor
from swift.utils import to_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/data/models/MiniCPM-4_5')
    parser.add_argument('--reference', required=True)
    parser.add_argument('--output-dir', default='/data/zyh/experiments/minicpmo45/parity')
    parser.add_argument('--sequence-parallel', action='store_true')
    parser.add_argument('--check-weights', action='store_true')
    parser.add_argument('--roundtrip-dir')
    parser.add_argument('--dtype', choices=['float32', 'bfloat16'], default='bfloat16')
    parser.add_argument('--attention-backend', choices=['flash', 'local'], default='flash')
    options = parser.parse_args()
    os.makedirs(options.output_dir, exist_ok=True)
    args = MegatronExportArguments(
        model=options.model,
        model_type='minicpmo',
        template='minicpmo4_5',
        to_mcore=True,
        output_dir=options.output_dir,
        exist_ok=True,
        torch_dtype=options.dtype,
        fp16=False,
        bf16=options.dtype == 'bfloat16',
        tensor_model_parallel_size=int(os.environ['WORLD_SIZE']),
        sequence_parallel=options.sequence_parallel,
        attention_backend=options.attention_backend,
        vit_attn_impl='eager',
        padding_free=False,
        gradient_accumulation_fusion=False,
        masked_softmax_fusion=False,
        bias_dropout_fusion=False,
        cross_entropy_loss_fusion=False)
    from swift.model.npu_patcher import apply_mindspeed_patches
    runtime_args = asdict(args)
    runtime_args['use_flash_attn'] = options.attention_backend == 'flash'
    apply_mindspeed_patches(runtime_args)
    _, processor = get_model_processor(
        options.model, model_type='minicpmo', load_model=False, torch_dtype=getattr(torch, options.dtype))
    model = get_mcore_model(args, processor.model_info.config)[0]
    model.config.bridge.load_weights([model], options.model)
    # Training's Float16Module casts the native normalization parameters too.
    model.cuda().to(dtype=args.torch_dtype)
    model.eval()
    print(f'DTYPE args={args.torch_dtype} params={model.config.params_dtype}', flush=True)
    if options.check_weights:
        with open(os.path.join(options.model, 'model.safetensors.index.json')) as stream:
            index = json.load(stream)['weight_map']
        checked = set()
        for name, tensor in model.config.bridge.export_weights([model], target_device='cpu', only_master_rank=True):
            if tensor is None:
                continue
            with safe_open(os.path.join(options.model, index[name]), framework='pt') as shard:
                expected_weight = shard.get_tensor(name)
            assert tensor.dtype == expected_weight.dtype, (name, tensor.dtype, expected_weight.dtype)
            torch.testing.assert_close(tensor, expected_weight, atol=0, rtol=0)
            checked.add(name)
        if dist.get_rank() == 0:
            assert checked == set(index), f'Missing weights: {set(index) - checked}'
            print(f'WEIGHT ROUNDTRIP PASS: {len(checked)} tensors', flush=True)
    if options.roundtrip_dir:
        model.config.bridge.save_weights([model],
                                         options.roundtrip_dir,
                                         args=args,
                                         processor=processor,
                                         save_missing_weights=options.model)
    reference = torch.load(options.reference, weights_only=False, map_location='cpu')
    batch = to_device(reference['batch'], torch.cuda.current_device())
    labels = batch.pop('labels')
    seq_len = batch['input_ids'].shape[1]
    device = batch['input_ids'].device
    batch['position_ids'] = torch.arange(seq_len, device=device).unsqueeze(0).expand_as(batch['input_ids'])
    # Match SWIFT's NPU training path: MindSpeed generates the compressed causal mask.
    batch['attention_mask'] = None if options.attention_backend == 'flash' else torch.triu(
        torch.ones(1, 1, seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    layer_outputs = []
    probes = {}

    def save_probe(name, *, before=False):

        def hook(module, inputs, output=None):
            value = inputs[0] if before else output
            if isinstance(value, tuple):
                value = value[0]
            probes[name] = value.detach().transpose(0, 1).cpu()
            if name == 'qkv' and hasattr(module, '_rmsnorm'):
                probes['norm0'] = module._rmsnorm(inputs[0]).detach().transpose(0, 1).cpu()

        return hook

    def save_layer_output(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if args.sequence_parallel:
            from megatron.core.tensor_parallel import gather_from_sequence_parallel_region
            hidden = gather_from_sequence_parallel_region(hidden.detach(), tensor_parallel_output_grad=False)
        layer_outputs.append(hidden.detach().transpose(0, 1).cpu())

    handles = [layer.register_forward_hook(save_layer_output) for layer in model.language_model.decoder.layers]
    if args.tensor_model_parallel_size == 1:
        first = model.language_model.decoder.layers[0]
        handles.append(first.self_attention.linear_qkv.register_forward_hook(save_probe('qkv')))
        for name in ['q', 'k']:
            handles.append(
                getattr(first.self_attention, name + '_layernorm').register_forward_hook(save_probe(name + '_norm')))
        handles.append(first.self_attention.linear_proj.register_forward_pre_hook(save_probe('context', before=True)))
        handles.append(first.mlp.linear_fc2.register_forward_pre_hook(save_probe('gated', before=True)))
    check_gradient = reference.get('embedding_gradient') is not None
    if check_gradient:
        model.requires_grad_(False)
        embedding_input = reference['embeddings'].to(
            device=device, dtype=args.torch_dtype).detach().requires_grad_(True)
        decoder_input = embedding_input.transpose(0, 1).contiguous()
        if args.sequence_parallel:
            from megatron.core.tensor_parallel import scatter_to_sequence_parallel_region
            decoder_input = scatter_to_sequence_parallel_region(decoder_input)
        batch = {name: batch[name] for name in ['input_ids', 'position_ids', 'attention_mask']}
        batch['decoder_input'] = decoder_input
    with nullcontext() if check_gradient else torch.no_grad():
        logits = model(**batch, runtime_gather_output=True)
    for handle in handles:
        handle.remove()
    expected_probes = reference.get('probes', {}).copy()
    if expected_probes and args.tensor_model_parallel_size == 1:
        groups = model.config.num_query_groups
        q, k, v = [expected_probes[name].reshape(1, seq_len, groups, -1) for name in ['q_proj', 'k_proj', 'v_proj']]
        expected_probes['qkv'] = torch.cat([q, k, v], dim=-1).flatten(2)
        for name in probes.keys() & expected_probes.keys():
            actual = probes[name]
            expected_probe = expected_probes[name].reshape_as(actual)
            difference = (actual.float() - expected_probe.float()).abs()
            print(f'PROBE {name}: max={difference.max().item()} mean={difference.mean().item()}', flush=True)
        torch.save({
            'actual': probes,
            'expected': expected_probes
        }, os.path.join(args.output_dir, 'first-layer-probes.pt'))
    for i, (actual, expected_layer) in enumerate(zip(layer_outputs, reference.get('layers', []))):
        difference = (actual.float() - expected_layer.float()).abs()
        print(f'LAYER {i}: max={difference.max().item()} mean={difference.mean().item()}', flush=True)
    expected = reference['logits'].to(logits)
    print(f'DTYPE logits={logits.dtype} reference={reference["logits"].dtype}', flush=True)
    error = (logits.float() - expected.float()).abs()
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=-100)
    if check_gradient:
        loss.backward()
        expected_gradient = reference['embedding_gradient'].to(embedding_input)
        cosine = torch.nn.functional.cosine_similarity(
            embedding_input.grad.float().flatten(), expected_gradient.float().flatten(), dim=0).item()
        print(f'GRADIENT cosine={cosine}', flush=True)
        assert torch.isfinite(embedding_input.grad).all() and cosine >= 0.99
    print(f'LOGITS max_error={error.max().item()} mean_error={error.mean().item()}', flush=True)
    print(f'LOSS mcore={loss.item()} hf={reference["loss"].item()}', flush=True)
    atol, rtol = (1e-4, 1e-4) if options.dtype == 'float32' else (0.05, 0.02)
    torch.testing.assert_close(logits, expected, atol=atol, rtol=rtol)
    assert abs(loss.item() - reference['loss'].item()) <= 0.05
    print('PARITY PASS', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
