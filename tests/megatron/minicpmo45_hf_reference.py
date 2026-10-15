# Copyright (c) ModelScope Contributors. All rights reserved.
"""Create an HF reference using a real local MiniCPMO checkpoint and audio."""
import argparse
import json
import torch
from contextlib import nullcontext

from swift.model import get_model_processor
from swift.template import get_template
from swift.utils import to_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='/data/models/MiniCPM-4_5')
    parser.add_argument('--dataset', default='/data/dataset/librispeech_asr_dummy/overfit8.jsonl')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='npu:0')
    parser.add_argument('--text-only', action='store_true')
    parser.add_argument('--attn-impl', choices=['eager', 'sdpa'], default='eager')
    parser.add_argument('--dtype', choices=['float32', 'bfloat16'], default='bfloat16')
    parser.add_argument('--enable_npu_model_patch', default='true')
    parser.add_argument('--llm-attn-impl', choices=['eager', 'sdpa', 'flash_attention_2'])
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--backward', action='store_true')
    options = parser.parse_args()
    model, processor = get_model_processor(
        options.model,
        model_type='minicpmo',
        torch_dtype=getattr(torch, options.dtype),
        device_map=options.device,
        attn_impl=options.attn_impl)
    if options.llm_attn_impl:
        # Keep the audio encoder's reference backend fixed while selecting the
        # language decoder backend through Transformers' public API.
        model.llm.set_attn_implementation(options.llm_attn_impl)
    template = get_template(processor, template_type='minicpmo4_5', max_length=1024, padding_free=False)
    template.set_mode('train')
    with open(options.dataset) as stream:
        row = json.loads(next(stream))
    if options.text_only:
        row = {
            'messages': [{
                'role': 'user',
                'content': 'What is two plus two?'
            }, {
                'role': 'assistant',
                'content': 'Four.'
            }]
        }
    batch = to_device(template.data_collator([template.encode(row)]), options.device)
    model.eval()
    layer_outputs = []
    probes = {}

    def save_probe(name, *, before=False):

        def hook(module, inputs, output=None):
            value = inputs[0] if before else output
            probes[name] = value.detach().cpu()

        return hook

    def save_layer_output(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        layer_outputs.append(hidden.detach().cpu())

    handles = [layer.register_forward_hook(save_layer_output) for layer in model.llm.model.layers]
    first = model.llm.model.layers[0]
    handles += [first.input_layernorm.register_forward_hook(save_probe('norm0'))]
    for name in ['q_proj', 'k_proj', 'v_proj', 'q_norm', 'k_norm']:
        handles.append(getattr(first.self_attn, name).register_forward_hook(save_probe(name)))
    handles.append(first.self_attn.o_proj.register_forward_pre_hook(save_probe('context', before=True)))
    handles.append(first.mlp.down_proj.register_forward_pre_hook(save_probe('gated', before=True)))
    with torch.no_grad():
        embeddings, _ = model.get_vllm_embedding(batch)
        embeddings = model.get_omni_embedding(batch, embeddings, chunk_length=model.config.audio_chunk_length)
    if options.backward:
        model.requires_grad_(False)
        embeddings = embeddings.detach().requires_grad_(True)
    with nullcontext() if options.backward else torch.no_grad():
        output = model.llm(
            inputs_embeds=embeddings,
            attention_mask=batch.get('attention_mask'),
            labels=batch['labels'],
            use_cache=False)
    if options.backward:
        output.loss.backward()
        assert embeddings.grad is not None and torch.isfinite(embeddings.grad).all()
    assert torch.isfinite(output.logits).all() and torch.isfinite(output.loss)
    if options.generate:
        # Remove the supervised answer from the prompt before generation.
        prompt_end = int((batch['labels'][0] != -100).nonzero()[0])
        with torch.no_grad():
            generated = model.llm.generate(
                inputs_embeds=embeddings[:, :prompt_end],
                attention_mask=batch['attention_mask'][:, :prompt_end],
                max_new_tokens=8,
                do_sample=False)
        assert generated.numel() > 0
        print(f'GENERATION PASS: {processor.tokenizer.decode(generated[0])!r}', flush=True)
    for handle in handles:
        handle.remove()
    torch.save(
        {
            'batch': to_device(batch, 'cpu'),
            'embeddings': embeddings.detach().cpu(),
            'logits': output.logits.detach().cpu(),
            'loss': output.loss.detach().cpu(),
            'layers': layer_outputs,
            'probes': probes,
            'embedding_gradient': embeddings.grad.cpu() if options.backward else None,
        }, options.output)
    print(f'HF REFERENCE PASS: loss={output.loss.item()}, output={options.output}', flush=True)


if __name__ == '__main__':
    main()
