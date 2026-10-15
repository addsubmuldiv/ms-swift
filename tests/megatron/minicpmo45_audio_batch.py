# Copyright (c) ModelScope Contributors. All rights reserved.
"""Compare real mixed-length audio embeddings with independent official encodes."""
import argparse
import copy
import json
import torch
from mcore_bridge.model.mm_gpts.minicpmo import MiniCPMOVit
from safetensors import safe_open
from types import SimpleNamespace

import swift.megatron  # Bootstrap MindSpeed before importing mcore-bridge.
from swift.model import get_model_processor
from swift.template import get_template
from swift.utils import to_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='/data/models/MiniCPM-4_5')
    parser.add_argument('--dataset', default='/data/dataset/librispeech_asr_dummy/mixed2.jsonl')
    parser.add_argument('--device', default='npu:0')
    options = parser.parse_args()
    _, processor = get_model_processor(
        options.model, model_type='minicpmo', load_model=False, torch_dtype=torch.bfloat16)
    config = SimpleNamespace(
        hf_config=processor.model_info.config,
        params_dtype=torch.bfloat16,
        vit_attn_impl='eager',
        language_model_only=False)
    visual = MiniCPMOVit(config).to(options.device).eval().requires_grad_(False)
    with open(options.model + '/model.safetensors.index.json') as stream:
        index = json.load(stream)['weight_map']
    state = {}
    embedding_weight = None
    for filename in sorted(set(index.values())):
        with safe_open(options.model + '/' + filename, framework='pt') as shard:
            for name in shard.keys():
                if name.split('.')[0] in visual.module_mapping:
                    state[name] = shard.get_tensor(name)
                elif name == 'llm.model.embed_tokens.weight':
                    embedding_weight = shard.get_tensor(name).to(options.device)
    visual.load_state_dict(state, strict=True)
    del state
    template = get_template(processor, template_type='minicpmo4_5', max_length=1024, padding_free=False)
    template.set_mode('train')
    with open(options.dataset) as stream:
        rows = [json.loads(line) for line in stream]
    encoded = [template.encode(row) for row in rows]
    batch = to_device(template.data_collator(copy.deepcopy(encoded)), options.device)
    inputs = torch.nn.functional.embedding(batch['input_ids'], embedding_weight)
    with torch.no_grad():
        actual = visual.get_inputs_embeds(inputs, **batch)
        expected = inputs.clone()
        for i, sample in enumerate(encoded):
            single = to_device(template.data_collator([copy.deepcopy(sample)]), options.device)
            with visual.patch_hf_config():
                segments = visual.model_cls.get_audio_embedding(visual, single, visual.hf_config.audio_chunk_length)[0]
            states = torch.cat(segments)
            offset = 0
            for start, end in single['audio_bounds'][0]:
                start, end = int(start), int(end)
                expected[i, start:end] = states[offset:offset + end - start]
                offset += end - start
            assert offset == states.shape[0]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    print(f'REAL AUDIO BATCH PASS: batch={len(rows)}, max_error={(actual - expected).abs().max().item()}', flush=True)


if __name__ == '__main__':
    main()
