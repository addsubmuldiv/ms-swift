# Copyright (c) ModelScope Contributors. All rights reserved.
"""Download the small ASR fixture without a proxy and create local SFT JSONL."""
import argparse
import hashlib
import io
import json
import pyarrow.parquet as pq
import soundfile as sf
import subprocess
from pathlib import Path

REPOSITORY = 'hf-internal-testing/librispeech_asr_dummy'
REVISION = '5be91486e11a2d616f4ec5db8d3fd248585ac07a'
SHA256 = '4e69a06fa5edc90921e5e7e39a7084881f8b3ed9c805c574f4f39c6fde27c603'
DATA_FILE = 'clean/validation-00000-of-00001.parquet'


def curl(url, *, output=None, check=True):
    command = ['curl', '--noproxy', '*', '-fsSL', '--retry', '2', '--connect-timeout', '10', '--max-time', '120', url]
    if output is not None:
        command += ['-o', str(output)]
    return subprocess.run(command, check=check, capture_output=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('/data/dataset/librispeech_asr_dummy'))
    options = parser.parse_args()
    root = options.output_dir.resolve()
    (root / 'raw').mkdir(parents=True, exist_ok=True)
    (root / 'audio').mkdir(exist_ok=True)
    # Prefer the same fixture on ModelScope. Do not fetch the >300 MB shards of
    # the full LibriSpeech corpus merely to obtain a small integration fixture.
    response = curl(
        f'https://modelscope.cn/api/v1/datasets/{REPOSITORY}/repo/tree?Revision=master&Root=&Recursive=true',
        check=False)
    metadata = json.loads(response.stdout) if response.stdout else {}
    files = (metadata.get('Data') or {}).get('Files', [])
    matching = [f for f in files if f['Path'] == DATA_FILE and f.get('Sha256') == SHA256]
    if matching:
        source = f'https://modelscope.cn/datasets/{REPOSITORY}/resolve/master/{DATA_FILE}'
    else:
        source = f'https://hf-mirror.com/datasets/{REPOSITORY}/resolve/{REVISION}/{DATA_FILE}'
    target = root / 'raw' / 'validation.parquet'
    if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != SHA256:
        temporary = target.with_suffix('.incomplete')
        curl(source, output=temporary)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != SHA256:
            raise ValueError('Dataset SHA256 does not match the pinned source.')
        temporary.replace(target)
    records, inventory = [], []
    rows = sorted(pq.read_table(target).to_pylist(), key=lambda row: row['id'])
    assert len(rows) == 73
    for row in rows:
        audio_bytes = row['audio']['bytes']
        samples, rate = sf.read(io.BytesIO(audio_bytes), dtype='float32')
        assert rate == 16000 and samples.ndim == 1 and row['text'].strip()
        duration = len(samples) / rate
        assert 0 < duration <= 30
        audio_path = root / 'audio' / f'{row["id"]}.flac'
        audio_path.write_bytes(audio_bytes)
        records.append({
            'messages': [{
                'role': 'user',
                'content': '<audio>Please transcribe this audio.'
            }, {
                'role': 'assistant',
                'content': row['text']
            }],
            'audios': [str(audio_path)]
        })
        inventory.append({'id': row['id'], 'duration': duration, 'text': row['text']})
    subsets = {
        'train':
        list(range(64)),
        'validation':
        list(range(64, 73)),
        'overfit8':
        sorted(range(64), key=lambda i: (inventory[i]['duration'], inventory[i]['id']))[:8],
        'mixed2': [
            min(range(64), key=lambda i: inventory[i]['duration']),
            max(range(64), key=lambda i: inventory[i]['duration'])
        ],
    }
    for name, indices in subsets.items():
        with (root / f'{name}.jsonl').open('w') as stream:
            for index in indices:
                stream.write(json.dumps(records[index]) + '\n')
    manifest = {
        'repository': REPOSITORY,
        'revision': REVISION,
        'source': source,
        'modelscope_status': metadata.get('Code', response.returncode),
        'proxy': False,
        'sha256': SHA256,
        'subsets': subsets,
        'rows': inventory
    }
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'DATASET PASS: {len(rows)} clips, {sum(r["duration"] for r in inventory):.3f} seconds, {root}')


if __name__ == '__main__':
    main()
