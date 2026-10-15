# Copyright (c) ModelScope Contributors. All rights reserved.
"""Prepare complete LibriSpeech train-clean-100/test-clean from ModelScope directly."""
import argparse
import hashlib
import io
import json
import pyarrow.parquet as pq
import soundfile as sf
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('/data/dataset/librispeech_train_clean_100'))
    parser.add_argument('--workers', type=int, default=4)
    options = parser.parse_args()
    root = options.output_dir.resolve()
    (root / 'raw').mkdir(parents=True, exist_ok=True)
    (root / 'audio').mkdir(exist_ok=True)
    pinned = json.loads(Path(__file__).with_name('librispeech_train100_manifest.json').read_text())

    def download(entry):
        target = root / 'raw' / entry['Path'].replace('/', '-')
        if target.exists() and target.stat().st_size == entry['Size'] and sha256(target) == entry['Sha256']:
            return entry, target
        temporary = target.with_suffix('.download')
        url = f'https://modelscope.cn/datasets/{pinned["repository"]}/resolve/master/{entry["Path"]}'
        subprocess.run([
            'curl', '--noproxy', '*', '-fsSL', '--retry', '3', '--connect-timeout', '15', '--max-time', '1800', url,
            '-o',
            str(temporary)
        ],
                       check=True)
        if temporary.stat().st_size != entry['Size'] or sha256(temporary) != entry['Sha256']:
            raise ValueError(f'Pinned dataset checksum mismatch: {entry["Path"]}')
        temporary.replace(target)
        print(f'DOWNLOAD PASS: {entry["Path"]}', flush=True)
        return entry, target

    with ThreadPoolExecutor(max_workers=options.workers) as pool:
        sources = list(pool.map(download, pinned['files']))
    records = {'train': [], 'test': []}
    inventory = {'train': [], 'test': []}
    identifiers = set()
    for entry, filename in sources:
        split = 'train' if '/train.clean.100/' in entry['Path'] else 'test'
        for batch in pq.ParquetFile(filename).iter_batches(batch_size=64):
            for row in batch.to_pylist():
                identifier = row['id']
                assert identifier not in identifiers, f'Duplicate audio ID: {identifier}'
                identifiers.add(identifier)
                audio = row['audio']['bytes']
                info = sf.info(io.BytesIO(audio))
                assert info.samplerate == 16000 and info.channels == 1 and info.frames > 0
                assert row['text'].strip()
                path = root / 'audio' / f'{identifier}.flac'
                path.write_bytes(audio)
                records[split].append((identifier, {
                    'messages': [
                        {
                            'role': 'user',
                            'content': '<audio>Please transcribe this audio.'
                        },
                        {
                            'role': 'assistant',
                            'content': row['text']
                        },
                    ],
                    'audios': [str(path)],
                }))
                inventory[split].append({
                    'id': identifier,
                    'seconds': info.frames / info.samplerate,
                    'sha256': hashlib.sha256(audio).hexdigest(),
                })
    assert len(records['train']) == 28539 and len(records['test']) == 2620
    summary = {}
    for split, rows in records.items():
        with (root / f'{split}.jsonl').open('w') as stream:
            for _, record in sorted(rows):
                stream.write(json.dumps(record) + '\n')
        inventory[split].sort(key=lambda row: row['id'])
        summary[split] = {
            'samples': len(rows),
            'hours': sum(row['seconds'] for row in inventory[split]) / 3600,
            'max_seconds': max(row['seconds'] for row in inventory[split]),
            'jsonl_sha256': sha256(root / f'{split}.jsonl'),
        }
    (root / 'manifest.json').write_text(
        json.dumps({
            'source': pinned,
            'proxy': False,
            'summary': summary,
            'audio': inventory,
        }, indent=2))
    print('LIBRISPEECH PASS: ' + json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
