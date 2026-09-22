"""Fetch pinned public safetensors, tokenizer, GLUE data and official LoRA source."""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fetch(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    temporary = path.with_name(path.name + '.partial')
    print('Downloading', path.name, flush=True)
    req = urllib.request.Request(url, headers={'User-Agent': 'AutoResearch-reproduction/1.0'})
    with urllib.request.urlopen(req, timeout=90) as response, temporary.open('wb') as target:
        received = 0
        expected = response.headers.get('Content-Length')
        while block := response.read(8 * 1024 * 1024):
            target.write(block)
            received += len(block)
        if expected and received != int(expected):
            raise ValueError('Incomplete download: ' + path.name)
    temporary.rename(path)
    print('Saved', path.name, received, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--asset-dir', type=Path, required=True)
    args = parser.parse_args()
    p = json.loads((ROOT / 'protocol.json').read_text())
    records = []
    for name in ['config.json', 'model.safetensors', 'tokenizer.json', 'merges.txt', 'vocab.json']:
        url = f"https://huggingface.co/{p['model']}/resolve/{p['model_revision']}/{name}"
        target = args.asset_dir / 'model' / name
        fetch(url, target)
        records.append({'relative_path': str(target.relative_to(args.asset_dir)), 'url': url, 'sha256': digest(target), 'bytes': target.stat().st_size})
    for split in ['train', 'validation']:
        name = f'{split}-00000-of-00001.parquet'
        url = f"https://huggingface.co/datasets/{p['dataset']}/resolve/{p['dataset_revision']}/mrpc/{name}"
        target = args.asset_dir / 'data' / name
        fetch(url, target)
        records.append({'relative_path': str(target.relative_to(args.asset_dir)), 'url': url, 'sha256': digest(target), 'bytes': target.stat().st_size})
    for name in ['loralib/__init__.py', 'loralib/layers.py', 'loralib/utils.py', 'LICENSE.md']:
        url = f"https://raw.githubusercontent.com/microsoft/LoRA/{p['upstream_lora_commit']}/{name}"
        target = ROOT / 'vendor' / name
        fetch(url, target)
        records.append({'source_path': str(target.relative_to(ROOT)), 'url': url, 'sha256': digest(target), 'bytes': target.stat().st_size})
    (args.asset_dir / 'asset_manifest.json').write_text(json.dumps({'assets': records, 'test_downloaded': False}, indent=2) + '\n')


if __name__ == '__main__':
    main()
