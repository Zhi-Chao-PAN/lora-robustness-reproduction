"""Pinned Google Research Datasets HF mirror after original GCS URL returned 403."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

REPO = 'google-research-datasets/paws'
REVISION = '161ece9501cf0a11f3e48bd356eaa82de46d6a09'
FILE = 'labeled_final/test-00000-of-00001.parquet'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a new output directory')
    args.out.mkdir(parents=True)
    sources = {
        'test.parquet': f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{FILE}',
        'DATASET_CARD.md': f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/README.md',
        'LICENSE': 'https://raw.githubusercontent.com/google-research-datasets/paws/master/LICENSE',
    }
    entries = []
    for name, url in sources.items():
        request = urllib.request.Request(url, headers={'User-Agent': 'reproducible-research/1.0'})
        with urllib.request.urlopen(request, timeout=90) as response:
            data = response.read()
        (args.out / name).write_bytes(data)
        entries.append({'file': name, 'url': url, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)})
    manifest = {'repository': REPO, 'revision': REVISION, 'config': 'labeled_final', 'split': 'test', 'expected_rows': 8000, 'original_gcs_download': 'HTTP 403 on 2026-09-23; no original-archive byte equivalence claimed', 'training_or_development_downloaded': False, 'purpose': 'Frozen zero-shot stress evaluation after MRPC checkpoint selection', 'assets': entries}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
