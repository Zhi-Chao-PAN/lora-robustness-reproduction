"""Compare actual training cache tensors with the independent raw tokenizer asset."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tokenizers import Tokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--asset-dir', type=Path, required=True)
    p.add_argument('--cache-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    metadata = json.loads((a.cache_dir / 'split_manifest.json').read_text())
    path = a.cache_dir / 'tokenized.npz'
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata['tokenized_sha256']:
        raise ValueError('Cache hash mismatch')
    if any(metadata['normalized_unordered_exact_pair_overlaps'].values()):
        raise ValueError('Sentence-pair overlap across splits')
    cached = np.load(path, allow_pickle=False)
    training = pd.read_parquet(a.asset_dir / 'data/train-00000-of-00001.parquet').set_index('idx')
    validation = pd.read_parquet(a.asset_dir / 'data/validation-00000-of-00001.parquet').set_index('idx')
    if set(cached['train_ids']) & set(cached['dev_ids']) or set(cached['train_ids']) | set(cached['dev_ids']) != set(training.index):
        raise ValueError('Incorrect internal partition')
    raw = Tokenizer.from_file(str(a.asset_dir / 'model/tokenizer.json'))
    raw.enable_truncation(max_length=128)
    raw.enable_padding(length=128, pad_id=1, pad_token='<pad>')
    counts = {}
    for split in ('train', 'dev', 'validation'):
        source = (validation if split == 'validation' else training).loc[cached[f'{split}_ids']]
        encodings = raw.encode_batch(list(zip(source.sentence1, source.sentence2)))
        checks = [np.array_equal(cached[f'{split}_input_ids'], np.array([e.ids for e in encodings])), np.array_equal(cached[f'{split}_attention_mask'], np.array([e.attention_mask for e in encodings])), np.array_equal(cached[f'{split}_labels'], source.label.to_numpy())]
        if not all(checks):
            raise ValueError('Actual cache differs from raw source: ' + split)
        counts[split] = len(source)
    result = {'status': 'PASS', 'rows_checked': counts, 'total_rows': sum(counts.values()), 'input_ids_attention_masks_and_labels_identical_to_independent_source': True, 'all_split_pair_overlaps_zero': True, 'tokenized_sha256': metadata['tokenized_sha256']}
    a.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
