"""Independent all-row parity check and diagnosis of the withdrawn v3 inputs."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tokenizers import Tokenizer
from transformers import RobertaTokenizerFast
from tokenization import load_tokenizer, asset_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--asset-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    model = args.asset_dir / 'model'
    good = load_tokenizer(model)
    bad = RobertaTokenizerFast.from_pretrained(str(model), local_files_only=True)
    reference = Tokenizer.from_file(str(model / 'tokenizer.json'))
    result = {'status': 'PASS', 'transformers': importlib.metadata.version('transformers'), 'tokenizers': importlib.metadata.version('tokenizers'), 'tokenizer_asset_sha256': asset_hash(model), 'splits': {}}
    for split in ('train', 'validation'):
        frame = pd.read_parquet(args.asset_dir / f'data/{split}-00000-of-00001.parquet')
        pairs = list(zip(frame.sentence1, frame.sentence2))
        raw = [e.ids for e in reference.encode_batch(pairs)]
        fixed = good(frame.sentence1.tolist(), frame.sentence2.tolist(), truncation=False, padding=False)['input_ids']
        legacy = bad(frame.sentence1.tolist(), frame.sentence2.tolist(), truncation=False, padding=False)['input_ids']
        if raw != fixed:
            raise AssertionError('All-row parity failed for ' + split)
        def stats(encodings):
            lengths = np.array([len(row) for row in encodings])
            return {'over_128': int((lengths > 128).sum()), 'length_median': float(np.median(lengths)), 'length_max': int(max(lengths)), 'unique_tokens': len({t for row in encodings for t in row})}
        result['splits'][split] = {'rows': len(pairs), 'all_fixed_rows_equal_raw_asset': True, 'mismatched_legacy_rows': sum(a != b for a, b in zip(raw, legacy)), 'fixed': stats(fixed), 'legacy': stats(legacy), 'fixed_encodings_sha256': hashlib.sha256(json.dumps(fixed).encode()).hexdigest()}
    text = 'The quick brown fox jumps over the lazy dog.'
    result['probe'] = {'text': text, 'legacy_ids': bad(text)['input_ids'], 'fixed_ids': good(text)['input_ids'], 'legacy_decoded': bad.decode(bad(text)['input_ids']), 'fixed_decoded': good.decode(good(text)['input_ids'])}
    result['backend_merges'] = {name: len(json.loads(tok.backend_tokenizer.to_str())['model']['merges']) for name, tok in [('legacy', bad), ('fixed', good)]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
