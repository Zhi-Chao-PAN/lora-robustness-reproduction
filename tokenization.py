"""Pinned tokenizer loader with guards against silent character-only fallback."""
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer
from transformers import AutoTokenizer


def load_tokenizer(model_dir):
    model_dir = Path(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
    source = json.loads((model_dir / 'tokenizer.json').read_text())
    actual = json.loads(tokenizer.backend_tokenizer.to_str())
    if actual['model']['vocab'] != source['model']['vocab']:
        raise ValueError('Tokenizer vocabulary differs from pinned source')
    # Older tokenizer.json encodes merge pairs as strings, current tokenizers as lists.
    normalize = lambda pairs: [tuple(p.split(' ', 1)) if isinstance(p, str) else tuple(p) for p in pairs]
    if normalize(actual['model']['merges']) != normalize(source['model']['merges']):
        raise ValueError('BPE merges differ from pinned source')
    raw = Tokenizer.from_file(str(model_dir / 'tokenizer.json'))
    probes = ['Hello world', 'The quick brown fox jumps over the lazy dog.', 'Café — 你好!']
    for text in probes:
        if tokenizer(text)['input_ids'] != raw.encode(text).ids:
            raise ValueError('Tokenizer probe differs from original tokenizer.json')
    if tokenizer('Hello world')['input_ids'] != [0, 31414, 232, 2]:
        raise ValueError('Documented RoBERTa probe failed')
    return tokenizer


def asset_hash(model_dir):
    return hashlib.sha256((Path(model_dir) / 'tokenizer.json').read_bytes()).hexdigest()
