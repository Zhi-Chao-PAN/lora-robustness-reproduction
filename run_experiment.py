"""Pinned RoBERTa MRPC training; no test labels or remote model code execution."""
import argparse
import contextlib
import copy
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file, save_file
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import RobertaForSequenceClassification
from transformers.utils import logging as hf_logging
from tokenization import load_tokenizer, asset_hash

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'vendor'))
import loralib as lora


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parameter_hash(model, frozen_only=False):
    h = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if frozen_only and parameter.requires_grad:
            continue
        h.update(name.encode())
        h.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def make_model(asset_dir, method, seed):
    seed_all(seed)
    hf_logging.set_verbosity_error()
    model = RobertaForSequenceClassification.from_pretrained(
        str(asset_dir / 'model'), num_labels=2, local_files_only=True,
        use_safetensors=True, attn_implementation='eager',
    )
    initial_head_hash = parameter_hash(model.classifier)
    if method != 'full':
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.classifier.parameters():
            parameter.requires_grad_(True)
    if method.startswith('lora_r'):
        rank = int(method.split('_r')[1])
        for layer in model.roberta.encoder.layer:
            for name in ['query', 'value']:
                original = getattr(layer.attention.self, name)
                replacement = lora.Linear(
                    original.in_features, original.out_features, r=rank,
                    lora_alpha=2 * rank, lora_dropout=0.0,
                    bias=original.bias is not None, merge_weights=False,
                )
                with torch.no_grad():
                    replacement.weight.copy_(original.weight)
                    if original.bias is not None:
                        replacement.bias.copy_(original.bias)
                        replacement.bias.requires_grad_(False)
                setattr(layer.attention.self, name, replacement)
    return model, initial_head_hash


def prepare_data(asset_dir, out_dir, protocol):
    cache = out_dir / 'tokenized.npz'
    metadata_path = out_dir / 'split_manifest.json'
    if cache.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get('tokenizer_asset_sha256') != asset_hash(asset_dir / 'model') or not metadata.get('correct_tokenizer_guard_passed'):
            raise ValueError('Cache lacks corrected tokenizer provenance')
        if metadata['protocol_sha256'] != file_hash(ROOT / 'protocol.json'):
            raise ValueError('Tokenized cache protocol differs')
        if metadata['tokenized_sha256'] != file_hash(cache):
            raise ValueError('Tokenized cache hash differs')
        return np.load(cache), metadata
    original = pd.read_parquet(asset_dir / 'data/train-00000-of-00001.parquet')
    external = pd.read_parquet(asset_dir / 'data/validation-00000-of-00001.parquet')
    train_indices, dev_indices = train_test_split(
        np.arange(len(original)), test_size=protocol['internal_dev_fraction'],
        random_state=protocol['split_seed'], stratify=original['label'],
    )
    frames = {'train': original.iloc[sorted(train_indices)], 'dev': original.iloc[sorted(dev_indices)], 'validation': external}
    def pair_key(row):
        return tuple(sorted((' '.join(str(row['sentence1']).lower().split()), ' '.join(str(row['sentence2']).lower().split()))))
    keys = {split: {pair_key(row) for _, row in frame.iterrows()} for split, frame in frames.items()}
    overlaps = {f'{a}_{b}': len(keys[a] & keys[b]) for a, b in [('train', 'dev'), ('train', 'validation'), ('dev', 'validation')]}
    if overlaps['train_dev']:
        raise ValueError('Exact sentence pair overlap in internal split; protocol amendment required')
    tokenizer = load_tokenizer(asset_dir / 'model')
    arrays = {}
    metadata = {'protocol_sha256': file_hash(ROOT / 'protocol.json'), 'splits': {}, 'normalized_unordered_exact_pair_overlaps': overlaps, 'tokenizer_asset_sha256': asset_hash(asset_dir / 'model'), 'correct_tokenizer_guard_passed': True}
    for split, frame in frames.items():
        encoded = tokenizer(
            frame['sentence1'].tolist(), frame['sentence2'].tolist(),
            truncation=True, max_length=protocol['max_length'], padding='max_length', return_tensors='np',
        )
        for key in ['input_ids', 'attention_mask']:
            arrays[f'{split}_{key}'] = encoded[key].astype(np.int64)
        arrays[f'{split}_labels'] = frame['label'].to_numpy(dtype=np.int64)
        arrays[f'{split}_ids'] = frame['idx'].to_numpy(dtype=np.int64)
        metadata['splits'][split] = {'rows': len(frame), 'ids': arrays[f'{split}_ids'].tolist(), 'labels': arrays[f'{split}_labels'].tolist()}
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **arrays)
    metadata['tokenized_sha256'] = file_hash(cache)
    dump(metadata_path, metadata)
    return np.load(cache), metadata


def amp():
    return torch.autocast(device_type='cuda', dtype=torch.bfloat16)


def batches(order, size):
    for start in range(0, len(order), size):
        yield order[start:start + size]


def tensor_inputs(data, split, indices):
    return {name: torch.from_numpy(data[f'{split}_{name}'][indices]).cuda() for name in ['input_ids', 'attention_mask']}


@torch.no_grad()
def encode_features(model, data, split, batch_size):
    model.eval()
    features = []
    for indices in batches(np.arange(len(data[f'{split}_labels'])), batch_size):
        with amp():
            encoded = model.roberta(**tensor_inputs(data, split, indices)).last_hidden_state[:, :1, :]
        features.append(encoded.cpu())
    return torch.cat(features)


@torch.no_grad()
def evaluate(model, data, split, batch_size, features=None):
    model.eval()
    all_logits = []
    for indices in batches(np.arange(len(data[f'{split}_labels'])), batch_size):
        with amp():
            logits = model.classifier(features[indices].cuda()) if features is not None else model(**tensor_inputs(data, split, indices)).logits
        all_logits.append(logits.float().cpu().numpy())
    logits = np.concatenate(all_logits)
    prediction = logits.argmax(axis=1)
    labels = data[f'{split}_labels']
    metrics = {'accuracy': float(accuracy_score(labels, prediction)), 'binary_f1': float(f1_score(labels, prediction, zero_division=0))}
    return metrics, logits, prediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--asset-dir', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--method', choices=['head', 'full', 'lora_r2', 'lora_r8', 'lora_r16'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--epochs', type=int, choices=[4, 12], required=True)
    parser.add_argument('--pilot-steps', type=int, default=0)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a new output directory')
    args.out.mkdir(parents=True)
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    protocol['epochs'] = args.epochs
    torch.set_num_threads(protocol['resource_budget']['torch_cpu_threads'])
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Frozen protocol needs a BF16-capable CUDA GPU')
    torch.cuda.set_per_process_memory_fraction(0.7)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter()
    data, split_manifest = prepare_data(args.asset_dir, args.cache_dir, protocol)
    dump(args.out / 'protocol.json', protocol)
    dump(args.out / 'split_manifest.json', split_manifest)
    source_hashes = {str(p.relative_to(ROOT)): file_hash(p) for p in [ROOT / 'run_experiment.py', ROOT / 'protocol.json', ROOT / 'tokenization.py', *sorted((ROOT / 'vendor').rglob('*.py'))]}
    environment = {
        'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0), 'cuda_memory_fraction': 0.7,
        'packages': {name: importlib.metadata.version(name) for name in ['transformers', 'numpy', 'pandas', 'pyarrow', 'scikit-learn', 'safetensors']},
        'source_sha256': source_hashes, 'asset_manifest_sha256': file_hash(args.asset_dir / 'asset_manifest.json'),
    }
    dump(args.out / 'environment.json', environment)
    model, head_hash = make_model(args.asset_dir, args.method, args.seed)
    frozen_before = parameter_hash(model, frozen_only=True)
    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())
    dump(args.out / 'trainable_parameters.json', {'names': trainable_names, 'trainable': trainable_count, 'total': total_count, 'initial_head_sha256': head_hash})
    torch.cuda.reset_peak_memory_stats()
    model.cuda()
    learning_rate = protocol['learning_rates']['lora' if args.method.startswith('lora') else args.method]
    groups = []
    for decay in [True, False]:
        selected = [p for name, p in model.named_parameters() if p.requires_grad and (not any(term in name for term in protocol['weight_decay_exclusions'])) == decay]
        groups.append({'params': selected, 'weight_decay': protocol['weight_decay'] if decay else 0.0})
    optimizer = torch.optim.AdamW(groups, lr=learning_rate, foreach=False)
    count = len(data['train_labels'])
    steps_per_epoch = math.ceil(count / protocol['effective_batch_size'])
    total_steps = protocol['epochs'] * steps_per_epoch
    warmup = math.ceil(total_steps * protocol['warmup_ratio'])
    def schedule(step):
        return step / max(1, warmup) if step < warmup else max(0.0, (total_steps - step) / max(1, total_steps - warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    features = {}
    feature_seconds = 0.0
    if args.method == 'head':
        feature_started = time.perf_counter()
        features = {split: encode_features(model, data, split, protocol['micro_batch_size']) for split in ['train', 'dev']}
        feature_seconds = time.perf_counter() - feature_started
    seed_all(args.seed)
    best_key = (-1.0, -1.0)
    best_epoch = None
    history = []
    step = 0
    training_started = time.perf_counter()
    print(json.dumps({'method': args.method, 'seed': args.seed, 'trainable': trainable_count, 'total': total_count, 'train_rows': count, 'epochs': protocol['epochs']}), flush=True)
    for epoch in range(1, protocol['epochs'] + 1):
        if args.method == 'head':
            model.eval()
            model.classifier.train()
        else:
            model.train()
        order = np.random.default_rng(args.seed + epoch * 100003).permutation(count)
        loss_sum = 0.0
        seen = 0
        for group in batches(order, protocol['effective_batch_size']):
            optimizer.zero_grad(set_to_none=True)
            for indices in batches(group, protocol['micro_batch_size']):
                labels = torch.from_numpy(data['train_labels'][indices]).cuda()
                with amp():
                    if args.method == 'head':
                        logits = model.classifier(features['train'][indices].cuda())
                        loss = torch.nn.functional.cross_entropy(logits, labels)
                    else:
                        loss = model(**tensor_inputs(data, 'train', indices), labels=labels).loss
                    weighted_loss = loss * (len(indices) / len(group))
                weighted_loss.backward()
                loss_sum += float(loss.detach()) * len(indices)
                seen += len(indices)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], protocol['gradient_clip_norm'])
            optimizer.step()
            scheduler.step()
            step += 1
            if step % 25 == 0:
                print(json.dumps({'epoch': epoch, 'step': step, 'loss': loss_sum / seen, 'training_seconds': time.perf_counter() - training_started}), flush=True)
            if args.pilot_steps and step >= args.pilot_steps:
                torch.cuda.synchronize()
                report = {'status': 'PILOT_ONLY', 'method': args.method, 'steps': step, 'training_seconds': time.perf_counter() - training_started, 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(), 'validation_evaluated': False}
                dump(args.out / 'pilot.json', report)
                print(json.dumps(report), flush=True)
                return
            if time.perf_counter() - started > protocol['resource_budget']['per_run_timeout_seconds']:
                raise TimeoutError('Frozen per-run time budget exceeded; preserve partial logs')
        dev_metrics, _, _ = evaluate(model, data, 'dev', protocol['micro_batch_size'], features.get('dev'))
        row = {'epoch': epoch, 'training_loss': loss_sum / seen, 'dev': dev_metrics, 'training_seconds': time.perf_counter() - training_started}
        history.append(row)
        print(json.dumps(row), flush=True)
        key = (dev_metrics['binary_f1'], dev_metrics['accuracy'])
        if key > best_key:
            best_key, best_epoch = key, epoch
            tensors = {name: p.detach().cpu().contiguous() for name, p in model.named_parameters() if p.requires_grad}
            save_file(tensors, str(args.out / 'best.safetensors'))
        dump(args.out / 'history.json', history)
    training_seconds = time.perf_counter() - training_started
    frozen_after = parameter_hash(model, frozen_only=True)
    if frozen_after != frozen_before:
        raise AssertionError('Frozen encoder parameters changed')
    weights = load_file(str(args.out / 'best.safetensors'))
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in weights:
                p.copy_(weights[name])
    if args.method == 'head':
        features['validation'] = encode_features(model, data, 'validation', protocol['micro_batch_size'])
    metrics, logits, prediction = evaluate(model, data, 'validation', protocol['micro_batch_size'], features.get('validation'))
    development_metrics, development_logits, development_prediction = evaluate(model, data, 'dev', protocol['micro_batch_size'], features.get('dev'))
    training_metrics, _, _ = evaluate(model, data, 'train', protocol['micro_batch_size'], features.get('train'))
    with (args.out / 'dev_predictions.jsonl').open('w') as f:
        for idx, label, pred, scores in zip(data['dev_ids'], data['dev_labels'], development_prediction, development_logits, strict=True):
            f.write(json.dumps({'idx': int(idx), 'label': int(label), 'prediction': int(pred), 'logits': scores.tolist()}) + '\n')
    if args.method.startswith('lora'):
        diagnostics = []
        for name, module in model.named_modules():
            if hasattr(module, 'lora_A'):
                a = module.lora_A.detach().float().cpu()
                b = module.lora_B.detach().float().cpu()
                _, rb = torch.linalg.qr(b, mode='reduced')
                _, ra = torch.linalg.qr(a.T, mode='reduced')
                singular = torch.linalg.svdvals(rb @ ra.T) * module.scaling
                base_norm = float(torch.linalg.vector_norm(module.weight.detach().float().cpu()))
                update_norm = float(torch.linalg.vector_norm(singular))
                diagnostics.append({'module': name, 'singular_values': singular.tolist(), 'base_frobenius_norm': base_norm, 'update_frobenius_norm': update_norm, 'relative_update_norm': update_norm / base_norm})
        dump(args.out / 'adapter_diagnostics.json', diagnostics)
    with (args.out / 'predictions.jsonl').open('w') as f:
        for idx, label, pred, scores in zip(data['validation_ids'], data['validation_labels'], prediction, logits, strict=True):
            f.write(json.dumps({'idx': int(idx), 'label': int(label), 'prediction': int(pred), 'logits': scores.tolist()}) + '\n')
    # Rebuild from base and saved trainable weights, then verify inference restoration.
    del model, optimizer, scheduler, weights
    gc.collect()
    torch.cuda.empty_cache()
    restored, _ = make_model(args.asset_dir, args.method, args.seed)
    weights = load_file(str(args.out / 'best.safetensors'))
    with torch.no_grad():
        for name, p in restored.named_parameters():
            if name in weights:
                p.copy_(weights[name])
    restored.cuda()
    replay_metrics, replay_logits, replay_prediction = evaluate(restored, data, 'validation', protocol['micro_batch_size'], features.get('validation'))
    if not np.array_equal(prediction, replay_prediction) or not np.allclose(logits, replay_logits, rtol=0, atol=1e-6):
        raise AssertionError('Checkpoint restoration differs')
    summary = {
        'status': 'COMPLETE', 'method': args.method, 'seed': args.seed, 'epochs': args.epochs,
        'metrics': metrics, 'selected_epoch': best_epoch, 'selected_internal_dev': {'binary_f1': best_key[0], 'accuracy': best_key[1]},
        'restored_development_metrics': development_metrics, 'selected_model_training_metrics': training_metrics,
        'correct_tokenizer_guard_passed': True,
        'trainable_parameters': trainable_count, 'total_parameters': total_count,
        'trainable_fraction': trainable_count / total_count,
        'peak_torch_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
        'wall_seconds': time.perf_counter() - started, 'training_and_dev_seconds': training_seconds,
        'frozen_feature_preparation_seconds': feature_seconds,
        'frozen_parameter_hash_before': frozen_before, 'frozen_parameter_hash_after': frozen_after,
        'initial_classifier_sha256': head_hash,
        'best_checkpoint_sha256': file_hash(args.out / 'best.safetensors'),
        'prediction_sha256': file_hash(args.out / 'predictions.jsonl'),
        'checkpoint_restore_predictions_equal': True,
        'checkpoint_restore_max_logit_abs_error': float(np.max(np.abs(logits - replay_logits))),
        'validation_rows': len(prediction), 'test_evaluated': False,
        'protocol_sha256': file_hash(ROOT / 'protocol.json'),
    }
    dump(args.out / 'summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
