"""Evaluate all predeclared MRPC-selected long-budget models on PAWS once."""
import argparse
import gc
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef, recall_score
import run_experiment as training
from tokenization import load_tokenizer

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--asset-dir', type=Path, required=True)
    p.add_argument('--paws-dir', type=Path, required=True)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--out-root', type=Path, required=True)
    a = p.parse_args()
    if a.out_root.exists():
        raise FileExistsError('Use a new stress-evaluation directory')
    manifest = json.loads((a.paws_dir / 'manifest.json').read_text())
    for record in manifest['assets']:
        if training.file_hash(a.paws_dir / record['file']) != record['sha256']:
            raise ValueError('Stress asset hash mismatch')
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    # Lock every selected checkpoint before reading/evaluating stress examples.
    selected = []
    for seed in protocol['primary_seeds']:
        for method in ('full', 'lora_r8'):
            name = f'{method}-e12-seed{seed}'
            folder = a.run_root / name
            summary = json.loads((folder / 'summary.json').read_text())
            if summary['status'] != 'COMPLETE' or training.file_hash(folder / 'best.safetensors') != summary['best_checkpoint_sha256']:
                raise ValueError('Checkpoint is not complete or differs: ' + name)
            selected.append({'run': name, 'method': method, 'seed': seed, 'checkpoint_sha256': summary['best_checkpoint_sha256'], 'selected_mrpc_dev_epoch': summary['selected_epoch']})
    a.out_root.mkdir(parents=True)
    training.dump(a.out_root / 'locked_model_selection.json', selected)
    frame = pd.read_parquet(a.paws_dir / 'test.parquet')
    if len(frame) != 8000 or frame.id.nunique() != 8000 or set(frame.label) != {0, 1}:
        raise ValueError('Unexpected PAWS public test schema')
    tokenizer = load_tokenizer(a.asset_dir / 'model')
    encoded = tokenizer(frame.sentence1.tolist(), frame.sentence2.tolist(), truncation=True, max_length=128, padding='max_length', return_tensors='np')
    data = {f'validation_{key}': encoded[key].astype(np.int64) for key in ('input_ids', 'attention_mask')}
    data['validation_labels'] = frame.label.to_numpy(dtype=np.int64)
    labels = data['validation_labels']
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.7)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for item in selected:
        start = time.perf_counter()
        folder = a.out_root / item['run']
        folder.mkdir()
        model, _ = training.make_model(a.asset_dir, item['method'], item['seed'])
        weights = load_file(str(a.run_root / item['run'] / 'best.safetensors'))
        expected = {name for name, p in model.named_parameters() if p.requires_grad}
        if set(weights) != expected:
            raise ValueError('Checkpoint trainable key mismatch')
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in weights:
                    parameter.copy_(weights[name])
        model.cuda()
        _, logits, prediction = training.evaluate(model, data, 'validation', 8)
        with (folder / 'predictions.jsonl').open('w') as stream:
            for idx, label, pred, scores in zip(frame.id, labels, prediction, logits, strict=True):
                stream.write(json.dumps({'idx': int(idx), 'label': int(label), 'prediction': int(pred), 'logits': scores.tolist()}) + '\n')
        metrics = {'accuracy': float(accuracy_score(labels, prediction)), 'binary_f1': float(f1_score(labels, prediction, zero_division=0)), 'balanced_accuracy': float(balanced_accuracy_score(labels, prediction)), 'mcc': float(matthews_corrcoef(labels, prediction)), 'negative_recall': float(recall_score(labels, prediction, pos_label=0)), 'positive_recall': float(recall_score(labels, prediction, pos_label=1))}
        result = {**item, 'status': 'COMPLETE', 'split': 'PAWS-Wiki labeled_final public test', 'rows': len(frame), 'metrics': metrics, 'prediction_sha256': training.file_hash(folder / 'predictions.jsonl'), 'data_sha256': training.file_hash(a.paws_dir / 'test.parquet'), 'evaluator_sha256': training.file_hash(Path(__file__)), 'selection': 'Frozen before stress inference; MRPC internal development only', 'no_paws_training': True, 'wall_seconds': time.perf_counter() - start}
        training.dump(folder / 'summary.json', result)
        print(json.dumps(result), flush=True)
        del model, weights
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
