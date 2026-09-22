"""Sequential GPU training and compact evidence export for the frozen v4 matrix."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--asset-dir', type=Path, required=True)
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--export-dir', type=Path, required=True)
    args = p.parse_args()
    assets, runtime, export = [path.resolve() for path in (args.asset_dir, args.out_root, args.export_dir)]
    if runtime.exists() or export.exists():
        raise FileExistsError('Use new runtime and export directories')
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    for relative, digest in json.loads((ROOT / 'pre_training_manifest.json').read_text())['sha256'].items():
        if sha(ROOT / relative) != digest:
            raise ValueError('Frozen source changed: ' + relative)
    for record in json.loads((ROOT / 'asset_manifest.json').read_text())['assets']:
        path = assets / record['relative_path'] if 'relative_path' in record else ROOT / record['source_path']
        if sha(path) != record['sha256']:
            raise ValueError('Frozen asset changed: ' + str(path))
    runtime.mkdir(parents=True)
    export.mkdir(parents=True)
    (export / 'logs').mkdir()
    jobs = [{'method': method, 'seed': seed, 'epochs': epochs} for epochs in protocol['budget_epochs'] for seed in protocol['primary_seeds'] for method in protocol['primary_methods']]
    jobs += protocol['diagnostic_ablations']
    jobs += [{**protocol['planned_repeat'], 'repeat': True}]
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '4'}
    progress = []
    for job in jobs:
        name = f"{job['method']}-e{job['epochs']}-seed{job['seed']}" + ('-repeat' if job.get('repeat') else '')
        folder = runtime / 'runs' / name
        log = export / 'logs' / f'{name}.log'
        cmd = [sys.executable, str(ROOT / 'run_experiment.py'), '--asset-dir', str(assets), '--cache-dir', str(runtime / 'cache'), '--out', str(folder), '--method', job['method'], '--seed', str(job['seed']), '--epochs', str(job['epochs'])]
        print('START ' + name, flush=True)
        with log.open('w') as stream:
            subprocess.run(cmd, env=env, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=protocol['resource_budget']['per_run_timeout_seconds'] + 60)
        summary = json.loads((folder / 'summary.json').read_text())
        if summary['status'] != 'COMPLETE' or not summary['correct_tokenizer_guard_passed']:
            raise ValueError('Incomplete or incorrect tokenizer run')
        target = export / 'results' / name
        target.mkdir(parents=True)
        for path in folder.iterdir():
            if path.is_file() and not (job['method'] == 'full' and path.name == 'best.safetensors'):
                shutil.copy2(path, target / path.name)
        if job['method'] == 'full':
            (target / 'checkpoint_note.json').write_text(json.dumps({'sha256': summary['best_checkpoint_sha256'], 'restoration_verified': True, 'distribution': 'Full weight kept in original runtime, reconstruct using frozen method/seed/epochs'}, indent=2) + '\n')
        progress.append({'run': name, 'status': 'COMPLETE', 'metrics': summary['metrics'], 'selected_epoch': summary['selected_epoch'], 'wall_seconds': summary['wall_seconds']})
        (export / 'suite_progress.json').write_text(json.dumps(progress, indent=2) + '\n')
        print(json.dumps(progress[-1]), flush=True)


if __name__ == '__main__':
    main()
