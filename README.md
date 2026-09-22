# LoRA robustness after a tokenizer correction

This repository is an auditable research artifact for one bounded study: after
correcting the RoBERTa input pipeline, how do training budget and
parameterization affect MRPC performance and zero-shot behavior on PAWS-Wiki?

[Public evidence audit](https://github.com/Zhi-Chao-PAN/lora-robustness-reproduction/actions/workflows/public-audit.yml) checks the published file hashes, aggregate evidence, and two post-release single-run replay records on every push. It does not train in CI. The first [source replay](PUBLIC_SOURCE_REPLAY_2026-09-23.md) used retained, hash-verified assets; the later [upstream-fetch replay](PUBLIC_UPSTREAM_REPLAY_2026-09-23.md) ran from a new asset directory using the pinned public download script. Both were AI-agent runs on the same machine, and both matched the archived `lora_r8` seed-42 run on all 24 non-timing summary fields.

The study covers an 18-run primary matrix (classification head, full fine
tuning, and query/value LoRA-r8; 4 and 12 epochs; seeds 42, 123, and 456), two
single-seed rank diagnostics, and one deterministic repeat. It is a controlled
method study and exploratory extension. It does **not** reproduce every result
in the LoRA paper.

## Main result and boundary

At 12 epochs, mean MRPC F1 was 0.9074 for LoRA-r8 and 0.9081 for full fine
tuning. That small difference is descriptive and is not evidence of statistical
equivalence or universal parity. On PAWS-Wiki, the same frozen model selections
had mean balanced accuracy of 0.5009 and 0.5003, respectively. Both were near
the constant-classifier baseline of 0.5 and almost always predicted the positive
class. The ordinary MRPC result therefore did not transfer to reliable word
order discrimination in this stress test.

MRPC's public validation labels had already been observed before the corrected
extensions were run. All corrected MRPC comparisons are consequently reported
as exploratory. There are three training seeds, one internal split, one backbone,
fixed method-specific learning rates, and one out-of-distribution stress set.

The earlier v3 pipeline instantiated a tokenizer backend with zero BPE merges;
all 4,076 MRPC rows disagreed with the pinned `tokenizer.json`. The normal-model
interpretation of those runs was withdrawn. The corrected pipeline checks every
cached row against the raw tokenizer asset. See [CORRECTION.md](CORRECTION.md).

This work was implemented, executed, and analyzed with AI assistance. The
artifact does not claim unaided personal authorship or prove any individual's
independent mastery.

## Quick audit (no downloads, training, GPU, or third-party packages)

The public artifact includes aggregate run summaries and analysis outputs. It
does not include pretrained assets, raw datasets, prediction rows, or weights.

```bash
python3 verify_public_bundle.py
python3 verify_public_source_replay.py
python3 verify_public_upstream_replay.py
sha256sum -c SHA256SUMS
```

The first command independently checks the run matrix, corrected-tokenizer
guards, summary/analysis agreement for the headline MRPC and PAWS metrics,
publication exclusions, local-path hygiene, and the Markdown links that point
to local files. The next two commands check the published single-run replay
records against the archived summary, their audits, and frozen source hashes.
The upstream-fetch check also compares the local download observation to the
fixed asset manifest and log, but cannot authenticate the network transfer or
re-hash unpublished assets. These are audits of included evidence, not retraining.

The detailed report is [reports/研究分析报告.md](reports/研究分析报告.md). JSON values
used to generate it are in [reports/analysis.json](reports/analysis.json), and
the per-run aggregate records are under `evidence/`. The exact scope of
publication-time source and input checks is in [REPRODUCTION_QA.md](REPRODUCTION_QA.md).

## Full rerun

The original pinned environment used Python 3.12, CUDA/BF16, one RTX 4060
Laptop GPU, and the exact versions in `requirements-lock.txt`. Later package or
GPU versions may not be bitwise deterministic. A full run downloads about
500 MB of pretrained model assets plus datasets, then creates substantially
larger runtime checkpoints and predictions.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-lock.txt

.venv/bin/python prepare_assets.py --asset-dir ./assets
.venv/bin/python prepare_paws_mirror.py --out ./paws-data
.venv/bin/python audit_tokenization.py --asset-dir ./assets --out ./replay-tokenization-audit.json
.venv/bin/python check_implementation.py --asset-dir ./assets --out ./replay-implementation-checks.json

.venv/bin/python run_suite.py \
  --asset-dir ./assets \
  --out-root ./runtime-replay \
  --export-dir ./experiments-replay

.venv/bin/python verify_cache.py \
  --asset-dir ./assets \
  --cache-dir ./runtime-replay/cache \
  --out ./replay-cache-verification.json

.venv/bin/python audit_stress_input.py \
  --asset-dir ./assets \
  --paws-dir ./paws-data \
  --out ./replay-stress-input-audit.json

.venv/bin/python evaluate_stress.py \
  --asset-dir ./assets \
  --paws-dir ./paws-data \
  --run-root ./runtime-replay/runs \
  --out-root ./experiments-replay/stress

.venv/bin/python analyze_results.py \
  --asset-dir ./assets \
  --runtime-root ./runtime-replay \
  --results-dir ./experiments-replay/results \
  --paws-dir ./paws-data \
  --out ./analysis-replay

.venv/bin/python make_report.py \
  --analysis ./analysis-replay/analysis.json \
  --out ./analysis-replay/report.md
```

Use new output directories as shown. `run_suite.py` refuses an existing runtime
or export directory and checks the frozen source and asset hashes before
training. Four-epoch and twelve-epoch runs have separately defined schedules;
the first four epochs of a twelve-epoch run are not the four-epoch experiment.
PAWS is evaluated only after MRPC internal-development checkpoint selection and
is not used for training, epoch selection, or threshold tuning.

## Repository contents

- `protocol.json`, `pre_training_manifest.json`, and `asset_manifest.json`
  freeze the protocol, source, upstream revisions, and expected asset hashes.
- `run_experiment.py` and `run_suite.py` implement the training matrix.
- `audit_*.py`, `verify_cache.py`, and `check_implementation.py` check the input
  and LoRA implementation.
- `analyze_results.py` and `make_report.py` rebuild the full analysis after a
  rerun with predictions and runtime checkpoints present.
- `evidence/` contains aggregate, non-textual result records only.
- `reports/` contains the public analysis JSON, report, and figures.
- `vendor/loralib/` is a pinned MIT-licensed upstream snapshot.

## Data, weights, and licensing

Raw MRPC/PAWS records, tokenized caches, predictions, pretrained assets, and
trained weights are deliberately absent. This avoids redistributing data and
model artifacts whose permissions or downstream terms require separate review.
The scripts download pinned assets from their upstream locations and verify the
recorded hashes. Access remains subject to each upstream provider's current
terms and availability.

The vendored Microsoft LoRA library retains its MIT license in
`vendor/LICENSE.md`. PAWS's upstream notice is preserved in `PAWS-LICENSE.txt`.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for provenance. No license
has been selected for the original study code and reports, so no permission to
reuse those original portions is granted by this repository until the rights
holder adds an explicit project license.

