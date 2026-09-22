# Post-release single-configuration source replay (2026-09-23)

After the [fixed reviewer snapshot](https://github.com/Zhi-Chao-PAN/lora-robustness-reproduction/releases/tag/reviewer-snapshot-2026-09-23) was published, Codex ran one new GPU training job in a separate scratch directory. The source checkout was the unchanged snapshot commit `603b04714e4b8219b55976fe1a172c08acb0d42d`. The fixed tag and its original training records were not modified. This addendum is later evidence, not part of that tagged snapshot.

The job used `lora_r8`, seed `42`, four epochs on MRPC. Its pretrained model and MRPC files were **copied from retained local assets**, then checked byte-for-byte against [asset_manifest.json](asset_manifest.json). This was not a fresh upstream download or a cross-machine test. The public repository still excludes raw data, tokenized rows, prediction rows, and weights.

## What was checked before and after training

- The copied `model.safetensors` SHA-256 was `5bde1d28afb363d0103324efeb5afc8b2b397fe5e04beabb9b1ef355255ade81`. The MRPC train and validation file hashes were `61fd41301e0e244b0420c4350a170c8e7cf64740335fc875a4af2d79af0df0af` and `33c007dbf5bfa8463d87a13e6226df8c0fcf2596c2cd39d0f3bb79754e00f50f`. All seven retained input files matched the public manifest's sizes and hashes; the observed values are in [asset_verification.json](evidence/public_source_replay_20260923/asset_verification.json).
- [Tokenization audit](evidence/public_source_replay_20260923/tokenization_audit.json): all 3,668 MRPC train rows and 408 validation rows agreed with the raw pinned tokenizer asset. The old tokenizer path had zero BPE merges and is excluded from the corrected study.
- [Implementation check](evidence/public_source_replay_20260923/implementation_check.json): the LoRA formula check passed, base parameters received no gradient or weight change, both low-rank factors received gradient, and 24 query/value layers were adapted.
- [Post-run cache audit](evidence/public_source_replay_20260923/cache_verification.json): all 4,076 train, internal-development, and validation rows had matching token IDs, masks, and labels, with no split-pair overlap.

The isolated environment was Python 3.12.14, Torch 2.13.0+cu130, Transformers 5.15.1, and one NVIDIA GeForce RTX 4060 Laptop GPU. Exact package versions and frozen source hashes are in [environment.json](evidence/public_source_replay_20260923/environment.json). The relative training invocation, using the verified assets in a directory beside the source checkout, was:

```bash
../.venv/bin/python run_experiment.py \
  --asset-dir ../assets --cache-dir ../runtime/cache \
  --out ../runtime/runs/lora_r8-e4-seed42 \
  --method lora_r8 --seed 42 --epochs 4
```

## Result

The new [summary](evidence/public_source_replay_20260923/summary.json), [four-epoch history](evidence/public_source_replay_20260923/history.json), and [training trace](evidence/public_source_replay_20260923/training_trace.jsonl) are included. Epoch 4 was selected by the internal development split. MRPC public validation accuracy was `0.8627450980392157` and binary F1 was `0.8985507246376812` on 408 rows. The model had 887,042 trainable parameters. Frozen base-weight hashes before and after training were equal; checkpoint-restore predictions and logits matched exactly. The MRPC test split was not evaluated.

Compared with the [original archived run summary](evidence/runs/lora_r8-e4-seed42/summary.json), **all 24 non-timing top-level fields matched exactly**. The newly measured wall and training times differed, as expected, and are not an equality criterion. The [offline verifier](verify_public_source_replay.py) checks that comparison, the trace/history agreement, the audit records, and the source hashes:

```bash
python3 verify_public_source_replay.py
sha256sum -c SHA256SUMS
```

The public CI executes these offline checks. It checks that the published asset observations agree with the manifest, but it cannot re-hash the local assets because they are not in the repository; it does not fetch assets or retrain. This one same-machine, same-asset replay supports same-environment repeatability of one frozen configuration. It does not prove that all 21 runs reproduce elsewhere, that the observed method differences are statistically equivalent, or that the applicant personally trained or debugged the model without AI assistance.
