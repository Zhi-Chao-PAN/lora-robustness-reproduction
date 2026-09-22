# Post-release upstream-fetch single-configuration replay (2026-09-23)

An AI agent cloned the public source at `38d40cc245d083be94d54771baeed21d5b8e00a7` into a new WSL ext4 work directory and made a new Python 3.12 virtual environment. This is a **second post-release replay**. Unlike the earlier [retained-asset replay](PUBLIC_SOURCE_REPLAY_2026-09-23.md), the agent started with no asset directory and ran the unchanged `prepare_assets.py` against its seven pinned model and MRPC URLs. The local [download log](evidence/public_upstream_replay_20260923/download_log.txt) records seven fetches. All seven resulting files were re-hashed locally and matched the [fixed asset manifest](asset_manifest.json) in both byte count and SHA-256; the four vendored LoRA files came from the fixed public checkout, not a new download. The [observation record](evidence/public_upstream_replay_20260923/fresh_asset_verification.json) includes the URLs, hashes, sizes, and download-script hash. “Upstream-fetch” names the agent's recorded procedure; the public records do not independently authenticate the network transfer.

The downloaded `model.safetensors` was 498,818,054 bytes with SHA-256 `5bde1d28afb363d0103324efeb5afc8b2b397fe5e04beabb9b1ef355255ade81`. The MRPC train and validation files matched `61fd41301e0e244b0420c4350a170c8e7cf64740335fc875a4af2d79af0df0af` and `33c007dbf5bfa8463d87a13e6226df8c0fcf2596c2cd39d0f3bb79754e00f50f`. The new [tokenization audit](evidence/public_upstream_replay_20260923/tokenization_audit.json) checked all 3,668 train and 408 validation rows; the [implementation check](evidence/public_upstream_replay_20260923/implementation_check.json) passed the frozen-base, LoRA-gradient, zero-adapter, and 24 query/value-layer checks. The isolated environment matched the earlier replay: Python 3.12.14, Torch 2.13.0+cu130, Transformers 5.15.1, and one RTX 4060 Laptop GPU.

## Frozen run and observed result

The agent ran the published `run_experiment.py` unchanged with `--method lora_r8 --seed 42 --epochs 4`. MRPC train was split by the frozen policy into 2,934 gradient-training and 734 internal-development rows; the official 408-row validation was used only for final evaluation. Epoch 4 was selected on internal development. The new [summary](evidence/public_upstream_replay_20260923/summary.json) reports validation accuracy `0.8627450980392157`, binary F1 `0.8985507246376812`, 887,042 trainable parameters, and `COMPLETE` status. Base-parameter hashes before and after training match. Restored-checkpoint predictions and logits match the in-memory model exactly; the MRPC test split was not downloaded or evaluated. A [post-run cache audit](evidence/public_upstream_replay_20260923/cache_verification.json) compared all 4,076 tokenized rows with the independent raw tokenizer asset and found no split-pair overlaps.

All **24 non-timing top-level summary fields** match both the [original archived run](evidence/runs/lora_r8-e4-seed42/summary.json) and the [retained-asset replay](evidence/public_source_replay_20260923/summary.json) exactly, including the best-checkpoint and prediction-file SHA-256 values. All four [epoch records](evidence/public_upstream_replay_20260923/history.json) match the earlier replay after excluding elapsed time; the environment record also matches. New wall time was 100.9 seconds. This is a stronger asset-acquisition check for one configuration, **not** a repeat of the complete 21-run matrix or a cross-machine replication.

The commands used the following structure; each output directory was new and separate from the public checkout:

```bash
uv venv --python 3.12 ../.venv
uv pip install --python ../.venv/bin/python -r requirements-lock.txt
../.venv/bin/python prepare_assets.py --asset-dir ../assets
../.venv/bin/python audit_tokenization.py --asset-dir ../assets --out ../tokenization_audit.json
../.venv/bin/python check_implementation.py --asset-dir ../assets --out ../implementation_check.json
../.venv/bin/python run_experiment.py --asset-dir ../assets --cache-dir ../runtime/cache \
  --out ../runtime/runs/lora_r8-e4-seed42 --method lora_r8 --seed 42 --epochs 4
../.venv/bin/python verify_cache.py --asset-dir ../assets --cache-dir ../runtime/cache \
  --out ../cache_verification.json
```

Run `python3 verify_public_upstream_replay.py` and `sha256sum -c SHA256SUMS` to check the **published records** offline. Public CI executes those checks without downloading data or retraining. The log and observation record document the agent's local execution; they are not independent network attestation. Public readers can check the script, fixed URLs, hashes, and numerical consistency but cannot re-hash the unpublished local assets, predictions, or checkpoint from this repository. The new run remained on the **same physical machine and software versions**, public MRPC validation had already been observed in this project, and the work was executed by an AI agent. It does not prove the applicant personally performed training, that all configurations reproduce elsewhere, or that LoRA and full fine tuning are statistically equivalent. Third-party model and dataset files are not redistributed here.
