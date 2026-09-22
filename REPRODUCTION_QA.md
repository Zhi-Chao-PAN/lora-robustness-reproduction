# Publication-time source and input checks (2026-09-23)

The public bundle was checked separately from the original training runs. `verify_public_bundle.py` passed with 21 completed run summaries, six PAWS stress summaries, and the four headline aggregate metrics recomputed from included summaries. `sha256sum -c SHA256SUMS` passed for every included file. All ten documented command-line scripts accepted `--help` in the pinned Python 3.12 environment.

We additionally ran the **published source files** against the retained original input assets, without copying those assets into this repository:

| Check | Result | Scope |
|---|---|---|
| `audit_tokenization.py` | `PASS`; 3,668 train and 408 validation encodings all matched the raw `tokenizer.json`; the legacy backend had zero BPE merges and the corrected backend had 50,000 | Input correctness on the retained MRPC snapshot |
| `check_implementation.py` | `PASS`; 24 query/value projections adapted, 887,042 trainable LoRA-r8 parameters including the classifier; zero-adapter logits matched and base weights remained frozen | Implementation invariants on the retained pretrained model |
| `verify_public_bundle.py` | `PASS`; 21/21 run summaries, six PAWS stress summaries, no prohibited source rows or model assets in the public package | Aggregate-record and package audit, **not** raw prediction recomputation |

This publication-time check did **not** retrain the 21 runs or independently download every upstream asset from the public README. The detailed frozen-result verification remains in the original local evidence package. Training outcomes depend on the recorded environment and data snapshot; a reviewer can attempt a fresh run with the README commands and upstream terms.

## Post-release addendum

After the fixed reviewer snapshot was published, one `lora_r8` seed-42 four-epoch run was trained from scratch using an unchanged copy of the published source and retained, hash-verified local assets. Its 24 non-timing summary fields matched the archived summary exactly. See [PUBLIC_SOURCE_REPLAY_2026-09-23.md](PUBLIC_SOURCE_REPLAY_2026-09-23.md) and run `python3 verify_public_source_replay.py` to audit the public records. This addendum does not change the tagged snapshot or the publication-time CI scope above; the new CI step checks the addendum offline without loading assets or training.

A second AI-agent-run replay started from a new asset directory and executed the unchanged `prepare_assets.py` against seven pinned public model/MRPC URLs. Seven observed sizes and SHA-256 values matched the fixed asset manifest; tokenizer, implementation, and post-run cache audits passed. The `lora_r8` seed-42 four-epoch result again matched the archived summary on all 24 non-timing fields and matched the first replay's four epoch records except elapsed time. See [PUBLIC_UPSTREAM_REPLAY_2026-09-23.md](PUBLIC_UPSTREAM_REPLAY_2026-09-23.md) and run `python3 verify_public_upstream_replay.py`. The download log is local execution evidence, not independent network attestation. Both replays used the same machine; neither re-ran the full matrix or demonstrates applicant hands-on ability.
