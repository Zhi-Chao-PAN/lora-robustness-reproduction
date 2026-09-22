"""Audit the published fresh-upstream single-run record without downloading or training.

This checks internal consistency of the public evidence. The download log is an
agent-generated record, not independent network attestation; the underlying
model/data files are excluded from Git and cannot be re-hashed by this audit.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence/public_upstream_replay_20260923"
ARCHIVED = ROOT / "evidence/runs/lora_r8-e4-seed42"
RETAINED_REPLAY = ROOT / "evidence/public_source_replay_20260923"
TIMING_KEYS = {"wall_seconds", "training_and_dev_seconds", "frozen_feature_preparation_seconds"}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    errors: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    current = read_json(EVIDENCE / "summary.json")
    archived = read_json(ARCHIVED / "summary.json")
    retained = read_json(RETAINED_REPLAY / "summary.json")
    history = read_json(EVIDENCE / "history.json")
    retained_history = read_json(RETAINED_REPLAY / "history.json")
    environment = read_json(EVIDENCE / "environment.json")
    observation = read_json(EVIDENCE / "fresh_asset_verification.json")
    tokenization = read_json(EVIDENCE / "tokenization_audit.json")
    implementation = read_json(EVIDENCE / "implementation_check.json")
    cache = read_json(EVIDENCE / "cache_verification.json")
    manifest = read_json(ROOT / "asset_manifest.json")
    frozen = read_json(ROOT / "pre_training_manifest.json")["sha256"]
    protocol = read_json(ROOT / "protocol.json")

    check(set(current) == set(archived) == set(retained), "summary fields differ")
    substantive = sorted(set(current) - TIMING_KEYS)
    check(len(substantive) == 24, "unexpected non-timing summary field count")
    for key in substantive:
        check(current[key] == archived[key], f"archived summary differs: {key}")
        check(current[key] == retained[key], f"retained replay differs: {key}")
    check((current["status"], current["method"], current["seed"], current["epochs"]) ==
          ("COMPLETE", "lora_r8", 42, 4), "run status/configuration differs")
    check(current["test_evaluated"] is False, "MRPC test was evaluated")
    check(current["correct_tokenizer_guard_passed"] is True, "correct tokenizer guard failed")
    check(current["validation_rows"] == 408, "validation row count differs")
    check(current["frozen_parameter_hash_before"] == current["frozen_parameter_hash_after"],
          "frozen base parameters changed")
    check(current["checkpoint_restore_predictions_equal"] is True and
          current["checkpoint_restore_max_logit_abs_error"] == 0.0,
          "checkpoint restore differs")

    check(len(history) == len(retained_history) == 4, "expected four epoch records")
    for index, (new, old) in enumerate(zip(history, retained_history), 1):
        check(new["epoch"] == old["epoch"] == index, f"epoch order differs: {index}")
        check(set(new) == set(old), f"epoch fields differ: {index}")
        for key in set(new) - {"training_seconds"}:
            check(new[key] == old[key], f"non-timing history differs: epoch {index}, {key}")
    check(current["selected_epoch"] == 4 and current["selected_internal_dev"] == history[-1]["dev"],
          "selected internal development epoch differs")
    check(current["restored_development_metrics"] == current["selected_internal_dev"],
          "restored development metrics differ")

    check(environment == read_json(RETAINED_REPLAY / "environment.json"),
          "environment differs from the retained-asset replay")
    for name, digest in environment["source_sha256"].items():
        check(sha256(ROOT / name) == digest, f"source hash differs: {name}")
    check(current["protocol_sha256"] == sha256(ROOT / "protocol.json"),
          "protocol hash differs")
    check(environment["asset_manifest_sha256"] == sha256(ROOT / "asset_manifest.json"),
          "environment asset manifest hash differs")
    for name in ("run_experiment.py", "protocol.json", "requirements-lock.txt", "asset_manifest.json"):
        check(frozen[name] == sha256(ROOT / name), f"pre-training manifest differs: {name}")

    check(observation["status"] == "PASS", "asset observation status differs")
    check(observation["source_commit"] == "38d40cc245d083be94d54771baeed21d5b8e00a7",
          "observed source checkout differs")
    check(observation["asset_manifest_sha256"] == sha256(ROOT / "asset_manifest.json"),
          "observed asset manifest differs")
    check(observation["prepare_assets_source_sha256"] == sha256(ROOT / "prepare_assets.py"),
          "asset download script hash differs")
    check(observation["requirements_lock_sha256"] == sha256(ROOT / "requirements-lock.txt"),
          "requirements lock hash differs")
    check(observation["prepare_assets_log_sha256"] == sha256(EVIDENCE / "download_log.txt"),
          "download log hash differs")
    expected_assets = manifest["assets"]
    observed_assets = observation["records"]
    check(len(expected_assets) == len(observed_assets) == 11, "asset record count differs")
    expected_downloads: list[tuple[str, int]] = []
    for expected, observed in zip(expected_assets, observed_assets):
        path_key = "relative_path" if "relative_path" in expected else "source_path"
        check(observed["path"] == expected[path_key], f"asset path differs: {expected[path_key]}")
        check(observed["url"] == expected["url"], f"asset URL differs: {expected[path_key]}")
        check(observed["sha256"] == expected["sha256"] and observed["bytes"] == expected["bytes"],
              f"asset bytes/hash observation differs: {expected[path_key]}")
        expected_source = "agent_logged_fetch" if path_key == "relative_path" else "fixed_public_checkout"
        check(observed["source"] == expected_source, f"asset source label differs: {expected[path_key]}")
        if path_key == "relative_path":
            expected_downloads.append((Path(expected[path_key]).name, expected["bytes"]))
    check(observation["agent_logged_download_count"] == len(expected_downloads) == 7,
          "logged download count differs")
    check(observation["fixed_vendor_file_count"] == 4, "fixed vendor source count differs")
    log_lines = (EVIDENCE / "download_log.txt").read_text(encoding="utf-8").splitlines()
    expected_log = [line for name, size in expected_downloads
                    for line in (f"Downloading {name}", f"Saved {name} {size}")]
    check(log_lines == expected_log, "local download log differs from the expected seven-file sequence")
    check(protocol["model_revision"] in expected_assets[0]["url"] and
          protocol["dataset_revision"] in expected_assets[5]["url"],
          "download revisions differ from protocol")

    check(tokenization["status"] == "PASS" and
          tokenization["backend_merges"] == {"legacy": 0, "fixed": 50000},
          "tokenization audit differs")
    check(tokenization["splits"]["train"]["rows"] == 3668 and
          tokenization["splits"]["validation"]["rows"] == 408 and
          all(row["all_fixed_rows_equal_raw_asset"] is True
              for row in tokenization["splits"].values()),
          "tokenization row audit differs")
    check(implementation["status"] == "PASS" and
          implementation["query_value_layers_adapted"] == 24 and
          implementation["lora_r8_trainable_parameters"] == current["trainable_parameters"],
          "implementation audit differs")
    for field in ("base_parameters_receive_no_gradient", "both_low_rank_factors_receive_gradient",
                  "base_weights_unchanged_after_step", "zero_adapter_real_model_logits_identical",
                  "same_seed_classifier_initialization_identical"):
        check(implementation[field] is True, f"implementation invariant failed: {field}")
    check(implementation["published_formula_max_abs_error"] == 0.0, "LoRA formula error")
    check(cache["status"] == "PASS" and cache["total_rows"] == 4076 and
          sum(cache["rows_checked"].values()) == 4076 and
          cache["input_ids_attention_masks_and_labels_identical_to_independent_source"] is True and
          cache["all_split_pair_overlaps_zero"] is True,
          "post-run cache audit differs")

    result = {
        "status": "PASS" if not errors else "FAIL",
        "non_timing_summary_fields_equal": len(substantive) if not errors else None,
        "epoch_records_compared": len(history),
        "reported_downloads": len(expected_downloads),
        "tokenization_rows_checked": 4076,
        "cache_rows_checked": cache["total_rows"],
        "scope": "Offline record audit only; cannot authenticate the network download or re-hash unpublished assets, predictions, and weights; does not retrain.",
        "errors": errors,
    }
    print(json.dumps(result, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
