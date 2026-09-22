"""Check the published one-configuration source replay without model assets."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPLAY = ROOT / "evidence/public_source_replay_20260923"
ORIGINAL = ROOT / "evidence/runs/lora_r8-e4-seed42"
TIMING_KEYS = {
    "wall_seconds",
    "training_and_dev_seconds",
    "frozen_feature_preparation_seconds",
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    errors: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    replay = read_json(REPLAY / "summary.json")
    original = read_json(ORIGINAL / "summary.json")
    history = read_json(REPLAY / "history.json")
    environment = read_json(REPLAY / "environment.json")
    tokenization = read_json(REPLAY / "tokenization_audit.json")
    implementation = read_json(REPLAY / "implementation_check.json")
    cache = read_json(REPLAY / "cache_verification.json")
    asset_observation = read_json(REPLAY / "asset_verification.json")
    trace = [json.loads(line) for line in (REPLAY / "training_trace.jsonl").read_text(encoding="utf-8").splitlines()]
    manifest = read_json(ROOT / "pre_training_manifest.json")["sha256"]
    assets = read_json(ROOT / "asset_manifest.json")["assets"]

    check(set(replay) == set(original), "replay and original summary fields differ")
    substantive = sorted(set(replay) - TIMING_KEYS)
    check(len(substantive) == 24, "unexpected substantive-field count")
    for key in sorted(set(replay) & set(original) - TIMING_KEYS):
        check(replay[key] == original[key], f"summary differs: {key}")

    check(
        (replay["status"], replay["method"], replay["seed"], replay["epochs"])
        == ("COMPLETE", "lora_r8", 42, 4),
        "unexpected run configuration or status",
    )
    check(replay["correct_tokenizer_guard_passed"] is True, "tokenizer guard failed")
    check(replay["test_evaluated"] is False, "MRPC test was evaluated")
    check(replay["selected_epoch"] == 4, "unexpected selected epoch")
    check(replay["validation_rows"] == 408, "unexpected validation row count")
    check(replay["frozen_parameter_hash_before"] == replay["frozen_parameter_hash_after"], "base parameter hash changed")
    check(replay["checkpoint_restore_predictions_equal"] is True, "checkpoint restore predictions differ")
    check(replay["checkpoint_restore_max_logit_abs_error"] == 0.0, "checkpoint restore logit error")

    check(len(history) == 4, "expected four epoch records")
    for epoch, row in enumerate(history, 1):
        check(row["epoch"] == epoch, f"bad history epoch {epoch}")
    check(history[-1]["dev"] == replay["selected_internal_dev"], "selected development result differs from history")
    check(history[-1]["training_seconds"] <= replay["training_and_dev_seconds"], "bad training timing")
    check(
        all(history[i]["dev"]["binary_f1"] <= history[-1]["dev"]["binary_f1"] for i in range(3)),
        "selected epoch is not the best reported development F1",
    )

    epoch_trace = [row for row in trace if "epoch" in row and "training_loss" in row]
    check(len(epoch_trace) == 4, "training trace lacks four epoch summaries")
    for epoch, (logged, recorded) in enumerate(zip(epoch_trace, history), 1):
        check(logged == recorded, f"training trace and history differ at epoch {epoch}")
    check(trace[-1] == replay, "training trace final summary differs")

    check(tokenization["status"] == "PASS", "tokenization audit failed")
    check(tokenization["backend_merges"] == {"legacy": 0, "fixed": 50000}, "tokenizer merge count differs")
    check(tokenization["splits"]["train"]["rows"] == 3668, "tokenization train row count differs")
    check(tokenization["splits"]["validation"]["rows"] == 408, "tokenization validation row count differs")
    check(all(s["all_fixed_rows_equal_raw_asset"] is True for s in tokenization["splits"].values()), "fixed tokenization differs from raw asset")
    tokenizer = next(row for row in assets if row.get("relative_path") == "model/tokenizer.json")
    check(tokenization["tokenizer_asset_sha256"] == tokenizer["sha256"], "tokenizer asset hash differs")

    check(implementation["status"] == "PASS", "implementation audit failed")
    for field in (
        "base_parameters_receive_no_gradient",
        "both_low_rank_factors_receive_gradient",
        "base_weights_unchanged_after_step",
        "zero_adapter_real_model_logits_identical",
        "same_seed_classifier_initialization_identical",
    ):
        check(implementation[field] is True, f"implementation invariant failed: {field}")
    check(implementation["published_formula_max_abs_error"] == 0.0, "LoRA formula error")
    check(implementation["query_value_layers_adapted"] == 24, "query/value layer count differs")
    check(implementation["lora_r8_trainable_parameters"] == replay["trainable_parameters"], "trainable parameter count differs")

    check(cache["status"] == "PASS", "post-run cache audit failed")
    check(cache["total_rows"] == 4076, "cache row count differs")
    check(sum(cache["rows_checked"].values()) == 4076, "cache split row count differs")
    check(cache["input_ids_attention_masks_and_labels_identical_to_independent_source"] is True, "cache differs from source")
    check(cache["all_split_pair_overlaps_zero"] is True, "cache split overlap")

    check(environment == read_json(ORIGINAL / "environment.json"), "run environment differs")
    for source_name, source_hash in environment["source_sha256"].items():
        check(sha256(ROOT / source_name) == source_hash, f"current source hash differs: {source_name}")
    check(replay["protocol_sha256"] == sha256(ROOT / "protocol.json"), "protocol hash differs")
    check(environment["asset_manifest_sha256"] == sha256(ROOT / "asset_manifest.json"), "asset manifest hash differs")
    check(asset_observation["status"] == "PASS", "asset observation failed")
    check(asset_observation["manifest_sha256"] == sha256(ROOT / "asset_manifest.json"), "observed asset manifest hash differs")
    published_assets = {
        row["relative_path"]: {"bytes": row["bytes"], "sha256": row["sha256"]}
        for row in assets if "relative_path" in row
    }
    observed_assets = {
        row["relative_path"]: {"bytes": row["bytes"], "sha256": row["sha256"]}
        for row in asset_observation["assets"]
    }
    check(observed_assets == published_assets, "observed asset sizes or hashes differ from manifest")
    for source_name in ("run_experiment.py", "protocol.json", "tokenization.py"):
        check(manifest[source_name] == environment["source_sha256"][source_name], f"frozen source manifest differs: {source_name}")

    print(json.dumps({
        "status": "PASS" if not errors else "FAIL",
        "substantive_summary_fields_equal": len(substantive) if not errors else None,
        "timing_fields_excluded_from_equality": sorted(TIMING_KEYS),
        "epoch_records": len(history),
        "tokenization_rows_checked": 4076,
        "post_run_cache_rows_checked": cache["total_rows"],
        "asset_observation_records": len(observed_assets),
        "scope": "Offline audit of published replay records; CI cannot re-hash unpublished local assets and does not retrain.",
        "errors": errors,
    }, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
