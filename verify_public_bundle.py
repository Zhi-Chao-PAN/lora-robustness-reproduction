"""Audit the small public bundle without downloading data or loading models."""

from __future__ import annotations

import json
import math
import re
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PRIMARY_METHODS = ("head", "full", "lora_r8")
PRIMARY_EPOCHS = (4, 12)
PRIMARY_SEEDS = (42, 123, 456)
FORBIDDEN_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".parquet", ".npz"}
ABSOLUTE_PATH = re.compile(
    r"(?:^|[\"'(=:\s])(?:/mnt/|/home/|/tmp/|[A-Za-z]:[\\/](?![\\/]))"
)
LOCAL_LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)|!\[[^\]]*\]\(([^)]+)\)")


def load(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def close(actual: float, expected: float, label: str, errors: list[str]) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        errors.append(f"{label}: {actual!r} != {expected!r}")


def main() -> int:
    errors: list[str] = []
    analysis = load("reports/analysis.json")
    progress = load("evidence/suite_progress.json")

    expected_primary = {
        f"{method}-e{epochs}-seed{seed}"
        for method in PRIMARY_METHODS
        for epochs in PRIMARY_EPOCHS
        for seed in PRIMARY_SEEDS
    }
    expected_extra = {
        "lora_r2-e12-seed42",
        "lora_r16-e12-seed42",
        "lora_r8-e4-seed42-repeat",
    }
    progress_names = {row["run"] for row in progress}
    if progress_names != expected_primary | expected_extra:
        errors.append("suite_progress.json does not contain the expected 21-run matrix")
    if any(row.get("status") != "COMPLETE" for row in progress):
        errors.append("one or more suite progress rows are incomplete")

    summaries = {}
    for name in sorted(progress_names):
        path = ROOT / "evidence" / "runs" / name / "summary.json"
        if not path.is_file():
            errors.append(f"missing run summary: {name}")
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        summaries[name] = row
        if row.get("status") != "COMPLETE":
            errors.append(f"incomplete run summary: {name}")
        if row.get("correct_tokenizer_guard_passed") is not True:
            errors.append(f"corrected-tokenizer guard absent or false: {name}")
        if row.get("test_evaluated") is not False:
            errors.append(f"unexpected MRPC test evaluation: {name}")

    aggregates = analysis["aggregates_complete_three_seed_only"]
    for method in PRIMARY_METHODS:
        for epochs in PRIMARY_EPOCHS:
            key = f"{method}-e{epochs}"
            values = [
                summaries[f"{key}-seed{seed}"]["metrics"]["binary_f1"]
                for seed in PRIMARY_SEEDS
                if f"{key}-seed{seed}" in summaries
            ]
            if len(values) == 3:
                close(statistics.mean(values), aggregates[key]["binary_f1"]["mean"], f"{key} mean F1", errors)

    headline = {
        "full-e12": 0.9081034721173129,
        "lora_r8-e12": 0.9073889667558408,
    }
    for key, expected in headline.items():
        close(aggregates[key]["binary_f1"]["mean"], expected, f"headline {key}", errors)

    stress_groups: dict[str, list[float]] = {"full": [], "lora_r8": []}
    for method in stress_groups:
        for seed in PRIMARY_SEEDS:
            relative = f"evidence/stress/{method}-e12-seed{seed}-summary.json"
            row = load(relative)
            if row.get("status") != "COMPLETE" or row.get("rows") != 8000:
                errors.append(f"bad stress summary: {relative}")
            if row.get("no_paws_training") is not True:
                errors.append(f"PAWS training exclusion missing: {relative}")
            stress_groups[method].append(row["metrics"]["balanced_accuracy"])
    close(statistics.mean(stress_groups["full"]), 0.5003029092806132, "full PAWS mean balanced accuracy", errors)
    close(statistics.mean(stress_groups["lora_r8"]), 0.5009418400666029, "LoRA PAWS mean balanced accuracy", errors)

    if analysis.get("expected_runs") != 21 or analysis.get("found_runs") != 21:
        errors.append("analysis does not report 21/21 runs")
    if analysis.get("missing_runs") or analysis.get("problems"):
        errors.append("analysis reports missing runs or problems")

    for path in ROOT.rglob("*"):
        if not path.is_file() or path.name in {"SHA256SUMS", Path(__file__).name}:
            continue
        relative = path.relative_to(ROOT).as_posix()
        lower = path.name.lower()
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or "predictions.jsonl" in lower:
            errors.append(f"forbidden artifact included: {relative}")
        if path.stat().st_size > 5 * 1024 * 1024:
            errors.append(f"unexpected file over 5 MiB: {relative}")
        if path.suffix.lower() in {".md", ".py", ".json", ".txt"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            for line_number, line in enumerate(text.splitlines(), 1):
                if ABSOLUTE_PATH.search(line):
                    errors.append(f"local absolute path in {relative}:{line_number}")

    for path in ROOT.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        for match in LOCAL_LINK.finditer(text):
            target = match.group(1) or match.group(2)
            target = target.strip().split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:") or "@" in target:
                continue
            if not (path.parent / target).resolve().exists():
                errors.append(f"broken local link in {path.relative_to(ROOT)}: {target}")

    status = "PASS" if not errors else "FAIL"
    print(json.dumps({
        "status": status,
        "checks": {
            "run_matrix": len(progress_names),
            "run_summaries": len(summaries),
            "stress_summaries": sum(len(v) for v in stress_groups.values()),
            "headline_metrics_recomputed": 4,
            "publication_exclusions_checked": True,
            "local_paths_checked": True,
            "local_markdown_links_checked": True,
        },
        "errors": errors,
        "scope": "Included aggregate evidence only; no raw-prediction audit, model loading, or retraining.",
    }, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())

