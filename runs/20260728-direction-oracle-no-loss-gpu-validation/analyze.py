#!/usr/bin/env python3
"""Check exact five-trial trajectories and reconstruct no-loss escape-check savings."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Sequence


RUN_ROOT = Path(__file__).resolve().parent
DEFAULT_REFERENCE_PATH = RUN_ROOT / "reference.json"
DEFAULT_OUTPUT_ROOT = RUN_ROOT / "output"
SYSTEMS = ("c60", "pdo")
MAX_TRIALS = 5


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _load_reference(reference_path: Path) -> dict[str, Any]:
    reference = _read_json(reference_path)
    if reference.get("schema_version") != 1 or reference.get("trial_count") != MAX_TRIALS:
        raise ValueError("reference schema or trial count mismatch")
    if set(reference.get("systems", ())) != set(SYSTEMS):
        raise ValueError("reference systems mismatch")
    return reference


def _consecutive_step_edges(direction_trace: list[dict[str, Any]]) -> int:
    last_step: dict[tuple[int, int], int] = {}
    edges = 0
    for row in direction_trace:
        key = (int(row["trial"]), int(row["proposal"]))
        step = int(row["step"])
        if last_step.get(key) == step - 1:
            edges += 1
        last_step[key] = step
    return edges


def _validate_native_candidate_only(summary: dict[str, Any]) -> None:
    config = summary["effective_config"]
    required = {
        "max_trials": MAX_TRIALS,
        "direction_selection_mode": "discrete",
        "direction_synthesis_mode": "none",
        "direction_probe_enabled": False,
        "plateau_evolution_enabled": False,
    }
    for key, value in required.items():
        if config.get(key) != value:
            raise ValueError(f"validation config mismatch: {key}")


def _analyze_system(reference: dict[str, Any], output_root: Path, system: str) -> dict[str, Any]:
    expected = reference["systems"][system]
    case_dir = output_root / system
    summary_path = case_dir / "summary.json"
    energy_path = case_dir / "energy_trace.json"
    walk_path = case_dir / "walk_records.json"
    direction_path = case_dir / "direction_trace.jsonl"
    summary = _read_json(summary_path)
    energy_trace = _read_json(energy_path)
    walk_records = _read_json(walk_path)
    direction_trace = _read_jsonl(direction_path)
    if summary.get("system") != system:
        raise ValueError(f"{system} summary identity mismatch")
    if summary.get("stats", {}).get("n_trials") != MAX_TRIALS:
        raise ValueError(f"{system} did not complete five trials")
    _validate_native_candidate_only(summary)
    for label, actual in (
        ("energy_trace", energy_trace),
        ("walk_records", walk_records),
        ("direction_trace", direction_trace),
    ):
        if actual != expected[label]:
            raise ValueError(f"{system} {label} differs from frozen old trajectory")

    purpose_counts = {key: int(value) for key, value in summary["purpose_counts"].items()}
    force_evaluations = int(summary["force_evaluations"])
    if sum(purpose_counts.values()) != force_evaluations:
        raise ValueError(f"{system} purpose accounting does not close")
    if purpose_counts.get("unattributed") != 0:
        raise ValueError(f"{system} purpose accounting contains unattributed evaluations")
    candidate_count = sum(int(row["candidate_count"]) for row in direction_trace)
    if purpose_counts["direction_oracle"] != 2 * candidate_count:
        raise ValueError(f"{system} direction oracle accounting does not match its central-HVP trace")
    mechanism_savings = {
        "true_curvature_hvp": 2 * len(direction_trace),
        "true_after_carry": _consecutive_step_edges(direction_trace),
    }
    escape_savings = sum(mechanism_savings.values())
    baseline_counts = dict(purpose_counts)
    baseline_counts["escape_true_pes_check"] += escape_savings
    savings = {key: baseline_counts[key] - value for key, value in purpose_counts.items()}
    if any(value != 0 for key, value in savings.items() if key != "escape_true_pes_check"):
        raise ValueError(f"{system} no-loss accounting claims a non-escape saving")
    if savings["escape_true_pes_check"] != escape_savings:
        raise ValueError(f"{system} escape saving does not match trace-derived mechanisms")
    return {
        "trajectory_exact": True,
        "old_execution_commit": expected["old_execution_commit"],
        "old_artifact_sha256": expected["old_artifact_sha256"],
        "new_raw_sha256": {
            "summary": _sha256(summary_path),
            "energy_trace": _sha256(energy_path),
            "walk_records": _sha256(walk_path),
            "direction_trace": _sha256(direction_path),
        },
        "force_evaluations": force_evaluations,
        "purpose_counts": purpose_counts,
        "counterfactual_baseline_purpose_counts": baseline_counts,
        "mechanism_savings": mechanism_savings,
        "savings": savings,
    }


def analyze(reference_path: Path, output_root: Path) -> dict[str, Any]:
    reference = _load_reference(Path(reference_path))
    return {
        "schema_version": 1,
        "trial_count": MAX_TRIALS,
        "reference_sha256": _sha256(Path(reference_path)),
        "cost_baseline_scope": (
            "counterfactual reconstruction from the exact frozen direction trace; "
            "the old 200-trial output did not persist per-trial purpose counts"
        ),
        "systems": {
            system: _analyze_system(reference, Path(output_root), system)
            for system in SYSTEMS
        },
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--evidence", type=Path, default=RUN_ROOT / "evidence.json")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    evidence = analyze(args.reference, args.output_root)
    args.evidence.write_text(
        json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
