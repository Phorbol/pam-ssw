#!/usr/bin/env python3
"""Close the three-seed archive-scaled versus fixed-target experiment."""

from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.util
import json
import math
from pathlib import Path
from statistics import mean, median
import sys
from typing import Any, Mapping, Sequence


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
PROTOCOL_PATH = RUN_ROOT / "protocol.py"
SYSTEMS = ("c60", "pdo", "cuo")
TARGET_MODES = ("archive_scaled", "fixed_reference")
SEEDS = (46, 47, 48)


def _load_protocol():
    name = "_fixed_target_analysis_protocol"
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(name, PROTOCOL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load protocol from {PROTOCOL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


protocol = _load_protocol()


def _sha256(path: Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _display_path(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(Path(path).resolve())


def first_best_force_evaluations(
    best_energy_eV: float,
    accepted_rows: Sequence[Mapping[str, Any]],
    *,
    initial_energy_eV: float | None = None,
    bootstrap_force_evaluations: int = 0,
) -> int:
    if (
        initial_energy_eV is not None
        and math.isclose(
            float(best_energy_eV),
            float(initial_energy_eV),
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
    ):
        return int(bootstrap_force_evaluations)
    for row in accepted_rows:
        if math.isclose(
            float(row["best_energy"]),
            float(best_energy_eV),
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            return int(row["force_evaluations"])
    raise ValueError("recorded best energy was never attained in accepted trace")


def target_distribution(values: Sequence[float]) -> dict[str, float | int]:
    targets = [float(value) for value in values]
    if not targets or any(not math.isfinite(value) for value in targets):
        raise ValueError("target history must be non-empty and finite")
    return {
        "count": len(targets),
        "mean_eV": float(mean(targets)),
        "median_eV": float(median(targets)),
        "min_eV": min(targets),
        "max_eV": max(targets),
    }


def combine_cases(
    initial_cases: Sequence[Mapping[str, Any]],
    repeat_cases: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    combined = list(initial_cases) + list(repeat_cases)
    observed = [
        (str(row["system"]), int(row["seed"]), str(row["target_mode"]))
        for row in combined
    ]
    required = {
        (system, seed, mode)
        for system in SYSTEMS
        for seed in SEEDS
        for mode in TARGET_MODES
    }
    if len(observed) != len(required) or set(observed) != required:
        raise ValueError("combined cases do not contain the required matrix")
    indexed = {key: row for key, row in zip(observed, combined)}
    return [
        indexed[(system, seed, mode)]
        for system in SYSTEMS
        for seed in SEEDS
        for mode in TARGET_MODES
    ]


def _trace_path(evidence_path: Path, row: Mapping[str, Any]) -> Path:
    return (
        evidence_path.parent
        / str(row["system"])
        / f"seed-{int(row['seed']):08d}"
        / str(row["target_mode"])
        / "accepted_trace.json"
    )


def _compact_case(
    row: Mapping[str, Any],
    *,
    evidence_path: Path,
) -> dict[str, Any]:
    trace_path = _trace_path(evidence_path, row)
    accepted_rows = _read_json(trace_path)
    best_fe = first_best_force_evaluations(
        float(row["best_energy_eV"]),
        accepted_rows,
        initial_energy_eV=float(row["initial_energy_eV"]),
        bootstrap_force_evaluations=int(row["shared_bootstrap_force_evaluations"]),
    )
    return {
        "system": str(row["system"]),
        "seed": int(row["seed"]),
        "target_mode": str(row["target_mode"]),
        "initial_energy_eV": float(row["initial_energy_eV"]),
        "best_energy_eV": float(row["best_energy_eV"]),
        "energy_drop_eV": float(row["energy_drop_eV"]),
        "gain_auc_eV": float(row["gain_auc_eV"]),
        "first_best_force_evaluations": best_fe,
        "force_evaluations": int(row["force_evaluations"]),
        "budget_residual": int(row["campaign_force_budget"])
        - int(row["force_evaluations"]),
        "purpose_counts": {
            str(key): int(value) for key, value in row["purpose_counts"].items()
        },
        "completed_trials": int(row["completed_trials"]),
        "archive_entries": int(row["archive_entries"]),
        "duplicate_rate": float(row["duplicate_rate"]),
        "wall_time_s": float(row["wall_time_s"]),
        "target_distribution": target_distribution(
            row["completed_trial_target_history_eV"]
        ),
        "accepted_trace_sha256": _sha256(trace_path),
    }


def _sum_purposes(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    purposes = sorted(
        {
            str(purpose)
            for row in rows
            for purpose in row["purpose_counts"]
        }
    )
    return {
        purpose: sum(int(row["purpose_counts"].get(purpose, 0)) for row in rows)
        for purpose in purposes
    }


def analyze(
    initial_evidence_path: Path,
    repeat_evidence_path: Path,
) -> dict[str, Any]:
    initial_path = Path(initial_evidence_path).resolve()
    repeat_path = Path(repeat_evidence_path).resolve()
    initial = _read_json(initial_path)
    repeats = _read_json(repeat_path)
    combined = combine_cases(initial["cases"], repeats["cases"])
    source_by_seed = {46: initial_path, 47: repeat_path, 48: repeat_path}
    cases = [
        _compact_case(row, evidence_path=source_by_seed[int(row["seed"])])
        for row in combined
    ]
    decision = protocol.ut2_decision(cases)
    indexed = {
        (row["system"], row["seed"], row["target_mode"]): row
        for row in cases
    }
    blocks = []
    for system in SYSTEMS:
        for seed in SEEDS:
            scaled = indexed[(system, seed, "archive_scaled")]
            fixed = indexed[(system, seed, "fixed_reference")]
            auc_delta = fixed["gain_auc_eV"] - scaled["gain_auc_eV"]
            blocks.append(
                {
                    "system": system,
                    "seed": seed,
                    "winner_by_gain_auc": (
                        "fixed_reference" if auc_delta > 0.0 else "archive_scaled"
                    ),
                    "fixed_minus_scaled_gain_auc_eV": auc_delta,
                    "fixed_energy_advantage_eV": (
                        scaled["best_energy_eV"] - fixed["best_energy_eV"]
                    ),
                    "fixed_minus_scaled_first_best_force_evaluations": (
                        fixed["first_best_force_evaluations"]
                        - scaled["first_best_force_evaluations"]
                    ),
                    "fixed_minus_scaled_wall_time_s": (
                        fixed["wall_time_s"] - scaled["wall_time_s"]
                    ),
                }
            )
    by_system = {}
    for system in SYSTEMS:
        rows = [row for row in blocks if row["system"] == system]
        deltas = [row["fixed_minus_scaled_gain_auc_eV"] for row in rows]
        by_system[system] = {
            "fixed_wins": sum(value > 0.0 for value in deltas),
            "scaled_wins": sum(value <= 0.0 for value in deltas),
            "median_fixed_minus_scaled_gain_auc_eV": float(median(deltas)),
            "mean_fixed_minus_scaled_gain_auc_eV": float(mean(deltas)),
        }
    by_mode = {}
    for mode in TARGET_MODES:
        rows = [row for row in cases if row["target_mode"] == mode]
        by_mode[mode] = {
            "case_count": len(rows),
            "force_evaluations": sum(row["force_evaluations"] for row in rows),
            "wall_time_s": sum(row["wall_time_s"] for row in rows),
            "completed_trials": sum(row["completed_trials"] for row in rows),
            "archive_entries": sum(row["archive_entries"] for row in rows),
            "mean_gain_auc_eV": float(mean(row["gain_auc_eV"] for row in rows)),
            "mean_energy_drop_eV": float(mean(row["energy_drop_eV"] for row in rows)),
            "purpose_counts": _sum_purposes(rows),
        }
    return {
        "schema_version": 1,
        "raw_evidence": [
            {"path": _display_path(initial_path), "sha256": _sha256(initial_path)},
            {"path": _display_path(repeat_path), "sha256": _sha256(repeat_path)},
        ],
        "execution_commits": [
            str(initial["execution_commit"]),
            str(repeats["execution_commit"]),
        ],
        "new_force_evaluations": sum(row["force_evaluations"] for row in cases),
        "unattributed_force_evaluations": sum(
            row["purpose_counts"].get("unattributed", 0) for row in cases
        ),
        "decision": decision,
        "paired_blocks": blocks,
        "by_system": by_system,
        "by_mode": by_mode,
        "cases": cases,
        "production_default_changed": False,
        "claim_ceiling": (
            "three systems and three paired seeds at 20k force evaluations per arm; "
            "not a universal barrier-scale or long-horizon production result"
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-evidence", type=Path, required=True)
    parser.add_argument("--repeat-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    evidence = analyze(args.initial_evidence, args.repeat_evidence)
    _write_json(args.output, evidence)
    print(json.dumps(evidence["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
