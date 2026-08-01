#!/usr/bin/env python3
"""Reconstruct macro-action uphill targets from an existing archive trace."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
DEFAULT_SOURCE = (
    REPO_ROOT
    / "runs"
    / "20260801-paired-continuation-restart-gate"
    / "output"
)


def sha256_file(path: Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def archive_scaled_target(
    energies: Sequence[float],
    *,
    fallback: float,
) -> float:
    values = np.asarray(energies, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("energies must be a nonempty one-dimensional sequence")
    if not np.all(np.isfinite(values)) or not math.isfinite(float(fallback)):
        raise ValueError("energies and fallback must be finite")
    if fallback <= 0.0:
        raise ValueError("fallback must be positive")
    if values.size < 2:
        return float(fallback)
    deltas = values - float(np.min(values))
    median = float(np.median(deltas))
    mad = float(np.median(np.abs(deltas - median)))
    positive = deltas[deltas > 1.0e-12]
    positive_median = float(np.median(positive)) if positive.size else 0.0
    scale = max(mad, positive_median)
    if scale <= 1.0e-12:
        return float(fallback)
    return float(np.clip(0.2 * scale, 0.05 * fallback, 5.0 * fallback))


def reconstruct_targets(
    *,
    bootstrap_energy_eV: float,
    completed_trials: int,
    accepted_rows: Sequence[Mapping[str, Any]],
    fallback_target_eV: float,
) -> dict[str, Any]:
    if completed_trials < 0:
        raise ValueError("completed_trials must be nonnegative")
    bootstrap = float(bootstrap_energy_eV)
    if not math.isfinite(bootstrap):
        raise ValueError("bootstrap energy must be finite")
    rows = [dict(row) for row in accepted_rows]
    indices = [int(row["trial_index"]) for row in rows]
    if indices != sorted(indices) or len(indices) != len(set(indices)):
        raise ValueError("accepted trial indices must be strictly increasing")
    if any(index < 1 or index > completed_trials for index in indices):
        raise ValueError("accepted trial lies outside completed trial range")
    accepted_by_trial: dict[int, float] = {}
    for row, trial_index in zip(rows, indices, strict=True):
        energy = float(row["energy"])
        if not math.isfinite(energy):
            raise ValueError("accepted energies must be finite")
        accepted_by_trial[trial_index] = energy

    archive_energies = [bootstrap]
    targets: list[float] = []
    for trial_index in range(1, completed_trials + 1):
        targets.append(
            archive_scaled_target(
                archive_energies,
                fallback=fallback_target_eV,
            )
        )
        accepted_energy = accepted_by_trial.get(trial_index)
        if accepted_energy is not None:
            archive_energies.append(accepted_energy)
    return {
        "completed_trial_targets_eV": targets,
        "next_attempt_target_eV": archive_scaled_target(
            archive_energies,
            fallback=fallback_target_eV,
        ),
        "reconstructed_archive_entries": len(archive_energies),
        "archive_energies_eV": archive_energies,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line
    ]


def analyze_case(
    raw_case: Mapping[str, Any],
    accepted_log: Path,
    *,
    expected_log_sha256: str | None = None,
) -> dict[str, Any]:
    accepted_log = Path(accepted_log)
    log_sha256 = sha256_file(accepted_log)
    if expected_log_sha256 is not None and log_sha256 != expected_log_sha256:
        raise ValueError("accepted-log SHA-256 mismatch")
    fallback = float(raw_case["effective_config"]["target_uphill_energy"])
    reconstructed = reconstruct_targets(
        bootstrap_energy_eV=float(raw_case["bootstrap_energy_eV"]),
        completed_trials=int(raw_case["completed_trials"]),
        accepted_rows=_read_jsonl(accepted_log),
        fallback_target_eV=fallback,
    )
    if reconstructed["reconstructed_archive_entries"] != int(
        raw_case["archive_entries"]
    ):
        raise ValueError("reconstructed archive size does not close")
    reported_next = float(raw_case["stats"]["adaptive_step_target"])
    if not math.isclose(
        reconstructed["next_attempt_target_eV"],
        reported_next,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("reconstructed next target does not close")
    targets = np.asarray(reconstructed["completed_trial_targets_eV"], dtype=float)
    if targets.size == 0:
        raise ValueError("case has no completed target history")
    away = ~np.isclose(targets, fallback, rtol=0.0, atol=1.0e-12)
    return {
        "system": str(raw_case["system"]),
        "starter_mode": str(raw_case["starter_mode"]),
        "seed": int(raw_case["seed"]),
        "accepted_log_path": accepted_log.as_posix(),
        "accepted_log_sha256": log_sha256,
        "completed_trials": int(raw_case["completed_trials"]),
        "archive_entries": int(raw_case["archive_entries"]),
        "archive_energy_range_eV": float(
            max(reconstructed["archive_energies_eV"])
            - min(reconstructed["archive_energies_eV"])
        ),
        "fallback_target_eV": fallback,
        "completed_trial_targets_eV": reconstructed[
            "completed_trial_targets_eV"
        ],
        "next_attempt_target_eV": reconstructed["next_attempt_target_eV"],
        "target_mean_eV": float(np.mean(targets)),
        "target_median_eV": float(np.median(targets)),
        "target_min_eV": float(np.min(targets)),
        "target_max_eV": float(np.max(targets)),
        "mean_target_reference_ratio": float(np.mean(targets) / fallback),
        "fraction_below_reference": float(np.mean(targets < fallback - 1.0e-12)),
        "fraction_equal_reference": float(np.mean(~away)),
        "fraction_above_reference": float(np.mean(targets > fallback + 1.0e-12)),
        "fraction_away_from_reference": float(np.mean(away)),
        "adaptive_step_multiplier": float(
            raw_case["stats"]["adaptive_step_multiplier"]
        ),
        "adaptive_progress_boost": float(
            raw_case["stats"]["adaptive_progress_boost"]
        ),
    }


def build_evidence(source: Path) -> dict[str, Any]:
    source = Path(source)
    raw_path = source / "evidence.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    cases = []
    for raw_case in raw["cases"]:
        accepted_log = (
            source
            / raw_case["system"]
            / f"seed-{int(raw_case['seed']):08d}"
            / raw_case["starter_mode"]
            / "accepted_structures.jsonl"
        )
        cases.append(analyze_case(raw_case, accepted_log))

    by_system: dict[str, dict[str, float | int]] = {}
    admitted_systems = []
    for system in sorted({row["system"] for row in cases}):
        rows = [row for row in cases if row["system"] == system]
        targets = np.asarray(
            [target for row in rows for target in row["completed_trial_targets_eV"]],
            dtype=float,
        )
        reference = float(rows[0]["fallback_target_eV"])
        away_fraction = float(
            np.mean(~np.isclose(targets, reference, rtol=0.0, atol=1.0e-12))
        )
        by_system[system] = {
            "completed_trial_count": int(targets.size),
            "target_mean_eV": float(np.mean(targets)),
            "target_median_eV": float(np.median(targets)),
            "target_min_eV": float(np.min(targets)),
            "target_max_eV": float(np.max(targets)),
            "fraction_away_from_reference": away_fraction,
        }
        if away_fraction > 0.75:
            admitted_systems.append(system)
    decision = "ADMIT_U_T1" if len(admitted_systems) >= 2 else "DO_NOT_ADMIT_U_T1"
    return {
        "schema_version": 1,
        "decision": decision,
        "new_force_evaluations": 0,
        "source_raw_evidence_path": raw_path.as_posix(),
        "source_raw_evidence_sha256": sha256_file(raw_path),
        "case_count": len(cases),
        "cases": cases,
        "aggregate_by_system": by_system,
        "systems_away_from_reference_above_75pct": admitted_systems,
        "all_multipliers_inactive": all(
            math.isclose(row["adaptive_step_multiplier"], 1.0)
            and math.isclose(row["adaptive_progress_boost"], 1.0)
            for row in cases
        ),
        "claim_ceiling": (
            "zero-FE reconstruction of the macro uphill target in twelve "
            "seed-45 S-CR1 traces; activity evidence, not fixed-target efficacy"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=RUN_ROOT / "evidence.json")
    args = parser.parse_args()
    evidence = build_evidence(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "decision": evidence["decision"],
        "aggregate_by_system": evidence["aggregate_by_system"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
