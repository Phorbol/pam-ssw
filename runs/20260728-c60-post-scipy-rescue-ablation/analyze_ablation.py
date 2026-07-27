#!/usr/bin/env python3
"""Analyze the fixed C60 post-SciPy accepted-endpoint rescue ablation."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


RUN_ROOT = Path(__file__).resolve().parent
RAW_DIR = RUN_ROOT / "output"
EXPECTED_TASK_COUNT = 143
ARMS = (
    ("scipy-lbfgsb-restart", "scipy-lbfgsb", None),
    ("safe-lbfgs-total-rescue", "safe-lbfgs-total", 10),
)
SCIPY_ARM = ARMS[0][0]
SAFE_ARM = ARMS[1][0]
INITIAL_FORCE_THRESHOLDS = (0.10, 0.05, 0.01)
SAME_BASIN_ENERGY_TOL_EV = 1.0e-3
SAME_BASIN_KABSCH_RMSD_TOL_A = 0.15


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _positions(value: object, label: str) -> np.ndarray:
    coordinates = np.asarray(value, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3 or coordinates.shape[0] == 0:
        raise ValueError(f"{label} must have shape (n_atoms, 3)")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError(f"{label} must be finite")
    return coordinates


def kabsch_rmsd(first: object, second: object) -> float:
    source = _positions(first, "first positions")
    target = _positions(second, "second positions")
    if source.shape != target.shape:
        raise ValueError("paired terminal positions have different shapes")
    source_centered = source - source.mean(axis=0, keepdims=True)
    target_centered = target - target.mean(axis=0, keepdims=True)
    covariance = source_centered.T @ target_centered
    left, _, right_t = np.linalg.svd(covariance)
    determinant = float(np.linalg.det(right_t.T @ left.T))
    correction = np.diag(
        [1.0, 1.0, np.sign(determinant) if determinant != 0.0 else 1.0]
    )
    rotation = left @ correction @ right_t
    difference = source_centered @ rotation - target_centered
    return float(np.sqrt(np.mean(np.sum(difference * difference, axis=1))))


def _certificate(force: float, threshold: float) -> bool:
    return bool(threshold > 0.0 and 0.0 <= force <= threshold)


def _validate_summary(summary: object) -> Mapping[str, Any]:
    if not isinstance(summary, Mapping):
        raise ValueError("summary.json must contain an object")
    if (
        summary.get("schema_version") != 1
        or summary.get("task_count") != EXPECTED_TASK_COUNT
        or summary.get("row_count") != EXPECTED_TASK_COUNT * len(ARMS)
    ):
        raise ValueError("summary does not describe the fixed 143-task, 286-row matrix")
    expected_arms = [
        {
            "arm_id": arm_id,
            "optimizer": optimizer,
            "safe_history_limit": history_limit,
        }
        for arm_id, optimizer, history_limit in ARMS
    ]
    if summary.get("arms") != expected_arms:
        raise ValueError("summary arms do not match the fixed two-arm contract")
    protocol = summary.get("protocol")
    if not isinstance(protocol, Mapping) or protocol != {
        "fmax_eV_per_A": 0.01,
        "maxiter": 400,
        "coordinate_trust_radius_A": None,
        "objective": "true_mace_pes_no_bias_no_softening",
        "task_selection": "all_accepted_endpoints_without_filtering",
    }:
        raise ValueError("summary protocol does not match the fixed rescue contract")
    if summary.get("interpretation_scope") != (
        "post_scipy_accepted_endpoint_refinement_rescue"
    ):
        raise ValueError("summary interpretation scope is not post-SciPy rescue")
    return summary


def _validate_row(
    row: object,
    expected_optimizer: str,
    expected_history_limit: int | None,
) -> Mapping[str, Any]:
    if not isinstance(row, Mapping):
        raise ValueError("raw row must be an object")
    arm_id = row.get("arm_id")
    task_index = _integer(row.get("task_index"), "task_index")
    if not 0 <= task_index < EXPECTED_TASK_COUNT:
        raise ValueError("task_index is outside the fixed matrix")
    if (
        row.get("optimizer") != expected_optimizer
        or row.get("safe_history_limit") != expected_history_limit
        or row.get("fmax_eV_per_A") != 0.01
        or row.get("maxiter") != 400
        or row.get("coordinate_trust_radius_A") is not None
        or row.get("objective") != "true_mace_pes_no_bias_no_softening"
    ):
        raise ValueError(f"row {arm_id}/{task_index} violates the fixed protocol")
    if row.get("trial_index") is None or row.get("discovered_entry_id") is None:
        raise ValueError(f"row {arm_id}/{task_index} has no endpoint identity")
    initial = row.get("initial")
    final = row.get("final")
    telemetry = row.get("telemetry")
    purpose_delta = row.get("purpose_count_delta")
    if not all(isinstance(item, Mapping) for item in (initial, final, telemetry, purpose_delta)):
        raise ValueError(f"row {arm_id}/{task_index} lacks result components")
    _finite(initial.get("energy_eV"), "initial energy")
    _finite(initial.get("max_active_force_eV_per_A"), "initial force")
    _positions(initial.get("positions"), "initial positions")
    _finite(final.get("energy_eV"), "final energy")
    _finite(final.get("max_active_force_eV_per_A"), "final force")
    _positions(final.get("positions"), "final positions")
    evaluator_calls = _integer(row.get("evaluator_calls"), "evaluator_calls")
    telemetry_calls = _integer(telemetry.get("evaluator_calls"), "telemetry evaluator_calls")
    post_relax_calls = _integer(
        purpose_delta.get("post_relax_validation"),
        "post_relax_validation calls",
    )
    unattributed = _integer(purpose_delta.get("unattributed"), "unattributed calls")
    purpose_total = sum(
        _integer(value, f"purpose count {purpose}")
        for purpose, value in purpose_delta.items()
    )
    if not (
        evaluator_calls
        == telemetry_calls
        == post_relax_calls
        == purpose_total
        and unattributed == 0
    ):
        raise ValueError(
            f"counter and telemetry evaluator calls do not close for "
            f"{arm_id}/{task_index}"
        )
    return row


def _validated_rows(
    raw_dir: Path,
    summary: Mapping[str, Any],
) -> dict[tuple[str, int], Mapping[str, Any]]:
    rows = _load_json(raw_dir / str(summary["rows_file"]))
    if not isinstance(rows, list) or len(rows) != EXPECTED_TASK_COUNT * len(ARMS):
        raise ValueError("rows.json does not contain the fixed 286-row matrix")
    arm_contract = {
        arm_id: (optimizer, history_limit)
        for arm_id, optimizer, history_limit in ARMS
    }
    by_key: dict[tuple[str, int], Mapping[str, Any]] = {}
    for item in rows:
        if not isinstance(item, Mapping) or item.get("arm_id") not in arm_contract:
            raise ValueError("raw row has an unknown arm")
        arm_id = str(item["arm_id"])
        optimizer, history_limit = arm_contract[arm_id]
        row = _validate_row(item, optimizer, history_limit)
        key = (arm_id, int(row["task_index"]))
        if key in by_key:
            raise ValueError(f"duplicate raw row: {key}")
        by_key[key] = row
    expected = {
        (arm_id, task_index)
        for arm_id, _, _ in ARMS
        for task_index in range(EXPECTED_TASK_COUNT)
    }
    if set(by_key) != expected:
        raise ValueError("raw rows do not match the fixed two-arm matrix")
    for task_index in range(EXPECTED_TASK_COUNT):
        scipy_row = by_key[(SCIPY_ARM, task_index)]
        safe_row = by_key[(SAFE_ARM, task_index)]
        for field in (
            "trial_index",
            "discovered_entry_id",
            "seed_entry_id",
            "source_sha256",
        ):
            if scipy_row.get(field) != safe_row.get(field):
                raise ValueError(f"paired task {task_index} differs in {field}")
        if scipy_row["initial"]["positions_sha256"] != safe_row["initial"]["positions_sha256"]:
            raise ValueError(f"paired task {task_index} has different initial positions")
    return by_key


def analyze(raw_dir: Path) -> dict[str, Any]:
    summary = _validate_summary(_load_json(raw_dir / "summary.json"))
    by_key = _validated_rows(raw_dir, summary)
    certificate_counts: dict[str, dict[str, int]] = {}
    terminal_by_arm: dict[str, Any] = {}
    for arm_id, _, _ in ARMS:
        arm_rows = [
            by_key[(arm_id, task_index)]
            for task_index in range(EXPECTED_TASK_COUNT)
        ]
        certificate_counts[arm_id] = {
            f"{threshold:.2f}": sum(
                _certificate(
                    float(row["initial"]["max_active_force_eV_per_A"]),
                    threshold,
                )
                for row in arm_rows
            )
            for threshold in INITIAL_FORCE_THRESHOLDS
        }
        terminal_by_arm[arm_id] = {
            "force_certificate_0.01_count": sum(
                _certificate(
                    float(row["final"]["max_active_force_eV_per_A"]),
                    0.01,
                )
                for row in arm_rows
            ),
            "evaluator_calls": sum(int(row["evaluator_calls"]) for row in arm_rows),
            "termination_reasons": dict(
                sorted(Counter(str(row["termination_reason"]) for row in arm_rows).items())
            ),
        }

    pairs = []
    same_basin_count = 0
    for task_index in range(EXPECTED_TASK_COUNT):
        scipy_row = by_key[(SCIPY_ARM, task_index)]
        safe_row = by_key[(SAFE_ARM, task_index)]
        energy_delta = abs(
            float(safe_row["final"]["energy_eV"])
            - float(scipy_row["final"]["energy_eV"])
        )
        rmsd = kabsch_rmsd(
            scipy_row["final"]["positions"],
            safe_row["final"]["positions"],
        )
        same_basin = bool(
            energy_delta <= SAME_BASIN_ENERGY_TOL_EV
            and rmsd <= SAME_BASIN_KABSCH_RMSD_TOL_A
        )
        same_basin_count += same_basin
        pairs.append(
            {
                "task_index": task_index,
                "trial_index": int(scipy_row["trial_index"]),
                "discovered_entry_id": int(scipy_row["discovered_entry_id"]),
                "absolute_terminal_energy_delta_eV": energy_delta,
                "terminal_kabsch_rmsd_A": rmsd,
                "same_basin_current_archive_semantics": same_basin,
            }
        )
    return {
        "schema_version": 1,
        "ledger": {
            "task_count": EXPECTED_TASK_COUNT,
            "row_count": EXPECTED_TASK_COUNT * len(ARMS),
            "counter_telemetry_closure_validated": True,
        },
        "initial_force_certificate_counts_by_arm": certificate_counts,
        "terminal_by_arm": terminal_by_arm,
        "terminal_pairing": {
            "same_basin_definition": (
                "abs_energy_delta_eV<=0.001_and_kabsch_rmsd_A<=0.15"
            ),
            "semantic_scope": "current_c60_archive_energy_plus_kabsch_thresholds",
            "same_basin_count": same_basin_count,
            "different_basin_count": EXPECTED_TASK_COUNT - same_basin_count,
            "pairs": pairs,
        },
        "claim_boundary": (
            "This is a post-SciPy accepted-endpoint refinement/rescue comparison, "
            "not a fair replacement comparison against the original landing process."
        ),
    }


def render_conclusion(evidence: Mapping[str, Any]) -> str:
    initial = evidence["initial_force_certificate_counts_by_arm"]
    terminal = evidence["terminal_by_arm"]
    pairing = evidence["terminal_pairing"]
    return (
        "# C60 post-SciPy accepted-endpoint refinement/rescue ablation\n\n"
        "This experiment re-relaxes all 143 accepted endpoints produced by the "
        "completed SciPy landing path. It is a post-SciPy accepted-endpoint "
        "refinement/rescue study, not a fair replacement comparison against the "
        "original landing process.\n\n"
        "## Initial force certificates\n\n"
        f"SciPy-arm initial counts at 0.10/0.05/0.01 eV/Å: "
        f"{initial[SCIPY_ARM]['0.10']}/{initial[SCIPY_ARM]['0.05']}/"
        f"{initial[SCIPY_ARM]['0.01']}. Safe-arm re-evaluations: "
        f"{initial[SAFE_ARM]['0.10']}/{initial[SAFE_ARM]['0.05']}/"
        f"{initial[SAFE_ARM]['0.01']}. These derived loose-threshold counts do "
        "not add a third optimizer arm; there is no third optimizer arm.\n\n"
        "## Terminal outcomes\n\n"
        f"0.01 eV/Å terminal certificates: SciPy restart "
        f"{terminal[SCIPY_ARM]['force_certificate_0.01_count']}/143; safe L-BFGS "
        f"rescue {terminal[SAFE_ARM]['force_certificate_0.01_count']}/143. Under "
        "the current archive semantics of |dE| <= 1e-3 eV plus Kabsch RMSD <= "
        f"0.15 Å, {pairing['same_basin_count']}/143 terminal pairs are same-basin.\n\n"
        "The basin label is limited to those current archive semantics. It is not "
        "a general chemical-equivalence certificate or evidence that either arm "
        "fairly replaces the original landing workflow.\n"
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=RUN_ROOT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = _parse_args(argv)
    evidence = analyze(arguments.raw_dir)
    conclusion = render_conclusion(evidence)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(arguments.output_dir / "evidence.json", evidence)
    (arguments.output_dir / "conclusion.md").write_text(
        conclusion,
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
