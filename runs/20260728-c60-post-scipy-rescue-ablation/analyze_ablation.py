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
FLOAT32_REFERENCE_ENERGY_EV = -500.0


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


def _percentile(values: Sequence[float | int], percentile: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    return float(np.percentile(np.asarray(values, dtype=float), percentile))


def _distribution(
    values: Sequence[float | int],
    *,
    total_as_integer: bool = False,
    max_as_integer: bool = False,
) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    total = sum(values)
    maximum = max(values)
    return {
        "total": int(total) if total_as_integer else float(total),
        "median": _percentile(values, 50.0),
        "p90": _percentile(values, 90.0),
        "max": int(maximum) if max_as_integer else float(maximum),
    }


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
    wall_time = _finite(row.get("wall_time_s"), "wall_time_s")
    if wall_time < 0.0:
        raise ValueError("wall_time_s must be non-negative")
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
        calls = [int(row["evaluator_calls"]) for row in arm_rows]
        terminal_forces = [
            float(row["final"]["max_active_force_eV_per_A"])
            for row in arm_rows
        ]
        energy_changes = [
            float(row["final"]["energy_eV"]) - float(row["initial"]["energy_eV"])
            for row in arm_rows
        ]
        terminal_by_arm[arm_id] = {
            "force_certificate_0.01_count": sum(
                _certificate(
                    float(row["final"]["max_active_force_eV_per_A"]),
                    0.01,
                )
                for row in arm_rows
            ),
            "evaluator_calls": _distribution(
                calls,
                total_as_integer=True,
                max_as_integer=True,
            ),
            "wall_time_s_total": float(
                sum(float(row["wall_time_s"]) for row in arm_rows)
            ),
            "terminal_force_eV_per_A": {
                "median": _percentile(terminal_forces, 50.0),
                "p90": _percentile(terminal_forces, 90.0),
                "max": max(terminal_forces),
            },
            "energy_change_eV": {
                "definition": "final_energy_eV_minus_initial_energy_eV",
                "total": float(sum(energy_changes)),
                "median": _percentile(energy_changes, 50.0),
                "p90": _percentile(energy_changes, 90.0),
                "min": min(energy_changes),
                "max": max(energy_changes),
                "decreased_count": sum(delta < 0.0 for delta in energy_changes),
                "unchanged_count": sum(delta == 0.0 for delta in energy_changes),
                "increased_count": sum(delta > 0.0 for delta in energy_changes),
            },
            "termination_reasons": dict(
                sorted(Counter(str(row["termination_reason"]) for row in arm_rows).items())
            ),
        }

    pairs = []
    same_basin_count = 0
    strict_contingency = {
        "safe_only": 0,
        "scipy_only": 0,
        "both": 0,
        "neither": 0,
    }
    for task_index in range(EXPECTED_TASK_COUNT):
        scipy_row = by_key[(SCIPY_ARM, task_index)]
        safe_row = by_key[(SAFE_ARM, task_index)]
        scipy_strict = _certificate(
            float(scipy_row["final"]["max_active_force_eV_per_A"]),
            0.01,
        )
        safe_strict = _certificate(
            float(safe_row["final"]["max_active_force_eV_per_A"]),
            0.01,
        )
        contingency_key = (
            "both"
            if scipy_strict and safe_strict
            else "scipy_only"
            if scipy_strict
            else "safe_only"
            if safe_strict
            else "neither"
        )
        strict_contingency[contingency_key] += 1
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

    safe_failures = [
        by_key[(SAFE_ARM, task_index)]
        for task_index in range(EXPECTED_TASK_COUNT)
        if by_key[(SAFE_ARM, task_index)]["termination_reason"]
        == "line_search_failed"
    ]
    failure_calls = [int(row["evaluator_calls"]) for row in safe_failures]
    safe_failure_summary = {
        "count": len(safe_failures),
        "evaluator_calls_total": sum(failure_calls),
        "evaluator_calls_median": (
            _percentile(failure_calls, 50.0) if failure_calls else None
        ),
        "accepted_steps_total": sum(
            _integer(row["telemetry"].get("accepted_steps"), "accepted_steps")
            for row in safe_failures
        ),
        "rejected_steps_total": sum(
            _integer(row["telemetry"].get("rejected_steps"), "rejected_steps")
            for row in safe_failures
        ),
        "line_search_evaluations_total": sum(
            _integer(
                row["telemetry"].get("line_search_evaluations"),
                "line_search_evaluations",
            )
            for row in safe_failures
        ),
    }

    all_energy_changes = [
        float(row["final"]["energy_eV"]) - float(row["initial"]["energy_eV"])
        for row in by_key.values()
    ]
    nonzero_energy_steps = [
        abs(delta) for delta in all_energy_changes if delta != 0.0
    ]
    if not nonzero_energy_steps:
        raise ValueError("all final-minus-initial energy changes are zero")
    minimum_energy_step = min(nonzero_energy_steps)
    float32_ulp = abs(
        float(np.spacing(np.float32(FLOAT32_REFERENCE_ENERGY_EV)))
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
        "safe_line_search_failed": safe_failure_summary,
        "energy_resolution": {
            "minimum_nonzero_absolute_final_minus_initial_energy_eV": (
                minimum_energy_step
            ),
            "float32_reference_energy_eV": FLOAT32_REFERENCE_ENERGY_EV,
            "float32_ulp_at_reference_energy_eV": float32_ulp,
            "minimum_step_equals_float32_ulp": minimum_energy_step == float32_ulp,
            "interpretation": (
                "This numerical match supports the inference that energy "
                "quantization may contribute to Armijo stalling near minima; "
                "it does not prove the root cause."
            ),
        },
        "terminal_pairing": {
            "same_basin_definition": (
                "abs_energy_delta_eV<=0.001_and_kabsch_rmsd_A<=0.15"
            ),
            "semantic_scope": "current_c60_archive_energy_plus_kabsch_thresholds",
            "same_basin_count": same_basin_count,
            "different_basin_count": EXPECTED_TASK_COUNT - same_basin_count,
            "strict_force_certificate_0.01_contingency": strict_contingency,
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
    scipy = terminal[SCIPY_ARM]
    safe = terminal[SAFE_ARM]
    failures = evidence["safe_line_search_failed"]
    resolution = evidence["energy_resolution"]
    contingency = pairing["strict_force_certificate_0.01_contingency"]
    failure_median = failures["evaluator_calls_median"]
    failure_median_text = "n/a" if failure_median is None else f"{failure_median:g}"
    minimum_step = resolution[
        "minimum_nonzero_absolute_final_minus_initial_energy_eV"
    ]
    float32_ulp = resolution["float32_ulp_at_reference_energy_eV"]
    if resolution["minimum_step_equals_float32_ulp"]:
        resolution_text = (
            f"The minimum non-zero absolute final-minus-initial energy step was "
            f"{minimum_step} eV, equal to the float32 ULP at approximately "
            f"-500 eV ({float32_ulp} eV). This numerical match supports the "
            "inference that energy quantization may contribute to Armijo stalling "
            "near minima; it does not prove the root cause."
        )
    else:
        resolution_text = (
            f"The minimum non-zero absolute final-minus-initial energy step was "
            f"{minimum_step} eV. It does not equal the float32 ULP at approximately "
            f"-500 eV ({float32_ulp} eV), so this comparison alone does not "
            "support the energy-quantization inference."
        )
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
        f"Safe L-BFGS reached {safe['force_certificate_0.01_count']}/143 versus "
        f"{scipy['force_certificate_0.01_count']}/143 for the SciPy restart, but "
        f"used {safe['evaluator_calls']['total']:,} versus "
        f"{scipy['evaluator_calls']['total']:,} evaluator calls. Per-task "
        "evaluator calls (median/p90/max) were "
        f"{safe['evaluator_calls']['median']:g}/"
        f"{safe['evaluator_calls']['p90']:g}/"
        f"{safe['evaluator_calls']['max']:g} for safe and "
        f"{scipy['evaluator_calls']['median']:g}/"
        f"{scipy['evaluator_calls']['p90']:g}/"
        f"{scipy['evaluator_calls']['max']:g} for SciPy; total wall times were "
        f"{safe['wall_time_s_total']:.6g} s and "
        f"{scipy['wall_time_s_total']:.6g} s, respectively. Terminal force "
        "(median/p90/max, eV/Å) was "
        f"{safe['terminal_force_eV_per_A']['median']:.6g}/"
        f"{safe['terminal_force_eV_per_A']['p90']:.6g}/"
        f"{safe['terminal_force_eV_per_A']['max']:.6g} for safe and "
        f"{scipy['terminal_force_eV_per_A']['median']:.6g}/"
        f"{scipy['terminal_force_eV_per_A']['p90']:.6g}/"
        f"{scipy['terminal_force_eV_per_A']['max']:.6g} for SciPy.\n\n"
        "The strict 0.01 eV/Å paired contingency "
        f"(safe-only/SciPy-only/both/neither) was "
        f"{contingency['safe_only']}/{contingency['scipy_only']}/"
        f"{contingency['both']}/{contingency['neither']}. "
        f"{failures['count']} line-search failures consumed "
        f"{failures['evaluator_calls_total']:,} evaluator calls "
        f"(median {failure_median_text}), with "
        f"{failures['accepted_steps_total']:,} accepted steps, "
        f"{failures['rejected_steps_total']:,} rejected steps, and "
        f"{failures['line_search_evaluations_total']:,} line-search "
        "evaluations.\n\n"
        "Initial-to-final energy changes (total/median/p90/min/max, eV) were "
        f"{safe['energy_change_eV']['total']:.12g}/"
        f"{safe['energy_change_eV']['median']:.12g}/"
        f"{safe['energy_change_eV']['p90']:.12g}/"
        f"{safe['energy_change_eV']['min']:.12g}/"
        f"{safe['energy_change_eV']['max']:.12g} for safe and "
        f"{scipy['energy_change_eV']['total']:.12g}/"
        f"{scipy['energy_change_eV']['median']:.12g}/"
        f"{scipy['energy_change_eV']['p90']:.12g}/"
        f"{scipy['energy_change_eV']['min']:.12g}/"
        f"{scipy['energy_change_eV']['max']:.12g} for SciPy.\n\n"
        "## Numerical resolution evidence\n\n"
        f"{resolution_text}\n\n"
        "Under "
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
