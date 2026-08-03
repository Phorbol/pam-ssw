#!/usr/bin/env python3
"""Analyze the paired low-rank residual-Ritz direction screen."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence


RUN_ROOT = Path(__file__).resolve().parent
RUNNER_PATH = RUN_ROOT / "run_residual_ritz_ablation.py"
SYSTEMS = ("c60", "pdo")
ARMS = (
    "transported_direction",
    "residual_ritz2",
    "fixed_intent_ritz",
)


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "_residual_ritz_protocol", RUNNER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _validate_case(row: Mapping[str, Any]) -> None:
    purposes = row.get("purpose_counts")
    if (
        row.get("status") != "completed"
        or row.get("arm") not in ARMS
        or row.get("system") not in SYSTEMS
        or not isinstance(purposes, Mapping)
        or int(row.get("force_evaluations", -1))
        != sum(int(value) for value in purposes.values())
        or int(purposes.get("unattributed", -1)) != 0
        or int(purposes.get("bootstrap_true_quench", 0)) != 0
        or not bool(row.get("landing_geometry_valid", False))
    ):
        raise ValueError("action row fails its execution or ledger contract")


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    later_directions = [
        direction
        for row in rows
        for direction in row.get("direction_trace", [])[1:]
    ]

    def optional_median(field: str) -> float | None:
        values = [
            float(direction[field])
            for direction in later_directions
            if direction.get(field) is not None
        ]
        return None if not values else float(statistics.median(values))

    return {
        "case_count": len(rows),
        "certificate_count": sum(
            bool(row.get("certificate", False)) for row in rows
        ),
        "new_basin_count": sum(
            bool(row.get("is_new_basin", False)) for row in rows
        ),
        "median_landing_delta_eV": float(
            statistics.median(
                float(row["landing_delta_eV"]) for row in rows
            )
        ),
        "median_escape_delta_eV": (
            None
            if any(
                row.get("escape_energy_eV") is None
                or row.get("starter_energy_eV") is None
                for row in rows
            )
            else float(
                statistics.median(
                    float(row["escape_energy_eV"])
                    - float(row["starter_energy_eV"])
                    for row in rows
                )
            )
        ),
        "total_force_evaluations": sum(
            int(row["force_evaluations"]) for row in rows
        ),
        "direction_force_evaluations": sum(
            int(row["purpose_counts"]["direction_oracle"])
            for row in rows
        ),
        "proposal_force_evaluations": sum(
            int(row["purpose_counts"]["biased_proposal_relax"])
            for row in rows
        ),
        "quench_force_evaluations": sum(
            int(row["purpose_counts"].get("landing_true_quench", 0))
            + int(row["purpose_counts"].get("terminal_true_quench", 0))
            for row in rows
        ),
        "wall_time_s": sum(
            float(row["generation_wall_time_s"])
            + float(row["quench_wall_time_s"])
            for row in rows
        ),
        "post_step_zero_direction_count": len(later_directions),
        "median_abs_consecutive_direction_cosine": optional_median(
            "selected_to_previous_selected_abs_cosine"
        ),
        "median_selected_curvature": optional_median(
            "selected_curvature"
        ),
        "median_true_curvature": optional_median("true_curvature"),
        "proposal_energy_exploded": sum(
            int(
                row.get("relaxation_diagnostics", {}).get(
                    "proposal_relax_outcome_energy_exploded", 0
                )
            )
            for row in rows
        ),
        "proposal_unconverged": sum(
            int(
                row.get("relaxation_diagnostics", {}).get(
                    "proposal_relax_unconverged", 0
                )
            )
            for row in rows
        ),
    }


def _paired(
    rows: Sequence[Mapping[str, Any]], other_arm: str
) -> dict[str, Any]:
    by_key = {
        (str(row["state_id"]), int(row["seed"]), str(row["arm"])): row
        for row in rows
    }
    wins = ties = losses = 0
    deltas = []
    for state_id, seed, arm in sorted(by_key):
        if arm != "residual_ritz2":
            continue
        residual = float(by_key[(state_id, seed, arm)]["landing_delta_eV"])
        other = float(
            by_key[(state_id, seed, other_arm)]["landing_delta_eV"]
        )
        difference = residual - other
        deltas.append(difference)
        if difference < -1.0e-12:
            wins += 1
        elif difference > 1.0e-12:
            losses += 1
        else:
            ties += 1
    return {
        "landing_delta_wins_ties_losses": [wins, ties, losses],
        "median_paired_landing_delta_difference_eV": float(
            statistics.median(deltas)
        ),
    }


def analyze_cases(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    runner = _load_runner()
    expected = {
        (
            str(case["system"]),
            str(case["state_id"]),
            int(case["seed"]),
            str(case["arm"]),
        )
        for system in SYSTEMS
        for case in runner.case_matrix(system)
    }
    observed = {
        (
            str(row["system"]),
            str(row["state_id"]),
            int(row["seed"]),
            str(row["arm"]),
        )
        for row in rows
    }
    if observed != expected or len(rows) != len(expected):
        raise ValueError("paired 90-action matrix is incomplete or duplicated")
    for row in rows:
        _validate_case(row)

    systems: dict[str, Any] = {}
    advances = []
    for system in SYSTEMS:
        system_rows = [
            row for row in rows if row["system"] == system
        ]
        arm_results = {
            arm: _summary(
                [row for row in system_rows if row["arm"] == arm]
            )
            for arm in ARMS
        }
        residual_vs_transport = _paired(
            system_rows, "transported_direction"
        )
        residual_vs_control = _paired(
            system_rows, "fixed_intent_ritz"
        )
        wins, _, losses = residual_vs_transport[
            "landing_delta_wins_ties_losses"
        ]
        residual = arm_results["residual_ritz2"]
        transport = arm_results["transported_direction"]
        control = arm_results["fixed_intent_ritz"]
        advance = bool(
            wins > losses
            and residual["median_landing_delta_eV"]
            <= transport["median_landing_delta_eV"]
            and transport["direction_force_evaluations"]
            < residual["direction_force_evaluations"]
            < control["direction_force_evaluations"]
            and residual["total_force_evaluations"]
            < control["total_force_evaluations"]
            and residual["certificate_count"]
            >= min(
                transport["certificate_count"],
                control["certificate_count"],
            )
        )
        advances.append(advance)
        systems[system] = {
            "arm_results": arm_results,
            "residual_vs_transport": residual_vs_transport,
            "residual_vs_control": residual_vs_control,
            "advance_gate": advance,
        }
    return {
        "schema_version": 1,
        "hypothesis": (
            "a two-HVP Rayleigh-Ritz correction in span(v, residual) "
            "recovers landing quality lost by pure transport at lower "
            "cost than a fresh fixed-intent Ritz solve"
        ),
        "systems": systems,
        "decision": (
            "advance_residual_ritz2"
            if all(advances)
            else "reject_residual_ritz2"
        ),
    }


def _closed_cost(
    record: Mapping[str, Any],
    *,
    allowed_positive_purposes: set[str],
) -> int:
    purposes = record.get("purpose_counts")
    if not isinstance(purposes, Mapping):
        raise ValueError("cost record lacks purpose counts")
    force_evaluations = int(record.get("force_evaluations", -1))
    if (
        force_evaluations != sum(int(value) for value in purposes.values())
        or int(purposes.get("unattributed", 0)) != 0
        or any(
            int(value) != 0
            for purpose, value in purposes.items()
            if purpose not in allowed_positive_purposes
        )
    ):
        raise ValueError("shared or bootstrap cost ledger does not close")
    return force_evaluations


def analyze_raw(
    c60_raw: Mapping[str, Any],
    pdo_raw: Mapping[str, Any],
) -> dict[str, Any]:
    execution_commit = str(c60_raw.get("execution_commit", ""))
    if (
        not execution_commit
        or execution_commit != str(pdo_raw.get("execution_commit", ""))
    ):
        raise ValueError("execution commit differs across systems")
    c60_model_sha = str(
        c60_raw.get("shared_provenance", {}).get("model_sha256", "")
    )
    pdo_model_sha = str(pdo_raw.get("model_sha256", ""))
    if not c60_model_sha or c60_model_sha != pdo_model_sha:
        raise ValueError("model sha256 differs across systems")

    c60_shared = c60_raw.get("shared_initial_directions")
    pdo_shared = pdo_raw.get("shared_initial_directions")
    bootstrap = pdo_raw.get("bootstrap")
    if (
        not isinstance(c60_shared, list)
        or len(c60_shared) != 20
        or not isinstance(pdo_shared, list)
        or len(pdo_shared) != 10
        or not isinstance(bootstrap, Mapping)
        or not bool(bootstrap.get("certificate", False))
        or not bool(bootstrap.get("geometry_valid", False))
    ):
        raise ValueError("shared direction or PdO bootstrap matrix is invalid")
    c60_shared_fe = sum(
        _closed_cost(
            record,
            allowed_positive_purposes={"direction_oracle"},
        )
        for record in c60_shared
    )
    pdo_shared_fe = sum(
        _closed_cost(
            record,
            allowed_positive_purposes={"direction_oracle"},
        )
        for record in pdo_shared
    )
    pdo_bootstrap_fe = _closed_cost(
        bootstrap,
        allowed_positive_purposes={
            "bootstrap_true_quench",
            "post_relax_validation",
        },
    )
    rows = [
        {**row, "system": system}
        for system, raw in (("c60", c60_raw), ("pdo", pdo_raw))
        for row in raw.get("cases", [])
    ]
    evidence = analyze_cases(rows)
    c60_action_fe = sum(
        int(row["force_evaluations"])
        for row in rows
        if row["system"] == "c60"
    )
    pdo_action_fe = sum(
        int(row["force_evaluations"])
        for row in rows
        if row["system"] == "pdo"
    )
    evidence["provenance"] = {
        "execution_commit": execution_commit,
        "model_sha256": c60_model_sha,
        "c60_raw_input_sha256": str(
            c60_raw.get("shared_provenance", {}).get(
                "raw_input_sha256", ""
            )
        ),
        "pdo_raw_input_sha256": str(
            pdo_raw.get("raw_input_sha256", "")
        ),
    }
    evidence["cost_scope"] = {
        "c60_action_fe": c60_action_fe,
        "c60_shared_direction_fe": c60_shared_fe,
        "c60_total_fe": c60_action_fe + c60_shared_fe,
        "pdo_action_fe": pdo_action_fe,
        "pdo_bootstrap_fe": pdo_bootstrap_fe,
        "pdo_shared_direction_fe": pdo_shared_fe,
        "pdo_total_fe": (
            pdo_action_fe + pdo_bootstrap_fe + pdo_shared_fe
        ),
    }
    return evidence


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c60-raw", type=Path, required=True)
    parser.add_argument("--pdo-raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    c60_raw = json.loads(args.c60_raw.read_text(encoding="utf-8"))
    pdo_raw = json.loads(args.pdo_raw.read_text(encoding="utf-8"))
    evidence = analyze_raw(c60_raw, pdo_raw)
    _write_json(args.output, evidence)
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
