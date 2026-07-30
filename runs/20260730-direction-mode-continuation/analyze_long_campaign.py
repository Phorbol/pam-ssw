#!/usr/bin/env python3
"""Analyze the seed42 200-step direction-transport campaign screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


RUN_ROOT = Path(__file__).resolve().parent
SYSTEMS = ("c60", "pdo")
ARMS = ("fixed_intent_ritz", "transported_direction")
SEED = 42
ENERGY_EQUIVALENCE_TOL_EV = 1.0e-3


def _validate(row: Mapping[str, Any]) -> None:
    purposes = row.get("purpose_counts")
    stats = row.get("stats")
    telemetry = row.get("optimizer_telemetry")
    if (
        not isinstance(purposes, Mapping)
        or not isinstance(stats, Mapping)
        or not isinstance(telemetry, Mapping)
        or int(row.get("completed_trials", -1)) != 200
        or int(stats.get("n_trials", -1)) != 200
        or int(purposes.get("unattributed", -1)) != 0
        or sum(int(value) for value in purposes.values())
        != int(row.get("force_evaluations", -1))
    ):
        raise ValueError("long-campaign row does not close")


def compare_system(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != 2 or {row.get("arm") for row in rows} != set(ARMS):
        raise ValueError("system comparison requires exactly two arms")
    for row in rows:
        _validate(row)
    by_arm = {str(row["arm"]): row for row in rows}
    control = by_arm["fixed_intent_ritz"]
    transported = by_arm["transported_direction"]
    lower_cost = bool(
        int(transported["force_evaluations"])
        < int(control["force_evaluations"])
    )
    final_not_worse = bool(
        float(transported["best_energy_eV"])
        <= float(control["best_energy_eV"]) + ENERGY_EQUIVALENCE_TOL_EV
    )
    complete_trajectories = bool(
        control.get("trajectory_complete", True)
        and transported.get("trajectory_complete", True)
        and control.get("mean_best_energy_improvement_eV") is not None
        and transported.get("mean_best_energy_improvement_eV") is not None
    )
    trajectory_not_worse = bool(
        complete_trajectories
        and float(transported["mean_best_energy_improvement_eV"])
        + ENERGY_EQUIVALENCE_TOL_EV
        >= float(control["mean_best_energy_improvement_eV"])
    )
    control_unconverged = int(
        control["optimizer_telemetry"].get("true_quench_unconverged", 0)
    )
    transported_unconverged = int(
        transported["optimizer_telemetry"].get("true_quench_unconverged", 0)
    )
    no_reliability_regression = bool(
        transported_unconverged <= control_unconverged
        and int(transported["stats"].get("fragment_rejections", 0))
        <= int(control["stats"].get("fragment_rejections", 0))
    )
    if (
        lower_cost
        and final_not_worse
        and trajectory_not_worse
        and no_reliability_regression
    ):
        decision = "pareto_dominates"
    elif lower_cost and final_not_worse and not complete_trajectories:
        decision = "cost_final_only_incomplete_trace"
    elif lower_cost and no_reliability_regression:
        decision = "cost_search_tradeoff"
    else:
        decision = "no_long_campaign_support"
    return {
        "decision": decision,
        "lower_cost": lower_cost,
        "final_best_not_worse": final_not_worse,
        "trajectory_complete": complete_trajectories,
        "trajectory_not_worse": trajectory_not_worse,
        "no_reliability_regression": no_reliability_regression,
        "force_evaluations_saved": (
            int(control["force_evaluations"])
            - int(transported["force_evaluations"])
        ),
        "direction_force_evaluations_saved": (
            int(control["purpose_counts"]["direction_oracle"])
            - int(transported["purpose_counts"]["direction_oracle"])
        ),
        "wall_time_saved_s": (
            float(control["wall_time_s"])
            - float(transported["wall_time_s"])
        ),
        "control": dict(control),
        "transported": dict(transported),
    }


def analyze(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected = {
        (system, SEED, arm)
        for system in SYSTEMS
        for arm in ARMS
    }
    observed = {
        (row.get("system"), row.get("seed"), row.get("arm"))
        for row in rows
    }
    if len(rows) != 4 or observed != expected:
        raise ValueError("seed42 screen requires the exact four campaigns")
    systems = {
        system: compare_system(
            [row for row in rows if row["system"] == system]
        )
        for system in SYSTEMS
    }
    if all(
        result["decision"] == "pareto_dominates"
        for result in systems.values()
    ):
        decision = "seed42_screen_survives"
    elif any(
        result["decision"] == "no_long_campaign_support"
        for result in systems.values()
    ):
        decision = "seed42_screen_rejects"
    else:
        decision = "seed42_screen_mixed"
    return {
        "schema_version": 1,
        "decision": decision,
        "systems": systems,
        "claim_ceiling": (
            "one 200-step campaign seed per system and arm; a survivor gate "
            "for multi-seed production, not a significance claim"
        ),
    }


def _markdown(evidence: Mapping[str, Any]) -> str:
    lines = [
        "# 200-step direction-transport seed42 screen",
        "",
        f"- Decision: `{evidence['decision']}`",
        "",
        "| system | decision | FE saved | direction FE saved | wall s saved |",
        "|---|---|---:|---:|---:|",
    ]
    for system in SYSTEMS:
        result = evidence["systems"][system]
        lines.append(
            f"| {system} | {result['decision']} | "
            f"{result['force_evaluations_saved']} | "
            f"{result['direction_force_evaluations_saved']} | "
            f"{result['wall_time_saved_s']:.3f} |"
        )
    lines.extend(["", f"Claim ceiling: {evidence['claim_ceiling']}.", ""])
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--conclusion", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = [
        json.loads(
            (
                args.root
                / f"{system}-seed{SEED}-{arm}"
                / "summary.json"
            ).read_text(encoding="utf-8")
        )
        for system in SYSTEMS
        for arm in ARMS
    ]
    evidence = analyze(rows)
    args.evidence.write_text(
        json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    args.conclusion.write_text(_markdown(evidence), encoding="utf-8")
    print(json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
