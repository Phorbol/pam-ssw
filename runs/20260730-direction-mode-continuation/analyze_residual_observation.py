#!/usr/bin/env python3
"""Analyze whether transported-mode residual predicts downstream cost."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence

import numpy as np


RUN_ROOT = Path(__file__).resolve().parent
RUNNER_PATH = RUN_ROOT / "run_residual_observation.py"
SYSTEMS = ("c60", "pdo")


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "_transport_residual_protocol",
        RUNNER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=float)
    start = 0
    while start < len(array):
        stop = start + 1
        while (
            stop < len(array)
            and array[order[stop]] == array[order[start]]
        ):
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _spearman(
    left: Sequence[float], right: Sequence[float]
) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_ranks = _ranks(left)
    right_ranks = _ranks(right)
    if (
        float(np.ptp(left_ranks)) == 0.0
        or float(np.ptp(right_ranks)) == 0.0
    ):
        return None
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def _median(values: Sequence[float]) -> float | None:
    return None if not values else float(statistics.median(values))


def _case_features(row: Mapping[str, Any]) -> dict[str, Any]:
    purposes = row.get("purpose_counts")
    trace = row.get("direction_trace")
    if (
        row.get("status") != "completed"
        or row.get("arm") != "transported_direction"
        or not isinstance(purposes, Mapping)
        or not isinstance(trace, list)
        or not trace
        or trace[0].get("step") != 0
        or trace[0].get("selected_kind") != "block_ritz"
        or sum(int(value) for value in purposes.values())
        != int(row.get("force_evaluations", -1))
        or int(purposes.get("unattributed", -1)) != 0
    ):
        raise ValueError("residual observation case does not close")
    transported = trace[1:]
    for item in transported:
        if (
            item.get("selected_kind") != "transported"
            or item.get("direction_hvp_count") != 1
            or item.get("oracle_selection_force_evaluations_delta")
            != 2
            or item.get("transported_relative_residual") is None
            or item.get("transported_true_relative_residual") is None
        ):
            raise ValueError("transported residual trace is incomplete")
    expected_direction_fe = 2 * len(transported)
    if int(purposes.get("direction_oracle", -1)) != expected_direction_fe:
        raise ValueError("transported direction FE contract changed")
    residuals = [
        float(item["transported_relative_residual"])
        for item in transported
    ]
    true_residuals = [
        float(item["transported_true_relative_residual"])
        for item in transported
    ]
    if any(
        not np.isfinite(value) or value < 0.0
        for value in residuals + true_residuals
    ):
        raise ValueError("residuals must be finite and nonnegative")
    diagnostics = row.get("relaxation_diagnostics", {})
    return {
        "system": str(row["system"]),
        "state_id": str(row["state_id"]),
        "seed": int(row["seed"]),
        "certificate": bool(row.get("certificate", False)),
        "transported_steps": len(transported),
        "first_relative_residual": (
            None if not residuals else residuals[0]
        ),
        "max_relative_residual": (
            None if not residuals else max(residuals)
        ),
        "mean_relative_residual": (
            None if not residuals else float(statistics.mean(residuals))
        ),
        "final_relative_residual": (
            None if not residuals else residuals[-1]
        ),
        "residual_change": (
            None if not residuals else residuals[-1] - residuals[0]
        ),
        "max_true_relative_residual": (
            None if not true_residuals else max(true_residuals)
        ),
        "force_evaluations": int(row["force_evaluations"]),
        "direction_force_evaluations": int(
            purposes["direction_oracle"]
        ),
        "proposal_force_evaluations": int(
            purposes["biased_proposal_relax"]
        ),
        "quench_force_evaluations": int(
            purposes["landing_true_quench"]
        ),
        "downstream_force_evaluations": int(
            purposes["biased_proposal_relax"]
            + purposes["landing_true_quench"]
        ),
        "quench_iterations": int(row["quench_iterations"]),
        "landing_delta_eV": float(row["landing_delta_eV"]),
        "proposal_energy_exploded": int(
            diagnostics.get(
                "proposal_relax_outcome_energy_exploded",
                0,
            )
        ),
        "proposal_unconverged": int(
            diagnostics.get("proposal_relax_unconverged", 0)
        ),
    }


def _system_summary(
    features: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    observable = [
        row
        for row in features
        if row["max_relative_residual"] is not None
    ]
    residuals = [
        float(row["max_relative_residual"]) for row in observable
    ]
    downstream = [
        float(row["downstream_force_evaluations"])
        for row in observable
    ]
    quench_iterations = [
        float(row["quench_iterations"]) for row in observable
    ]
    changes = [
        float(row["residual_change"]) for row in observable
    ]
    cost_median = _median(downstream)
    low = [
        residual
        for residual, cost in zip(residuals, downstream)
        if cost_median is not None and cost <= cost_median
    ]
    high = [
        residual
        for residual, cost in zip(residuals, downstream)
        if cost_median is not None and cost > cost_median
    ]
    return {
        "case_count": len(features),
        "observable_case_count": len(observable),
        "certificate_count": sum(
            bool(row["certificate"]) for row in features
        ),
        "median_first_relative_residual": _median(
            [
                float(row["first_relative_residual"])
                for row in observable
            ]
        ),
        "median_max_relative_residual": _median(residuals),
        "median_residual_change": _median(changes),
        "spearman_max_residual_vs_downstream_fe": _spearman(
            residuals,
            downstream,
        ),
        "spearman_max_residual_vs_quench_iterations": _spearman(
            residuals,
            quench_iterations,
        ),
        "low_cost_median_max_residual": _median(low),
        "high_cost_median_max_residual": _median(high),
        "proposal_energy_exploded": sum(
            int(row["proposal_energy_exploded"])
            for row in features
        ),
        "total_force_evaluations": sum(
            int(row["force_evaluations"]) for row in features
        ),
        "direction_force_evaluations": sum(
            int(row["direction_force_evaluations"])
            for row in features
        ),
    }


def analyze_cases(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    runner = _load_runner()
    expected = {
        (
            case["system"],
            case["state_id"],
            case["seed"],
            case["arm"],
        )
        for system in SYSTEMS
        for case in runner.case_matrix(system)
    }
    observed = {
        (
            row.get("system"),
            row.get("state_id"),
            row.get("seed"),
            row.get("arm"),
        )
        for row in rows
    }
    if len(rows) != 30 or observed != expected:
        raise ValueError("analysis requires the exact 30-action cohort")
    features = [_case_features(row) for row in rows]
    systems = {
        system: _system_summary(
            [row for row in features if row["system"] == system]
        )
        for system in SYSTEMS
    }
    consistent = all(
        systems[system][
            "spearman_max_residual_vs_downstream_fe"
        ]
        is not None
        and systems[system][
            "spearman_max_residual_vs_downstream_fe"
        ]
        > 0.0
        and systems[system]["median_residual_change"] is not None
        and systems[system]["median_residual_change"] > 0.0
        and systems[system]["low_cost_median_max_residual"] is not None
        and systems[system]["high_cost_median_max_residual"] is not None
        and systems[system]["high_cost_median_max_residual"]
        > systems[system]["low_cost_median_max_residual"]
        for system in SYSTEMS
    )
    return {
        "schema_version": 1,
        "decision": (
            "residual_signal_consistent"
            if consistent
            else "residual_signal_inconsistent"
        ),
        "zero_extra_direction_fe": all(
            int(row["direction_force_evaluations"])
            == 2 * int(row["transported_steps"])
            for row in features
        ),
        "systems": systems,
        "cases": features,
        "claim_ceiling": (
            "transport-only observational survivor gate; no refresh "
            "threshold, causal claim, or significance claim"
        ),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _markdown(evidence: Mapping[str, Any]) -> str:
    lines = [
        "# Transported-mode residual observation",
        "",
        f"- Decision: `{evidence['decision']}`",
        f"- Zero extra direction FE: `{evidence['zero_extra_direction_fe']}`",
        "",
        "| system | observable | median first residual | median max residual | median change | rho residual/downstream FE | rho residual/quench iter | low-cost residual | high-cost residual |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for system in SYSTEMS:
        row = evidence["systems"][system]
        lines.append(
            f"| {system} | {row['observable_case_count']}/"
            f"{row['case_count']} | "
            f"{row['median_first_relative_residual']:.6f} | "
            f"{row['median_max_relative_residual']:.6f} | "
            f"{row['median_residual_change']:.6f} | "
            f"{row['spearman_max_residual_vs_downstream_fe']:.6f} | "
            f"{row['spearman_max_residual_vs_quench_iterations']:.6f} | "
            f"{row['low_cost_median_max_residual']:.6f} | "
            f"{row['high_cost_median_max_residual']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"Claim ceiling: {evidence['claim_ceiling']}.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c60-raw", type=Path, required=True)
    parser.add_argument("--pdo-raw", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--conclusion", type=Path, required=True)
    args = parser.parse_args(argv)
    raw_by_system = {
        "c60": json.loads(args.c60_raw.read_text(encoding="utf-8")),
        "pdo": json.loads(args.pdo_raw.read_text(encoding="utf-8")),
    }
    rows = [
        {**row, "system": system}
        for system in SYSTEMS
        for row in raw_by_system[system]["cases"]
    ]
    evidence = analyze_cases(rows)
    evidence["provenance"] = {
        system: {
            "execution_commit": raw_by_system[system][
                "execution_commit"
            ],
        }
        for system in SYSTEMS
    }
    _write_json(args.evidence, evidence)
    args.conclusion.write_text(_markdown(evidence), encoding="utf-8")
    print(json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
