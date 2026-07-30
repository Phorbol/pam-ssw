#!/usr/bin/env python3
"""Fixed-starter C60 gate for eight versus fourteen uphill microsteps."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Mapping

import numpy as np

from pamssw.accounting import EvaluationPurpose
from pamssw.archive import MinimaArchive
from pamssw.io import read_state, write_state
from pamssw.relax import has_force_convergence_certificate
from pamssw.walker import SurfaceWalker


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
U4_RUNNER = (
    REPO_ROOT
    / "runs"
    / "20260731-uphill-mechanism-closure"
    / "run_u4.py"
)
SYSTEM_STATES = ("late", "mid")
SEEDS = (42, 43, 44)
HORIZONS = (8, 14)
TOTAL_FORCE_CEILING = 12_000


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


u4 = _load_module(U4_RUNNER, "_uphill_horizon_u4_support")


def paired_config_diff(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, tuple[Any, Any]]:
    keys = set(left) | set(right)
    difference = {
        key: (left.get(key), right.get(key))
        for key in sorted(keys)
        if left.get(key) != right.get(key)
    }
    if set(difference) != {"max_steps_per_walk"}:
        raise ValueError(
            "paired horizon configs may differ only in "
            "max_steps_per_walk"
        )
    return difference


def walk_termination_reason(stats: Mapping[str, Any]) -> str:
    if int(stats.get("walk_terminations", 0)) != 1:
        raise ValueError("a completed arm must record exactly one termination")
    prefix = "walk_termination_"
    reasons = [
        key.removeprefix(prefix)
        for key, value in stats.items()
        if key.startswith(prefix)
        and key != "walk_termination_last_reason"
        and int(value) > 0
    ]
    if len(reasons) != 1:
        raise ValueError(
            "a completed arm must record exactly one termination reason"
        )
    reason = reasons[0]
    if stats.get("walk_termination_last_reason") != reason:
        raise ValueError("last termination reason disagrees with counts")
    return reason


def _build_config(
    *,
    state_id: str,
    seed: int,
    horizon: int,
    output: Path,
):
    config_builder = u4._load_module(
        u4.CONFIG_RUNNER,
        f"_horizon_config_{state_id}_{seed}_{horizon}",
    )
    base = config_builder.build_production_config(
        "c60",
        output / "configs" / state_id / f"seed-{seed}" / f"h{horizon}",
        master_seed=seed,
    )
    return replace(
        base,
        max_trials=1,
        max_force_evals=None,
        max_steps_per_walk=horizon,
        local_softening_scope="both",
        direction_diagnostics_enabled=False,
        direction_diagnostics_path=None,
    )


def _run_arm(
    *,
    state,
    state_id: str,
    seed: int,
    horizon: int,
    calculator,
    output: Path,
) -> dict[str, Any]:
    config = _build_config(
        state_id=state_id,
        seed=seed,
        horizon=horizon,
        output=output,
    )
    walker = SurfaceWalker(
        calculator=calculator,
        config=config,
        softening_enabled=True,
    )
    before = walker.calculator.snapshot()
    started = perf_counter()
    escape = walker._walk_candidate_from_seed(
        state,
        trial_index=0,
        proposal_index=0,
    )
    after_walk = walker.calculator.snapshot()
    with walker.calculator.purpose(EvaluationPurpose.ESCAPE_TRUE_PES_CHECK):
        escape_evaluation = walker.calculator.evaluate(escape)
    landing = walker.relax_true_minimum(escape)
    after = walker.calculator.snapshot()
    wall_time = float(perf_counter() - started)
    stats = walker._direction_stats_summary()
    termination = walk_termination_reason(stats)
    arm_dir = output / "cases" / state_id / f"seed-{seed}" / f"h{horizon}"
    arm_dir.mkdir(parents=True, exist_ok=True)
    write_state(arm_dir / "escape.xyz", escape)
    write_state(arm_dir / "landing.xyz", landing.state)
    row = {
        "state_id": state_id,
        "seed": seed,
        "horizon": horizon,
        "config": asdict(config),
        "termination_reason": termination,
        "executed_microsteps": int(stats["uphill_control_steps"]),
        "walk_displacement_clips": int(stats["walk_displacement_clips"]),
        "requested_sigma_mean": float(
            stats["uphill_requested_sigma_mean"]
        ),
        "executed_sigma_mean": float(
            stats["uphill_executed_sigma_mean"]
        ),
        "sigma_capped_steps": int(stats["uphill_sigma_capped_steps"]),
        "base_weight_at_config_max_steps": int(
            stats["uphill_base_weight_at_config_max_steps"]
        ),
        "final_weight_above_config_max_steps": int(
            stats["uphill_final_weight_above_config_max_steps"]
        ),
        "true_curvature_mean": float(
            stats["uphill_true_curvature_mean"]
        ),
        "inner_curvature_mean": float(
            stats["uphill_inner_curvature_mean"]
        ),
        "escape_energy_eV": float(escape_evaluation.energy),
        "landing_energy_eV": float(landing.energy),
        "landing_gradient_norm_eV_per_A": float(
            landing.gradient_norm
        ),
        "landing_iterations": int(landing.n_iter),
        "landing_certificate": bool(
            has_force_convergence_certificate(
                landing,
                config.quench_fmax,
            )
        ),
        "walk_counts": u4._counts_delta(before, after_walk),
        "validation_counts": u4._counts_delta(after_walk, after),
        "total_counts": u4._counts_delta(before, after),
        "wall_time_s": wall_time,
    }
    u4._write_json(arm_dir / "summary.json", row)
    return row


def _pair_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    indexed = {
        (row["state_id"], row["seed"], row["horizon"]): row
        for row in rows
    }
    pairs = []
    for state_id in SYSTEM_STATES:
        for seed in SEEDS:
            h8 = indexed[(state_id, seed, 8)]
            h14 = indexed[(state_id, seed, 14)]
            paired_config_diff(h8["config"], h14["config"])
            rmsd = MinimaArchive._rmsd(
                _landing_state(h8),
                _landing_state(h14),
            )
            same_landing = bool(
                np.isfinite(rmsd)
                and abs(
                    h14["landing_energy_eV"]
                    - h8["landing_energy_eV"]
                )
                <= h8["config"]["dedup_energy_tol"]
                and rmsd <= h8["config"]["dedup_rmsd_tol"]
            )
            pairs.append(
                {
                    "state_id": state_id,
                    "seed": seed,
                    "h8": h8,
                    "h14": h14,
                    "landing_energy_delta_eV": (
                        h14["landing_energy_eV"]
                        - h8["landing_energy_eV"]
                    ),
                    "landing_rmsd_A": (
                        float(rmsd) if np.isfinite(rmsd) else None
                    ),
                    "same_landing": same_landing,
                    "additional_force_evaluations": (
                        h14["total_counts"]["total"]
                        - h8["total_counts"]["total"]
                    ),
                    "additional_microsteps": (
                        h14["executed_microsteps"]
                        - h8["executed_microsteps"]
                    ),
                }
            )
    return pairs


def _landing_state(row: Mapping[str, Any]):
    return read_state(
        Path(row["output_directory"]) / "landing.xyz"
    )


def _summary(
    pairs: list[dict[str, Any]],
    *,
    total_force_evaluations: int,
) -> dict[str, Any]:
    return {
        "pair_count": len(pairs),
        "same_landing_count": sum(pair["same_landing"] for pair in pairs),
        "h14_lower_count": sum(
            pair["landing_energy_delta_eV"] < -1.0e-3
            for pair in pairs
        ),
        "h14_higher_count": sum(
            pair["landing_energy_delta_eV"] > 1.0e-3
            for pair in pairs
        ),
        "h14_additional_force_evaluations": sum(
            pair["additional_force_evaluations"] for pair in pairs
        ),
        "h14_additional_microsteps": sum(
            pair["additional_microsteps"] for pair in pairs
        ),
        "radius_censored_pairs": sum(
            pair["h8"]["termination_reason"]
            == "walk_displacement_clipped"
            or pair["h14"]["termination_reason"]
            == "walk_displacement_clipped"
            for pair in pairs
        ),
        "total_force_evaluations": total_force_evaluations,
        "pairs": pairs,
    }


def _write_conclusion(path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# H=8 versus H=14 Uphill Capacity Gate",
        "",
        f"- Paired fixed-starter runs: {summary['pair_count']}",
        (
            "- Same strict landing basin: "
            f"{summary['same_landing_count']}/{summary['pair_count']}"
        ),
        (
            "- H=14 lower/higher landing counts: "
            f"{summary['h14_lower_count']}/"
            f"{summary['h14_higher_count']}"
        ),
        (
            "- H=14 additional force evaluations: "
            f"{summary['h14_additional_force_evaluations']}"
        ),
        (
            "- H=14 additional executed microsteps: "
            f"{summary['h14_additional_microsteps']}"
        ),
        (
            "- Radius-censored pairs: "
            f"{summary['radius_censored_pairs']}"
        ),
        (
            "- Aggregate force evaluations: "
            f"{summary['total_force_evaluations']}"
        ),
        "",
        (
            "The gate changes only max_steps_per_walk. It does not establish "
            "a cross-system production default."
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(
    *,
    output: Path,
    expected_commit: str,
) -> dict[str, Any]:
    preflight = u4._preflight(expected_commit)
    ls_gate = u4._load_module(
        u4.LS_GATE_RUNNER,
        "_uphill_horizon_calculator",
    )
    calculator, _production = ls_gate._calculator()
    rows = []
    for state_id in SYSTEM_STATES:
        state = read_state(u4.STATE_FILES[state_id])
        for seed in SEEDS:
            pair_rows = []
            for horizon in HORIZONS:
                row = _run_arm(
                    state=state,
                    state_id=state_id,
                    seed=seed,
                    horizon=horizon,
                    calculator=calculator,
                    output=output,
                )
                row["output_directory"] = str(
                    output
                    / "cases"
                    / state_id
                    / f"seed-{seed}"
                    / f"h{horizon}"
                )
                pair_rows.append(row)
            paired_config_diff(
                pair_rows[0]["config"],
                pair_rows[1]["config"],
            )
            rows.extend(pair_rows)
    total_force_evaluations = sum(
        row["total_counts"]["total"] for row in rows
    )
    if total_force_evaluations > TOTAL_FORCE_CEILING:
        raise RuntimeError(
            f"force ceiling exceeded: {total_force_evaluations} > "
            f"{TOTAL_FORCE_CEILING}"
        )
    pairs = _pair_rows(rows)
    summary = _summary(
        pairs,
        total_force_evaluations=total_force_evaluations,
    )
    payload = {
        "schema_version": 1,
        "preflight": preflight,
        "horizons": list(HORIZONS),
        "systems": ["c60"],
        "state_ids": list(SYSTEM_STATES),
        "seeds": list(SEEDS),
        "force_ceiling": TOTAL_FORCE_CEILING,
        "rows": rows,
        "summary": summary,
    }
    u4._write_json(output / "evidence.json", payload)
    _write_conclusion(output / "conclusion.md", summary)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-git-commit", required=True)
    args = parser.parse_args()
    payload = run(
        output=args.output,
        expected_commit=args.expected_git_commit,
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in payload["summary"].items()
                if key != "pairs"
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
