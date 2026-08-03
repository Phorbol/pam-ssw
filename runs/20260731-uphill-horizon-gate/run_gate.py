#!/usr/bin/env python3
"""Single-trajectory C60 gate for uphill checkpoints eight and fourteen."""

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

from pamssw.accounting import EvaluationCounts, EvaluationPurpose
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
CHECKPOINTS = (8, 14)
TOTAL_FORCE_CEILING = 12_000
NO_WALK_BALL_RADIUS = 1.0e6


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


class CheckpointWalker(SurfaceWalker):
    """Observe accepted uphill states without changing the production walk."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.checkpoint_states = []
        self.checkpoint_counts: list[EvaluationCounts] = []
        self._pending_checkpoint_state = None

    def _clip_walk_displacement(
        self,
        reference,
        candidate,
        max_displacement,
    ):
        state, clipped = SurfaceWalker._clip_walk_displacement(
            reference,
            candidate,
            max_displacement,
        )
        self._pending_checkpoint_state = state
        return state, clipped

    def _record_trust_update(self, update) -> None:
        super()._record_trust_update(update)
        if self._pending_checkpoint_state is None:
            raise RuntimeError("trust update has no accepted checkpoint state")
        self.checkpoint_states.append(self._pending_checkpoint_state)
        self.checkpoint_counts.append(self.calculator.snapshot())
        self._pending_checkpoint_state = None


def _build_config(
    *,
    state_id: str,
    seed: int,
    output: Path,
    no_walk_ball: bool,
):
    config_builder = u4._load_module(
        u4.CONFIG_RUNNER,
        f"_horizon_config_{state_id}_{seed}_{int(no_walk_ball)}",
    )
    base = config_builder.build_production_config(
        "c60",
        output / "configs" / state_id / f"seed-{seed}",
        master_seed=seed,
    )
    return replace(
        base,
        max_trials=1,
        max_force_evals=None,
        max_steps_per_walk=14,
        walk_trust_radius=(
            NO_WALK_BALL_RADIUS
            if no_walk_ball
            else base.walk_trust_radius
        ),
        local_softening_scope="both",
        direction_diagnostics_enabled=False,
        direction_diagnostics_path=None,
    )


def _quench_checkpoint(
    *,
    walker: CheckpointWalker,
    checkpoint: int,
    output: Path,
) -> dict[str, Any]:
    state = walker.checkpoint_states[checkpoint - 1]
    before = walker.calculator.snapshot()
    started = perf_counter()
    with walker.calculator.purpose(EvaluationPurpose.ESCAPE_TRUE_PES_CHECK):
        escape_evaluation = walker.calculator.evaluate(state)
    landing = walker.relax_true_minimum(state)
    after = walker.calculator.snapshot()
    wall_time = float(perf_counter() - started)
    output.mkdir(parents=True, exist_ok=True)
    write_state(output / "escape.xyz", state)
    write_state(output / "landing.xyz", landing.state)
    return {
        "checkpoint": checkpoint,
        "escape_energy_eV": float(escape_evaluation.energy),
        "landing_energy_eV": float(landing.energy),
        "landing_gradient_norm_eV_per_A": float(
            landing.gradient_norm
        ),
        "landing_iterations": int(landing.n_iter),
        "landing_certificate": bool(
            has_force_convergence_certificate(
                landing,
                walker.config.quench_fmax,
            )
        ),
        "validation_counts": u4._counts_delta(before, after),
        "validation_wall_time_s": wall_time,
        "output_directory": str(output),
    }


def _run_trajectory(
    *,
    state,
    state_id: str,
    seed: int,
    calculator,
    output: Path,
    no_walk_ball: bool,
) -> dict[str, Any]:
    config = _build_config(
        state_id=state_id,
        seed=seed,
        output=output,
        no_walk_ball=no_walk_ball,
    )
    walker = CheckpointWalker(
        calculator=calculator,
        config=config,
        softening_enabled=True,
    )
    before = walker.calculator.snapshot()
    started = perf_counter()
    walker._walk_candidate_from_seed(
        state,
        trial_index=0,
        proposal_index=0,
    )
    after_walk = walker.calculator.snapshot()
    walk_wall_time = float(perf_counter() - started)
    stats = walker._direction_stats_summary()
    trust_stats = walker._trust_stats_summary()
    termination = walk_termination_reason(stats)
    accepted_steps = len(walker.checkpoint_states)
    if accepted_steps != int(trust_stats["trust_region_steps"]):
        raise RuntimeError(
            "accepted checkpoint count disagrees with trust updates"
        )
    trajectory_id = "no_walk_ball" if no_walk_ball else "standard"
    trajectory_dir = (
        output
        / "cases"
        / state_id
        / f"seed-{seed}"
        / trajectory_id
    )
    checkpoints = {}
    for checkpoint in CHECKPOINTS:
        if accepted_steps >= checkpoint:
            checkpoints[str(checkpoint)] = _quench_checkpoint(
                walker=walker,
                checkpoint=checkpoint,
                output=trajectory_dir / f"checkpoint-{checkpoint:02d}",
            )
    after = walker.calculator.snapshot()
    row = {
        "state_id": state_id,
        "seed": seed,
        "trajectory": trajectory_id,
        "config": asdict(config),
        "termination_reason": termination,
        "attempted_microsteps": int(stats["uphill_control_steps"]),
        "accepted_microsteps": accepted_steps,
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
        "walk_counts": u4._counts_delta(before, after_walk),
        "total_counts": u4._counts_delta(before, after),
        "walk_wall_time_s": walk_wall_time,
        "checkpoints": checkpoints,
    }
    if accepted_steps >= 8:
        row["to_checkpoint_8_counts"] = u4._counts_delta(
            before,
            walker.checkpoint_counts[7],
        )
    if accepted_steps >= 14:
        row["checkpoint_8_to_14_counts"] = u4._counts_delta(
            walker.checkpoint_counts[7],
            walker.checkpoint_counts[13],
        )
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    u4._write_json(trajectory_dir / "summary.json", row)
    return row


def _landing_state(checkpoint: Mapping[str, Any]):
    return read_state(
        Path(checkpoint["output_directory"]) / "landing.xyz"
    )


def _pair_checkpoints(row: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint8 = row["checkpoints"]["8"]
    checkpoint14 = row["checkpoints"]["14"]
    rmsd = MinimaArchive._rmsd(
        _landing_state(checkpoint8),
        _landing_state(checkpoint14),
    )
    energy_delta = (
        checkpoint14["landing_energy_eV"]
        - checkpoint8["landing_energy_eV"]
    )
    same_landing = bool(
        np.isfinite(rmsd)
        and abs(energy_delta) <= row["config"]["dedup_energy_tol"]
        and rmsd <= row["config"]["dedup_rmsd_tol"]
    )
    return {
        "state_id": row["state_id"],
        "seed": row["seed"],
        "trajectory": row["trajectory"],
        "checkpoint8": checkpoint8,
        "checkpoint14": checkpoint14,
        "landing_energy_delta_eV": float(energy_delta),
        "landing_rmsd_A": float(rmsd) if np.isfinite(rmsd) else None,
        "same_landing": same_landing,
        "additional_uphill_force_evaluations": (
            row["checkpoint_8_to_14_counts"]["total"]
        ),
    }


def _summary(
    *,
    rows: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    total_force_evaluations: int,
) -> dict[str, Any]:
    standard_rows = [
        row for row in rows if row["trajectory"] == "standard"
    ]
    return {
        "standard_trajectory_count": len(standard_rows),
        "complete_pair_count": len(pairs),
        "same_landing_count": sum(pair["same_landing"] for pair in pairs),
        "checkpoint14_lower_count": sum(
            pair["landing_energy_delta_eV"] < -1.0e-3
            for pair in pairs
        ),
        "checkpoint14_higher_count": sum(
            pair["landing_energy_delta_eV"] > 1.0e-3
            for pair in pairs
        ),
        "additional_uphill_force_evaluations": sum(
            pair["additional_uphill_force_evaluations"]
            for pair in pairs
        ),
        "standard_radius_censored_count": sum(
            row["walk_displacement_clips"] > 0
            and row["accepted_microsteps"] < 14
            for row in standard_rows
        ),
        "no_walk_ball_trajectory_count": sum(
            row["trajectory"] == "no_walk_ball" for row in rows
        ),
        "unresolved_before_checkpoint14": sum(
            row["accepted_microsteps"] < 14
            for row in rows
            if (
                row["trajectory"] == "no_walk_ball"
                or (
                    row["trajectory"] == "standard"
                    and row["walk_displacement_clips"] == 0
                )
            )
        ),
        "total_force_evaluations": total_force_evaluations,
        "pairs": pairs,
    }


def _write_conclusion(path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# Single-Trajectory H=8 versus H=14 Capacity Gate",
        "",
        (
            "- Complete checkpoint pairs: "
            f"{summary['complete_pair_count']}/"
            f"{summary['standard_trajectory_count']}"
        ),
        (
            "- Same strict landing basin: "
            f"{summary['same_landing_count']}/"
            f"{summary['complete_pair_count']}"
        ),
        (
            "- Checkpoint 14 lower/higher landing counts: "
            f"{summary['checkpoint14_lower_count']}/"
            f"{summary['checkpoint14_higher_count']}"
        ),
        (
            "- Additional uphill force evaluations after checkpoint 8: "
            f"{summary['additional_uphill_force_evaluations']}"
        ),
        (
            "- Standard radius-censored trajectories: "
            f"{summary['standard_radius_censored_count']}"
        ),
        (
            "- Fixed no-walk-ball diagnostic trajectories: "
            f"{summary['no_walk_ball_trajectory_count']}"
        ),
        (
            "- Unresolved before checkpoint 14: "
            f"{summary['unresolved_before_checkpoint14']}"
        ),
        (
            "- Aggregate force evaluations: "
            f"{summary['total_force_evaluations']}"
        ),
        "",
        (
            "Checkpoint 8 and 14 are quenched from one shared uphill "
            "trajectory. No independent-arm prefix is compared."
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
            standard = _run_trajectory(
                state=state,
                state_id=state_id,
                seed=seed,
                calculator=calculator,
                output=output,
                no_walk_ball=False,
            )
            rows.append(standard)
            if (
                standard["accepted_microsteps"] < 14
                and standard["walk_displacement_clips"] > 0
            ):
                rows.append(
                    _run_trajectory(
                        state=state,
                        state_id=state_id,
                        seed=seed,
                        calculator=calculator,
                        output=output,
                        no_walk_ball=True,
                    )
                )
    pair_rows = [
        row
        for row in rows
        if row["accepted_microsteps"] >= 14
    ]
    pairs = [_pair_checkpoints(row) for row in pair_rows]
    total_force_evaluations = sum(
        row["total_counts"]["total"] for row in rows
    )
    if total_force_evaluations > TOTAL_FORCE_CEILING:
        raise RuntimeError(
            f"force ceiling exceeded: {total_force_evaluations} > "
            f"{TOTAL_FORCE_CEILING}"
        )
    summary = _summary(
        rows=rows,
        pairs=pairs,
        total_force_evaluations=total_force_evaluations,
    )
    payload = {
        "schema_version": 2,
        "preflight": preflight,
        "checkpoints": list(CHECKPOINTS),
        "state_ids": list(SYSTEM_STATES),
        "seeds": list(SEEDS),
        "force_ceiling": TOTAL_FORCE_CEILING,
        "no_walk_ball_radius_A": NO_WALK_BALL_RADIUS,
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
