#!/usr/bin/env python3
"""Fixed-task CuO diagnosis for proposal line-search failures."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

from pamssw.accounting import EvaluationPurpose
from pamssw.proposal_replay import (
    ProposalTaskNotCaptured,
    capture_proposal_task,
    replay_proposal_task_observed,
)
from pamssw.walker import SurfaceWalker


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
SOURCE_GATE = (
    RUN_ROOT.parent / "20260731-starter-selection-mechanism-gate" / "run_gate.py"
)
DEFAULT_DEPTHS = (1, 2, 4, 8)
SEED = 42
FORCE_BUDGET = 20_000


def _load_source_gate():
    name = "_cuo_line_search_source_gate"
    module = sys.modules.get(name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(name, SOURCE_GATE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load source gate: {SOURCE_GATE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _replay_row(*, depth: int, variant: str, task, calculator) -> dict:
    replay = replay_proposal_task_observed(
        task,
        calculator,
        optimizer="ase-fire" if variant == "fire_with_ls" else "safe-lbfgs-total",
    )
    telemetry = replay.result.telemetry
    return {
        "bias_depth": depth,
        "variant": variant,
        "softening_terms": 0 if task.softening is None else len(task.softening.terms),
        "force_evaluations": replay.evaluation_counts.total,
        "wall_time_s": replay.wall_time_s,
        "certificate_satisfied": replay.certificate_satisfied,
        "termination_reason": telemetry.termination_reason,
        "accepted_steps": telemetry.accepted_steps,
        "rejected_steps": telemetry.rejected_steps,
        "line_search_evaluations": telemetry.line_search_evaluations,
        "accepted_secants": telemetry.accepted_secants,
        "rejected_secants": telemetry.rejected_secants,
        "initial": asdict(replay.initial),
        "final": asdict(replay.final),
        "delta_total_energy_eV": replay.final.total_energy - replay.initial.total_energy,
        "delta_true_energy_eV": replay.final.true_energy - replay.initial.true_energy,
        "direction_progress_A": replay.direction_progress,
        "orthogonal_displacement_norm_A": replay.orthogonal_displacement_norm,
        "gradient_norm_eV_per_A": replay.result.gradient_norm,
    }


def run(output_directory: Path, depths: tuple[int, ...]) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    output_directory.mkdir(parents=True, exist_ok=False)
    gate = _load_source_gate()
    resources = gate._materialize_cuo_resources(
        gate.CUO_ARCHIVE_PATH,
        output_directory / "cuo-input",
    )
    state = gate._load_state("cuo", resources)
    bootstrap_config = gate.build_config(
        "cuo",
        output_directory / "bootstrap-case",
        seed=SEED,
        starter_mode="uniform_archive",
        force_budget=FORCE_BUDGET,
    )
    bootstrap_walker = SurfaceWalker(
        calculator=gate._calculator("cuo", resources),
        config=bootstrap_config,
        softening_enabled=True,
    )
    started = perf_counter()
    bootstrap = bootstrap_walker.relax_true_minimum(
        state,
        trajectory_name="shared_initial_true_quench",
        quench_purpose=EvaluationPurpose.BOOTSTRAP_TRUE_QUENCH,
    )
    bootstrap_wall_time_s = perf_counter() - started
    bootstrap_counts = bootstrap_walker.calculator.snapshot()

    evidence = {
        "schema_version": 1,
        "commit": _git_commit(),
        "seed": SEED,
        "depths": list(depths),
        "hypothesis": (
            "Separate accumulated-bias depth, LS pair softening, and optimizer "
            "mechanism on fixed CuO proposal objectives."
        ),
        "cuo_archive_sha256": _sha256(gate.CUO_ARCHIVE_PATH),
        "cuo_model_sha256": _sha256(resources["model"]),
        "bootstrap": {
            "energy_eV": bootstrap.energy,
            "gradient_norm_eV_per_A": bootstrap.gradient_norm,
            "force_evaluations": bootstrap_counts.total,
            "purpose_counts": bootstrap_counts.as_dict(),
            "wall_time_s": bootstrap_wall_time_s,
            "n_atoms": bootstrap.state.n_atoms,
            "n_fixed_atoms": int(bootstrap.state.fixed_mask.sum()),
        },
        "production_config": asdict(bootstrap_config),
        "captures": [],
        "replays": [],
    }
    _write_json(output_directory / "evidence.json", evidence)

    for depth in depths:
        capture_config = replace(
            bootstrap_config,
            max_force_evals=FORCE_BUDGET,
            rng_seed=SEED,
        )
        capture_started = perf_counter()
        try:
            captured = capture_proposal_task(
                bootstrap.state,
                gate._calculator("cuo", resources),
                capture_config,
                target_bias_count=depth,
                softening_enabled=True,
            )
        except ProposalTaskNotCaptured as exc:
            evidence["captures"].append(
                {
                    "bias_depth": depth,
                    "captured": False,
                    "capture_force_evaluations": exc.evaluation_counts.total,
                    "capture_purpose_counts": exc.evaluation_counts.as_dict(),
                    "capture_wall_time_s": perf_counter() - capture_started,
                }
            )
            _write_json(output_directory / "evidence.json", evidence)
            continue

        task = captured.task
        evidence["captures"].append(
            {
                "bias_depth": depth,
                "captured": True,
                "bias_count": len(task.biases),
                "softening_terms": 0 if task.softening is None else len(task.softening.terms),
                "proposal_maxiter": task.maxiter,
                "proposal_fmax": task.fmax,
                "capture_force_evaluations": captured.evaluation_counts.total,
                "capture_purpose_counts": captured.evaluation_counts.as_dict(),
                "capture_wall_time_s": perf_counter() - capture_started,
            }
        )
        variants = (
            ("safe_lbfgs_with_ls", task),
            ("safe_lbfgs_without_ls", replace(task, softening=None)),
            ("fire_with_ls", task),
        )
        for variant, variant_task in variants:
            row = _replay_row(
                depth=depth,
                variant=variant,
                task=variant_task,
                calculator=gate._calculator("cuo", resources),
            )
            evidence["replays"].append(row)
            _write_json(output_directory / "evidence.json", evidence)
            print(
                f"depth={depth} variant={variant} "
                f"termination={row['termination_reason']} "
                f"FE={row['force_evaluations']} "
                f"rejected={row['rejected_steps']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--depths",
        type=int,
        nargs="+",
        default=DEFAULT_DEPTHS,
    )
    args = parser.parse_args()
    depths = tuple(args.depths)
    if not depths or any(depth not in DEFAULT_DEPTHS for depth in depths):
        raise ValueError(f"depths must be a non-empty subset of {DEFAULT_DEPTHS}")
    run(args.output, depths)


if __name__ == "__main__":
    main()
