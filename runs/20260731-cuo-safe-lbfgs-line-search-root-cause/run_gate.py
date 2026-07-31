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

import numpy as np

from pamssw.accounting import EvalCounter, EvaluationPurpose
from pamssw.proposal_replay import (
    ProposalTaskNotCaptured,
    _RecordingProposalPotential,
    capture_proposal_task,
    replay_proposal_task_observed,
)
from pamssw.relax import Relaxer, _SAFE_LBFGS_ARMIJO_C1, _SAFE_LBFGS_MAX_LINE_TRIALS
from pamssw.softening import LocalSofteningModel
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


def _replay_row(*, depth: int, variant: str, task, calculator, calculator_dtype: str) -> dict:
    if variant.startswith("safe_lbfgs") and task.softening is not None:
        return _diagnose_safe_lbfgs_with_ls(
            depth=depth,
            variant=variant,
            task=task,
            calculator=calculator,
            calculator_dtype=calculator_dtype,
        )
    replay = replay_proposal_task_observed(
        task,
        calculator,
        optimizer="ase-fire" if variant == "fire_with_ls" else "safe-lbfgs-total",
    )
    telemetry = replay.result.telemetry
    return {
        "bias_depth": depth,
        "variant": variant,
        "calculator_dtype": calculator_dtype,
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


def _diagnose_safe_lbfgs_with_ls(
    *, depth: int, variant: str, task, calculator, calculator_dtype: str
) -> dict:
    counter = EvalCounter(calculator)
    proposal = _RecordingProposalPotential(
        counter,
        biases=list(task.biases),
        softening=task.softening,
    )
    relaxer = Relaxer(
        proposal.evaluate,
        optimizer="safe-lbfgs-total",
        component_evaluator=proposal.evaluate_parts,
    )
    started = perf_counter()
    with counter.purpose(EvaluationPurpose.BIASED_PROPOSAL_RELAX):
        result = relaxer.relax(
            task.initial_state,
            fmax=task.fmax,
            maxiter=task.maxiter,
            coordinate_trust_radius=task.coordinate_trust_radius,
        )
    wall_time_s = perf_counter() - started
    final_flat = result.state.flatten_positions()
    records = tuple(proposal.records)
    initial_observation = records[0][1]
    final_observation = records[-1][1]
    last_bias = task.biases[-1]
    endpoint_delta = final_flat - last_bias.center
    direction_progress = float(np.dot(endpoint_delta, last_bias.direction))
    orthogonal = endpoint_delta - direction_progress * last_bias.direction
    row = {
        "bias_depth": depth,
        "variant": variant,
        "calculator_dtype": calculator_dtype,
        "softening_terms": 0 if task.softening is None else len(task.softening.terms),
        "force_evaluations": counter.snapshot().total,
        "wall_time_s": wall_time_s,
        "certificate_satisfied": bool(result.gradient_norm <= task.fmax),
        "termination_reason": result.telemetry.termination_reason,
        "accepted_steps": result.telemetry.accepted_steps,
        "rejected_steps": result.telemetry.rejected_steps,
        "line_search_evaluations": result.telemetry.line_search_evaluations,
        "accepted_secants": result.telemetry.accepted_secants,
        "rejected_secants": result.telemetry.rejected_secants,
        "initial": asdict(initial_observation),
        "final": asdict(final_observation),
        "delta_total_energy_eV": final_observation.total_energy - initial_observation.total_energy,
        "delta_true_energy_eV": final_observation.true_energy - initial_observation.true_energy,
        "direction_progress_A": direction_progress,
        "orthogonal_displacement_norm_A": float(np.linalg.norm(orthogonal)),
        "gradient_norm_eV_per_A": result.gradient_norm,
    }
    if result.telemetry.termination_reason != "line_search_failed":
        return row

    if not np.array_equal(records[-1][0], final_flat):
        raise RuntimeError("finalization record does not match the returned failed-line state")
    trial_records = records[-(_SAFE_LBFGS_MAX_LINE_TRIALS + 1) : -1]
    if len(trial_records) != _SAFE_LBFGS_MAX_LINE_TRIALS:
        raise RuntimeError("failed line-search record is incomplete")
    base_positions, base_observation = records[-(_SAFE_LBFGS_MAX_LINE_TRIALS + 2)]
    base_parts = proposal.evaluate_parts(base_positions, task.initial_state)
    active = np.repeat(result.state.movable_mask, 3)
    component_gradients = {
        "true": base_parts.true_gradient[active],
        "bias": base_parts.bias_gradient[active],
        "softening": base_parts.softening_gradient[active],
        "total": base_parts.total_gradient[active],
    }
    base_energies = {
        "true": base_observation.true_energy,
        "bias": base_observation.bias_energy,
        "softening": base_observation.softening_energy,
        "total": base_observation.total_energy,
    }
    line_scan = []
    for trial_index, (positions, observation) in enumerate(trial_records):
        delta = (positions - base_positions)[active]
        observed_energies = {
            "true": observation.true_energy,
            "bias": observation.bias_energy,
            "softening": observation.softening_energy,
            "total": observation.total_energy,
        }
        predicted = {
            name: float(np.dot(gradient, delta))
            for name, gradient in component_gradients.items()
        }
        actual = {
            name: observed_energies[name] - base_energies[name]
            for name in base_energies
        }
        line_scan.append(
            {
                "trial_index": trial_index,
                "relative_alpha": 2.0 ** (-trial_index),
                "active_displacement_norm_A": float(np.linalg.norm(delta)),
                "max_atomic_displacement_A": float(
                    np.linalg.norm(delta.reshape(-1, 3), axis=1).max()
                ),
                "predicted_first_order_change_eV": predicted,
                "actual_change_eV": actual,
                "armijo_residual_eV": (
                    actual["total"]
                    - _SAFE_LBFGS_ARMIJO_C1 * predicted["total"]
                ),
            }
        )
    row["failed_line_search"] = {
        "base_energy_eV": base_energies,
        "repeat_base_energy_eV": {
            "true": base_parts.true_energy,
            "bias": base_parts.bias_energy,
            "softening": base_parts.softening_energy,
            "total": base_parts.total_energy,
        },
        "base_energy_repeat_error_eV": {
            name: getattr(base_parts, f"{name}_energy") - value
            for name, value in base_energies.items()
        },
        "line_scan": line_scan,
    }
    return row


def _without_softening_cutoff(task):
    source = task.softening
    if source is None:
        raise ValueError("task has no local-softening model")
    softening = LocalSofteningModel(
        list(source.terms),
        cell=source.cell,
        pbc=source.pbc,
        penalty=source.penalty,
        xi=source.xi,
        reference_scaled_xi=source.reference_scaled_xi,
        cutoff=None,
        adaptive_strength=source.adaptive_strength,
        max_strength_scale=source.max_strength_scale,
        deviation_scale=source.deviation_scale,
    )
    return replace(task, softening=softening)


def _calculator(gate, resources, *, dtype: str):
    if dtype == "float32":
        return gate._calculator("cuo", resources)
    if dtype != "float64":
        raise ValueError("dtype must be float32 or float64")
    from mace.calculators import MACECalculator

    from pamssw.mace_batch import MACEBatchCalculator

    source_gate = gate._load_source_gate()
    prior = source_gate._load_prior_harness()
    production = prior._load_production_runner()
    settings = dict(production.CALCULATOR_CONFIG)
    settings.update(default_dtype="float64", inference_precision="float64")
    return MACEBatchCalculator(
        MACECalculator(model_paths=str(resources["model"]), **settings)
    )


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
        no_cutoff_task = _without_softening_cutoff(task)
        variants = (
            ("safe_lbfgs_with_ls", task, "float32"),
            ("safe_lbfgs_ls_no_cutoff", no_cutoff_task, "float32"),
            ("safe_lbfgs_with_ls_float64", task, "float64"),
            ("safe_lbfgs_ls_no_cutoff_float64", no_cutoff_task, "float64"),
            ("safe_lbfgs_without_ls", replace(task, softening=None), "float32"),
            ("fire_with_ls", task, "float32"),
        )
        for variant, variant_task, calculator_dtype in variants:
            row = _replay_row(
                depth=depth,
                variant=variant,
                task=variant_task,
                calculator=_calculator(
                    gate,
                    resources,
                    dtype=calculator_dtype,
                ),
                calculator_dtype=calculator_dtype,
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
