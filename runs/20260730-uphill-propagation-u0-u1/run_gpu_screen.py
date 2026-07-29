#!/usr/bin/env python3
"""Run the pre-registered fixed-prefix C60/PdO GPU mechanism screen."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
from types import ModuleType
from typing import Any, Sequence

import numpy as np

from pamssw import validated_ls_ssw_config
from pamssw.accounting import EvaluationPurpose
from pamssw.calculators import ASECalculator
from pamssw.proposal_replay import capture_proposal_task
from pamssw.relax import has_force_convergence_certificate
from pamssw.walker import SurfaceWalker


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
FROZEN_RUNNER_PATH = (
    REPO_ROOT
    / "runs"
    / "20260728-safe-lbfgs-200-production"
    / "run_production.py"
)
C60_PROFILE = "c60_direction_efficient_validated_20260729"
SYSTEMS = ("c60", "pdo")
CALIBRATION_SEEDS = (1001, 1002, 1003, 1004)
MIN_CALIBRATION_TASKS = 2
EVALUATION_SPECS = (
    (2001, 1),
    (2002, 3),
    (2003, 5),
    (2004, 8),
    (2005, 1),
    (2006, 3),
    (2007, 5),
    (2008, 8),
)

sys.path.insert(0, str(RUN_ROOT))

from analyze_results import analyze_rows  # noqa: E402
from run_ablation import (  # noqa: E402
    base_sigma_from_task,
    calibrate_fixed_parameters,
    measure_inner_curvature,
    run_task_arms,
    task_sha256,
)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _base_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_uphill_u0_u1_frozen_runner",
        FROZEN_RUNNER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen production runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config(system: str, seed: int, base: ModuleType):
    scratch = RUN_ROOT / "scratch"
    if system == "c60":
        source = validated_ls_ssw_config(
            C60_PROFILE,
            output_dir=scratch,
            max_trials=1,
            rng_seed=seed,
            max_force_evals=None,
        )
    elif system == "pdo":
        source = base.build_config(system, scratch)
    else:
        raise ValueError(f"unknown system: {system}")
    return replace(
        source,
        max_trials=1,
        max_force_evals=None,
        oracle_candidates=4,
        rng_seed=seed,
        direction_diagnostics_enabled=False,
        direction_diagnostics_path=None,
        accepted_structures_log=None,
        accepted_structures_dir=None,
        write_proposal_minima=False,
        proposal_minima_dir=None,
        write_relaxation_trajectories=False,
        relaxation_trajectory_dir=None,
    )


def _config_projection(config) -> dict[str, Any]:
    fields = (
        "max_steps_per_walk",
        "target_uphill_energy",
        "target_negative_curvature",
        "oracle_candidates",
        "proposal_relax_steps",
        "proposal_fmax",
        "proposal_optimizer",
        "proposal_trust_radius",
        "target_step_rms",
        "max_step_rms",
        "step_rms_scope",
        "step_active_threshold",
        "hvp_epsilon",
        "min_step_scale",
        "max_step_scale",
        "step_error_tolerance",
        "step_gamma_down",
        "step_gamma_up",
        "quench_fmax",
        "quench_maxiter",
        "quench_optimizer",
        "quench_fallback_optimizer",
    )
    values = asdict(config)
    return {field: values[field] for field in fields}


def _runtime_provenance(base: ModuleType) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    packages = {}
    for distribution in ("numpy", "scipy", "ase", "torch", "mace-torch"):
        packages[distribution] = importlib.metadata.version(distribution)
    return {
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "packages": packages,
        "cuda_runtime": str(torch.version.cuda),
        "cuda_device": str(torch.cuda.get_device_name(0)),
        "model_path": str(base.MODEL_PATH),
        "model_sha256": _sha256(base.MODEL_PATH),
        "calculator": dict(base.CALCULATOR_CONFIG),
        "runner_path": str(FROZEN_RUNNER_PATH.relative_to(REPO_ROOT)),
        "runner_sha256": _sha256(FROZEN_RUNNER_PATH),
        "script_sha256": _sha256(Path(__file__)),
        "inputs": {
            system: {
                "path": str(base.INPUT_PATHS[system]),
                "sha256": _sha256(base.INPUT_PATHS[system]),
            }
            for system in SYSTEMS
        },
    }


def _bootstrap(system: str, calculator, config, base: ModuleType):
    raw = base.load_state(system)
    walker = SurfaceWalker(
        calculator=calculator,
        config=config,
        softening_enabled=False,
    )
    started = perf_counter()
    result = walker.relax_true_minimum(
        raw,
        quench_purpose=EvaluationPurpose.BOOTSTRAP_TRUE_QUENCH,
    )
    wall_time_s = perf_counter() - started
    counts = walker.calculator.snapshot()
    if counts.count(EvaluationPurpose.UNATTRIBUTED) != 0:
        raise RuntimeError("bootstrap contains unattributed evaluations")
    return result.state, _bootstrap_record(
        result,
        counts,
        wall_time_s=wall_time_s,
        fmax=config.quench_fmax,
    )


def _bootstrap_record(
    result,
    counts,
    *,
    wall_time_s: float,
    fmax: float,
) -> dict[str, Any]:
    return {
        "energy_eV": float(result.energy),
        "gradient_norm": float(result.gradient_norm),
        "certificate_satisfied": has_force_convergence_certificate(
            result,
            fmax,
        ),
        "force_evaluations": counts.total,
        "purpose_counts": counts.as_dict(),
        "wall_time_s": wall_time_s,
    }


def _capture_record(
    *,
    system: str,
    task_id: str,
    seed: int,
    bias_count: int,
    captured,
) -> dict[str, Any]:
    counts = captured.evaluation_counts
    return {
        "system": system,
        "task_id": task_id,
        "seed": seed,
        "target_bias_count": bias_count,
        "source_task_sha256": task_sha256(captured.task),
        "sigma": float(captured.task.biases[-1].sigma),
        "weight": float(captured.task.biases[-1].weight),
        "force_evaluations": counts.total,
        "purpose_counts": counts.as_dict(),
    }


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run(*, systems: Sequence[str], output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    invalid = set(systems) - set(SYSTEMS)
    if invalid:
        raise ValueError(f"unknown systems: {sorted(invalid)}")

    base = _base_runner()
    payload: dict[str, Any] = {
        "schema_version": 2,
        "scope": (
            "conditional fixed-prefix Gaussian propagation screen; "
            "no local-softening term; not a full-search comparison"
        ),
        "protocol": {
            "systems": list(systems),
            "calibration_seeds": list(CALIBRATION_SEEDS),
            "evaluation_specs": [
                {"seed": seed, "target_bias_count": count}
                for seed, count in EVALUATION_SPECS
            ],
            "arms": [
                "current_full",
                "curvature_matched_no_feedback",
                "fixed_calibrated",
            ],
            "observer_only_force_evaluations_expected": 0,
            "minimum_calibration_tasks": MIN_CALIBRATION_TASKS,
            "curvature_arm_weight_bounds": (
                "shared production bias_weight_min/bias_weight_max"
            ),
            "right_censor_rule": (
                "a calibration or evaluation prefix that terminates before "
                "its requested bias count is recorded as capture_failed and "
                "is not replaced"
            ),
        },
        "provenance": _runtime_provenance(base),
        "systems": {},
        "rows": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)

    raw_mace = base._calculator()
    calculator = ASECalculator(raw_mace)
    for system in systems:
        print(f"[{system}] bootstrap", flush=True)
        bootstrap_config = _config(system, 0, base)
        seed_state, bootstrap = _bootstrap(
            system,
            calculator,
            bootstrap_config,
            base,
        )
        system_payload: dict[str, Any] = {
            "config": _config_projection(bootstrap_config),
            "input_state": {
                "atom_count": seed_state.n_atoms,
                "fixed_count": int(np.count_nonzero(seed_state.fixed_mask)),
                "pbc": list(seed_state.pbc),
            },
            "bootstrap": bootstrap,
            "calibration_tasks": [],
            "calibration_failures": [],
            "evaluation_tasks": [],
            "capture_failures": [],
        }
        payload["systems"][system] = system_payload
        _write(output, payload)

        print(f"[{system}] calibration", flush=True)
        calibration_records = []
        for seed in CALIBRATION_SEEDS:
            config = _config(system, seed, base)
            try:
                captured = capture_proposal_task(
                    seed_state,
                    calculator,
                    config,
                    target_bias_count=1,
                )
            except RuntimeError as error:
                failure = {
                    "task_id": f"{system}-cal-{seed}",
                    "seed": seed,
                    "target_bias_count": 1,
                    "error": str(error),
                }
                system_payload["calibration_failures"].append(failure)
                print(
                    f"[{system}] calibration right-censored "
                    f"seed={seed}: {error}",
                    flush=True,
                )
                _write(output, payload)
                continue
            record = _capture_record(
                system=system,
                task_id=f"{system}-cal-{seed}",
                seed=seed,
                bias_count=1,
                captured=captured,
            )
            calibration_records.append(record)
            system_payload["calibration_tasks"].append(record)
            print(
                f"[{system}] calibration seed={seed} "
                f"calls={record['force_evaluations']}",
                flush=True,
            )
            _write(output, payload)

        if len(calibration_records) < MIN_CALIBRATION_TASKS:
            raise RuntimeError(
                f"{system} produced only {len(calibration_records)} "
                "calibration tasks"
            )
        calibration = calibrate_fixed_parameters(calibration_records)[system]
        system_payload["fixed_calibration"] = calibration
        fixed_sigma = float(calibration["sigma"])
        fixed_weight = float(calibration["weight"])

        for seed, bias_count in EVALUATION_SPECS:
            task_id = f"{system}-eval-{seed}-bias{bias_count}"
            print(f"[{system}] capture {task_id}", flush=True)
            config = _config(system, seed, base)
            try:
                captured = capture_proposal_task(
                    seed_state,
                    calculator,
                    config,
                    target_bias_count=bias_count,
                )
            except RuntimeError as error:
                failure = {
                    "task_id": task_id,
                    "seed": seed,
                    "target_bias_count": bias_count,
                    "error": str(error),
                }
                system_payload["capture_failures"].append(failure)
                print(f"[{system}] right-censored {task_id}: {error}", flush=True)
                _write(output, payload)
                continue

            task = captured.task
            capture = _capture_record(
                system=system,
                task_id=task_id,
                seed=seed,
                bias_count=bias_count,
                captured=captured,
            )
            base_sigma = base_sigma_from_task(
                task,
                target_step_rms=config.target_step_rms,
                max_step_rms=config.max_step_rms,
                step_rms_scope=config.step_rms_scope,
                active_threshold=config.step_active_threshold,
            )
            inner_curvature, curvature_counts = measure_inner_curvature(
                task,
                calculator,
                hvp_epsilon=config.hvp_epsilon,
            )
            if curvature_counts.count(EvaluationPurpose.UNATTRIBUTED) != 0:
                raise RuntimeError("curvature measurement is unattributed")
            characterization = {
                "base_sigma": base_sigma,
                "inner_curvature": inner_curvature,
                "target_negative_curvature": (
                    config.target_negative_curvature
                ),
                "force_evaluations": curvature_counts.total,
                "purpose_counts": curvature_counts.as_dict(),
            }
            rows = run_task_arms(
                task,
                system=system,
                task_id=task_id,
                calculator_factory=lambda: calculator,
                optimizer=config.proposal_optimizer,
                base_sigma=base_sigma,
                inner_curvature=inner_curvature,
                target_negative_curvature=(
                    config.target_negative_curvature
                ),
                fixed_sigma=fixed_sigma,
                fixed_weight=fixed_weight,
                bias_weight_min=config.bias_weight_min,
                bias_weight_max=config.bias_weight_max,
            )
            system_payload["evaluation_tasks"].append(
                {
                    **capture,
                    "characterization": characterization,
                }
            )
            payload["rows"].extend(rows)
            print(
                f"[{system}] replayed {task_id}: "
                + ", ".join(
                    f"{row['arm_id']}={row['force_evaluations']}"
                    for row in rows
                ),
                flush=True,
            )
            _write(output, payload)

    payload["summary"] = analyze_rows(payload["rows"])
    payload["validation"] = {
        "attempted_task_count": len(systems) * len(EVALUATION_SPECS),
        "completed_task_count": len(payload["rows"]) // 3,
        "row_count": len(payload["rows"]),
        "observer_only_force_evaluations": sum(
            int(row["observer_only_force_evaluations"])
            for row in payload["rows"]
        ),
        "unattributed_force_evaluations": sum(
            int(
                row["purpose_counts"][
                    EvaluationPurpose.UNATTRIBUTED.value
                ]
            )
            for row in payload["rows"]
        ),
    }
    _write(output, payload)
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=SYSTEMS,
        default=list(SYSTEMS),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RUN_ROOT / "gpu_screen.json",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = run(systems=args.systems, output=args.output)
    print(
        json.dumps(result["validation"], indent=2, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
