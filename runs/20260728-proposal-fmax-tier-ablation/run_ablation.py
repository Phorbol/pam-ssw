#!/usr/bin/env python3
"""Run the fixed one-bias proposal-fmax and landing-quench ablation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from pamssw.accounting import EvaluationCounts, EvaluationPurpose
from pamssw.calculators import ASECalculator
from pamssw.proposal_replay import (
    capture_proposal_task,
    proposal_task_to_payload,
    replay_proposal_task,
)
from pamssw.state import State
from pamssw.walker import ProposalRelaxationTask, SurfaceWalker


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
PRODUCTION_RUNNER_PATH = (
    REPO_ROOT
    / "runs"
    / "20260728-safe-lbfgs-200-production"
    / "run_production.py"
)
SYSTEMS = ("c60", "pdo")
SEEDS = tuple(range(42, 50))
PROPOSAL_OPTIMIZER = "safe-lbfgs-total"
SAFE_HISTORY_LIMIT = 10
TARGET_BIAS_COUNT = 1
SOFTENING_ENABLED = False
LANDING_OPTIMIZER = "scipy-lbfgsb"
LANDING_FMAX = 0.05
LANDING_MAXITER = 400
TRUE_PES_OBJECTIVE = "true_mace_pes_no_bias_no_softening"
RUNTIME_PACKAGES = {
    "numpy": "numpy",
    "scipy": "scipy",
    "ase": "ase",
    "torch": "torch",
    "mace": "mace-torch",
}


@dataclass(frozen=True)
class Arm:
    arm_id: str
    fmax_eV_per_A: float


ARMS = (Arm("fmax-0.05", 0.05), Arm("fmax-0.10", 0.10))


@dataclass(frozen=True)
class Preflight:
    metadata: Mapping[str, Any]
    production_states: Mapping[str, State]


@dataclass(frozen=True)
class BootstrapResult:
    state: State
    energy_eV: float
    gradient_norm: float
    n_iter: int
    wall_time_s: float
    evaluation_counts: EvaluationCounts
    termination_reason: str
    certificate_satisfied: bool


def _production_module():
    spec = importlib.util.spec_from_file_location(
        "_proposal_fmax_production_runner", PRODUCTION_RUNNER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the current production runner")
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def position_sha256(positions: object) -> str:
    coordinates = np.asarray(positions, dtype=np.dtype("<f8"))
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("positions must have shape (n_atoms, 3)")
    canonical = np.array(
        coordinates, dtype=np.dtype("<f8"), order="C", copy=True
    )
    digest = sha256()
    digest.update(str(canonical.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical.tobytes())
    return digest.hexdigest()


def state_payload(state: State) -> dict[str, Any]:
    return {
        "numbers": state.numbers.tolist(),
        "positions": state.positions.tolist(),
        "cell": None if state.cell is None else state.cell.tolist(),
        "pbc": list(state.pbc),
        "fixed_mask": state.fixed_mask.tolist(),
    }


def state_from_payload(payload: Mapping[str, Any]) -> State:
    return State(
        numbers=np.asarray(payload["numbers"], dtype=int),
        positions=np.asarray(payload["positions"], dtype=float),
        cell=(
            None
            if payload.get("cell") is None
            else np.asarray(payload["cell"], dtype=float)
        ),
        pbc=tuple(bool(value) for value in payload["pbc"]),
        fixed_mask=np.asarray(payload["fixed_mask"], dtype=bool),
    )


def source_task_sha256(task: ProposalRelaxationTask) -> str:
    return _canonical_sha256(proposal_task_to_payload(task))


def fixed_biased_pes_sha256(task: ProposalRelaxationTask) -> str:
    payload = proposal_task_to_payload(task)
    del payload["fmax"]
    return _canonical_sha256(payload)


def task_for_arm(
    source: ProposalRelaxationTask, arm: Arm
) -> ProposalRelaxationTask:
    if source.softening is not None:
        raise ValueError("proposal-fmax ablation requires softening=False")
    return replace(source, fmax=arm.fmax_eV_per_A)


def _current_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _tracked_worktree_clean() -> bool:
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return not completed.stdout.strip()


def _runtime_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for name, distribution in RUNTIME_PACKAGES.items():
        versions[name] = importlib.metadata.version(distribution)
    return versions


def _cuda_info() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return {
        "available": True,
        "runtime_version": str(torch.version.cuda),
        "device_name": str(torch.cuda.get_device_name(0)),
    }


def _safe_lbfgs_default_history_limit() -> int:
    from pamssw.relax import _SAFE_LBFGS_MEMORY

    return int(_SAFE_LBFGS_MEMORY)


def preflight(
    *,
    expected_git_commit: str,
    runtime_probe: Callable[[], Mapping[str, Any]] = _runtime_versions,
    cuda_probe: Callable[[], Mapping[str, Any]] = _cuda_info,
) -> Preflight:
    actual_commit = _current_commit()
    if actual_commit != expected_git_commit:
        raise RuntimeError(
            f"execution commit mismatch: expected {expected_git_commit}, "
            f"got {actual_commit}"
        )
    if not _tracked_worktree_clean():
        raise RuntimeError("tracked worktree is not clean")
    if _safe_lbfgs_default_history_limit() != SAFE_HISTORY_LIMIT:
        raise RuntimeError("safe L-BFGS history is not the fixed value 10")
    production = _production_module()
    for path in (*production.INPUT_PATHS.values(), production.MODEL_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)
    cuda = dict(cuda_probe())
    if cuda.get("available") is not True:
        raise RuntimeError("CUDA is unavailable")
    states = {
        system: production.load_state(system) for system in SYSTEMS
    }
    return Preflight(
        metadata={
            "schema_version": 1,
            "execution_commit": actual_commit,
            "model": {
                "path": str(production.MODEL_PATH),
                "sha256": _sha256(production.MODEL_PATH),
            },
            "inputs": {
                system: {
                    "path": str(production.INPUT_PATHS[system]),
                    "sha256": _sha256(production.INPUT_PATHS[system]),
                }
                for system in SYSTEMS
            },
            "runtime_versions": {
                key: str(value) for key, value in runtime_probe().items()
            },
            "cuda": cuda,
            "calculator": dict(production.CALCULATOR_CONFIG),
            "safe_lbfgs_default_history_limit": SAFE_HISTORY_LIMIT,
        },
        production_states=states,
    )


def build_capture_config(system: str, case_dir: Path, *, seed: int):
    production = _production_module()
    return replace(
        production.build_config(system, case_dir),
        max_steps_per_walk=TARGET_BIAS_COUNT,
        rng_seed=seed,
    )


def _calculator():
    production = _production_module()
    from mace.calculators import MACECalculator

    raw = MACECalculator(
        model_paths=str(production.MODEL_PATH),
        **production.CALCULATOR_CONFIG,
    )
    return ASECalculator(raw)


def _counts_close(counts: EvaluationCounts) -> bool:
    return (
        counts.total == sum(counts.as_dict().values())
        and counts.count(EvaluationPurpose.UNATTRIBUTED) == 0
    )


def bootstrap_minimum(
    system: str,
    raw_state: State,
    config,
    calculator: object,
    *,
    walker_factory=SurfaceWalker,
) -> BootstrapResult:
    bootstrap_config = replace(
        config,
        quench_optimizer=LANDING_OPTIMIZER,
        quench_fmax=LANDING_FMAX,
        quench_maxiter=LANDING_MAXITER,
    )
    walker = walker_factory(
        calculator=calculator,
        config=bootstrap_config,
        softening_enabled=SOFTENING_ENABLED,
    )
    started = perf_counter()
    relaxation = walker.relax_true_minimum(
        raw_state,
        quench_purpose=EvaluationPurpose.STARTER_TRUE_QUENCH,
    )
    wall_time_s = perf_counter() - started
    evaluation_counts = walker.calculator.snapshot()
    certificate = bool(
        math.isfinite(relaxation.energy)
        and math.isfinite(relaxation.gradient_norm)
        and relaxation.gradient_norm <= LANDING_FMAX
        and np.all(np.isfinite(relaxation.state.positions))
    )
    if not _counts_close(evaluation_counts):
        raise RuntimeError(f"{system} bootstrap purpose accounting does not close")
    if not certificate:
        raise RuntimeError(f"{system} bootstrap did not reach its force certificate")
    return BootstrapResult(
        state=relaxation.state,
        energy_eV=float(relaxation.energy),
        gradient_norm=float(relaxation.gradient_norm),
        n_iter=int(relaxation.n_iter),
        wall_time_s=wall_time_s,
        evaluation_counts=evaluation_counts,
        termination_reason=relaxation.telemetry.termination_reason,
        certificate_satisfied=certificate,
    )


def execute_proposal(
    source: ProposalRelaxationTask,
    arm: Arm,
    calculator: object,
    *,
    replay_fn=None,
) -> dict[str, Any]:
    task = task_for_arm(source, arm)
    if replay_fn is None:
        replay_fn = replay_proposal_task
    replay = replay_fn(task, calculator, optimizer=PROPOSAL_OPTIMIZER)
    result = replay.result
    counts = replay.evaluation_counts
    if (
        not _counts_close(counts)
        or counts.count(EvaluationPurpose.BIASED_PROPOSAL_RELAX)
        != counts.total
        or counts.total != result.telemetry.evaluator_calls
    ):
        raise RuntimeError("proposal purpose accounting does not close")
    return {
        "optimizer": PROPOSAL_OPTIMIZER,
        "safe_history_limit": SAFE_HISTORY_LIMIT,
        "fmax_eV_per_A": arm.fmax_eV_per_A,
        "maxiter": int(task.maxiter),
        "coordinate_trust_radius_A": task.coordinate_trust_radius,
        "initial_positions_sha256": position_sha256(
            task.initial_state.positions
        ),
        "final": {
            "biased_energy_eV": float(result.energy),
            "max_active_force_eV_per_A": float(result.gradient_norm),
            "positions_sha256": position_sha256(result.state.positions),
            "state": state_payload(result.state),
        },
        "certificate_satisfied": bool(replay.certificate_satisfied),
        "n_iter": int(result.n_iter),
        "termination_reason": result.telemetry.termination_reason,
        "outcome_class": result.outcome_class.value,
        "telemetry": asdict(result.telemetry),
        "force_evaluations": counts.total,
        "purpose_counts": counts.as_dict(),
        "wall_time_s": float(replay.wall_time_s),
    }


def execute_landing(
    initial_state: State,
    config,
    calculator: object,
    *,
    walker_factory=SurfaceWalker,
) -> dict[str, Any]:
    landing_config = replace(
        config,
        quench_optimizer=LANDING_OPTIMIZER,
        quench_fmax=LANDING_FMAX,
        quench_maxiter=LANDING_MAXITER,
    )
    walker = walker_factory(
        calculator=calculator,
        config=landing_config,
        softening_enabled=SOFTENING_ENABLED,
    )
    started = perf_counter()
    result = walker.relax_true_minimum(
        initial_state,
        quench_purpose=EvaluationPurpose.LANDING_TRUE_QUENCH,
    )
    wall_time_s = perf_counter() - started
    counts = walker.calculator.snapshot()
    certificate = bool(
        math.isfinite(result.energy)
        and math.isfinite(result.gradient_norm)
        and result.gradient_norm <= LANDING_FMAX
        and np.all(np.isfinite(result.state.positions))
    )
    if (
        not _counts_close(counts)
        or counts.count(EvaluationPurpose.LANDING_TRUE_QUENCH)
        + counts.count(EvaluationPurpose.POST_RELAX_VALIDATION)
        != counts.total
        or counts.count(EvaluationPurpose.LANDING_TRUE_QUENCH)
        != result.telemetry.evaluator_calls
    ):
        raise RuntimeError("landing purpose accounting does not close")
    return {
        "optimizer": LANDING_OPTIMIZER,
        "fmax_eV_per_A": LANDING_FMAX,
        "maxiter": LANDING_MAXITER,
        "objective": TRUE_PES_OBJECTIVE,
        "initial": {
            "positions_sha256": position_sha256(initial_state.positions),
            "state": state_payload(initial_state),
        },
        "final": {
            "energy_eV": float(result.energy),
            "max_active_force_eV_per_A": float(result.gradient_norm),
            "positions_sha256": position_sha256(result.state.positions),
            "state": state_payload(result.state),
        },
        "certificate_satisfied": certificate,
        "n_iter": int(result.n_iter),
        "termination_reason": result.telemetry.termination_reason,
        "outcome_class": result.outcome_class.value,
        "telemetry": asdict(result.telemetry),
        "force_evaluations": counts.total,
        "purpose_counts": counts.as_dict(),
        "wall_time_s": wall_time_s,
    }


def _stable_config_payload(config: object, partial_dir: Path) -> dict[str, Any]:
    payload = asdict(config)
    prefix = f"{partial_dir}{Path('/')}"
    return {
        key: (
            value.removeprefix(prefix)
            if isinstance(value, str) and value.startswith(prefix)
            else value
        )
        for key, value in payload.items()
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run(
    *,
    output_dir: Path,
    expected_git_commit: str,
    preflight_only: bool = False,
    preflight_fn: Callable[..., Preflight] = preflight,
    calculator_factory: Callable[[], object] = _calculator,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    partial_dir = output_dir.with_name(f"{output_dir.name}.partial")
    for path in (output_dir, partial_dir):
        if path.exists():
            raise FileExistsError(path)
    checked = preflight_fn(expected_git_commit=expected_git_commit)
    if preflight_only:
        return dict(checked.metadata)

    partial_dir.mkdir(parents=True)
    systems: dict[str, Any] = {}
    for system in SYSTEMS:
        system_dir = partial_dir / system
        system_dir.mkdir()
        bootstrap_calculator = calculator_factory()
        capture_calculator = calculator_factory()
        proposal_calculators = {
            arm.arm_id: calculator_factory() for arm in ARMS
        }
        landing_calculators = {
            arm.arm_id: calculator_factory() for arm in ARMS
        }
        bootstrap_config = build_capture_config(
            system, system_dir / "seed-42", seed=42
        )
        bootstrap = bootstrap_minimum(
            system,
            checked.production_states[system],
            bootstrap_config,
            bootstrap_calculator,
        )
        task_records = []
        rows = []
        for seed in SEEDS:
            case_dir = system_dir / f"seed-{seed}"
            case_dir.mkdir(parents=True, exist_ok=True)
            config = build_capture_config(system, case_dir, seed=seed)
            captured = capture_proposal_task(
                bootstrap.state,
                capture_calculator,
                config,
                target_bias_count=TARGET_BIAS_COUNT,
            )
            source = captured.task
            capture_counts = captured.evaluation_counts
            if (
                len(source.biases) != TARGET_BIAS_COUNT
                or source.softening is not None
                or source.maxiter != config.proposal_relax_steps
                or source.fmax != config.proposal_fmax
                or not _counts_close(capture_counts)
            ):
                raise RuntimeError("captured fixed proposal task violates protocol")
            source_hash = source_task_sha256(source)
            pes_hash = fixed_biased_pes_sha256(source)
            task_records.append(
                {
                    "system": system,
                    "seed": seed,
                    "source_task_sha256": source_hash,
                    "fixed_biased_pes_sha256": pes_hash,
                    "capture_evaluation_counts": capture_counts.as_dict(),
                    "task": proposal_task_to_payload(source),
                }
            )
            for arm in ARMS:
                proposal = execute_proposal(
                    source, arm, proposal_calculators[arm.arm_id]
                )
                landing = execute_landing(
                    state_from_payload(proposal["final"]["state"]),
                    config,
                    landing_calculators[arm.arm_id],
                )
                if (
                    landing["initial"]["positions_sha256"]
                    != proposal["final"]["positions_sha256"]
                ):
                    raise RuntimeError(
                        "landing quench did not start from proposal endpoint"
                    )
                rows.append(
                    {
                        "system": system,
                        "seed": seed,
                        "arm_id": arm.arm_id,
                        "source_task_sha256": source_hash,
                        "fixed_biased_pes_sha256": pes_hash,
                        "proposal": proposal,
                        "landing": landing,
                    }
                )
        systems[system] = {
            "dedup_energy_tol_eV": float(
                bootstrap_config.dedup_energy_tol
            ),
            "dedup_rmsd_tol_A": float(bootstrap_config.dedup_rmsd_tol),
            "effective_capture_config_seed42": _stable_config_payload(
                bootstrap_config, partial_dir
            ),
            "bootstrap": {
                "optimizer": LANDING_OPTIMIZER,
                "fmax_eV_per_A": LANDING_FMAX,
                "maxiter": LANDING_MAXITER,
                "objective": TRUE_PES_OBJECTIVE,
                "energy_eV": bootstrap.energy_eV,
                "max_active_force_eV_per_A": bootstrap.gradient_norm,
                "positions_sha256": position_sha256(
                    bootstrap.state.positions
                ),
                "state": state_payload(bootstrap.state),
                "n_iter": bootstrap.n_iter,
                "wall_time_s": bootstrap.wall_time_s,
                "purpose_counts": bootstrap.evaluation_counts.as_dict(),
                "force_evaluations": bootstrap.evaluation_counts.total,
                "termination_reason": bootstrap.termination_reason,
                "certificate_satisfied": (
                    bootstrap.certificate_satisfied
                ),
            },
            "tasks": task_records,
            "rows": rows,
        }

    summary = {
        **dict(checked.metadata),
        "protocol": {
            "systems": list(SYSTEMS),
            "seeds": list(SEEDS),
            "target_bias_count": TARGET_BIAS_COUNT,
            "softening_enabled": SOFTENING_ENABLED,
            "proposal_optimizer": PROPOSAL_OPTIMIZER,
            "safe_history_limit": SAFE_HISTORY_LIMIT,
            "proposal_arms": [asdict(arm) for arm in ARMS],
            "proposal_maxiter": "unchanged from current production config",
            "bootstrap_optimizer": LANDING_OPTIMIZER,
            "bootstrap_fmax_eV_per_A": LANDING_FMAX,
            "bootstrap_maxiter": LANDING_MAXITER,
            "bootstrap_shared_by_all_seeds_and_arms": True,
            "landing_optimizer": LANDING_OPTIMIZER,
            "landing_fmax_eV_per_A": LANDING_FMAX,
            "landing_maxiter": LANDING_MAXITER,
            "landing_objective": TRUE_PES_OBJECTIVE,
        },
        "task_count": len(SYSTEMS) * len(SEEDS),
        "row_count": len(SYSTEMS) * len(SEEDS) * len(ARMS),
        "systems": systems,
        "claim_boundary": (
            "This is a capture-policy-conditioned fixed one-bias local "
            "proposal-fmax ablation with softening disabled and one shared "
            "SciPy true-PES fmax=0.05 bootstrap per system. It tests whether "
            "a looser proposal certificate reduces proposal cost without "
            "changing the subsequent true-PES landing under the current "
            "archive matcher; it does not support full-SSW superiority claims."
        ),
    }
    _write_json(partial_dir / "summary.json", summary)
    partial_dir.rename(output_dir)
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RUN_ROOT / "output")
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    payload = run(
        output_dir=args.output,
        expected_git_commit=args.expected_git_commit,
        preflight_only=args.preflight_only,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
