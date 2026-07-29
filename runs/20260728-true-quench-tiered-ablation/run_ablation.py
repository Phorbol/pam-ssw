#!/usr/bin/env python3
"""Run the fixed raw-landing, two-tier true-quench optimizer ablation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

from ase.io import read
import numpy as np

from pamssw.accounting import EvalCounter, EvaluationCounts, EvaluationPurpose
from pamssw.calculators import ASECalculator
from pamssw.relax import Relaxer
from pamssw.result import RelaxResult
from pamssw.state import State
from pamssw.walker import SurfaceWalker


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
PRODUCTION_RUNNER_PATH = (
    REPO_ROOT
    / "runs"
    / "20260728-safe-lbfgs-200-production"
    / "run_production.py"
)
SYSTEMS = ("c60", "pdo")
CAPTURE_TRIALS = 16
MAXITER = 400
STAGES = (("loose", 0.05), ("refine", 0.01))
OBJECTIVE = "true_mace_pes_no_bias_no_softening"
SAFE_LBFGS_DEFAULT_HISTORY_LIMIT = 10
RUNTIME_PACKAGES = {
    "numpy": "numpy",
    "scipy": "scipy",
    "ase": "ase",
    "torch": "torch",
    "mace": "mace-torch",
}
TRUE_QUENCH_PATTERN = re.compile(
    r"^trial(?P<trial>\d{4})_proposal(?P<proposal>\d{3})_true_quench\.xyz$"
)


@dataclass(frozen=True)
class Arm:
    arm_id: str
    optimizer: str
    safe_history_limit: int | None


ARMS = (
    Arm("scipy-lbfgsb", "scipy-lbfgsb", None),
    Arm("safe-lbfgs-total", "safe-lbfgs-total", 10),
    Arm("ase-fire2", "ase-fire2", None),
)


@dataclass(frozen=True)
class LandingTask:
    system: str
    task_index: int
    trial_index: int
    proposal_index: int
    source_trajectory_path: Path
    source_trajectory_sha256: str
    source_frame_index: int
    initial_positions_sha256: str
    state: State


@dataclass(frozen=True)
class Preflight:
    metadata: Mapping[str, Any]
    production_states: Mapping[str, State]


def _production_module():
    spec = importlib.util.spec_from_file_location(
        "_tiered_ablation_production_runner", PRODUCTION_RUNNER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the frozen production runner")
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


def position_sha256(positions: object) -> str:
    coordinates = np.asarray(positions, dtype=np.dtype("<f8"))
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("positions must have shape (n_atoms, 3)")
    canonical = np.array(coordinates, dtype=np.dtype("<f8"), order="C", copy=True)
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
    if _safe_lbfgs_default_history_limit() != SAFE_LBFGS_DEFAULT_HISTORY_LIMIT:
        raise RuntimeError("safe L-BFGS default history limit is not 10")
    production = _production_module()
    if not production.MODEL_PATH.is_file():
        raise FileNotFoundError(production.MODEL_PATH)
    states = {system: production.load_state(system) for system in SYSTEMS}
    inputs = {
        system: {
            "path": str(production.INPUT_PATHS[system]),
            "sha256": _sha256(production.INPUT_PATHS[system]),
        }
        for system in SYSTEMS
    }
    cuda = dict(cuda_probe())
    if cuda.get("available") is not True:
        raise RuntimeError("CUDA is unavailable")
    return Preflight(
        metadata={
            "schema_version": 1,
            "execution_commit": actual_commit,
            "inputs": inputs,
            "model": {
                "path": str(production.MODEL_PATH),
                "sha256": _sha256(production.MODEL_PATH),
            },
            "runtime_versions": {
                key: str(value) for key, value in runtime_probe().items()
            },
            "cuda": cuda,
            "calculator": dict(production.CALCULATOR_CONFIG),
            "safe_lbfgs_default_history_limit": (
                SAFE_LBFGS_DEFAULT_HISTORY_LIMIT
            ),
        },
        production_states=states,
    )


def build_capture_config(system: str, case_dir: Path):
    production = _production_module()
    baseline = production.build_config(system, case_dir)
    return replace(
        baseline,
        max_trials=CAPTURE_TRIALS,
        write_relaxation_trajectories=True,
        relaxation_trajectory_dir=str(case_dir / "trajectories"),
        relaxation_trajectory_stride=1,
    )


def extract_raw_landing_corpus(
    system: str,
    trajectory_dir: Path,
    template_state: State,
) -> tuple[tuple[LandingTask, ...], dict[str, Any]]:
    candidates = []
    for path in sorted(Path(trajectory_dir).glob("*_true_quench.xyz")):
        match = TRUE_QUENCH_PATTERN.match(path.name)
        if match is None:
            continue
        candidates.append(
            (
                int(match.group("trial")),
                int(match.group("proposal")),
                path,
            )
        )
    expected = [(trial, 1) for trial in range(1, CAPTURE_TRIALS + 1)]
    if [(trial, proposal) for trial, proposal, _ in candidates] != expected:
        raise ValueError(
            f"{system} capture must contain one true-quench trajectory "
            f"for each of the first 16 trials"
        )

    tasks = []
    entries = []
    for task_index, (trial_index, proposal_index, path) in enumerate(candidates):
        atoms = read(path, index=0)
        numbers = np.asarray(atoms.numbers, dtype=int)
        if not np.array_equal(numbers, template_state.numbers):
            raise ValueError(f"{system} captured atom identities changed")
        state = State(
            numbers=numbers,
            positions=np.asarray(atoms.positions, dtype=float),
            cell=(
                None
                if template_state.cell is None
                else template_state.cell.copy()
            ),
            pbc=template_state.pbc,
            fixed_mask=template_state.fixed_mask.copy(),
            metadata={
                "system": system,
                "trial_index": trial_index,
                "proposal_index": proposal_index,
                "capture_policy": (
                    "first frame of proposal true-quench trajectory"
                ),
            },
        )
        positions_hash = position_sha256(state.positions)
        task = LandingTask(
            system=system,
            task_index=task_index,
            trial_index=trial_index,
            proposal_index=proposal_index,
            source_trajectory_path=path,
            source_trajectory_sha256=_sha256(path),
            source_frame_index=0,
            initial_positions_sha256=positions_hash,
            state=state,
        )
        tasks.append(task)
        entries.append(
            {
                "system": system,
                "task_index": task_index,
                "trial_index": trial_index,
                "proposal_index": proposal_index,
                "source_trajectory_path": (
                    f"stage_a/{system}/trajectories/{path.name}"
                ),
                "source_trajectory_sha256": task.source_trajectory_sha256,
                "source_frame_index": 0,
                "initial_positions_sha256": positions_hash,
                "state": state_payload(state),
            }
        )
    return tuple(tasks), {
        "system": system,
        "task_count": len(tasks),
        "capture_policy": (
            "first frame of each proposal true-quench trajectory from the "
            "fixed first-16-trial production configuration"
        ),
        "capture_policy_conditioned": True,
        "entries": entries,
    }


def capture_stage_a(
    system: str,
    state: State,
    config,
    calculator: object,
) -> tuple[
    tuple[LandingTask, ...],
    dict[str, Any],
    dict[str, Any],
    EvaluationCounts,
]:
    walker = SurfaceWalker(
        calculator=ASECalculator(calculator),
        config=config,
        softening_enabled=True,
    )
    result = walker.run(state)
    if result.stats.get("n_trials") != CAPTURE_TRIALS:
        raise RuntimeError(f"{system} Stage A did not complete 16 trials")
    tasks, manifest = extract_raw_landing_corpus(
        system, Path(config.relaxation_trajectory_dir), state
    )
    return tasks, manifest, dict(result.stats), walker.calculator.snapshot()


def _counter(calculator: object) -> EvalCounter:
    return EvalCounter(ASECalculator(calculator))


def _calculator():
    production = _production_module()
    from mace.calculators import MACECalculator

    return MACECalculator(
        model_paths=str(production.MODEL_PATH),
        **production.CALCULATOR_CONFIG,
    )


def _count_delta(
    before: EvaluationCounts, after: EvaluationCounts
) -> EvaluationCounts:
    return EvaluationCounts(
        tuple(
            right - left
            for left, right in zip(before.values, after.values, strict=True)
        )
    )


def _active_max_force(gradient: np.ndarray, state: State) -> float:
    matrix = np.asarray(gradient, dtype=float).reshape(state.n_atoms, 3)
    active = matrix[state.movable_mask]
    if active.size == 0:
        return 0.0
    return float(np.max(np.linalg.norm(active, axis=1)))


class _InitialEvaluator:
    def __init__(self, counter: EvalCounter, state: State) -> None:
        self.counter = counter
        self.positions = state.positions.copy()
        self.energy: float | None = None
        self.gradient: np.ndarray | None = None

    def __call__(
        self, flat_positions: np.ndarray, template: State
    ) -> tuple[float, np.ndarray]:
        energy, gradient = self.counter.evaluate_flat(flat_positions, template)
        if self.energy is None:
            positions = np.asarray(flat_positions, dtype=float).reshape(-1, 3)
            if not np.array_equal(positions, self.positions):
                raise RuntimeError("optimizer did not evaluate the fixed start first")
            self.energy = float(energy)
            self.gradient = np.asarray(gradient, dtype=float).reshape(-1).copy()
        return float(energy), np.asarray(gradient, dtype=float).reshape(-1).copy()


def execute_true_quench(
    task: LandingTask,
    stage: str,
    fmax: float,
    arm: Arm,
    counter: EvalCounter,
) -> dict[str, Any]:
    before = counter.snapshot()
    evaluator = _InitialEvaluator(counter, task.state)
    kwargs = {
        "fmax": fmax,
        "maxiter": MAXITER,
        "coordinate_trust_radius": None,
    }
    if arm.safe_history_limit is not None:
        kwargs["_safe_lbfgs_history_limit"] = arm.safe_history_limit
    started = perf_counter()
    with counter.purpose(EvaluationPurpose.LANDING_TRUE_QUENCH):
        result = Relaxer(evaluator, optimizer=arm.optimizer).relax(
            task.state, **kwargs
        )
    wall_time_s = perf_counter() - started
    delta = _count_delta(before, counter.snapshot())
    if evaluator.energy is None or evaluator.gradient is None:
        raise RuntimeError("true quench did not evaluate its fixed initial state")
    if (
        delta.total != result.telemetry.evaluator_calls
        or delta.count(EvaluationPurpose.LANDING_TRUE_QUENCH) != delta.total
        or delta.count(EvaluationPurpose.UNATTRIBUTED) != 0
    ):
        raise RuntimeError("true-quench task purpose accounting does not close")
    return {
        "system": task.system,
        "task_index": task.task_index,
        "trial_index": task.trial_index,
        "proposal_index": task.proposal_index,
        "stage": stage,
        "arm_id": arm.arm_id,
        "optimizer": arm.optimizer,
        "safe_history_limit": arm.safe_history_limit,
        "fmax_eV_per_A": fmax,
        "maxiter": MAXITER,
        "coordinate_trust_radius_A": None,
        "objective": OBJECTIVE,
        "initial": {
            "energy_eV": evaluator.energy,
            "max_active_force_eV_per_A": _active_max_force(
                evaluator.gradient, task.state
            ),
            "positions_sha256": position_sha256(task.state.positions),
            "state": state_payload(task.state),
        },
        "final": {
            "energy_eV": float(result.energy),
            "max_active_force_eV_per_A": float(result.gradient_norm),
            "positions_sha256": position_sha256(result.state.positions),
            "state": state_payload(result.state),
        },
        "n_iter": int(result.n_iter),
        "termination_reason": result.telemetry.termination_reason,
        "outcome_class": result.outcome_class.value,
        "telemetry": asdict(result.telemetry),
        "evaluator_calls": delta.total,
        "purpose_count_delta": delta.as_dict(),
        "wall_time_s": wall_time_s,
    }


def _refine_task(source: LandingTask, loose_scipy_row: Mapping[str, Any]) -> LandingTask:
    state = state_from_payload(loose_scipy_row["final"]["state"])
    return replace(
        source,
        initial_positions_sha256=position_sha256(state.positions),
        state=state,
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


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


def run(
    *,
    output_dir: Path,
    expected_git_commit: str,
    preflight_only: bool = False,
    preflight_fn: Callable[..., Preflight] = preflight,
    calculator_factory: Callable[[], object] = _calculator,
    counter_factory: Callable[[object], EvalCounter] = _counter,
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
    corpus: dict[str, Any] = {
        "capture_policy_conditioned": True,
        "claim_boundary": (
            "The raw-landing corpus is conditioned on the fixed seed-42, "
            "safe-LBFGS proposal, first-16-trial capture policy."
        ),
        "systems": {},
    }
    source_tasks: dict[str, tuple[LandingTask, ...]] = {}
    stage_a: dict[str, Any] = {}
    for system in SYSTEMS:
        case_dir = partial_dir / "stage_a" / system
        config = build_capture_config(system, case_dir)
        tasks, manifest, stats, counts = capture_stage_a(
            system,
            checked.production_states[system],
            config,
            calculator_factory(),
        )
        if len(tasks) != CAPTURE_TRIALS:
            raise RuntimeError(f"{system} Stage A corpus is incomplete")
        source_tasks[system] = tasks
        corpus["systems"][system] = manifest
        stage_a[system] = {
            "effective_config": _stable_config_payload(config, partial_dir),
            "stats": stats,
            "purpose_counts": counts.as_dict(),
        }

    rows: list[dict[str, Any]] = []
    loose_scipy: dict[tuple[str, int], Mapping[str, Any]] = {}
    for system in SYSTEMS:
        for arm in ARMS:
            counter = counter_factory(calculator_factory())
            for task in source_tasks[system]:
                row = execute_true_quench(task, "loose", 0.05, arm, counter)
                rows.append(row)
                if arm.arm_id == "scipy-lbfgsb":
                    loose_scipy[(system, task.task_index)] = row

    for system in SYSTEMS:
        for arm in ARMS:
            counter = counter_factory(calculator_factory())
            for source in source_tasks[system]:
                task = _refine_task(
                    source, loose_scipy[(system, source.task_index)]
                )
                rows.append(
                    execute_true_quench(task, "refine", 0.01, arm, counter)
                )

    expected_rows = len(SYSTEMS) * CAPTURE_TRIALS * len(ARMS)
    if len(rows) != 2 * expected_rows:
        raise RuntimeError("tiered true-quench matrix is incomplete")
    summary = {
        **dict(checked.metadata),
        "protocol": {
            "capture_trials_per_system": CAPTURE_TRIALS,
            "capture_policy_conditioned": True,
            "objective": OBJECTIVE,
            "maxiter": MAXITER,
            "stages": [
                {"stage": stage, "fmax_eV_per_A": fmax}
                for stage, fmax in STAGES
            ],
            "refine_common_start": "per-task loose scipy-lbfgsb endpoint",
        },
        "arms": [asdict(arm) for arm in ARMS],
        "stage_a_task_count": len(SYSTEMS) * CAPTURE_TRIALS,
        "stage_b_row_count": expected_rows,
        "stage_c_row_count": expected_rows,
        "corpus_file": "corpus.json",
        "rows_file": "rows.json",
        "stage_a": stage_a,
        "claim_boundary": (
            "Stage A is capture-policy-conditioned; Stage B compares local "
            "true-PES loose quenches; Stage C is conditional on each task's "
            "SciPy loose endpoint."
        ),
    }
    _write_json(partial_dir / "corpus.json", corpus)
    _write_json(partial_dir / "rows.json", rows)
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
    result = run(
        output_dir=args.output,
        expected_git_commit=args.expected_git_commit,
        preflight_only=args.preflight_only,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
