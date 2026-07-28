#!/usr/bin/env python3
"""Replay the fixed raw-landing corpus with strict true-PES quenches."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
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

import numpy as np

from pamssw.accounting import EvalCounter, EvaluationCounts, EvaluationPurpose
from pamssw.calculators import ASECalculator
from pamssw.relax import Relaxer
from pamssw.state import State


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
SOURCE_CORPUS_PATH = (
    REPO_ROOT
    / "runs"
    / "20260728-true-quench-tiered-ablation"
    / "output"
    / "corpus.json"
)
PRODUCTION_RUNNER_PATH = (
    REPO_ROOT / "runs" / "20260728-safe-lbfgs-200-production" / "run_production.py"
)
SYSTEMS = ("c60", "pdo")
TASKS_PER_SYSTEM = 16
STRICT_FMAX = 0.01
MAXITER = 400
OBJECTIVE = "true_mace_pes_no_bias_no_softening"
SAFE_LBFGS_HISTORY_LIMIT = 10
FROZEN_CORPUS_SHA256 = "100759e1871cdefb54972f91751c452763f74d0e34ee576f268a5808219a552b"
EXPECTED_MODEL_SHA256 = "0abfde07862cf1e93b8b4d03cb702f29ce9c344ff2fc4de2ec0d7166d6c113a5"
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
    optimizer: str
    safe_history_limit: int | None


ARMS = (
    Arm("scipy-lbfgsb", "scipy-lbfgsb", None),
    Arm("safe-lbfgs-total", "safe-lbfgs-total", 10),
    Arm("ase-fire", "ase-fire", None),
    Arm("ase-fire2", "ase-fire2", None),
    Arm("ase-lbfgs", "ase-lbfgs", None),
)


@dataclass(frozen=True)
class LandingTask:
    system: str
    task_index: int
    trial_index: int
    proposal_index: int
    source_trajectory_path: str
    source_trajectory_sha256: str
    source_frame_index: int
    initial_positions_sha256: str
    state: State


@dataclass(frozen=True)
class Corpus:
    source_path: Path
    source_sha256: str
    tasks_by_system: Mapping[str, tuple[LandingTask, ...]]


@dataclass(frozen=True)
class Preflight:
    metadata: Mapping[str, Any]
    corpus: Corpus


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


def _state_from_payload(payload: Mapping[str, Any]) -> State:
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


def _sha256_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def load_corpus(path: Path) -> Corpus:
    """Load only a complete, hash-self-consistent fixed raw-landing corpus."""

    source_path = Path(path)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("corpus root must be an object")
    if payload.get("capture_policy_conditioned") is not True:
        raise ValueError("corpus does not state its fixed capture policy")
    manifests = payload.get("systems")
    if not isinstance(manifests, dict) or set(manifests) != set(SYSTEMS):
        raise ValueError("corpus systems are incomplete or unexpected")

    tasks_by_system: dict[str, tuple[LandingTask, ...]] = {}
    for system in SYSTEMS:
        manifest = manifests[system]
        if not isinstance(manifest, dict) or manifest.get("system") != system:
            raise ValueError(f"{system} corpus system identity is invalid")
        if manifest.get("capture_policy_conditioned") is not True:
            raise ValueError(f"{system} corpus does not state its capture policy")
        if manifest.get("task_count") != TASKS_PER_SYSTEM:
            raise ValueError(f"{system} corpus task count is not {TASKS_PER_SYSTEM}")
        entries = manifest.get("entries")
        if not isinstance(entries, list) or len(entries) != TASKS_PER_SYSTEM:
            raise ValueError(f"{system} corpus entry count is not {TASKS_PER_SYSTEM}")

        tasks: list[LandingTask] = []
        for task_index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ValueError(f"{system} task {task_index} is not an object")
            if entry.get("system") != system or entry.get("task_index") != task_index:
                raise ValueError(f"{system} task {task_index} identity is invalid")
            if entry.get("trial_index") != task_index + 1 or entry.get("proposal_index") != 1:
                raise ValueError(f"{system} task {task_index} capture index is invalid")
            if entry.get("source_frame_index") != 0:
                raise ValueError(f"{system} task {task_index} source frame is invalid")
            source_trajectory_path = entry.get("source_trajectory_path")
            if not isinstance(source_trajectory_path, str) or not source_trajectory_path:
                raise ValueError(f"{system} task {task_index} source path is invalid")
            source_trajectory_sha256 = _sha256_text(
                entry.get("source_trajectory_sha256"),
                field=f"{system} task {task_index} source trajectory hash",
            )
            expected_positions_sha256 = _sha256_text(
                entry.get("initial_positions_sha256"),
                field=f"{system} task {task_index} positions hash",
            )
            state_payload = entry.get("state")
            if not isinstance(state_payload, dict):
                raise ValueError(f"{system} task {task_index} state is invalid")
            try:
                state = _state_from_payload(state_payload)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{system} task {task_index} state is invalid") from error
            actual_positions_sha256 = position_sha256(state.positions)
            if actual_positions_sha256 != expected_positions_sha256:
                raise ValueError(f"{system} task {task_index} positions hash mismatch")
            tasks.append(
                LandingTask(
                    system=system,
                    task_index=task_index,
                    trial_index=task_index + 1,
                    proposal_index=1,
                    source_trajectory_path=source_trajectory_path,
                    source_trajectory_sha256=source_trajectory_sha256,
                    source_frame_index=0,
                    initial_positions_sha256=expected_positions_sha256,
                    state=state,
                )
            )
        tasks_by_system[system] = tuple(tasks)
    return Corpus(
        source_path=source_path,
        source_sha256=_sha256(source_path),
        tasks_by_system=tasks_by_system,
    )


def _production_module():
    spec = importlib.util.spec_from_file_location(
        "_raw_strict_replay_production_runner", PRODUCTION_RUNNER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the frozen production runner")
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


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


def _is_tracked_file(path: Path) -> bool:
    try:
        relative = Path(path).resolve().relative_to(REPO_ROOT.resolve())
    except ValueError:
        return False
    completed = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", str(relative)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


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


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def preflight(
    *,
    expected_git_commit: str,
    corpus_path: Path = SOURCE_CORPUS_PATH,
    runtime_probe=_runtime_versions,
    cuda_probe=_cuda_info,
) -> Preflight:
    """Refuse execution unless identity, corpus, and CUDA provenance are fixed."""

    actual_commit = _current_commit()
    if actual_commit != expected_git_commit:
        raise RuntimeError(
            f"execution commit mismatch: expected {expected_git_commit}, got {actual_commit}"
        )
    if not _tracked_worktree_clean():
        raise RuntimeError("tracked worktree is not clean")
    corpus_path = Path(corpus_path)
    if not _is_tracked_file(corpus_path):
        raise RuntimeError("raw-landing corpus must be a tracked file")
    if _safe_lbfgs_default_history_limit() != SAFE_LBFGS_HISTORY_LIMIT:
        raise RuntimeError("safe L-BFGS default history limit is not 10")
    corpus = load_corpus(corpus_path)
    if corpus.source_sha256 != FROZEN_CORPUS_SHA256:
        raise ValueError("frozen corpus SHA-256 mismatch")
    production = _production_module()
    if not production.MODEL_PATH.is_file():
        raise FileNotFoundError(production.MODEL_PATH)
    model_sha256 = _sha256(production.MODEL_PATH)
    if model_sha256 != EXPECTED_MODEL_SHA256:
        raise ValueError("frozen model SHA-256 mismatch")
    for system in SYSTEMS:
        if not production.INPUT_PATHS[system].is_file():
            raise FileNotFoundError(production.INPUT_PATHS[system])
    cuda = dict(cuda_probe())
    if cuda.get("available") is not True:
        raise RuntimeError("CUDA is unavailable")
    return Preflight(
        corpus=corpus,
        metadata={
            "schema_version": 1,
            "execution_commit": actual_commit,
            "corpus": {
                "path": _display_path(corpus.source_path),
                "sha256": corpus.source_sha256,
                "task_count_per_system": TASKS_PER_SYSTEM,
                "total_task_count": len(SYSTEMS) * TASKS_PER_SYSTEM,
            },
            "model": {
                "path": _display_path(production.MODEL_PATH),
                "sha256": model_sha256,
            },
            "inputs": {
                system: {
                    "path": _display_path(production.INPUT_PATHS[system]),
                    "sha256": _sha256(production.INPUT_PATHS[system]),
                }
                for system in SYSTEMS
            },
            "runtime_versions": {
                key: str(value) for key, value in runtime_probe().items()
            },
            "cuda": cuda,
            "calculator": dict(production.CALCULATOR_CONFIG),
            "safe_lbfgs_default_history_limit": SAFE_LBFGS_HISTORY_LIMIT,
        },
    )


def state_payload(state: State) -> dict[str, Any]:
    return {
        "numbers": state.numbers.tolist(),
        "positions": state.positions.tolist(),
        "cell": None if state.cell is None else state.cell.tolist(),
        "pbc": list(state.pbc),
        "fixed_mask": state.fixed_mask.tolist(),
    }


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
    """Record the first real true-PES call and reject shifted starts."""

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
                raise RuntimeError("optimizer did not evaluate the fixed raw start first")
            self.energy = float(energy)
            self.gradient = np.asarray(gradient, dtype=float).reshape(-1).copy()
        return float(energy), np.asarray(gradient, dtype=float).reshape(-1).copy()


def execute_strict_true_quench(
    task: LandingTask,
    arm: Arm,
    counter: EvalCounter,
) -> dict[str, Any]:
    """Run exactly one independent strict true-PES quench from a raw state."""

    before = counter.snapshot()
    evaluator = _InitialEvaluator(counter, task.state)
    kwargs: dict[str, Any] = {
        "fmax": STRICT_FMAX,
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
        raise RuntimeError("strict true quench did not evaluate its fixed raw start")
    if (
        delta.total != result.telemetry.evaluator_calls
        or delta.count(EvaluationPurpose.LANDING_TRUE_QUENCH) != delta.total
        or delta.count(EvaluationPurpose.UNATTRIBUTED) != 0
    ):
        raise RuntimeError("strict true-quench task purpose accounting does not close")
    return {
        "system": task.system,
        "task_index": task.task_index,
        "trial_index": task.trial_index,
        "proposal_index": task.proposal_index,
        "source_trajectory_path": task.source_trajectory_path,
        "source_trajectory_sha256": task.source_trajectory_sha256,
        "source_frame_index": task.source_frame_index,
        "arm_id": arm.arm_id,
        "optimizer": arm.optimizer,
        "safe_history_limit": arm.safe_history_limit,
        "fmax_eV_per_A": STRICT_FMAX,
        "maxiter": MAXITER,
        "coordinate_trust_radius_A": None,
        "objective": OBJECTIVE,
        "initial": {
            "energy_eV": evaluator.energy,
            "max_active_force_eV_per_A": _active_max_force(
                evaluator.gradient, task.state
            ),
            "positions_sha256": task.initial_positions_sha256,
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
        "force_evaluations": delta.total,
        "purpose_count_delta": delta.as_dict(),
        "wall_time_s": wall_time_s,
    }


def _calculator():
    production = _production_module()
    from mace.calculators import MACECalculator

    return MACECalculator(
        model_paths=str(production.MODEL_PATH),
        **production.CALCULATOR_CONFIG,
    )


def _counter(calculator: object) -> EvalCounter:
    return EvalCounter(ASECalculator(calculator))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _validate_complete_matrix(
    rows: Sequence[Mapping[str, Any]], corpus: Corpus
) -> EvaluationCounts:
    expected = len(SYSTEMS) * TASKS_PER_SYSTEM * len(ARMS)
    if len(rows) != expected:
        raise RuntimeError("strict true-quench matrix is incomplete")
    seen: set[tuple[str, int, str]] = set()
    counts: list[EvaluationCounts] = []
    for row in rows:
        key = (str(row["system"]), int(row["task_index"]), str(row["arm_id"]))
        if key in seen:
            raise RuntimeError("strict true-quench matrix contains a duplicate row")
        seen.add(key)
        system, task_index, _ = key
        if system not in corpus.tasks_by_system or not 0 <= task_index < TASKS_PER_SYSTEM:
            raise RuntimeError("strict true-quench matrix contains an unknown raw task")
        task = corpus.tasks_by_system[system][task_index]
        if row["initial"]["positions_sha256"] != task.initial_positions_sha256:
            raise RuntimeError("strict true-quench row did not use its raw corpus state")
        if (
            row["fmax_eV_per_A"] != STRICT_FMAX
            or row["maxiter"] != MAXITER
            or row["objective"] != OBJECTIVE
        ):
            raise RuntimeError("strict true-quench row violates the fixed protocol")
        delta = EvaluationCounts.from_mapping(row["purpose_count_delta"])
        if (
            delta.total != row["force_evaluations"]
            or delta.total != row["evaluator_calls"]
            or delta.count(EvaluationPurpose.LANDING_TRUE_QUENCH) != delta.total
            or delta.count(EvaluationPurpose.UNATTRIBUTED) != 0
        ):
            raise RuntimeError("strict true-quench row has an open evaluation ledger")
        counts.append(delta)
    expected_keys = {
        (system, task_index, arm.arm_id)
        for system in SYSTEMS
        for task_index in range(TASKS_PER_SYSTEM)
        for arm in ARMS
    }
    if seen != expected_keys:
        raise RuntimeError("strict true-quench matrix does not cover every system-task-arm")
    return EvaluationCounts.sum(counts)


def run(
    *,
    output_dir: Path,
    expected_git_commit: str,
    preflight_fn: Callable[..., Preflight] = preflight,
    calculator_factory: Callable[[], object] = _calculator,
    counter_factory: Callable[[object], EvalCounter] = _counter,
) -> dict[str, Any]:
    """Run the 2 x 16 x 5 strict matrix and atomically publish its artifacts."""

    output_dir = Path(output_dir)
    partial_dir = output_dir.with_name(f"{output_dir.name}.partial")
    for path in (output_dir, partial_dir):
        if path.exists():
            raise FileExistsError(path)
    checked = preflight_fn(expected_git_commit=expected_git_commit)
    partial_dir.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    for system in SYSTEMS:
        for arm in ARMS:
            calculator = calculator_factory()
            counter = counter_factory(calculator)
            for task in checked.corpus.tasks_by_system[system]:
                rows.append(execute_strict_true_quench(task, arm, counter))
    total_counts = _validate_complete_matrix(rows, checked.corpus)
    summary = {
        **dict(checked.metadata),
        "strict_protocol": {
            "fmax_eV_per_A": STRICT_FMAX,
            "maxiter": MAXITER,
            "objective": OBJECTIVE,
        },
        "arms": [asdict(arm) for arm in ARMS],
        "row_count": len(rows),
        "calculator_instances": len(SYSTEMS) * len(ARMS),
        "evaluation_counts": total_counts.as_dict(),
        "rows_file": "rows.json",
        "claim_boundary": (
            "This is a fixed raw-landing local true-PES strict-quench replay; "
            "it does not recapture proposals or measure global-search performance."
        ),
    }
    _write_json(partial_dir / "rows.json", rows)
    _write_json(partial_dir / "summary.json", summary)
    partial_dir.rename(output_dir)
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RUN_ROOT / "output")
    parser.add_argument("--expected-git-commit", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = run(
        output_dir=args.output,
        expected_git_commit=args.expected_git_commit,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
