#!/usr/bin/env python3
"""Replay strict true-PES fallbacks from primary ASE-LBFGS certificate failures."""

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
CORPUS_PATH = RUN_ROOT / "corpus.json"
PRODUCTION_RUNNER_PATH = (
    REPO_ROOT / "runs" / "20260728-safe-lbfgs-200-production" / "run_production.py"
)
SYSTEMS = ("c60", "pdo")
EXPECTED_FAILURE_KEYS = (("c60", 5), ("c60", 11), ("pdo", 3), ("pdo", 4))
STRICT_FMAX = 0.01
MAXITER = 400
OBJECTIVE = "true_mace_pes_no_bias_no_softening"
SAFE_LBFGS_HISTORY_LIMIT = 10
EXPECTED_SOURCE_ROWS_SHA256 = (
    "b9c12ca40f2fa183d8573a46b66ec630aeb25ab54ea8eacfc3e88640d163e1ac"
)
EXPECTED_SOURCE_SUMMARY_SHA256 = (
    "3e15c150ef20ad85e143346a55918aa7510ec3eeadae727168cd36a6e33c9b36"
)
FROZEN_CORPUS_SHA256 = (
    "82b475eb0bcf0a6fb631bff0ee47cdd0223401db0db6262090c6f721d8624e67"
)
EXPECTED_MODEL_SHA256 = "0abfde07862cf1e93b8b4d03cb702f29ce9c344ff2fc4de2ec0d7166d6c113a5"
EXPECTED_PRIMARY_COHORT = {
    "overall": {
        "task_count": 32,
        "certificate_success_count": 28,
        "trigger_count": 4,
        "total_force_evaluations": 5096,
        "total_wall_time_s": 98.84687816997757,
    },
    "by_system": {
        "c60": {
            "task_count": 16,
            "certificate_success_count": 14,
            "trigger_count": 2,
            "total_force_evaluations": 2490,
            "total_wall_time_s": 42.58624181896448,
        },
        "pdo": {
            "task_count": 16,
            "certificate_success_count": 14,
            "trigger_count": 2,
            "total_force_evaluations": 2606,
            "total_wall_time_s": 56.26063635101309,
        },
    },
}
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
    Arm("ase-lbfgs-restart", "ase-lbfgs", None),
    Arm("safe-lbfgs-total", "safe-lbfgs-total", 10),
    Arm("ase-fire", "ase-fire", None),
)


@dataclass(frozen=True)
class RescueTask:
    system: str
    task_index: int
    trial_index: int
    proposal_index: int
    source_trajectory_path: str
    source_trajectory_sha256: str
    source_frame_index: int
    primary: Mapping[str, Any]
    start_energy: float
    start_max_force: float
    start_positions_sha256: str
    state: State


@dataclass(frozen=True)
class Corpus:
    source_path: Path
    source_sha256: str
    provenance: Mapping[str, Any]
    tasks_by_system: Mapping[str, tuple[RescueTask, ...]]


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


def _digest(value: object, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _strict_int(value: object, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field} has invalid integer identity")
    return value


def _finite_float(
    value: object, field: str, *, nonnegative: bool = False
) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be finite")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be finite") from error
    if not np.isfinite(numeric) or (nonnegative and numeric < 0.0):
        raise ValueError(f"{field} must be finite and valid")
    return numeric


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


def load_corpus(path: Path) -> Corpus:
    """Load only the frozen four-state rescue corpus with intact identities."""

    source_path = Path(path)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("rescue corpus schema is invalid")
    selection = payload.get("selection")
    if (
        not isinstance(selection, dict)
        or selection.get("primary_arm_id") != "ase-lbfgs"
        or selection.get("strict_fmax_eV_per_A") != STRICT_FMAX
        or selection.get("expected_failure_keys")
        != [list(key) for key in EXPECTED_FAILURE_KEYS]
    ):
        raise ValueError("rescue corpus selection identity is invalid")
    primary_cohort = selection.get("primary_cohort")
    if primary_cohort != EXPECTED_PRIMARY_COHORT:
        raise ValueError("rescue corpus primary cohort identity is invalid")
    by_system_cohort = primary_cohort["by_system"]
    for field in (
        "task_count",
        "certificate_success_count",
        "trigger_count",
        "total_force_evaluations",
        "total_wall_time_s",
    ):
        if primary_cohort["overall"][field] != sum(
            by_system_cohort[system][field] for system in SYSTEMS
        ):
            raise ValueError("rescue corpus primary cohort inner/outer mismatch")
    provenance = payload.get("source")
    if not isinstance(provenance, dict):
        raise ValueError("rescue corpus source provenance is missing")
    rows_provenance = provenance.get("rows")
    summary_provenance = provenance.get("summary")
    if (
        not isinstance(rows_provenance, dict)
        or _digest(rows_provenance.get("sha256"), "source rows hash")
        != EXPECTED_SOURCE_ROWS_SHA256
        or not isinstance(summary_provenance, dict)
        or _digest(summary_provenance.get("sha256"), "source summary hash")
        != EXPECTED_SOURCE_SUMMARY_SHA256
    ):
        raise ValueError("rescue corpus source replay identity is invalid")
    model = provenance.get("model")
    if (
        not isinstance(model, dict)
        or _digest(model.get("sha256"), "source model hash")
        != EXPECTED_MODEL_SHA256
    ):
        raise ValueError("rescue corpus source model identity is invalid")

    entries = payload.get("entries")
    if payload.get("entry_count") != 4 or not isinstance(entries, list) or len(entries) != 4:
        raise ValueError("rescue corpus must contain exactly four entries")
    tasks_by_system: dict[str, list[RescueTask]] = {system: [] for system in SYSTEMS}
    seen: list[tuple[str, int]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("rescue corpus entry must be an object")
        system = entry.get("system")
        if system not in SYSTEMS:
            raise ValueError(f"rescue corpus contains unknown system {system!r}")
        task_index = _strict_int(entry.get("task_index"), "task_index")
        trial_index = _strict_int(entry.get("trial_index"), "trial_index")
        proposal_index = _strict_int(entry.get("proposal_index"), "proposal_index")
        source_frame_index = _strict_int(
            entry.get("source_frame_index"), "source_frame_index"
        )
        key = (system, task_index)
        seen.append(key)
        primary = entry.get("primary")
        start = entry.get("fallback_start")
        if (
            not isinstance(primary, dict)
            or primary.get("arm_id") != "ase-lbfgs"
            or primary.get("optimizer") != "ase-lbfgs"
            or primary.get("certificate_passed") is not False
        ):
            raise ValueError("rescue corpus primary failure identity is invalid")
        primary_force_evaluations = _strict_int(
            primary.get("force_evaluations"), "primary force_evaluations"
        )
        if primary_force_evaluations <= 0:
            raise ValueError("rescue corpus primary numeric fields are invalid")
        primary_initial_energy = _finite_float(
            primary.get("initial_energy_eV"), "primary numeric initial energy"
        )
        primary_final_energy = _finite_float(
            primary.get("final_energy_eV"), "primary numeric final energy"
        )
        primary_initial_force = _finite_float(
            primary.get("initial_max_active_force_eV_per_A"),
            "primary numeric initial force",
            nonnegative=True,
        )
        primary_final_force = _finite_float(
            primary.get("final_max_active_force_eV_per_A"),
            "primary numeric final force",
            nonnegative=True,
        )
        primary_wall_time = _finite_float(
            primary.get("wall_time_s"),
            "primary numeric wall time",
            nonnegative=True,
        )
        if primary_final_force <= STRICT_FMAX:
            raise ValueError("rescue corpus primary failure identity is invalid")
        if not isinstance(start, dict) or not isinstance(start.get("state"), dict):
            raise ValueError("rescue corpus fallback start is missing")
        try:
            state = _state_from_payload(start["state"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("rescue corpus fallback state is invalid") from error
        if not np.all(np.isfinite(state.positions)) or (
            state.cell is not None and not np.all(np.isfinite(state.cell))
        ):
            raise ValueError("rescue corpus fallback state is nonfinite")
        start_energy = _finite_float(
            start.get("energy_eV"), "fallback-start energy"
        )
        start_max_force = _finite_float(
            start.get("max_active_force_eV_per_A"),
            "fallback-start force",
            nonnegative=True,
        )
        if (
            start_energy != primary_final_energy
            or start_max_force != primary_final_force
        ):
            raise ValueError("rescue corpus primary/fallback numeric identity differs")
        positions_hash = _digest(
            start.get("positions_sha256"), "fallback-start positions hash"
        )
        if position_sha256(state.positions) != positions_hash:
            raise ValueError("rescue corpus fallback-start positions hash mismatch")
        tasks_by_system[key[0]].append(
            RescueTask(
                system=key[0],
                task_index=key[1],
                trial_index=trial_index,
                proposal_index=proposal_index,
                source_trajectory_path=str(entry["source_trajectory_path"]),
                source_trajectory_sha256=_digest(
                    entry["source_trajectory_sha256"], "source trajectory hash"
                ),
                source_frame_index=source_frame_index,
                primary={
                    **dict(primary),
                    "force_evaluations": primary_force_evaluations,
                    "initial_energy_eV": primary_initial_energy,
                    "final_energy_eV": primary_final_energy,
                    "initial_max_active_force_eV_per_A": primary_initial_force,
                    "final_max_active_force_eV_per_A": primary_final_force,
                    "wall_time_s": primary_wall_time,
                },
                start_energy=start_energy,
                start_max_force=start_max_force,
                start_positions_sha256=positions_hash,
                state=state,
            )
        )
    if tuple(seen) != EXPECTED_FAILURE_KEYS:
        raise ValueError("rescue corpus entry identities are invalid")
    for system in SYSTEMS:
        if (
            by_system_cohort[system]["trigger_count"]
            != len(tasks_by_system[system])
        ):
            raise ValueError("rescue corpus primary cohort trigger count mismatch")
    return Corpus(
        source_path=source_path,
        source_sha256=_sha256(source_path),
        provenance={"source": dict(provenance), "selection": dict(selection)},
        tasks_by_system={
            system: tuple(tasks_by_system[system]) for system in SYSTEMS
        },
    )


def _production_module():
    spec = importlib.util.spec_from_file_location(
        "_sequential_rescue_production_runner", PRODUCTION_RUNNER_PATH
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
    corpus_path: Path = CORPUS_PATH,
    runtime_probe=_runtime_versions,
    cuda_probe=_cuda_info,
) -> Preflight:
    """Fail closed unless commit, tracked corpus, model, calculator, and CUDA agree."""

    actual_commit = _current_commit()
    if actual_commit != expected_git_commit:
        raise RuntimeError(
            f"execution commit mismatch: expected {expected_git_commit}, got {actual_commit}"
        )
    if not _tracked_worktree_clean():
        raise RuntimeError("tracked worktree is not clean")
    corpus_path = Path(corpus_path)
    if not _is_tracked_file(corpus_path):
        raise RuntimeError("rescue corpus must be a tracked file")
    corpus = load_corpus(corpus_path)
    if corpus.source_sha256 != FROZEN_CORPUS_SHA256:
        raise ValueError("frozen rescue corpus SHA-256 mismatch")
    if _safe_lbfgs_default_history_limit() != SAFE_LBFGS_HISTORY_LIMIT:
        raise RuntimeError("safe L-BFGS default history limit is not 10")
    production = _production_module()
    if not production.MODEL_PATH.is_file():
        raise FileNotFoundError(production.MODEL_PATH)
    model_hash = _sha256(production.MODEL_PATH)
    if model_hash != EXPECTED_MODEL_SHA256:
        raise ValueError("frozen model SHA-256 mismatch")
    source = corpus.provenance["source"]
    if source["model"]["sha256"] != model_hash:
        raise ValueError("source replay and current model identities differ")
    if dict(source["calculator"]) != dict(production.CALCULATOR_CONFIG):
        raise ValueError("source replay and current calculator configs differ")
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
                "entry_count": 4,
            },
            "source_replay": dict(source),
            "primary_cohort": dict(corpus.provenance["selection"]["primary_cohort"]),
            "model": {
                "path": _display_path(production.MODEL_PATH),
                "sha256": model_hash,
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
    """Capture the first true-PES call and require the frozen terminal state."""

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
                raise RuntimeError(
                    "fallback optimizer did not evaluate the frozen terminal start first"
                )
            self.energy = float(energy)
            self.gradient = np.asarray(gradient, dtype=float).reshape(-1).copy()
        return float(energy), np.asarray(gradient, dtype=float).reshape(-1).copy()


def execute_fallback(
    task: RescueTask,
    arm: Arm,
    counter: EvalCounter,
) -> dict[str, Any]:
    """Run one independent fallback from the frozen primary terminal state."""

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
        result = Relaxer(evaluator, optimizer=arm.optimizer).relax(task.state, **kwargs)
    wall_time_s = perf_counter() - started
    delta = _count_delta(before, counter.snapshot())
    if evaluator.energy is None or evaluator.gradient is None:
        raise RuntimeError("fallback did not evaluate its frozen terminal start")
    if (
        delta.total != result.telemetry.evaluator_calls
        or delta.count(EvaluationPurpose.LANDING_TRUE_QUENCH) != delta.total
        or delta.count(EvaluationPurpose.UNATTRIBUTED) != 0
    ):
        raise RuntimeError("fallback purpose accounting does not close")
    primary_force_evaluations = int(task.primary["force_evaluations"])
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
        "primary": dict(task.primary),
        "fallback": {
            "initial": {
                "energy_eV": evaluator.energy,
                "max_active_force_eV_per_A": _active_max_force(
                    evaluator.gradient, task.state
                ),
                "positions_sha256": task.start_positions_sha256,
                "state": state_payload(task.state),
            },
            "final": {
                "energy_eV": float(result.energy),
                "max_active_force_eV_per_A": float(result.gradient_norm),
                "positions_sha256": position_sha256(result.state.positions),
                "state": state_payload(result.state),
            },
            "energy_change_eV": float(result.energy - evaluator.energy),
            "certificate_passed": bool(result.gradient_norm <= STRICT_FMAX),
            "n_iter": int(result.n_iter),
            "termination_reason": result.telemetry.termination_reason,
            "outcome_class": result.outcome_class.value,
            "telemetry": asdict(result.telemetry),
            "evaluator_calls": delta.total,
            "force_evaluations": delta.total,
            "purpose_count_delta": delta.as_dict(),
            "wall_time_s": wall_time_s,
        },
        "cost": {
            "primary_force_evaluations": primary_force_evaluations,
            "offline_fallback_force_evaluations": delta.total,
            "offline_combined_replay_force_evaluations": (
                primary_force_evaluations + delta.total
            ),
            "offline_repeats_primary_terminal_evaluation": True,
            "continuous_implementation_avoidable_force_evaluations": 1,
        },
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
    if len(rows) != 12:
        raise RuntimeError("fallback matrix is incomplete")
    expected = {
        (system, task.task_index, arm.arm_id)
        for system, tasks in corpus.tasks_by_system.items()
        for task in tasks
        for arm in ARMS
    }
    seen: set[tuple[str, int, str]] = set()
    counts: list[EvaluationCounts] = []
    for row in rows:
        key = (str(row["system"]), int(row["task_index"]), str(row["arm_id"]))
        if key in seen:
            raise RuntimeError("fallback matrix contains a duplicate row")
        seen.add(key)
        if (
            row["fmax_eV_per_A"] != STRICT_FMAX
            or row["maxiter"] != MAXITER
            or row["coordinate_trust_radius_A"] is not None
            or row["objective"] != OBJECTIVE
        ):
            raise RuntimeError("fallback row violates the fixed protocol")
        fallback = row["fallback"]
        initial = fallback.get("initial")
        final = fallback.get("final")
        telemetry = fallback.get("telemetry")
        if (
            not isinstance(initial, Mapping)
            or not isinstance(final, Mapping)
            or not isinstance(telemetry, Mapping)
        ):
            raise RuntimeError("fallback row numeric or telemetry payload is missing")
        numeric_values = (
            ("initial energy", initial.get("energy_eV"), False),
            (
                "initial force",
                initial.get("max_active_force_eV_per_A"),
                True,
            ),
            ("final energy", final.get("energy_eV"), False),
            ("final force", final.get("max_active_force_eV_per_A"), True),
            ("wall time", fallback.get("wall_time_s"), True),
        )
        checked_numeric: dict[str, float] = {}
        for field, value, nonnegative in numeric_values:
            try:
                checked_numeric[field] = _finite_float(
                    value, field, nonnegative=nonnegative
                )
            except ValueError as error:
                raise RuntimeError("fallback row finite numeric validation failed") from error
        certificate = checked_numeric["final force"] <= STRICT_FMAX
        recorded_certificate = fallback.get("certificate_passed")
        if (
            type(recorded_certificate) is not bool
            or recorded_certificate != certificate
        ):
            raise RuntimeError("fallback row certificate is inconsistent")
        telemetry_converged = telemetry.get("converged")
        if (
            type(telemetry_converged) is not bool
            or telemetry_converged != certificate
        ):
            raise RuntimeError("fallback telemetry convergence is inconsistent")
        termination_reason = fallback.get("termination_reason")
        telemetry_termination = telemetry.get("termination_reason")
        if termination_reason != telemetry_termination:
            raise RuntimeError("fallback telemetry termination is inconsistent")
        if (termination_reason == "converged") != certificate:
            raise RuntimeError("fallback termination is inconsistent")
        delta = EvaluationCounts.from_mapping(fallback["purpose_count_delta"])
        if (
            delta.total != fallback["force_evaluations"]
            or delta.total != fallback["evaluator_calls"]
            or delta.count(EvaluationPurpose.LANDING_TRUE_QUENCH) != delta.total
            or delta.count(EvaluationPurpose.UNATTRIBUTED) != 0
        ):
            raise RuntimeError("fallback row has an open evaluation ledger")
        counts.append(delta)
    if seen != expected:
        raise RuntimeError("fallback matrix does not cover every trigger and arm")
    return EvaluationCounts.sum(counts)


def run(
    *,
    output_dir: Path,
    expected_git_commit: str,
    preflight_fn: Callable[..., Preflight] = preflight,
    calculator_factory: Callable[[], object] = _calculator,
    counter_factory: Callable[[object], EvalCounter] = _counter,
) -> dict[str, Any]:
    """Run the 4 x 3 fallback matrix and atomically publish complete artifacts."""

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
                rows.append(execute_fallback(task, arm, counter))
    fallback_counts = _validate_complete_matrix(rows, checked.corpus)
    primary_cohort = checked.corpus.provenance["selection"]["primary_cohort"]
    primary_cohort_force_evaluations = int(
        primary_cohort["overall"]["total_force_evaluations"]
    )
    trigger_count = int(primary_cohort["overall"]["trigger_count"])
    pipeline_cost_by_fallback_arm = {}
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm_id"] == arm.arm_id]
        fallback_force_evaluations = sum(
            int(row["fallback"]["force_evaluations"]) for row in arm_rows
        )
        offline_total = primary_cohort_force_evaluations + fallback_force_evaluations
        pipeline_cost_by_fallback_arm[arm.arm_id] = {
            "trigger_count": trigger_count,
            "fallback_certificate_success_count": sum(
                bool(row["fallback"]["certificate_passed"]) for row in arm_rows
            ),
            "offline_fallback_force_evaluations": fallback_force_evaluations,
            "offline_pipeline_total_force_evaluations": offline_total,
            "continuous_pipeline_projected_force_evaluations": (
                offline_total - trigger_count
            ),
        }
    summary = {
        **dict(checked.metadata),
        "strict_protocol": {
            "fmax_eV_per_A": STRICT_FMAX,
            "maxiter": MAXITER,
            "coordinate_trust_radius_A": None,
            "objective": OBJECTIVE,
        },
        "arms": [asdict(arm) for arm in ARMS],
        "trigger_count": 4,
        "row_count": len(rows),
        "calculator_instances": len(SYSTEMS) * len(ARMS),
        "fallback_evaluation_counts": fallback_counts.as_dict(),
        "primary_cohort": primary_cohort,
        "pipeline_cost_by_fallback_arm": pipeline_cost_by_fallback_arm,
        "rows_file": "rows.json",
        "cost_interpretation": (
            "Each offline fallback re-evaluates the primary terminal state. A true "
            "certificate-triggered continuation can pass that cached terminal "
            "evaluation to the fallback and avoid exactly one force evaluation per "
            "trigger; this runner records the repeated call and does not remove it."
        ),
        "claim_boundary": (
            "This fixed offline replay compares strict fallback behavior only; it "
            "does not implement production fallback selection or measure global search."
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
