#!/usr/bin/env python3
"""Run the fixed C60 post-SciPy accepted-endpoint rescue ablation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

from ase.io import read
import numpy as np

from pamssw.accounting import EvalCounter, EvaluationCounts, EvaluationPurpose
from pamssw.calculators import ASECalculator
from pamssw.relax import Relaxer
from pamssw.state import State


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
SOURCE_CASE_DIR = (
    REPO_ROOT / "runs" / "20260728-safe-lbfgs-200-production" / "output" / "c60"
)
SOURCE_SUMMARY_PATH = SOURCE_CASE_DIR / "summary.json"
SOURCE_ACCEPTED_LOG = SOURCE_CASE_DIR / "accepted_structures.jsonl"
SOURCE_ACCEPTED_DIR = SOURCE_CASE_DIR / "accepted_minima"
MODEL_PATH = Path("/root/.cache/mace/mace-omat-0-small.model")
EXPECTED_MODEL_SHA256 = (
    "0abfde07862cf1e93b8b4d03cb702f29ce9c344ff2fc4de2ec0d7166d6c113a5"
)

EXPECTED_SOURCE_SUMMARY_SHA256 = (
    "90f169327ef315ef4447c702de5f89b62bd1d38b02affaa987dbf6a0917d5c2c"
)
EXPECTED_ACCEPTED_LOG_SHA256 = (
    "599788fff09b453bbd39b3cba3475c420cc4fb14235fa87aec05529d29d802d4"
)
EXPECTED_INPUT_MANIFEST_SHA256 = (
    "17a1aba6764bf5463746b781755646baf558b64d098aa83de2b02634850cb2d0"
)
EXPECTED_ACCEPTED_COUNT = 143
FMAX = 0.01
MAXITER = 400
COORDINATE_TRUST_RADIUS = None
OBJECTIVE = "true_mace_pes_no_bias_no_softening"
SAFE_LBFGS_DEFAULT_HISTORY_LIMIT = 10
CALCULATOR_CONFIG = {
    "device": "cuda",
    "default_dtype": "float32",
    "inference_precision": "float32",
    "enable_cueq": False,
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
    Arm("scipy-lbfgsb-restart", "scipy-lbfgsb", None),
    Arm("safe-lbfgs-total-rescue", "safe-lbfgs-total", 10),
)


@dataclass(frozen=True)
class EndpointTask:
    task_index: int
    trial_index: int
    discovered_entry_id: int
    seed_entry_id: int
    source_path: Path
    source_sha256: str
    source_archive_energy_eV: float
    state: State


@dataclass(frozen=True)
class Preflight:
    tasks: tuple[EndpointTask, ...]
    metadata: Mapping[str, Any]


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


def _canonical_manifest_sha256(entries: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(
        list(entries),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _source_state(path: Path) -> State:
    atoms = read(path)
    numbers = np.asarray(atoms.numbers, dtype=int)
    positions = np.asarray(atoms.positions, dtype=float)
    if len(numbers) != 60 or not np.all(numbers == 6):
        raise ValueError(f"accepted endpoint is not C60: {path}")
    if bool(np.any(atoms.pbc)):
        raise ValueError(f"accepted C60 endpoint must be nonperiodic: {path}")
    return State(
        numbers=numbers,
        positions=positions,
        cell=None,
        pbc=(False, False, False),
        fixed_mask=None,
        metadata={"source": str(path), "stage": "post_scipy_accepted_endpoint"},
    )


def load_input_manifest() -> tuple[tuple[EndpointTask, ...], dict[str, Any]]:
    for path in (SOURCE_SUMMARY_PATH, SOURCE_ACCEPTED_LOG):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not SOURCE_ACCEPTED_DIR.is_dir():
        raise FileNotFoundError(SOURCE_ACCEPTED_DIR)
    summary_sha = _sha256(SOURCE_SUMMARY_PATH)
    accepted_log_sha = _sha256(SOURCE_ACCEPTED_LOG)
    if summary_sha != EXPECTED_SOURCE_SUMMARY_SHA256:
        raise ValueError("source production summary SHA256 mismatch")
    if accepted_log_sha != EXPECTED_ACCEPTED_LOG_SHA256:
        raise ValueError("accepted-structures log SHA256 mismatch")

    source_summary = json.loads(SOURCE_SUMMARY_PATH.read_text(encoding="utf-8"))
    config = source_summary.get("effective_config", {})
    if (
        source_summary.get("system") != "c60"
        or config.get("quench_optimizer") != "scipy-lbfgsb"
        or config.get("quench_fmax") != FMAX
        or config.get("quench_maxiter") != MAXITER
    ):
        raise ValueError("source production summary is not the pinned C60 SciPy run")

    records = [
        json.loads(line)
        for line in SOURCE_ACCEPTED_LOG.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != EXPECTED_ACCEPTED_COUNT:
        raise ValueError("accepted-structures log must contain exactly 143 rows")
    records.sort(key=lambda row: (row["trial_index"], row["discovered_entry_id"]))
    xyz_paths = tuple(sorted(SOURCE_ACCEPTED_DIR.glob("*.xyz")))
    if len(xyz_paths) != EXPECTED_ACCEPTED_COUNT:
        raise ValueError("accepted-minima directory must contain exactly 143 XYZ files")

    tasks: list[EndpointTask] = []
    entries: list[dict[str, Any]] = []
    expected_paths: set[Path] = set()
    identities: set[tuple[int, int]] = set()
    for task_index, record in enumerate(records):
        trial_index = int(record["trial_index"])
        entry_id = int(record["discovered_entry_id"])
        identity = (trial_index, entry_id)
        if identity in identities:
            raise ValueError(f"duplicate accepted endpoint identity: {identity}")
        identities.add(identity)
        path = SOURCE_ACCEPTED_DIR / (
            f"trial{trial_index:04d}_entry{entry_id:04d}_accepted.xyz"
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        expected_paths.add(path)
        path_sha = _sha256(path)
        source_energy = float(record["energy"])
        state = _source_state(path)
        tasks.append(
            EndpointTask(
                task_index=task_index,
                trial_index=trial_index,
                discovered_entry_id=entry_id,
                seed_entry_id=int(record["seed_entry_id"]),
                source_path=path,
                source_sha256=path_sha,
                source_archive_energy_eV=source_energy,
                state=state,
            )
        )
        entries.append(
            {
                "trial_index": trial_index,
                "discovered_entry_id": entry_id,
                "seed_entry_id": int(record["seed_entry_id"]),
                "source_archive_energy_eV": source_energy,
                "filename": path.name,
                "sha256": path_sha,
            }
        )
    if set(xyz_paths) != expected_paths:
        raise ValueError("accepted-minima files do not exactly match the accepted log")
    manifest_sha = _canonical_manifest_sha256(entries)
    if manifest_sha != EXPECTED_INPUT_MANIFEST_SHA256:
        raise ValueError("accepted-endpoint input manifest SHA256 mismatch")
    return tuple(tasks), {
        "production_summary_path": str(SOURCE_SUMMARY_PATH),
        "production_summary_sha256": summary_sha,
        "accepted_structures_path": str(SOURCE_ACCEPTED_LOG),
        "accepted_structures_sha256": accepted_log_sha,
        "accepted_minima_dir": str(SOURCE_ACCEPTED_DIR),
        "accepted_input_manifest_sha256": manifest_sha,
        "accepted_input_count": len(entries),
        "ordering": "trial_index,discovered_entry_id",
        "entries": entries,
    }


def preflight(
    *,
    expected_git_commit: str,
    input_loader: Callable[
        [], tuple[tuple[EndpointTask, ...], Mapping[str, Any]]
    ] = load_input_manifest,
    runtime_probe: Callable[[], Mapping[str, Any]] = _runtime_versions,
    cuda_probe: Callable[[], Mapping[str, Any]] = _cuda_info,
) -> Preflight:
    actual_commit = _current_commit()
    if actual_commit != expected_git_commit:
        raise RuntimeError(
            f"execution commit mismatch: expected {expected_git_commit}, got {actual_commit}"
        )
    if not _tracked_worktree_clean():
        raise RuntimeError("tracked worktree is not clean")
    if _safe_lbfgs_default_history_limit() != SAFE_LBFGS_DEFAULT_HISTORY_LIMIT:
        raise RuntimeError("safe L-BFGS default history limit is not 10")
    tasks, source_manifest = input_loader()
    if len(tasks) != EXPECTED_ACCEPTED_COUNT:
        raise ValueError("input manifest must contain all 143 accepted endpoints")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)
    model_sha = _sha256(MODEL_PATH)
    if model_sha != EXPECTED_MODEL_SHA256:
        raise RuntimeError("model SHA256 does not match the frozen MACE model")
    runtime_versions = {key: str(value) for key, value in runtime_probe().items()}
    cuda = dict(cuda_probe())
    if cuda.get("available") is not True:
        raise RuntimeError("CUDA is unavailable")
    return Preflight(
        tasks=tasks,
        metadata={
            "schema_version": 1,
            "execution_commit": actual_commit,
            "source": dict(source_manifest),
            "model": {"path": str(MODEL_PATH), "sha256": model_sha},
            "runtime_versions": runtime_versions,
            "cuda": cuda,
            "calculator": dict(CALCULATOR_CONFIG),
            "safe_lbfgs_default_history_limit": SAFE_LBFGS_DEFAULT_HISTORY_LIMIT,
        },
    )


def _calculator():
    from mace.calculators import MACECalculator

    return MACECalculator(model_paths=str(MODEL_PATH), **CALCULATOR_CONFIG)


def _counter(calculator: object) -> EvalCounter:
    return EvalCounter(ASECalculator(calculator))


def _active_max_force(flat_gradient: np.ndarray, state: State) -> float:
    matrix = np.asarray(flat_gradient, dtype=float).reshape(state.n_atoms, 3)
    active = matrix[state.movable_mask]
    if active.size == 0:
        return 0.0
    return float(np.max(np.linalg.norm(active, axis=1)))


def _count_delta(before: EvaluationCounts, after: EvaluationCounts) -> EvaluationCounts:
    return EvaluationCounts(
        tuple(
            after_value - before_value
            for before_value, after_value in zip(
                before.values, after.values, strict=True
            )
        )
    )


class _InitialRecordingEvaluator:
    def __init__(self, counter: EvalCounter, expected_positions: np.ndarray) -> None:
        self.counter = counter
        self.expected_positions = np.asarray(expected_positions, dtype=float)
        self.initial_energy: float | None = None
        self.initial_gradient: np.ndarray | None = None

    def __call__(
        self,
        flat_positions: np.ndarray,
        template: State,
    ) -> tuple[float, np.ndarray]:
        energy, gradient = self.counter.evaluate_flat(flat_positions, template)
        if self.initial_energy is None:
            positions = np.asarray(flat_positions, dtype=float).reshape(-1, 3)
            if not np.array_equal(positions, self.expected_positions):
                raise RuntimeError("optimizer did not evaluate the accepted endpoint first")
            self.initial_energy = float(energy)
            self.initial_gradient = np.asarray(gradient, dtype=float).reshape(-1).copy()
        return float(energy), np.asarray(gradient, dtype=float).reshape(-1).copy()


def execute_task(
    task: EndpointTask,
    arm: Arm,
    counter: EvalCounter,
) -> dict[str, Any]:
    before = counter.snapshot()
    evaluator = _InitialRecordingEvaluator(counter, task.state.positions)
    started = perf_counter()
    with counter.purpose(EvaluationPurpose.POST_RELAX_VALIDATION):
        result = Relaxer(evaluator, optimizer=arm.optimizer).relax(
            task.state,
            fmax=FMAX,
            maxiter=MAXITER,
            coordinate_trust_radius=COORDINATE_TRUST_RADIUS,
        )
    wall_time_s = perf_counter() - started
    after = counter.snapshot()
    delta = _count_delta(before, after)
    if evaluator.initial_energy is None or evaluator.initial_gradient is None:
        raise RuntimeError("relaxation did not evaluate its accepted-endpoint input")
    if (
        delta.total != result.telemetry.evaluator_calls
        or delta.count(EvaluationPurpose.POST_RELAX_VALIDATION) != delta.total
        or delta.count(EvaluationPurpose.UNATTRIBUTED) != 0
    ):
        raise RuntimeError("per-task counter delta does not close to telemetry evaluator calls")
    expected_history = (
        SAFE_LBFGS_DEFAULT_HISTORY_LIMIT
        if arm.optimizer == "safe-lbfgs-total"
        else None
    )
    if arm.safe_history_limit != expected_history:
        raise RuntimeError("arm history contract is inconsistent")
    return {
        "task_index": task.task_index,
        "trial_index": task.trial_index,
        "discovered_entry_id": task.discovered_entry_id,
        "seed_entry_id": task.seed_entry_id,
        "source_path": str(task.source_path),
        "source_sha256": task.source_sha256,
        "source_archive_energy_eV": task.source_archive_energy_eV,
        "arm_id": arm.arm_id,
        "optimizer": arm.optimizer,
        "safe_history_limit": arm.safe_history_limit,
        "fmax_eV_per_A": FMAX,
        "maxiter": MAXITER,
        "coordinate_trust_radius_A": COORDINATE_TRUST_RADIUS,
        "objective": OBJECTIVE,
        "initial": {
            "energy_eV": evaluator.initial_energy,
            "max_active_force_eV_per_A": _active_max_force(
                evaluator.initial_gradient, task.state
            ),
            "positions_sha256": position_sha256(task.state.positions),
            "positions": task.state.positions.tolist(),
        },
        "final": {
            "energy_eV": float(result.energy),
            "max_active_force_eV_per_A": float(result.gradient_norm),
            "positions_sha256": position_sha256(result.state.positions),
            "positions": result.state.positions.tolist(),
        },
        "n_iter": int(result.n_iter),
        "termination_reason": result.telemetry.termination_reason,
        "outcome_class": result.outcome_class.value,
        "telemetry": asdict(result.telemetry),
        "evaluator_calls": delta.total,
        "purpose_count_delta": delta.as_dict(),
        "wall_time_s": wall_time_s,
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
    counter_factory: Callable[[object], EvalCounter] = _counter,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    partial_dir = output_dir.with_name(f"{output_dir.name}.partial")
    for path in (output_dir, partial_dir):
        if path.exists():
            raise FileExistsError(path)
    checked = preflight_fn(expected_git_commit=expected_git_commit)
    if len(checked.tasks) != EXPECTED_ACCEPTED_COUNT:
        raise ValueError("preflight did not return all 143 accepted endpoints")
    if preflight_only:
        return dict(checked.metadata)

    partial_dir.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        counter = counter_factory(calculator_factory())
        for task in checked.tasks:
            rows.append(execute_task(task, arm, counter))
    if len(rows) != EXPECTED_ACCEPTED_COUNT * len(ARMS):
        raise RuntimeError("rescue ablation row count is incomplete")
    summary = {
        **dict(checked.metadata),
        "task_count": EXPECTED_ACCEPTED_COUNT,
        "row_count": EXPECTED_ACCEPTED_COUNT * len(ARMS),
        "protocol": {
            "fmax_eV_per_A": FMAX,
            "maxiter": MAXITER,
            "coordinate_trust_radius_A": COORDINATE_TRUST_RADIUS,
            "objective": OBJECTIVE,
            "task_selection": "all_accepted_endpoints_without_filtering",
        },
        "arms": [asdict(arm) for arm in ARMS],
        "rows_file": "rows.json",
        "interpretation_scope": "post_scipy_accepted_endpoint_refinement_rescue",
    }
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
    arguments = _parse_args(argv)
    result = run(
        output_dir=arguments.output,
        expected_git_commit=arguments.expected_git_commit,
        preflight_only=arguments.preflight_only,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
