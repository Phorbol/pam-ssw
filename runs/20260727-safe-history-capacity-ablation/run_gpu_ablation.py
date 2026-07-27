#!/usr/bin/env python3
"""Fail-closed, one-shot safe L-BFGS history-capacity GPU ablation.

This runner intentionally evaluates only two fixed arms over the 16 frozen
one-bias proposal-relaxation tasks.  It is an execution artifact: it neither
retunes nor retries a negative result.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import importlib
import importlib.util
import inspect
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
from time import perf_counter
from typing import Any, Callable, Mapping, NamedTuple, Sequence

import numpy as np

from pamssw.accounting import EvalCounter, EvaluationCounts, EvaluationPurpose
from pamssw.proposal_replay import proposal_task_from_payload
from pamssw.relax import Relaxer
from pamssw.result import RelaxResult
from pamssw.state import State
from pamssw.walker import ProposalRelaxationTask


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
PAMSSW_SOURCE_ROOT = REPO_ROOT / "pamssw"
OUTPUT_DIR = RUN_ROOT / "output" / "ledger"
SOURCE_SUMMARY_PATH = Path(
    "/tmp/SSW-worktrees/fixed-proposal-replay/"
    "runs/20260727-023234-fixed-proposal-replay-gpu/output/summary.json"
)
FIXED_REPLAY_DRIVER = SOURCE_SUMMARY_PATH.parent.parent / "run_fixed_replay.py"
G1_DRIVER_PATH = (
    FIXED_REPLAY_DRIVER.parent.parent
    / "20260727-014407-bias-relaxation-gpu-g1"
    / "run_g1.py"
)
TRACE_RECORDER_PATH = (
    RUN_ROOT.parent / "20260727-proposal-energy-traces" / "trace_recorder.py"
)

SYSTEMS = ("c60", "pdo")
SEEDS = tuple(range(42, 50))
MAXITER = 400
EXPECTED_SOURCE_SUMMARY_SHA256 = "62cc771e2aa24e9addef0e870d0524f901f02bddc34152cf2a2eeea91a671b04"
EXPECTED_PAMSSW_BUNDLE_SHA256 = "459a3ed173afbde50c499a2653702796ec1f08f022cc0a93ed2e028222d62636"
EXPECTED_FIXED_REPLAY_DRIVER_SHA256 = "f9c9602e42985891a6ca2a84ca70dda69c52b9d794a6ea6c76857c398345fa8f"
EXPECTED_G1_DRIVER_SHA256 = "0e69736d5d92372a2f4e449c440c09307613a55f0028126c7bb36d77296bbfbb"
EXPECTED_TRACE_RECORDER_SHA256 = "c6feeaabf0062f6654f8ea4b4fff610b258dcab754e1907dd3f4c3779e7165de"
EXPECTED_SAFE_KERNEL_DESCRIPTOR = {
    "optimizer": "safe-lbfgs-total",
    "safe_lbfgs_memory": 10,
    "kernel_constants": {
        "_SAFE_LBFGS_EMPTY_HISTORY_SCALE": 1.0 / 70.0,
        "_SAFE_LBFGS_MAX_ATOM_STEP": 0.2,
        "_SAFE_LBFGS_ARMIJO_C1": 1.0e-4,
        "_SAFE_LBFGS_BACKTRACK": 0.5,
        "_SAFE_LBFGS_MAX_LINE_TRIALS": 20,
        "_SAFE_LBFGS_MIN_ALPHA": 2.0**-20,
        "_SAFE_LBFGS_CURVATURE_REL": 1.4901161193847656e-08,
    },
}
EXPECTED_SAFE_KERNEL_DESCRIPTOR_SHA256 = "1846e340e762b79f50897dfacd40a16288ed52bb481f36f24ab01594c4724102"
_TRACE_RECORDER_MODULE_NAME = "_safe_history_capacity_trace_recorder"


@dataclass(frozen=True)
class Arm:
    arm_id: str
    kernel: str
    history_limit: int


ARMS = (
    Arm(
        arm_id="safe-total-gradient-history10",
        kernel="safe-lbfgs-total",
        history_limit=10,
    ),
    Arm(
        arm_id="safe-total-gradient-history0",
        kernel="safe-lbfgs-total",
        history_limit=0,
    ),
)


@dataclass(frozen=True)
class FrozenTask:
    system: str
    task_id: str
    seed: int
    task_payload: Mapping[str, Any]
    task_sha256: str
    relax_task: ProposalRelaxationTask


class ReplayResult(NamedTuple):
    result: RelaxResult
    evaluation_counts: EvaluationCounts
    wall_time_s: float
    trace_records: list[dict[str, Any]]
    accepted_callback_hashes: tuple[str, ...]
    certificate_satisfied: bool


@dataclass(frozen=True)
class Preflight:
    tasks_by_system: Mapping[str, tuple[FrozenTask, ...]]
    source_summary_sha256: str
    source: Mapping[str, Any]
    safe_kernel_descriptor: Mapping[str, Any]
    safe_kernel_descriptor_sha256: str
    helper_provenance: Mapping[str, Any]
    pamssw_source_provenance: Mapping[str, Any]
    repo_provenance: Mapping[str, Any]
    cuda_provenance: Mapping[str, Any]


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load {label} JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def canonical_task_sha256(task_payload: Mapping[str, Any]) -> str:
    """Return the canonical hash recorded for the source task payload."""

    if not isinstance(task_payload, Mapping):
        raise ValueError("task payload must be a mapping")
    try:
        serialized = json.dumps(
            task_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("task payload is not canonical JSON") from error
    return sha256(serialized.encode("utf-8")).hexdigest()


def canonical_descriptor_sha256(descriptor: Mapping[str, Any]) -> str:
    """Return the canonical SHA-256 of a serializable kernel descriptor."""

    if not isinstance(descriptor, Mapping):
        raise ValueError("kernel descriptor must be a mapping")
    try:
        serialized = json.dumps(
            descriptor,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("kernel descriptor is not canonical JSON") from error
    return sha256(serialized.encode("utf-8")).hexdigest()


def _validated_source_summary(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_sha256 = _sha256(path)
    if actual_sha256 != EXPECTED_SOURCE_SUMMARY_SHA256:
        raise ValueError(
            "source summary SHA256 mismatch: "
            f"expected {EXPECTED_SOURCE_SUMMARY_SHA256}, got {actual_sha256}"
        )
    return _load_json_object(path, label="source summary"), actual_sha256


def load_fixed_tasks(path: Path) -> tuple[dict[str, tuple[FrozenTask, ...]], str]:
    """Fail closed unless the source contains exactly the reviewed 16 tasks."""

    summary, summary_sha256 = _validated_source_summary(path)
    entries = summary.get("systems")
    if not isinstance(entries, list) or len(entries) != len(SYSTEMS):
        raise ValueError("source summary must contain exactly c60 and pdo")
    if tuple(entry.get("system") if isinstance(entry, Mapping) else None for entry in entries) != SYSTEMS:
        raise ValueError("source summary systems must be ordered exactly as c60, pdo")

    tasks_by_system: dict[str, tuple[FrozenTask, ...]] = {}
    for system, entry in zip(SYSTEMS, entries, strict=True):
        assert isinstance(entry, Mapping)  # checked while constructing the tuple above
        items = entry.get("tasks")
        if not isinstance(items, list) or len(items) != len(SEEDS):
            raise ValueError(f"source summary {system} must contain exactly 8 tasks")
        expected_ids = tuple(f"{system}-seed-{seed}-bias-1" for seed in SEEDS)
        frozen: list[FrozenTask] = []
        for item, expected_id, expected_seed in zip(items, expected_ids, SEEDS, strict=True):
            if not isinstance(item, Mapping):
                raise ValueError(f"source summary {system} task must be an object")
            task_id = item.get("task_id")
            seed = item.get("seed")
            payload = item.get("task")
            if task_id != expected_id:
                raise ValueError(
                    f"source summary {system} task ID mismatch: "
                    f"expected {expected_id!r}, got {task_id!r}"
                )
            if isinstance(seed, bool) or seed != expected_seed:
                raise ValueError(
                    f"source summary {system}/{task_id} seed mismatch: "
                    f"expected {expected_seed!r}, got {seed!r}"
                )
            if not isinstance(payload, Mapping):
                raise ValueError(f"source summary {system}/{task_id} has no task payload")
            try:
                task = proposal_task_from_payload(payload)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid frozen task: {system}/{task_id}") from error
            if len(task.biases) != 1:
                raise ValueError(f"source summary {system}/{task_id} is not a one-bias task")
            frozen.append(
                FrozenTask(
                    system=system,
                    task_id=expected_id,
                    seed=expected_seed,
                    task_payload=dict(payload),
                    task_sha256=canonical_task_sha256(payload),
                    relax_task=replace(task, maxiter=MAXITER),
                )
            )
        tasks_by_system[system] = tuple(frozen)
    return tasks_by_system, summary_sha256


def objective_descriptor() -> dict[str, Any]:
    """Bind the ledger to the currently imported safe-L-BFGS objective."""

    from pamssw import relax as relax_module

    constant_names = (
        "_SAFE_LBFGS_EMPTY_HISTORY_SCALE",
        "_SAFE_LBFGS_MAX_ATOM_STEP",
        "_SAFE_LBFGS_ARMIJO_C1",
        "_SAFE_LBFGS_BACKTRACK",
        "_SAFE_LBFGS_MAX_LINE_TRIALS",
        "_SAFE_LBFGS_MIN_ALPHA",
        "_SAFE_LBFGS_CURVATURE_REL",
    )
    return {
        "optimizer": "safe-lbfgs-total",
        "safe_lbfgs_memory": int(relax_module._SAFE_LBFGS_MEMORY),
        "kernel_constants": {
            name: getattr(relax_module, name) for name in constant_names
        },
    }


def _verified_safe_kernel_descriptor() -> tuple[dict[str, Any], str]:
    descriptor = objective_descriptor()
    descriptor_sha256 = canonical_descriptor_sha256(descriptor)
    if descriptor_sha256 != EXPECTED_SAFE_KERNEL_DESCRIPTOR_SHA256:
        raise ValueError(
            "safe kernel descriptor SHA256 mismatch: "
            f"expected {EXPECTED_SAFE_KERNEL_DESCRIPTOR_SHA256}, got {descriptor_sha256}"
        )
    return descriptor, descriptor_sha256


def _verified_helper_file(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{label} SHA256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    return {"path": str(path), "sha256": actual_sha256}


def _verified_helper_provenance() -> dict[str, dict[str, str]]:
    return {
        "fixed_replay_driver": _verified_helper_file(
            FIXED_REPLAY_DRIVER,
            expected_sha256=EXPECTED_FIXED_REPLAY_DRIVER_SHA256,
            label="fixed replay driver",
        ),
        "g1_driver": _verified_helper_file(
            G1_DRIVER_PATH,
            expected_sha256=EXPECTED_G1_DRIVER_SHA256,
            label="G1 driver",
        ),
        "trace_recorder": _verified_helper_file(
            TRACE_RECORDER_PATH,
            expected_sha256=EXPECTED_TRACE_RECORDER_SHA256,
            label="trace recorder",
        ),
    }


def _pamssw_bundle_sha256(source_root: Path) -> str:
    """Hash sorted repo-relative Python sources, excluding bytecode caches."""

    if not source_root.is_dir():
        raise FileNotFoundError(f"pamssw source root does not exist: {source_root}")
    source_files = sorted(
        path
        for path in source_root.rglob("*.py")
        if path.is_file() and "__pycache__" not in path.parts
    )
    if not source_files:
        raise ValueError(f"pamssw source root contains no Python files: {source_root}")
    digest = sha256()
    for path in source_files:
        relative_path = path.relative_to(source_root).as_posix().encode("utf-8")
        digest.update(relative_path)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _imported_pamssw_source_paths() -> dict[str, Path]:
    """Return filesystem paths for every pamssw module used by this replay chain."""

    module_names = (
        "pamssw",
        "pamssw.accounting",
        "pamssw.calculators",
        "pamssw.exploration",
        "pamssw.exploration.runner",
        "pamssw.pbc",
        "pamssw.proposal_replay",
        "pamssw.relax",
        "pamssw.result",
        "pamssw.state",
        "pamssw.walker",
    )
    paths: dict[str, Path] = {}
    for module_name in module_names:
        module = importlib.import_module(module_name)
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str):
            raise RuntimeError(f"pamssw module has no source file: {module_name}")
        paths[module_name] = Path(module_file).resolve()
    return paths


def _pamssw_symbol_source_paths() -> dict[str, Path]:
    return {
        "EvalCounter": Path(inspect.getfile(EvalCounter)).resolve(),
        "Relaxer": Path(inspect.getfile(Relaxer)).resolve(),
        "ProposalRelaxationTask": Path(inspect.getfile(ProposalRelaxationTask)).resolve(),
    }


def _verified_pamssw_source() -> dict[str, Any]:
    source_root = PAMSSW_SOURCE_ROOT.resolve()
    imported_module_paths = _imported_pamssw_source_paths()
    imported_symbol_paths = _pamssw_symbol_source_paths()
    imported_paths = {**imported_module_paths, **imported_symbol_paths}
    for name, path in imported_paths.items():
        if not path.is_relative_to(source_root):
            raise RuntimeError(
                "pamssw import path is outside the current repo source tree: "
                f"{name} -> {path}"
            )
    bundle_sha256 = _pamssw_bundle_sha256(source_root)
    if bundle_sha256 != EXPECTED_PAMSSW_BUNDLE_SHA256:
        raise ValueError(
            "pamssw source bundle SHA256 mismatch: "
            f"expected {EXPECTED_PAMSSW_BUNDLE_SHA256}, got {bundle_sha256}"
        )
    return {
        "source_root": str(source_root),
        "bundle_sha256": bundle_sha256,
        "imported_module_paths": {
            name: str(path) for name, path in imported_module_paths.items()
        },
        "imported_symbol_paths": {
            name: str(path) for name, path in imported_symbol_paths.items()
        },
    }


def _worktree_is_clean() -> bool:
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return not completed.stdout.strip()


def _verified_repo_provenance(expected_git_commit: str) -> dict[str, Any]:
    if (
        not isinstance(expected_git_commit, str)
        or len(expected_git_commit) != 40
        or any(character not in "0123456789abcdef" for character in expected_git_commit)
    ):
        raise ValueError("expected git commit must be a 40-character lowercase SHA-1")
    actual_git_commit = _current_commit()
    if actual_git_commit != expected_git_commit:
        raise RuntimeError(
            "git commit mismatch: "
            f"expected {expected_git_commit}, got {actual_git_commit}"
        )
    if not _worktree_is_clean():
        raise RuntimeError("git worktree is not clean")
    return {
        "repo_root": str(REPO_ROOT),
        "actual_git_commit": actual_git_commit,
        "expected_git_commit": expected_git_commit,
        "worktree_clean": True,
    }


def _fixed_replay_source() -> Mapping[str, Any]:
    _verified_helper_file(
        FIXED_REPLAY_DRIVER,
        expected_sha256=EXPECTED_FIXED_REPLAY_DRIVER_SHA256,
        label="fixed replay driver",
    )
    _verified_helper_file(
        G1_DRIVER_PATH,
        expected_sha256=EXPECTED_G1_DRIVER_SHA256,
        label="G1 driver",
    )
    helpers = runpy.run_path(str(FIXED_REPLAY_DRIVER))
    source_factory = helpers.get("_source")
    if not callable(source_factory):
        raise RuntimeError("fixed replay driver does not expose _source()")
    source = source_factory()
    if not isinstance(source, Mapping):
        raise RuntimeError("fixed replay source must be a mapping")
    if not callable(source.get("_calculator")):
        raise RuntimeError("fixed replay source does not expose _calculator")
    return source


def _verified_file_provenance(
    path_value: object,
    declared_sha256: object,
    *,
    label: str,
) -> tuple[str, str, str]:
    if not isinstance(path_value, (str, Path)):
        raise ValueError(f"{label} path is required")
    path = Path(path_value)
    if not path.is_file():
        raise FileNotFoundError(f"{label} path does not exist: {path}")
    if not isinstance(declared_sha256, str) or len(declared_sha256) != 64:
        raise ValueError(f"{label} declared SHA256 is required")
    verified_sha256 = _sha256(path)
    if verified_sha256 != declared_sha256:
        raise ValueError(
            f"{label} SHA256 mismatch: expected {declared_sha256}, got {verified_sha256}"
        )
    return str(path), declared_sha256, verified_sha256


def _verified_model_provenance(source: Mapping[str, Any]) -> dict[str, Any]:
    """Validate model and both source inputs before a calculator exists."""

    model_path, model_declared_sha256, model_verified_sha256 = _verified_file_provenance(
        source.get("MODEL"), source.get("MODEL_SHA256"), label="model"
    )
    source_systems = source.get("SYSTEMS")
    if not isinstance(source_systems, Mapping) or tuple(source_systems) != SYSTEMS:
        raise ValueError("source SYSTEMS must be ordered exactly as c60, pdo")
    system_inputs: dict[str, dict[str, str]] = {}
    for system in SYSTEMS:
        specification = source_systems[system]
        if not isinstance(specification, Mapping):
            raise ValueError(f"source system specification must be an object: {system!r}")
        input_path, input_declared_sha256, input_verified_sha256 = _verified_file_provenance(
            specification.get("input"),
            specification.get("sha256"),
            label=f"input {system}",
        )
        system_inputs[system] = {
            "input": input_path,
            "declared_sha256": input_declared_sha256,
            "sha256": input_verified_sha256,
        }
    return {
        "model": model_path,
        "model_declared_sha256": model_declared_sha256,
        "model_sha256": model_verified_sha256,
        "source_system_inputs": system_inputs,
    }


def _cuda_provenance(source: Mapping[str, Any]) -> dict[str, Any]:
    del source
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch with CUDA support is required") from error
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False")
    return {
        "device": "cuda",
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
    }


def preflight(
    *,
    source_summary_path: Path,
    expected_git_commit: str,
    source_loader: Callable[[], Mapping[str, Any]] = _fixed_replay_source,
    cuda_probe: Callable[[Mapping[str, Any]], Mapping[str, Any]] = _cuda_provenance,
) -> Preflight:
    """Complete all source/provenance/CUDA checks before calculator creation."""

    tasks_by_system, source_summary_sha256 = load_fixed_tasks(source_summary_path)
    safe_kernel_descriptor, safe_kernel_descriptor_sha256 = _verified_safe_kernel_descriptor()
    helper_provenance = _verified_helper_provenance()
    pamssw_source_provenance = _verified_pamssw_source()
    repo_provenance = _verified_repo_provenance(expected_git_commit)
    source = source_loader()
    if not isinstance(source, Mapping) or not callable(source.get("_calculator")):
        raise RuntimeError("fixed replay source does not expose _calculator")
    model_input_provenance = _verified_model_provenance(source)
    cuda_provenance = cuda_probe(source)
    if not isinstance(cuda_provenance, Mapping) or cuda_provenance.get("device") != "cuda":
        raise RuntimeError("CUDA provenance must confirm device='cuda'")
    verified_provenance = {**model_input_provenance, **dict(cuda_provenance)}
    return Preflight(
        tasks_by_system=tasks_by_system,
        source_summary_sha256=source_summary_sha256,
        source=source,
        safe_kernel_descriptor=safe_kernel_descriptor,
        safe_kernel_descriptor_sha256=safe_kernel_descriptor_sha256,
        helper_provenance=helper_provenance,
        pamssw_source_provenance=pamssw_source_provenance,
        repo_provenance=repo_provenance,
        cuda_provenance=verified_provenance,
    )


def _trace_recorder_module():
    """Load the existing recorder only by its experiment-local file path."""

    _verified_helper_file(
        TRACE_RECORDER_PATH,
        expected_sha256=EXPECTED_TRACE_RECORDER_SHA256,
        label="trace recorder",
    )
    module = sys.modules.get(_TRACE_RECORDER_MODULE_NAME)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(
        _TRACE_RECORDER_MODULE_NAME,
        TRACE_RECORDER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load trace recorder: {TRACE_RECORDER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_TRACE_RECORDER_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def replay_task_with_trace(
    task: ProposalRelaxationTask,
    calculator,
    *,
    history_limit: int,
) -> ReplayResult:
    """Run one arm while the observer records no PES calls of its own."""

    trace_recorder = _trace_recorder_module()
    counter = EvalCounter(calculator)
    proposal = trace_recorder.RecordingProposalPotential(
        counter,
        biases=list(task.biases),
        softening=task.softening,
    )
    relaxer = Relaxer(
        proposal.evaluate,
        optimizer="safe-lbfgs-total",
        component_evaluator=proposal.evaluate_parts,
    )
    accepted_callback_hashes: list[str] = []

    def record_accepted_state(state: State) -> None:
        accepted_callback_hashes.append(trace_recorder.position_hash(state))

    started = perf_counter()
    with counter.purpose(EvaluationPurpose.BIASED_PROPOSAL_RELAX):
        result = relaxer.relax(
            task.initial_state,
            fmax=task.fmax,
            maxiter=MAXITER,
            coordinate_trust_radius=task.coordinate_trust_radius,
            _safe_lbfgs_history_limit=history_limit,
            trajectory_callback=record_accepted_state,
        )
    wall_time_s = perf_counter() - started
    evaluation_counts = counter.snapshot()
    trace_records = trace_recorder.mark_accepted_state_evaluations(
        proposal.records,
        accepted_callback_hashes,
    )
    if len(trace_records) != evaluation_counts.total:
        raise RuntimeError(
            "trace recorder/counter mismatch: "
            f"records={len(trace_records)} counter={evaluation_counts.total}"
        )
    if len(trace_records) != result.telemetry.evaluator_calls:
        raise RuntimeError(
            "trace recorder/telemetry mismatch: "
            f"records={len(trace_records)} telemetry={result.telemetry.evaluator_calls}"
        )
    certificate_satisfied = bool(
        np.isfinite(result.energy)
        and np.isfinite(result.gradient_norm)
        and np.all(np.isfinite(result.state.positions))
        and result.gradient_norm <= task.fmax
    )
    return ReplayResult(
        result=result,
        evaluation_counts=evaluation_counts,
        wall_time_s=wall_time_s,
        trace_records=trace_records,
        accepted_callback_hashes=tuple(accepted_callback_hashes),
        certificate_satisfied=certificate_satisfied,
    )


def _trace_values_are_finite(records: Sequence[Mapping[str, Any]]) -> bool:
    fields = (
        "true_energy_eV",
        "bias_energy_eV",
        "softening_energy_eV",
        "total_energy_eV",
        "active_max_total_force_eV_per_A",
    )
    try:
        return all(np.isfinite(float(record[field])) for record in records for field in fields)
    except (KeyError, TypeError, ValueError):
        return False


def _expected_row_keys() -> list[dict[str, str]]:
    return [
        {"system": system, "task_id": f"{system}-seed-{seed}-bias-1", "arm_id": arm.arm_id}
        for system in SYSTEMS
        for seed in SEEDS
        for arm in ARMS
    ]


def _validate_full_ledger_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    expected = {
        (row["system"], row["task_id"], row["arm_id"])
        for row in _expected_row_keys()
    }
    try:
        observed = {(row["system"], row["task_id"], row["arm_id"]) for row in rows}
    except (KeyError, TypeError) as error:
        raise ValueError("32-row ledger rows require system, task_id, and arm_id") from error
    if len(rows) != 32 or len(observed) != 32 or observed != expected:
        raise ValueError("32-row ledger does not contain exactly every fixed task and arm")


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def publish_ledger_atomically(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    summary_payload: Mapping[str, Any],
    *,
    write_json: Callable[[Path, Any], None] = _write_json_atomic,
) -> None:
    """Publish only a complete 32-row ledger with one directory rename."""

    if output_dir.exists():
        raise FileExistsError(output_dir)
    _validate_full_ledger_rows(rows)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    try:
        for system in SYSTEMS:
            system_rows = [dict(row) for row in rows if row["system"] == system]
            write_json(
                staging_dir / f"{system}.json",
                {
                    "system": system,
                    "task_ids": [f"{system}-seed-{seed}-bias-1" for seed in SEEDS],
                    "rows": system_rows,
                },
            )
        write_json(staging_dir / "summary.json", summary_payload)
        if output_dir.exists():
            raise FileExistsError(output_dir)
        staging_dir.rename(output_dir)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)


def _current_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _row_payload(
    frozen: FrozenTask,
    arm: Arm,
    replay: ReplayResult,
    descriptor: Mapping[str, Any],
) -> dict[str, Any]:
    result = replay.result
    return {
        "system": frozen.system,
        "task_id": frozen.task_id,
        "seed": frozen.seed,
        "task_sha256": frozen.task_sha256,
        "source_task_maxiter": int(frozen.task_payload["maxiter"]),
        "replay_maxiter": MAXITER,
        "arm_id": arm.arm_id,
        "kernel": arm.kernel,
        "history_limit": arm.history_limit,
        "objective_descriptor": dict(descriptor),
        "force_evaluations": replay.evaluation_counts.total,
        "purpose_counts": replay.evaluation_counts.as_dict(),
        "wall_time_s": replay.wall_time_s,
        "certificate_satisfied": replay.certificate_satisfied,
        "termination_reason": result.telemetry.termination_reason,
        "endpoint": {
            "biased_energy_eV": float(result.energy),
            "max_active_atom_force_eV_per_A": float(result.gradient_norm),
            "positions": result.state.positions.tolist(),
        },
        "telemetry": asdict(result.telemetry),
        "accepted_callback_hashes": list(replay.accepted_callback_hashes),
        "trace_records": replay.trace_records,
    }


def run(
    *,
    source_summary_path: Path = SOURCE_SUMMARY_PATH,
    output_dir: Path = OUTPUT_DIR,
    expected_git_commit: str,
) -> dict[str, Any]:
    """Execute every fixed task once per arm and atomically publish the ledger."""

    if output_dir.exists():
        raise FileExistsError(output_dir)
    checked = preflight(
        source_summary_path=source_summary_path,
        expected_git_commit=expected_git_commit,
    )
    descriptor = checked.safe_kernel_descriptor
    calculator_factory = checked.source["_calculator"]
    started = perf_counter()
    rows: list[dict[str, Any]] = []
    for system in SYSTEMS:
        calculators = {arm.arm_id: calculator_factory() for arm in ARMS}
        for frozen in checked.tasks_by_system[system]:
            for arm in ARMS:
                replay = replay_task_with_trace(
                    frozen.relax_task,
                    calculators[arm.arm_id],
                    history_limit=arm.history_limit,
                )
                if not _trace_values_are_finite(replay.trace_records):
                    raise RuntimeError(f"non-finite trace value: {system}/{frozen.task_id}/{arm.arm_id}")
                if not replay.certificate_satisfied:
                    raise RuntimeError(f"certificate not satisfied: {system}/{frozen.task_id}/{arm.arm_id}")
                rows.append(_row_payload(frozen, arm, replay, descriptor))
    _validate_full_ledger_rows(rows)
    summary = {
        "schema_version": 1,
        "claim_ceiling": (
            "Fixed-task C60/PdO CUDA replay evidence only; no endpoint-equivalence, "
            "SSW-performance, statistical, or default-change claim."
        ),
        "source_summary": str(source_summary_path),
        "source_summary_sha256": checked.source_summary_sha256,
        "systems": list(SYSTEMS),
        "task_count": 16,
        "row_count": 32,
        "arms": [asdict(arm) for arm in ARMS],
        "safe_kernel_descriptor": descriptor,
        "safe_kernel_descriptor_sha256": checked.safe_kernel_descriptor_sha256,
        "runner_helper_provenance": dict(checked.helper_provenance),
        "pamssw_source_provenance": dict(checked.pamssw_source_provenance),
        "git_provenance": dict(checked.repo_provenance),
        "cuda_model_input_provenance": dict(checked.cuda_provenance),
        "current_git_commit": checked.repo_provenance["actual_git_commit"],
        "wall_time_total_s": perf_counter() - started,
    }
    publish_ledger_atomically(output_dir, rows, summary)
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-summary", type=Path, default=SOURCE_SUMMARY_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--expected-git-commit", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    run(
        source_summary_path=args.source_summary,
        output_dir=args.output_dir,
        expected_git_commit=args.expected_git_commit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
