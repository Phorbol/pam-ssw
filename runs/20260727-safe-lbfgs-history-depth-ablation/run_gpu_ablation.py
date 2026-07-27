#!/usr/bin/env python3
"""Fail-closed, one-shot safe L-BFGS history-depth GPU ablation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
from pathlib import Path
import platform
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
OUTPUT_DIR = RUN_ROOT / "output"
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
EXPECTED_PAMSSW_BUNDLE_SHA256 = "83a2b845c4bbe0235e8c584a579e1b9f1e89690084c8abdd9ef3aa20c2a09b50"
EXPECTED_FIXED_REPLAY_DRIVER_SHA256 = "f9c9602e42985891a6ca2a84ca70dda69c52b9d794a6ea6c76857c398345fa8f"
EXPECTED_G1_DRIVER_SHA256 = "0e69736d5d92372a2f4e449c440c09307613a55f0028126c7bb36d77296bbfbb"
EXPECTED_TRACE_RECORDER_SHA256 = "c6feeaabf0062f6654f8ea4b4fff610b258dcab754e1907dd3f4c3779e7165de"
EXPECTED_TASK_SHA256 = {
    "c60-seed-42-bias-1": "2837438681d2343e3fd5eef9339a26c8b4e1746459c166092110a1ead9df3bfa",
    "c60-seed-43-bias-1": "6bebcffbe675eae301173760186ac08ee7f8339afa4cc7099e2b1e413690b38f",
    "c60-seed-44-bias-1": "4ac7a3949bc1c72a24a79b4f1ac87000dd82f39f8e91e1ad1d0abe480e3ea98b",
    "c60-seed-45-bias-1": "3d3a552793967194e3a98bd2d6fdf028fc1cdc02853b9a0dbf22fd5048f0c6ef",
    "c60-seed-46-bias-1": "822b04923ef5b0b0450706ac279d12ad2c8bcb48032510fdef0b38619643fa1c",
    "c60-seed-47-bias-1": "75392b6f59908b81cada5e2acc24518ee73369a54087794d8e1efe2a86a11380",
    "c60-seed-48-bias-1": "caa0ea5a95ad4ac229ff82f5e569c1a4a4c60199bb343f4a5fdc0677b807569a",
    "c60-seed-49-bias-1": "e5fcb74aac18248389f0ca0033454558698710be9c6c8434403e2ff7e8af655a",
    "pdo-seed-42-bias-1": "ca24fb95083d970446ceb5cadf37c910c57ed6ec12dd3c275ec82027c7bfe051",
    "pdo-seed-43-bias-1": "813f61a3478a4e64ba224cda061e61991d4386fb2fcfa574c16846e84928048e",
    "pdo-seed-44-bias-1": "3fa440c3255aefdbe3cf3797ce1d02762b8e66b729ede1ca613969c38df62c6e",
    "pdo-seed-45-bias-1": "273268331fd7453f4c697494bd6456f3c27e753e6efcfe34b1b62ee4113a3510",
    "pdo-seed-46-bias-1": "dc4dd4e9141a4c51678d7c08b95aac5cd589d29cc7b3ec0076c598f77eef997a",
    "pdo-seed-47-bias-1": "2fad468b328fc1b0480ebba151abd8746df5b29a6c2f4ce09e526163d53d8d3a",
    "pdo-seed-48-bias-1": "089555d5b46c1785e54a67174bdfcd35f5cd3621a523ae4e8f982153b567bd2b",
    "pdo-seed-49-bias-1": "9e7c662b926315cb2969c71ea81ba15bb67a2a0ca9a3c6789796cb48e1ed6857",
}
EXPECTED_SAFE_KERNEL_DESCRIPTOR = {
    "optimizer": "safe-lbfgs-total",
    "safe_lbfgs_memory": 10,
    "secant_policy": "total-biased-gradient",
    "scale_policies": {
        "latest-history-pair-gamma-plus-one-two-loop-correction": (
            "history_limit=1; gamma=(s.T@y)/(y.T@y) from latest accepted "
            "total-gradient secant; one two-loop correction"
        ),
        "latest-history-pair-gamma-plus-up-to-ten-two-loop-corrections": (
            "history_limit=10; gamma=(s.T@y)/(y.T@y) from latest accepted "
            "total-gradient secant; up to ten two-loop corrections"
        ),
    },
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
EXPECTED_SAFE_KERNEL_DESCRIPTOR_SHA256 = "72c2cfe8299c5950f3f0085cfc95810680e0d2a087685d878ef5e259a5918bbf"
TRACE_MODULE_NAME = "_safe_lbfgs_history_depth_trace_recorder"
RUNTIME_VERSION_KEYS = (
    "python",
    "python_implementation",
    "numpy",
    "scipy",
    "ase",
    "torch",
    "mace",
)
PLATFORM_PROVENANCE_KEYS = ("sys_platform", "system", "release", "machine")
PURPOSE_KEYS = tuple(purpose.value for purpose in EvaluationPurpose)
TRACE_KEYS = (
    "evaluation_index",
    "positions_sha256",
    "true_energy_eV",
    "bias_energy_eV",
    "softening_energy_eV",
    "total_energy_eV",
    "active_max_total_force_eV_per_A",
    "accepted_state",
)
TELEMETRY_KEYS = (
    "backend",
    "evaluator_calls",
    "backend_evaluations",
    "reporting_cache_hits",
    "reporting_evaluator_calls",
    "finalization_requests",
    "explicit_finalization_calls",
    "gradient_measure",
    "converged",
    "termination_reason",
    "optimizer_success",
    "accepted_steps",
    "rejected_steps",
    "accepted_secants",
    "rejected_secants",
    "line_search_evaluations",
    "mic_branch_resets",
    "bias_secant_curvature_sum",
)
ROW_KEYS = (
    "system",
    "task_id",
    "seed",
    "task_payload",
    "task_sha256",
    "source_task_maxiter",
    "replay_maxiter",
    "arm_id",
    "arm",
    "result",
    "certificate",
    "termination",
    "force_evaluations",
    "purpose_counts",
    "unattributed_calls",
    "zero_extra_call_trace",
    "accepted_callback_hashes",
    "telemetry",
    "wall_time_s",
    "endpoint",
)
SUMMARY_KEYS = (
    "schema_version",
    "claim_ceiling",
    "source_summary_sha256",
    "systems",
    "task_count",
    "row_count",
    "certificate_all_satisfied",
    "certificate_unsatisfied_count",
    "termination_reason_counts",
    "arms",
    "safe_kernel_descriptor",
    "safe_kernel_descriptor_sha256",
    "runner_helper_provenance",
    "pamssw_source_provenance",
    "git_provenance",
    "runtime_versions",
    "platform_provenance",
    "cuda_model_input_provenance",
    "wall_time_total_s",
)


@dataclass(frozen=True)
class Arm:
    arm_id: str
    kernel: str
    history_limit: int
    scale_policy: str
    secant_policy: str
    adaptive_scale_without_history: bool = False


ARMS = (
    Arm(
        arm_id="adaptive-scale-history1",
        kernel="safe-lbfgs-total",
        history_limit=1,
        scale_policy="latest-history-pair-gamma-plus-one-two-loop-correction",
        secant_policy="total-biased-gradient",
    ),
    Arm(
        arm_id="adaptive-scale-history10",
        kernel="safe-lbfgs-total",
        history_limit=10,
        scale_policy="latest-history-pair-gamma-plus-up-to-ten-two-loop-corrections",
        secant_policy="total-biased-gradient",
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
    resolved_arm: Mapping[str, Any]


@dataclass(frozen=True)
class Preflight:
    tasks_by_system: Mapping[str, tuple[FrozenTask, ...]]
    source_summary_sha256: str
    source: Mapping[str, Any]
    safe_kernel_descriptor: Mapping[str, Any]
    safe_kernel_descriptor_sha256: str
    helper_provenance: Mapping[str, Any]
    pamssw_source_provenance: Mapping[str, Any]
    git_provenance: Mapping[str, Any]
    runtime_versions: Mapping[str, Any]
    platform_provenance: Mapping[str, Any]
    cuda_model_input_provenance: Mapping[str, Any]


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any, *, label: str) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not canonical finite JSON") from error


def canonical_task_sha256(task_payload: Mapping[str, Any]) -> str:
    if not isinstance(task_payload, Mapping):
        raise ValueError("task payload must be a mapping")
    return sha256(_canonical_json(task_payload, label="task payload").encode("utf-8")).hexdigest()


def canonical_descriptor_sha256(descriptor: Mapping[str, Any]) -> str:
    if not isinstance(descriptor, Mapping):
        raise ValueError("kernel descriptor must be a mapping")
    return sha256(_canonical_json(descriptor, label="kernel descriptor").encode("utf-8")).hexdigest()


def _exact_keys(value: object, keys: Sequence[str], *, label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError(f"{label} schema keys mismatch")
    return value


def _require_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    value = _require_string(value, label=label)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def _require_git_sha1(value: object, *, label: str) -> str:
    value = _require_string(value, label=label)
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-1")
    return value


def _require_finite_json(value: Any, *, label: str) -> None:
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"{label} contains a non-finite float")
    elif isinstance(value, (str, int, bool)) or value is None:
        return
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _require_finite_json(item, label=f"{label}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{label} has a non-string key")
            _require_finite_json(item, label=f"{label}.{key}")
    else:
        raise ValueError(f"{label} has a non-JSON value")


def _validated_source_summary(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    measured = _sha256(path)
    if measured != EXPECTED_SOURCE_SUMMARY_SHA256:
        raise ValueError("source summary SHA256 mismatch")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("source summary is not valid JSON") from error
    if type(payload) is not dict:
        raise ValueError("source summary must be an object")
    _require_finite_json(payload, label="source summary")
    return payload, measured


def _tasks_from_summary(summary: Mapping[str, Any]) -> dict[str, tuple[FrozenTask, ...]]:
    entries = summary.get("systems")
    if not isinstance(entries, list) or len(entries) != len(SYSTEMS):
        raise ValueError("source summary must contain exactly c60 and pdo")
    if tuple(entry.get("system") if isinstance(entry, Mapping) else None for entry in entries) != SYSTEMS:
        raise ValueError("source summary systems must be ordered exactly as c60, pdo")
    tasks_by_system: dict[str, tuple[FrozenTask, ...]] = {}
    for system, entry in zip(SYSTEMS, entries, strict=True):
        if not isinstance(entry, Mapping):
            raise ValueError(f"source summary {system} must be an object")
        raw_tasks = entry.get("tasks")
        if not isinstance(raw_tasks, list) or len(raw_tasks) != len(SEEDS):
            raise ValueError(f"source summary {system} must contain exactly eight tasks")
        tasks: list[FrozenTask] = []
        for raw_task, seed in zip(raw_tasks, SEEDS, strict=True):
            if not isinstance(raw_task, Mapping):
                raise ValueError(f"source summary {system} task must be an object")
            task_id = f"{system}-seed-{seed}-bias-1"
            if raw_task.get("task_id") != task_id or raw_task.get("seed") != seed:
                raise ValueError(f"source summary task identity mismatch: {task_id}")
            payload = raw_task.get("task")
            if not isinstance(payload, Mapping):
                raise ValueError(f"source summary {task_id} lacks a task payload")
            measured_sha256 = canonical_task_sha256(payload)
            if measured_sha256 != EXPECTED_TASK_SHA256[task_id]:
                raise ValueError(f"canonical task SHA256 mismatch: {task_id}")
            try:
                replay_task = proposal_task_from_payload(payload)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid frozen task: {task_id}") from error
            if len(replay_task.biases) != 1:
                raise ValueError(f"frozen task is not a one-bias task: {task_id}")
            tasks.append(
                FrozenTask(
                    system=system,
                    task_id=task_id,
                    seed=seed,
                    task_payload=json.loads(_canonical_json(payload, label=task_id)),
                    task_sha256=measured_sha256,
                    relax_task=replace(replay_task, maxiter=MAXITER),
                )
            )
        tasks_by_system[system] = tuple(tasks)
    return tasks_by_system


def load_fixed_tasks(path: Path) -> tuple[dict[str, tuple[FrozenTask, ...]], str]:
    summary, measured = _validated_source_summary(path)
    return _tasks_from_summary(summary), measured


def objective_descriptor() -> dict[str, Any]:
    from pamssw import relax as relax_module

    constant_names = tuple(EXPECTED_SAFE_KERNEL_DESCRIPTOR["kernel_constants"])
    return {
        "optimizer": "safe-lbfgs-total",
        "safe_lbfgs_memory": int(relax_module._SAFE_LBFGS_MEMORY),
        "secant_policy": "total-biased-gradient",
        "scale_policies": dict(EXPECTED_SAFE_KERNEL_DESCRIPTOR["scale_policies"]),
        "kernel_constants": {name: getattr(relax_module, name) for name in constant_names},
    }


def _verified_safe_kernel_descriptor() -> tuple[dict[str, Any], str]:
    descriptor = objective_descriptor()
    measured = canonical_descriptor_sha256(descriptor)
    if descriptor != EXPECTED_SAFE_KERNEL_DESCRIPTOR or measured != EXPECTED_SAFE_KERNEL_DESCRIPTOR_SHA256:
        raise ValueError("safe kernel descriptor SHA256 mismatch")
    return descriptor, measured


def _verified_helper_file(path: Path, *, expected_sha256: str, label: str) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    measured = _sha256(path)
    if measured != expected_sha256:
        raise ValueError(f"{label} SHA256 mismatch")
    return {"path": str(path.resolve()), "sha256": measured}


def _verified_helper_provenance() -> dict[str, dict[str, str]]:
    return {
        "fixed_replay_driver": _verified_helper_file(FIXED_REPLAY_DRIVER, expected_sha256=EXPECTED_FIXED_REPLAY_DRIVER_SHA256, label="fixed replay driver"),
        "g1_driver": _verified_helper_file(G1_DRIVER_PATH, expected_sha256=EXPECTED_G1_DRIVER_SHA256, label="G1 driver"),
        "trace_recorder": _verified_helper_file(TRACE_RECORDER_PATH, expected_sha256=EXPECTED_TRACE_RECORDER_SHA256, label="trace recorder"),
    }


def _pamssw_bundle_sha256(source_root: Path) -> str:
    if not source_root.is_dir():
        raise FileNotFoundError(f"pamssw source root does not exist: {source_root}")
    paths = sorted(path for path in source_root.rglob("*.py") if path.is_file() and "__pycache__" not in path.parts)
    if not paths:
        raise ValueError("pamssw source root contains no Python files")
    digest = sha256()
    for path in paths:
        digest.update(path.relative_to(source_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _imported_pamssw_source_paths() -> dict[str, Path]:
    names = (
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
    for name in names:
        module = importlib.import_module(name)
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str):
            raise RuntimeError(f"pamssw module has no source path: {name}")
        paths[name] = Path(module_file).resolve()
    return paths


def _pamssw_symbol_source_paths() -> dict[str, Path]:
    return {
        "EvalCounter": Path(inspect.getfile(EvalCounter)).resolve(),
        "Relaxer": Path(inspect.getfile(Relaxer)).resolve(),
        "ProposalRelaxationTask": Path(inspect.getfile(ProposalRelaxationTask)).resolve(),
    }


def _verified_pamssw_source() -> dict[str, Any]:
    source_root = PAMSSW_SOURCE_ROOT.resolve()
    modules = _imported_pamssw_source_paths()
    symbols = _pamssw_symbol_source_paths()
    for name, path in {**modules, **symbols}.items():
        if not path.is_relative_to(source_root):
            raise RuntimeError(f"pamssw import path outside repo source tree: {name} -> {path}")
    measured = _pamssw_bundle_sha256(source_root)
    if measured != EXPECTED_PAMSSW_BUNDLE_SHA256:
        raise ValueError("pamssw source bundle SHA256 mismatch")
    return {
        "source_root": str(source_root),
        "bundle_sha256": measured,
        "imported_module_paths": {name: str(path) for name, path in modules.items()},
        "imported_symbol_paths": {name: str(path) for name, path in symbols.items()},
    }


def _current_commit() -> str:
    completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def _worktree_is_clean() -> bool:
    completed = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=REPO_ROOT, check=True, capture_output=True, text=True)
    return not completed.stdout.strip()


def _verified_git_provenance(expected_git_commit: str) -> dict[str, Any]:
    if not isinstance(expected_git_commit, str) or len(expected_git_commit) != 40 or any(c not in "0123456789abcdef" for c in expected_git_commit):
        raise ValueError("expected git commit must be a 40-character lowercase SHA-1")
    actual = _current_commit()
    if actual != expected_git_commit:
        raise RuntimeError(f"git commit mismatch: expected {expected_git_commit}, got {actual}")
    if not _worktree_is_clean():
        raise RuntimeError("git worktree is not clean")
    return {
        "expected_git_commit": expected_git_commit,
        "actual_git_commit": actual,
        "repo_root": str(REPO_ROOT),
        "worktree_clean": True,
    }


def _runtime_versions() -> dict[str, str]:
    distributions = {"numpy": "numpy", "scipy": "scipy", "ase": "ase", "torch": "torch", "mace": "mace-torch"}
    versions = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
    }
    for key, distribution in distributions.items():
        try:
            versions[key] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(f"required runtime distribution is missing: {distribution}") from error
    return _validated_runtime_versions(versions)


def _validated_runtime_versions(value: object) -> dict[str, str]:
    value = _exact_keys(value, RUNTIME_VERSION_KEYS, label="runtime versions")
    return {key: _require_string(value[key], label=f"runtime versions.{key}") for key in RUNTIME_VERSION_KEYS}


def _platform_provenance() -> dict[str, str]:
    return _validated_platform_provenance({
        "sys_platform": sys.platform,
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
    })


def _validated_platform_provenance(value: object) -> dict[str, str]:
    value = _exact_keys(value, PLATFORM_PROVENANCE_KEYS, label="platform provenance")
    return {key: _require_string(value[key], label=f"platform provenance.{key}") for key in PLATFORM_PROVENANCE_KEYS}


def _fixed_replay_source() -> Mapping[str, Any]:
    helpers = runpy.run_path(str(FIXED_REPLAY_DRIVER))
    factory = helpers.get("_source")
    if not callable(factory):
        raise RuntimeError("fixed replay driver does not expose _source()")
    source = factory()
    if not isinstance(source, Mapping) or not callable(source.get("_calculator")):
        raise RuntimeError("fixed replay source does not expose _calculator")
    return source


def _verified_file_provenance(path_value: object, declared_sha256: object, *, label: str) -> dict[str, str]:
    if not isinstance(path_value, (str, Path)):
        raise ValueError(f"{label} path is required")
    path = Path(path_value)
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    declared = _require_sha256(declared_sha256, label=f"{label} declared SHA256")
    measured = _sha256(path)
    if measured != declared:
        raise ValueError(f"{label} SHA256 mismatch")
    return {"path": str(path.resolve()), "declared_sha256": declared, "measured_sha256": measured}


def _verified_model_input_provenance(source: Mapping[str, Any]) -> dict[str, Any]:
    model = _verified_file_provenance(source.get("MODEL"), source.get("MODEL_SHA256"), label="model")
    systems = source.get("SYSTEMS")
    if not isinstance(systems, Mapping) or tuple(systems) != SYSTEMS:
        raise ValueError("source SYSTEMS must be ordered exactly as c60, pdo")
    inputs: dict[str, dict[str, str]] = {}
    for system in SYSTEMS:
        spec = systems[system]
        if not isinstance(spec, Mapping):
            raise ValueError(f"source system specification must be an object: {system}")
        inputs[system] = _verified_file_provenance(spec.get("input"), spec.get("sha256"), label=f"input {system}")
    return {"model": model, "inputs": inputs}


def _cuda_provenance(_: Mapping[str, Any]) -> dict[str, str]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch with CUDA support is required") from error
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False")
    runtime = torch.version.cuda
    if not isinstance(runtime, str) or not runtime:
        raise RuntimeError("torch CUDA runtime version is unavailable")
    return {
        "requested_device": "cuda",
        "cuda_device_name": str(torch.cuda.get_device_name(0)),
        "cuda_runtime_version": runtime,
    }


def _validated_cuda_model_input_provenance(value: object) -> dict[str, Any]:
    value = _exact_keys(value, ("requested_device", "cuda_device_name", "cuda_runtime_version", "model", "inputs"), label="cuda/model/input provenance")
    if value["requested_device"] != "cuda":
        raise RuntimeError("CUDA provenance must confirm requested_device='cuda'")
    result = {
        "requested_device": "cuda",
        "cuda_device_name": _require_string(value["cuda_device_name"], label="cuda device name"),
        "cuda_runtime_version": _require_string(value["cuda_runtime_version"], label="cuda runtime version"),
    }
    for key in ("model",):
        item = _exact_keys(value[key], ("path", "declared_sha256", "measured_sha256"), label=key)
        result[key] = {
            "path": _require_string(item["path"], label=f"{key}.path"),
            "declared_sha256": _require_sha256(item["declared_sha256"], label=f"{key}.declared_sha256"),
            "measured_sha256": _require_sha256(item["measured_sha256"], label=f"{key}.measured_sha256"),
        }
        if not Path(result[key]["path"]).is_absolute():
            raise ValueError(f"{key} path must be absolute")
        if result[key]["declared_sha256"] != result[key]["measured_sha256"]:
            raise ValueError(f"{key} SHA256 mismatch")
    inputs = _exact_keys(value["inputs"], SYSTEMS, label="input provenance")
    result["inputs"] = {}
    for system in SYSTEMS:
        item = _exact_keys(inputs[system], ("path", "declared_sha256", "measured_sha256"), label=f"input {system}")
        result["inputs"][system] = {
            "path": _require_string(item["path"], label=f"input {system}.path"),
            "declared_sha256": _require_sha256(item["declared_sha256"], label=f"input {system}.declared_sha256"),
            "measured_sha256": _require_sha256(item["measured_sha256"], label=f"input {system}.measured_sha256"),
        }
        if not Path(result["inputs"][system]["path"]).is_absolute():
            raise ValueError(f"input {system} path must be absolute")
        if result["inputs"][system]["declared_sha256"] != result["inputs"][system]["measured_sha256"]:
            raise ValueError(f"input {system} SHA256 mismatch")
    return result


def _check_execution_args(output_dir: Path, expected_git_commit: str) -> None:
    if not isinstance(output_dir, Path):
        raise ValueError("output directory must be a Path")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not isinstance(expected_git_commit, str) or len(expected_git_commit) != 40:
        raise ValueError("expected git commit must be a 40-character SHA-1")


def preflight(
    *,
    source_summary_path: Path,
    output_dir: Path,
    expected_git_commit: str,
    source_loader: Callable[[], Mapping[str, Any]] = _fixed_replay_source,
    runtime_probe: Callable[[], Mapping[str, Any]] = _runtime_versions,
    platform_probe: Callable[[], Mapping[str, Any]] = _platform_provenance,
    cuda_probe: Callable[[Mapping[str, Any]], Mapping[str, Any]] = _cuda_provenance,
) -> Preflight:
    """Run every non-calculator gate in frozen contract order."""

    _check_execution_args(output_dir, expected_git_commit)
    tasks_by_system, source_summary_sha256 = load_fixed_tasks(source_summary_path)
    helper_provenance = _verified_helper_provenance()
    pamssw_source_provenance = _verified_pamssw_source()
    safe_descriptor, safe_descriptor_sha256 = _verified_safe_kernel_descriptor()
    git_provenance = _verified_git_provenance(expected_git_commit)
    runtime_versions = _validated_runtime_versions(runtime_probe())
    platform_provenance = _validated_platform_provenance(platform_probe())
    source = source_loader()
    if not isinstance(source, Mapping) or not callable(source.get("_calculator")):
        raise RuntimeError("fixed replay source does not expose _calculator")
    model_input = _verified_model_input_provenance(source)
    cuda = cuda_probe(source)
    cuda_model_input_provenance = _validated_cuda_model_input_provenance({**dict(cuda), **model_input})
    return Preflight(
        tasks_by_system=tasks_by_system,
        source_summary_sha256=source_summary_sha256,
        source=source,
        safe_kernel_descriptor=safe_descriptor,
        safe_kernel_descriptor_sha256=safe_descriptor_sha256,
        helper_provenance=helper_provenance,
        pamssw_source_provenance=pamssw_source_provenance,
        git_provenance=git_provenance,
        runtime_versions=runtime_versions,
        platform_provenance=platform_provenance,
        cuda_model_input_provenance=cuda_model_input_provenance,
    )


def _trace_recorder_module():
    module = sys.modules.get(TRACE_MODULE_NAME)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(TRACE_MODULE_NAME, TRACE_RECORDER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load trace recorder: {TRACE_RECORDER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[TRACE_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def resolve_arm(arm: Arm) -> dict[str, Any]:
    if arm not in ARMS or arm.kernel != "safe-lbfgs-total" or arm.history_limit not in {1, 10}:
        raise ValueError("unreviewed safe L-BFGS history-depth arm")
    if arm.scale_policy not in EXPECTED_SAFE_KERNEL_DESCRIPTOR["scale_policies"] or arm.secant_policy != "total-biased-gradient":
        raise ValueError("unreviewed adaptive scale or secant policy")
    if arm.adaptive_scale_without_history is not False:
        raise ValueError("history-depth arms must not enable adaptive scale without history")
    return {
        "history_limit": arm.history_limit,
        "scale_policy": arm.scale_policy,
        "secant_policy": arm.secant_policy,
        "adaptive_scale_without_history": False,
    }


def replay_task_with_trace(task: ProposalRelaxationTask, calculator, *, arm: Arm) -> ReplayResult:
    """Run one frozen arm while recording only its already-required PES calls."""

    resolved_arm = resolve_arm(arm)
    recorder = _trace_recorder_module()
    counter = EvalCounter(calculator)
    proposal = recorder.RecordingProposalPotential(counter, biases=list(task.biases), softening=task.softening)
    relaxer = Relaxer(proposal.evaluate, optimizer="safe-lbfgs-total", component_evaluator=proposal.evaluate_parts)
    callback_hashes: list[str] = []

    def record_accepted_state(state: State) -> None:
        callback_hashes.append(recorder.position_hash(state))

    started = perf_counter()
    with counter.purpose(EvaluationPurpose.BIASED_PROPOSAL_RELAX):
        result = relaxer.relax(
            task.initial_state,
            fmax=task.fmax,
            maxiter=MAXITER,
            coordinate_trust_radius=task.coordinate_trust_radius,
            _safe_lbfgs_history_limit=arm.history_limit,
            _safe_lbfgs_adaptive_scale_without_history=False,
            trajectory_callback=record_accepted_state,
        )
    trace_records = recorder.mark_accepted_state_evaluations(proposal.records, callback_hashes)
    replay = ReplayResult(
        result=result,
        evaluation_counts=counter.snapshot(),
        wall_time_s=perf_counter() - started,
        trace_records=trace_records,
        accepted_callback_hashes=tuple(callback_hashes),
        certificate_satisfied=bool(np.isfinite(result.energy) and np.isfinite(result.gradient_norm) and np.all(np.isfinite(result.state.positions)) and result.gradient_norm <= task.fmax),
        resolved_arm=resolved_arm,
    )
    _validate_replay(replay)
    return replay


def _trace_values_are_finite(records: Sequence[Mapping[str, Any]]) -> bool:
    try:
        for index, record in enumerate(records, start=1):
            if type(record) is not dict or set(record) != set(TRACE_KEYS) or type(record["evaluation_index"]) is not int or record["evaluation_index"] != index:
                return False
            if not _require_sha256(record["positions_sha256"], label="trace positions SHA256"):
                return False
            if type(record["accepted_state"]) is not bool:
                return False
            for key in TRACE_KEYS[2:-1]:
                if type(record[key]) is not float or not np.isfinite(record[key]):
                    return False
        return True
    except (TypeError, ValueError):
        return False


def _validate_replay(replay: ReplayResult) -> None:
    counts = replay.evaluation_counts
    purposes = counts.as_dict()
    if not _trace_values_are_finite(replay.trace_records):
        raise RuntimeError("non-finite or malformed zero-extra-call trace")
    if not np.isfinite(replay.wall_time_s) or replay.wall_time_s < 0.0:
        raise RuntimeError("non-finite wall time")
    if not np.isfinite(replay.result.energy) or not np.isfinite(replay.result.gradient_norm) or not np.all(np.isfinite(replay.result.state.positions)):
        raise RuntimeError("non-finite replay endpoint")
    if len(replay.trace_records) != counts.total or counts.total != replay.result.telemetry.backend_evaluations or purposes["biased_proposal_relax"] != counts.total or purposes["unattributed"] != 0:
        raise RuntimeError("ledger mismatch")
    if any(value != 0 for key, value in purposes.items() if key != "biased_proposal_relax"):
        raise RuntimeError("ledger has non-relax evaluation purposes")
    if replay.result.telemetry.termination_reason not in {"converged", "maxiter", "line_search_failed"}:
        raise RuntimeError(f"unexpected termination reason: {replay.result.telemetry.termination_reason}")


def _expected_row_keys() -> list[dict[str, str]]:
    return [
        {"system": system, "task_id": f"{system}-seed-{seed}-bias-1", "arm_id": arm.arm_id}
        for system in SYSTEMS
        for arm in ARMS
        for seed in SEEDS
    ]


def _endpoint_positions_sha256(positions: Sequence[Sequence[float]]) -> str:
    return sha256(_canonical_json(positions, label="endpoint positions").encode("utf-8")).hexdigest()


def _row_payload(frozen: FrozenTask, arm: Arm, replay: ReplayResult) -> dict[str, Any]:
    result = replay.result
    resolved = dict(replay.resolved_arm)
    return {
        "system": frozen.system,
        "task_id": frozen.task_id,
        "seed": frozen.seed,
        "task_payload": dict(frozen.task_payload),
        "task_sha256": frozen.task_sha256,
        "source_task_maxiter": int(frozen.task_payload["maxiter"]),
        "replay_maxiter": MAXITER,
        "arm_id": arm.arm_id,
        "arm": {
            "requested_history_limit": arm.history_limit,
            "resolved_history_limit": resolved["history_limit"],
            "requested_scale_policy": arm.scale_policy,
            "resolved_scale_policy": resolved["scale_policy"],
            "requested_secant_policy": arm.secant_policy,
            "resolved_secant_policy": resolved["secant_policy"],
            "adaptive_scale_without_history": resolved["adaptive_scale_without_history"],
        },
        "result": {
            "biased_energy_eV": float(result.energy),
            "max_active_atom_force_eV_per_A": float(result.gradient_norm),
            "iterations": int(result.n_iter),
        },
        "certificate": {"satisfied": replay.certificate_satisfied, "fmax_eV_per_A": float(frozen.relax_task.fmax)},
        "termination": {"reason": result.telemetry.termination_reason, "converged": bool(result.telemetry.converged)},
        "force_evaluations": replay.evaluation_counts.total,
        "purpose_counts": replay.evaluation_counts.as_dict(),
        "unattributed_calls": replay.evaluation_counts.as_dict()["unattributed"],
        "zero_extra_call_trace": replay.trace_records,
        "accepted_callback_hashes": list(replay.accepted_callback_hashes),
        "telemetry": asdict(result.telemetry),
        "wall_time_s": replay.wall_time_s,
        "endpoint": {
            "positions": result.state.positions.tolist(),
            "positions_sha256": _endpoint_positions_sha256(result.state.positions.tolist()),
        },
    }


def _validate_row(row: object) -> None:
    row = _exact_keys(row, ROW_KEYS, label="row")
    if type(row["seed"]) is not int or row["seed"] not in SEEDS or row["system"] not in SYSTEMS or row["task_id"] != f"{row['system']}-seed-{row['seed']}-bias-1":
        raise ValueError("row task identity schema mismatch")
    if type(row["task_payload"]) is not dict or row["replay_maxiter"] != MAXITER or type(row["source_task_maxiter"]) is not int:
        raise ValueError("row maxiter schema mismatch")
    if canonical_task_sha256(row["task_payload"]) != row["task_sha256"] or row["task_sha256"] != EXPECTED_TASK_SHA256[row["task_id"]]:
        raise ValueError("row task SHA256 mismatch")
    arm = _exact_keys(row["arm"], ("requested_history_limit", "resolved_history_limit", "requested_scale_policy", "resolved_scale_policy", "requested_secant_policy", "resolved_secant_policy", "adaptive_scale_without_history"), label="row arm")
    matched = next((candidate for candidate in ARMS if candidate.arm_id == row["arm_id"]), None)
    if matched is None or arm["requested_history_limit"] != matched.history_limit or arm["resolved_history_limit"] != matched.history_limit or arm["requested_scale_policy"] != matched.scale_policy or arm["resolved_scale_policy"] != matched.scale_policy or arm["requested_secant_policy"] != "total-biased-gradient" or arm["resolved_secant_policy"] != "total-biased-gradient" or arm["adaptive_scale_without_history"] is not False:
        raise ValueError("row arm schema mismatch")
    result = _exact_keys(row["result"], ("biased_energy_eV", "max_active_atom_force_eV_per_A", "iterations"), label="row result")
    certificate = _exact_keys(row["certificate"], ("satisfied", "fmax_eV_per_A"), label="row certificate")
    termination = _exact_keys(row["termination"], ("reason", "converged"), label="row termination")
    if type(certificate["satisfied"]) is not bool or type(termination["converged"]) is not bool or termination["reason"] not in {"converged", "maxiter", "line_search_failed"} or type(result["iterations"]) is not int or type(result["biased_energy_eV"]) is not float or type(result["max_active_atom_force_eV_per_A"]) is not float or type(certificate["fmax_eV_per_A"]) is not float:
        raise ValueError("row result/certificate/termination schema mismatch")
    purposes = _exact_keys(row["purpose_counts"], PURPOSE_KEYS, label="row purpose counts")
    if type(row["force_evaluations"]) is not int or row["force_evaluations"] < 0 or type(row["unattributed_calls"]) is not int:
        raise ValueError("row evaluation count schema mismatch")
    if any(type(value) is not int or value < 0 for value in purposes.values()) or row["unattributed_calls"] != purposes["unattributed"]:
        raise ValueError("row purpose count schema mismatch")
    telemetry = _exact_keys(row["telemetry"], TELEMETRY_KEYS, label="row telemetry")
    counter_fields = tuple(key for key in TELEMETRY_KEYS if key.endswith("calls") or key.endswith("evaluations") or key.endswith("steps") or key.endswith("secants") or key == "mic_branch_resets")
    if not isinstance(telemetry["backend"], str) or not isinstance(telemetry["gradient_measure"], str) or type(telemetry["converged"]) is not bool or telemetry["optimizer_success"] not in {True, False, None} or any(type(telemetry[key]) is not int or telemetry[key] < 0 for key in counter_fields) or type(telemetry["bias_secant_curvature_sum"]) is not float:
        raise ValueError("row telemetry schema mismatch")
    trace = row["zero_extra_call_trace"]
    if not isinstance(trace, list) or not _trace_values_are_finite(trace) or len(trace) != row["force_evaluations"] or telemetry["backend_evaluations"] != row["force_evaluations"] or purposes["biased_proposal_relax"] != row["force_evaluations"] or purposes["unattributed"] != 0:
        raise ValueError("row ledger schema mismatch")
    if not isinstance(row["accepted_callback_hashes"], list) or any(not _require_sha256(value, label="accepted callback SHA256") for value in row["accepted_callback_hashes"]) or type(row["wall_time_s"]) is not float or row["wall_time_s"] < 0.0:
        raise ValueError("row telemetry schema mismatch")
    endpoint = _exact_keys(row["endpoint"], ("positions", "positions_sha256"), label="row endpoint")
    if _endpoint_positions_sha256(endpoint["positions"]) != endpoint["positions_sha256"]:
        raise ValueError("row endpoint position SHA256 mismatch")
    _require_finite_json(row, label="row")


def _validate_full_ledger_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    expected = {(item["system"], item["task_id"], item["arm_id"]) for item in _expected_row_keys()}
    if len(rows) != 32:
        raise ValueError("32-row ledger schema mismatch")
    observed: set[tuple[str, str, str]] = set()
    for row in rows:
        _validate_row(row)
        observed.add((row["system"], row["task_id"], row["arm_id"]))
    if len(observed) != 32 or observed != expected:
        raise ValueError("32-row ledger task matrix mismatch")


def _validate_helper_provenance(value: object) -> None:
    value = _exact_keys(value, ("fixed_replay_driver", "g1_driver", "trace_recorder"), label="runner helper provenance")
    for key, entry in value.items():
        entry = _exact_keys(entry, ("path", "sha256"), label=f"runner helper provenance.{key}")
        if not Path(_require_string(entry["path"], label=f"helper {key} path")).is_absolute():
            raise ValueError("helper provenance paths must be absolute")
        _require_sha256(entry["sha256"], label=f"helper {key} SHA256")


def _validate_summary(summary: object) -> None:
    summary = _exact_keys(summary, SUMMARY_KEYS, label="summary")
    if summary["schema_version"] != 1 or not isinstance(summary["claim_ceiling"], str) or not summary["claim_ceiling"] or summary["systems"] != list(SYSTEMS) or summary["task_count"] != 16 or summary["row_count"] != 32 or type(summary["certificate_all_satisfied"]) is not bool or type(summary["certificate_unsatisfied_count"]) is not int or not 0 <= summary["certificate_unsatisfied_count"] <= 32 or type(summary["wall_time_total_s"]) is not float or summary["wall_time_total_s"] < 0.0:
        raise ValueError("summary schema mismatch")
    if _require_sha256(summary["source_summary_sha256"], label="source summary SHA256") != EXPECTED_SOURCE_SUMMARY_SHA256:
        raise ValueError("source summary SHA256 mismatch")
    if summary["safe_kernel_descriptor"] != EXPECTED_SAFE_KERNEL_DESCRIPTOR or _require_sha256(summary["safe_kernel_descriptor_sha256"], label="safe kernel descriptor SHA256") != EXPECTED_SAFE_KERNEL_DESCRIPTOR_SHA256 or canonical_descriptor_sha256(summary["safe_kernel_descriptor"]) != summary["safe_kernel_descriptor_sha256"]:
        raise ValueError("safe kernel descriptor schema mismatch")
    if summary["arms"] != [asdict(arm) for arm in ARMS]:
        raise ValueError("summary arms schema mismatch")
    terminations = summary["termination_reason_counts"]
    if type(terminations) is not dict or not terminations or any(reason not in {"converged", "maxiter", "line_search_failed"} or type(count) is not int or count < 0 for reason, count in terminations.items()) or sum(terminations.values()) != 32:
        raise ValueError("summary termination counts schema mismatch")
    _validate_helper_provenance(summary["runner_helper_provenance"])
    pamssw = _exact_keys(summary["pamssw_source_provenance"], ("source_root", "bundle_sha256", "imported_module_paths", "imported_symbol_paths"), label="pamssw source provenance")
    if not Path(_require_string(pamssw["source_root"], label="pamssw source root")).is_absolute():
        raise ValueError("pamssw source root must be absolute")
    _require_sha256(pamssw["bundle_sha256"], label="pamssw source bundle SHA256")
    for key in ("imported_module_paths", "imported_symbol_paths"):
        if type(pamssw[key]) is not dict or not all(isinstance(name, str) and isinstance(path, str) and Path(path).is_absolute() for name, path in pamssw[key].items()):
            raise ValueError(f"pamssw {key} schema mismatch")
    git = _exact_keys(summary["git_provenance"], ("expected_git_commit", "actual_git_commit", "repo_root", "worktree_clean"), label="git provenance")
    if _require_git_sha1(git["expected_git_commit"], label="expected git commit") != _require_git_sha1(git["actual_git_commit"], label="actual git commit") or not Path(_require_string(git["repo_root"], label="repo root")).is_absolute() or git["worktree_clean"] is not True:
        raise ValueError("git provenance schema mismatch")
    _validated_runtime_versions(summary["runtime_versions"])
    _validated_platform_provenance(summary["platform_provenance"])
    _validated_cuda_model_input_provenance(summary["cuda_model_input_provenance"])
    _require_finite_json(summary, label="summary")


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def publish_ledger_atomically(output_dir: Path, rows: Sequence[Mapping[str, Any]], summary_payload: Mapping[str, Any], *, write_json: Callable[[Path, Any], None] = _write_json_atomic) -> None:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    _validate_full_ledger_rows(rows)
    _validate_summary(summary_payload)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    try:
        for system in SYSTEMS:
            system_rows = [dict(row) for row in rows if row["system"] == system]
            write_json(staging_dir / f"{system}.json", {"system": system, "task_ids": [f"{system}-seed-{seed}-bias-1" for seed in SEEDS], "rows": system_rows})
        write_json(staging_dir / "summary.json", summary_payload)
        if output_dir.exists():
            raise FileExistsError(output_dir)
        staging_dir.rename(output_dir)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)


def run(*, source_summary_path: Path = SOURCE_SUMMARY_PATH, output_dir: Path = OUTPUT_DIR, expected_git_commit: str, preflight_only: bool = False) -> dict[str, Any]:
    """Execute every frozen row once, or stop after provenance preflight."""

    checked = preflight(source_summary_path=source_summary_path, output_dir=output_dir, expected_git_commit=expected_git_commit)
    if preflight_only:
        return {"preflight_only": True, "git_provenance": dict(checked.git_provenance)}
    started = perf_counter()
    calculator_factory = checked.source["_calculator"]
    rows: list[dict[str, Any]] = []
    for system in SYSTEMS:
        for arm in ARMS:
            calculator = calculator_factory()
            for frozen in checked.tasks_by_system[system]:
                replay = replay_task_with_trace(frozen.relax_task, calculator, arm=arm)
                _validate_replay(replay)
                rows.append(_row_payload(frozen, arm, replay))
    _validate_full_ledger_rows(rows)
    termination_counts: dict[str, int] = {}
    for row in rows:
        reason = row["termination"]["reason"]
        termination_counts[reason] = termination_counts.get(reason, 0) + 1
    unsatisfied = sum(not row["certificate"]["satisfied"] for row in rows)
    summary = {
        "schema_version": 1,
        "claim_ceiling": "Fixed-task C60/PdO CUDA replay evidence only; no endpoint-equivalence, SSW-performance, statistical, or default-change claim.",
        "source_summary_sha256": checked.source_summary_sha256,
        "systems": list(SYSTEMS),
        "task_count": 16,
        "row_count": 32,
        "certificate_all_satisfied": unsatisfied == 0,
        "certificate_unsatisfied_count": unsatisfied,
        "termination_reason_counts": dict(sorted(termination_counts.items())),
        "arms": [asdict(arm) for arm in ARMS],
        "safe_kernel_descriptor": dict(checked.safe_kernel_descriptor),
        "safe_kernel_descriptor_sha256": checked.safe_kernel_descriptor_sha256,
        "runner_helper_provenance": dict(checked.helper_provenance),
        "pamssw_source_provenance": dict(checked.pamssw_source_provenance),
        "git_provenance": dict(checked.git_provenance),
        "runtime_versions": dict(checked.runtime_versions),
        "platform_provenance": dict(checked.platform_provenance),
        "cuda_model_input_provenance": dict(checked.cuda_model_input_provenance),
        "wall_time_total_s": perf_counter() - started,
    }
    publish_ledger_atomically(output_dir, rows, summary)
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-summary", type=Path, default=SOURCE_SUMMARY_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    run(source_summary_path=args.source_summary, output_dir=args.output_dir, expected_git_commit=args.expected_git_commit, preflight_only=args.preflight_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
