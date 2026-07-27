#!/usr/bin/env python3
"""Run the frozen two-arm safe L-BFGS history-depth GPU ablation."""

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
import runpy
import subprocess
import sys
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
TRACE_RECORDER_PATH = (
    RUN_ROOT.parent / "20260727-proposal-energy-traces" / "trace_recorder.py"
)

SYSTEMS = ("c60", "pdo")
SEEDS = tuple(range(42, 50))
MAXITER = 400
EXPECTED_SOURCE_SUMMARY_SHA256 = (
    "62cc771e2aa24e9addef0e870d0524f901f02bddc34152cf2a2eeea91a671b04"
)
EXPECTED_PAMSSW_BUNDLE_SHA256 = (
    "83a2b845c4bbe0235e8c584a579e1b9f1e89690084c8abdd9ef3aa20c2a09b50"
)
EXPECTED_FIXED_REPLAY_DRIVER_SHA256 = (
    "f9c9602e42985891a6ca2a84ca70dda69c52b9d794a6ea6c76857c398345fa8f"
)
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
RUNTIME_PACKAGES = {
    "numpy": "numpy",
    "scipy": "scipy",
    "ase": "ase",
    "torch": "torch",
    "mace": "mace-torch",
}
TRACE_MODULE_NAME = "_history_depth_trace_recorder"


@dataclass(frozen=True)
class Arm:
    arm_id: str
    history_limit: int


ARMS = (
    Arm("adaptive-scale-history1", 1),
    Arm("adaptive-scale-history10", 10),
)


@dataclass(frozen=True)
class FrozenTask:
    system: str
    seed: int
    task_id: str
    task_sha256: str
    relax_task: ProposalRelaxationTask


class ReplayResult(NamedTuple):
    result: RelaxResult
    evaluation_counts: EvaluationCounts
    wall_time_s: float
    trace_records: list[dict[str, Any]]


@dataclass(frozen=True)
class Preflight:
    tasks_by_system: Mapping[str, tuple[FrozenTask, ...]]
    source: Mapping[str, Any]
    metadata: Mapping[str, Any]


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_task_sha256(payload: Mapping[str, Any]) -> str:
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("task payload is not finite canonical JSON") from error
    return sha256(encoded).hexdigest()


def _load_fixed_tasks(
    path: Path,
) -> tuple[dict[str, tuple[FrozenTask, ...]], str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    source_sha256 = _sha256(path)
    if source_sha256 != EXPECTED_SOURCE_SUMMARY_SHA256:
        raise ValueError("source summary SHA256 mismatch")
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("source summary is not valid JSON") from error
    entries = summary.get("systems")
    if not isinstance(entries, list) or [
        entry.get("system") for entry in entries if isinstance(entry, dict)
    ] != list(SYSTEMS):
        raise ValueError("source summary must contain c60 then pdo")

    tasks_by_system: dict[str, tuple[FrozenTask, ...]] = {}
    for system, entry in zip(SYSTEMS, entries, strict=True):
        items = entry.get("tasks")
        if not isinstance(items, list) or len(items) != len(SEEDS):
            raise ValueError(f"{system} must contain eight frozen tasks")
        tasks: list[FrozenTask] = []
        for item, seed in zip(items, SEEDS, strict=True):
            task_id = f"{system}-seed-{seed}-bias-1"
            if item.get("task_id") != task_id or item.get("seed") != seed:
                raise ValueError(f"frozen task identity mismatch: {task_id}")
            payload = item.get("task")
            if not isinstance(payload, dict):
                raise ValueError(f"missing task payload: {task_id}")
            task_sha256 = canonical_task_sha256(payload)
            if task_sha256 != EXPECTED_TASK_SHA256[task_id]:
                raise ValueError(f"canonical task SHA256 mismatch: {task_id}")
            try:
                relax_task = proposal_task_from_payload(payload)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid task payload: {task_id}") from error
            if len(relax_task.biases) != 1:
                raise ValueError(f"frozen task must have one bias: {task_id}")
            tasks.append(
                FrozenTask(
                    system=system,
                    seed=seed,
                    task_id=task_id,
                    task_sha256=task_sha256,
                    relax_task=replace(relax_task, maxiter=MAXITER),
                )
            )
        tasks_by_system[system] = tuple(tasks)
    return tasks_by_system, source_sha256


def _pamssw_bundle_sha256(source_root: Path) -> str:
    paths = sorted(
        path
        for path in source_root.rglob("*.py")
        if path.is_file() and "__pycache__" not in path.parts
    )
    if not paths:
        raise ValueError("pamssw source bundle is empty")
    digest = sha256()
    for path in paths:
        digest.update(path.relative_to(source_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
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


def _fixed_replay_source() -> Mapping[str, Any]:
    namespace = runpy.run_path(str(FIXED_REPLAY_DRIVER))
    source_loader = namespace.get("_source")
    if not callable(source_loader):
        raise RuntimeError("fixed replay helper does not expose _source")
    source = source_loader()
    if not isinstance(source, Mapping):
        raise RuntimeError("fixed replay source must be a mapping")
    return source


def _runtime_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for key, distribution in RUNTIME_PACKAGES.items():
        try:
            versions[key] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(
                f"required runtime distribution is missing: {distribution}"
            ) from error
    return versions


def _cuda_info(_: Mapping[str, Any]) -> dict[str, Any]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required") from error
    available = bool(torch.cuda.is_available())
    if not available:
        raise RuntimeError("CUDA is unavailable")
    runtime_version = torch.version.cuda
    if not isinstance(runtime_version, str) or not runtime_version:
        raise RuntimeError("CUDA runtime version is unavailable")
    return {
        "requested_device": "cuda",
        "available": True,
        "runtime_version": runtime_version,
        "device_name": str(torch.cuda.get_device_name(0)),
    }


def _verified_file(path_value: object, declared_hash: object, label: str) -> str:
    if not isinstance(path_value, (str, Path)):
        raise ValueError(f"{label} path is missing")
    path = Path(path_value)
    if not path.is_file():
        raise FileNotFoundError(path)
    if not isinstance(declared_hash, str):
        raise ValueError(f"{label} declared SHA256 is missing")
    measured = _sha256(path)
    if measured != declared_hash:
        raise ValueError(f"{label} SHA256 mismatch")
    return measured


def preflight(
    *,
    source_summary_path: Path,
    expected_git_commit: str,
    source_loader: Callable[[], Mapping[str, Any]] = _fixed_replay_source,
    runtime_probe: Callable[[], Mapping[str, Any]] = _runtime_versions,
    cuda_probe: Callable[[Mapping[str, Any]], Mapping[str, Any]] = _cuda_info,
) -> Preflight:
    actual_commit = _current_commit()
    if expected_git_commit != actual_commit:
        raise RuntimeError(
            f"execution commit mismatch: expected {expected_git_commit}, "
            f"got {actual_commit}"
        )
    if not _tracked_worktree_clean():
        raise RuntimeError("tracked worktree is not clean")

    tasks_by_system, source_summary_sha256 = _load_fixed_tasks(
        source_summary_path
    )
    pamssw_bundle_sha256 = _pamssw_bundle_sha256(PAMSSW_SOURCE_ROOT)
    if pamssw_bundle_sha256 != EXPECTED_PAMSSW_BUNDLE_SHA256:
        raise ValueError("pamssw source bundle SHA256 mismatch")
    helper_sha256 = _sha256(FIXED_REPLAY_DRIVER)
    if helper_sha256 != EXPECTED_FIXED_REPLAY_DRIVER_SHA256:
        raise ValueError("fixed replay helper SHA256 mismatch")

    source = source_loader()
    if not isinstance(source, Mapping) or not callable(
        source.get("_calculator")
    ):
        raise RuntimeError("fixed replay source lacks calculator factory")
    model_sha256 = _verified_file(
        source.get("MODEL"), source.get("MODEL_SHA256"), "model"
    )
    source_systems = source.get("SYSTEMS")
    if not isinstance(source_systems, Mapping):
        raise ValueError("source SYSTEMS is missing")
    input_sha256 = {
        system: _verified_file(
            source_systems[system].get("input"),
            source_systems[system].get("sha256"),
            f"{system} input",
        )
        for system in SYSTEMS
    }

    raw_versions = runtime_probe()
    runtime_versions = {
        key: str(raw_versions[key])
        for key in ("python", "numpy", "scipy", "ase", "torch", "mace")
    }
    raw_cuda = cuda_probe(source)
    if (
        raw_cuda.get("requested_device") != "cuda"
        or raw_cuda.get("available") is not True
        or not raw_cuda.get("runtime_version")
        or not raw_cuda.get("device_name")
    ):
        raise RuntimeError("CUDA provenance is incomplete")
    cuda = {
        key: raw_cuda[key]
        for key in (
            "requested_device",
            "available",
            "runtime_version",
            "device_name",
        )
    }
    metadata = {
        "schema_version": 1,
        "execution_commit": actual_commit,
        "source_summary_sha256": source_summary_sha256,
        "pamssw_bundle_sha256": pamssw_bundle_sha256,
        "model_sha256": model_sha256,
        "input_sha256": input_sha256,
        "helper_sha256": helper_sha256,
        "runtime_versions": runtime_versions,
        "cuda": cuda,
        "systems": list(SYSTEMS),
        "task_count": len(SYSTEMS) * len(SEEDS),
        "row_count": len(SYSTEMS) * len(SEEDS) * len(ARMS),
        "arms": [asdict(arm) for arm in ARMS],
    }
    return Preflight(
        tasks_by_system=tasks_by_system,
        source=source,
        metadata=metadata,
    )


def _trace_recorder_module():
    module = sys.modules.get(TRACE_MODULE_NAME)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(
        TRACE_MODULE_NAME, TRACE_RECORDER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load trace recorder: {TRACE_RECORDER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[TRACE_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def replay_task_with_trace(
    task: ProposalRelaxationTask,
    calculator,
    *,
    arm: Arm,
) -> ReplayResult:
    recorder = _trace_recorder_module()
    counter = EvalCounter(calculator)
    proposal = recorder.RecordingProposalPotential(
        counter,
        biases=list(task.biases),
        softening=task.softening,
    )
    relaxer = Relaxer(
        proposal.evaluate,
        optimizer="safe-lbfgs-total",
        component_evaluator=proposal.evaluate_parts,
    )
    accepted_hashes: list[str] = []

    def record_accepted_state(current: State) -> None:
        accepted_hashes.append(recorder.position_hash(current))

    started = perf_counter()
    with counter.purpose(EvaluationPurpose.BIASED_PROPOSAL_RELAX):
        result = relaxer.relax(
            task.initial_state,
            fmax=task.fmax,
            maxiter=MAXITER,
            coordinate_trust_radius=task.coordinate_trust_radius,
            trajectory_callback=record_accepted_state,
            _safe_lbfgs_history_limit=arm.history_limit,
        )
    trace_records = recorder.mark_accepted_state_evaluations(
        proposal.records,
        accepted_hashes,
    )
    return ReplayResult(
        result=result,
        evaluation_counts=counter.snapshot(),
        wall_time_s=perf_counter() - started,
        trace_records=trace_records,
    )


def _row_payload(
    frozen: FrozenTask,
    arm: Arm,
    replay: ReplayResult,
) -> dict[str, Any]:
    result = replay.result
    recorder = _trace_recorder_module()
    return {
        "system": frozen.system,
        "seed": frozen.seed,
        "task_id": frozen.task_id,
        "task_sha256": frozen.task_sha256,
        "arm_id": arm.arm_id,
        "history_limit": arm.history_limit,
        "fmax_eV_per_A": float(frozen.relax_task.fmax),
        "final_total_biased_energy_eV": float(result.energy),
        "final_active_max_force_eV_per_A": float(result.gradient_norm),
        "iterations": int(result.n_iter),
        "termination_reason": result.telemetry.termination_reason,
        "displacement_rms_A": float(result.displacement_rms),
        "displacement_max_A": float(result.displacement_max),
        "outcome_class": result.outcome_class.value,
        "force_evaluations": replay.evaluation_counts.total,
        "purpose_counts": replay.evaluation_counts.as_dict(),
        "telemetry": asdict(result.telemetry),
        "trace": replay.trace_records,
        "final_positions": result.state.positions.tolist(),
        "final_positions_sha256": recorder.position_hash(result.state),
        "wall_time_s": float(replay.wall_time_s),
    }


def _finite_numbers(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(_finite_numbers(item) for item in value.values())
    if isinstance(value, Sequence):
        return all(_finite_numbers(item) for item in value)
    return False


def _expected_row_keys() -> list[tuple[str, int, str]]:
    return [
        (system, seed, arm.arm_id)
        for system in SYSTEMS
        for arm in ARMS
        for seed in SEEDS
    ]


def _validate_row(row: Mapping[str, Any]) -> None:
    key = (row.get("system"), row.get("seed"), row.get("arm_id"))
    if key not in set(_expected_row_keys()):
        raise ValueError("row is outside the frozen task-arm matrix")
    task_id = f"{key[0]}-seed-{key[1]}-bias-1"
    if (
        row.get("task_id") != task_id
        or row.get("task_sha256") != EXPECTED_TASK_SHA256[task_id]
    ):
        raise ValueError("row task identity is not frozen")
    arm = next(item for item in ARMS if item.arm_id == key[2])
    if row.get("history_limit") != arm.history_limit:
        raise ValueError("row history limit does not match its arm")

    trace = row.get("trace")
    purpose_counts = row.get("purpose_counts")
    telemetry = row.get("telemetry")
    if (
        not isinstance(trace, list)
        or not isinstance(purpose_counts, Mapping)
        or not isinstance(telemetry, Mapping)
    ):
        raise ValueError("row accounting fields are missing")
    count = row.get("force_evaluations")
    if not (
        len(trace)
        == count
        == telemetry.get("evaluator_calls")
        == purpose_counts.get("biased_proposal_relax")
    ):
        raise ValueError("row evaluator accounting is open")
    if purpose_counts.get("unattributed") != 0:
        raise ValueError("row has unattributed evaluator calls")
    if not _finite_numbers(row):
        raise ValueError("row contains non-finite numeric values")

    positions = row.get("final_positions")
    position_hash = row.get("final_positions_sha256")
    recorder = _trace_recorder_module()
    if (
        not isinstance(positions, list)
        or not isinstance(position_hash, str)
        or recorder.position_hash(np.asarray(positions, dtype=float))
        != position_hash
    ):
        raise ValueError("row final positions or hash are invalid")


def _validate_complete_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    for row in rows:
        _validate_row(row)
    observed = {
        (row["system"], row["seed"], row["arm_id"])
        for row in rows
    }
    expected = set(_expected_row_keys())
    if len(rows) != len(expected) or observed != expected:
        raise ValueError("ledger does not contain the complete 32-row matrix")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _partial_path(output_dir: Path) -> Path:
    return output_dir.with_name(output_dir.name + ".partial")


def _refuse_existing_output(output_dir: Path) -> None:
    partial_dir = _partial_path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if partial_dir.exists():
        raise FileExistsError(partial_dir)


def _write_output(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> None:
    _refuse_existing_output(output_dir)
    _validate_complete_rows(rows)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    partial_dir = _partial_path(output_dir)
    partial_dir.mkdir()
    for system in SYSTEMS:
        system_rows = [dict(row) for row in rows if row["system"] == system]
        _write_json(
            partial_dir / f"{system}.json",
            {"system": system, "rows": system_rows},
        )
    _write_json(partial_dir / "summary.json", dict(metadata))
    partial_dir.rename(output_dir)


def run(
    *,
    source_summary_path: Path = SOURCE_SUMMARY_PATH,
    output_dir: Path = OUTPUT_DIR,
    expected_git_commit: str,
    preflight_only: bool = False,
) -> dict[str, Any]:
    _refuse_existing_output(output_dir)
    checked = preflight(
        source_summary_path=source_summary_path,
        expected_git_commit=expected_git_commit,
    )
    if preflight_only:
        return dict(checked.metadata)

    calculator_factory = checked.source["_calculator"]
    rows: list[dict[str, Any]] = []
    for system in SYSTEMS:
        for arm in ARMS:
            calculator = calculator_factory()
            for frozen in checked.tasks_by_system[system]:
                measured = replay_task_with_trace(
                    frozen.relax_task,
                    calculator,
                    arm=arm,
                )
                row = _row_payload(frozen, arm, measured)
                _validate_row(row)
                rows.append(row)
    _write_output(output_dir, rows, checked.metadata)
    return dict(checked.metadata)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-summary",
        type=Path,
        default=SOURCE_SUMMARY_PATH,
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    run(
        source_summary_path=args.source_summary,
        output_dir=args.output_dir,
        expected_git_commit=args.expected_git_commit,
        preflight_only=args.preflight_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
