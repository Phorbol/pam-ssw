#!/usr/bin/env python3
"""Fail-closed, zero-extra-call energy traces for fixed GPU proposal replays.

The driver deliberately replays only the frozen one-bias tasks and validates
each replay against the already-recorded shared-cap-400 ledger.  Trace labels
come only from the Relaxer trajectory callback; they never trigger a PES call.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
from time import perf_counter
from typing import Any, Mapping, NamedTuple, Sequence

import numpy as np

from pamssw.accounting import EvalCounter, EvaluationCounts, EvaluationPurpose
from pamssw.proposal_replay import proposal_task_from_payload
from pamssw.relax import Relaxer
from pamssw.result import RelaxResult
from pamssw.state import State
from pamssw.walker import ProposalRelaxationTask


RUN_ROOT = Path(__file__).resolve().parent
FIXED_REPLAY_ROOT = RUN_ROOT.parent / "20260727-023234-fixed-proposal-replay-gpu"
FIXED_REPLAY_DRIVER = FIXED_REPLAY_ROOT / "run_fixed_replay.py"
TRACE_RECORDER_PATH = RUN_ROOT / "trace_recorder.py"

SYSTEMS = ("c60", "pdo")
BACKENDS = ("ase-fire", "safe-lbfgs-total")
# Preserve the prior driver's selected-backend ordering without executing FIRE2.
REFERENCE_BACKEND_ORDER = ("ase-fire", "ase-fire2", "safe-lbfgs-total")
OBSERVATION_MAXITER = 400

EXPECTED_SOURCE_SUMMARY_SHA256 = "62cc771e2aa24e9addef0e870d0524f901f02bddc34152cf2a2eeea91a671b04"
EXPECTED_REFERENCE_SUMMARY_SHA256 = "a11cd8ee1a1dae9cc71cac0038008149b1c0f0cb67ae72ceba8afc821cbdf370"

# The original cap-400 replay is deterministic on its recorded GPU/model.  A
# 1e-8 eV / Angstrom absolute tolerance admits only normal floating-point
# roundoff while rejecting materially different endpoints or objectives.
ENERGY_ABSOLUTE_TOLERANCE_EV = 1.0e-8
POSITION_ABSOLUTE_TOLERANCE_A = 1.0e-8

_TRACE_RECORDER_MODULE_NAME = "_proposal_energy_trace_recorder"


class TraceReplayResult(NamedTuple):
    """One traced replay plus the closed evaluator ledger used to create it."""

    result: RelaxResult
    evaluation_counts: EvaluationCounts
    wall_time_s: float
    trace_records: list[dict[str, Any]]
    accepted_callback_hashes: tuple[str, ...]
    certificate_satisfied: bool


class ReplayMismatchError(RuntimeError):
    """Raised when a replay does not reproduce a frozen cap-400 reference row."""


def _trace_recorder_module():
    """Load the adjacent benchmark-local recorder without importing a package path."""

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
    optimizer: str,
) -> TraceReplayResult:
    """Replay one frozen task while recording only the evaluations already made.

    The optimizer and component-evaluator selection intentionally mirrors
    :func:`pamssw.proposal_replay.replay_proposal_task`.  The callback observes
    accepted Relaxer states; the recorder classifies existing component
    evaluations after relaxation and never calls ``calculator`` for tracing.
    """

    trace_recorder = _trace_recorder_module()
    counter = EvalCounter(calculator)
    proposal = trace_recorder.RecordingProposalPotential(
        counter,
        biases=list(task.biases),
        softening=task.softening,
    )
    custom_lbfgs = optimizer in {
        "safe-lbfgs-total",
        "bias-separated-lbfgs",
    }
    relaxer = Relaxer(
        proposal.evaluate,
        optimizer=optimizer,
        component_evaluator=proposal.evaluate_parts if custom_lbfgs else None,
    )
    accepted_callback_hashes: list[str] = []

    def record_accepted_state(state: State) -> None:
        accepted_callback_hashes.append(trace_recorder.position_hash(state))

    started = perf_counter()
    with counter.purpose(EvaluationPurpose.BIASED_PROPOSAL_RELAX):
        result = relaxer.relax(
            task.initial_state,
            fmax=task.fmax,
            maxiter=task.maxiter,
            coordinate_trust_radius=task.coordinate_trust_radius,
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
    return TraceReplayResult(
        result=result,
        evaluation_counts=evaluation_counts,
        wall_time_s=wall_time_s,
        trace_records=trace_records,
        accepted_callback_hashes=tuple(accepted_callback_hashes),
        certificate_satisfied=certificate_satisfied,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
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


def _validated_summary(
    path: Path,
    *,
    label: str,
    expected_sha256: str,
) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{label} SHA256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    return _load_json_object(path, label=label), actual_sha256


def _system_entries(summary: Mapping[str, Any], *, label: str) -> dict[str, Mapping[str, Any]]:
    entries = summary.get("systems")
    if not isinstance(entries, list) or len(entries) != len(SYSTEMS):
        raise ValueError(f"{label} must contain exactly {len(SYSTEMS)} systems")
    by_system: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError(f"{label} systems entries must be objects")
        system = entry.get("system")
        if system not in SYSTEMS or system in by_system:
            raise ValueError(f"{label} has invalid or duplicate system: {system!r}")
        by_system[str(system)] = entry
    if tuple(sorted(by_system)) != tuple(sorted(SYSTEMS)):
        raise ValueError(f"{label} systems must be {SYSTEMS}")
    return by_system


def _source_tasks(source_summary: Mapping[str, Any]) -> dict[str, list[tuple[Mapping[str, Any], ProposalRelaxationTask]]]:
    tasks_by_system: dict[str, list[tuple[Mapping[str, Any], ProposalRelaxationTask]]] = {}
    for system, entry in _system_entries(source_summary, label="source summary").items():
        task_items = entry.get("tasks")
        if not isinstance(task_items, list) or len(task_items) != 8:
            raise ValueError(f"source summary {system} must contain exactly 8 tasks")
        seen_ids: set[str] = set()
        restored: list[tuple[Mapping[str, Any], ProposalRelaxationTask]] = []
        for item in task_items:
            if not isinstance(item, Mapping):
                raise ValueError(f"source summary {system} task must be an object")
            task_id = item.get("task_id")
            payload = item.get("task")
            if not isinstance(task_id, str) or not task_id or task_id in seen_ids:
                raise ValueError(f"source summary {system} task_id must be unique and nonempty")
            if not isinstance(payload, Mapping):
                raise ValueError(f"source summary {system}/{task_id} has no task payload")
            task = proposal_task_from_payload(payload)
            if len(task.biases) != 1:
                raise ValueError(f"source summary {system}/{task_id} is not a one-bias task")
            seen_ids.add(task_id)
            restored.append((item, replace(task, maxiter=OBSERVATION_MAXITER)))
        tasks_by_system[system] = restored
    return tasks_by_system


def _reference_rows(reference_summary: Mapping[str, Any]) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    rows_by_key: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for system, entry in _system_entries(reference_summary, label="reference summary").items():
        rows = entry.get("rows")
        if not isinstance(rows, list):
            raise ValueError(f"reference summary {system} has no rows")
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"reference summary {system} row must be an object")
            task_id = row.get("task_id")
            backend = row.get("backend")
            if not isinstance(task_id, str) or backend not in BACKENDS:
                continue
            key = (system, task_id, str(backend))
            if key in rows_by_key:
                raise ValueError(f"duplicate reference row: {key}")
            rows_by_key[key] = row
    return rows_by_key


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


def _require_reference_match(
    *,
    system: str,
    task_id: str,
    backend: str,
    replay: TraceReplayResult,
    reference: Mapping[str, Any],
) -> None:
    context = f"{system}/{task_id}/{backend}"
    if replay.evaluation_counts.total != reference.get("force_evaluations"):
        raise ReplayMismatchError(
            f"{context}: force calls {replay.evaluation_counts.total} != "
            f"reference {reference.get('force_evaluations')!r}"
        )
    if replay.certificate_satisfied is not reference.get("certificate_satisfied"):
        raise ReplayMismatchError(f"{context}: certificate does not reproduce reference")
    if replay.result.telemetry.termination_reason != reference.get("termination_reason"):
        raise ReplayMismatchError(f"{context}: termination reason does not reproduce reference")
    reference_energy = reference.get("final_biased_energy_eV")
    if not isinstance(reference_energy, (int, float)) or not np.isclose(
        replay.result.energy,
        float(reference_energy),
        rtol=0.0,
        atol=ENERGY_ABSOLUTE_TOLERANCE_EV,
    ):
        raise ReplayMismatchError(f"{context}: final biased energy does not reproduce reference")
    reference_positions = np.asarray(reference.get("final_positions"), dtype=float)
    if reference_positions.shape != replay.result.state.positions.shape or not np.allclose(
        replay.result.state.positions,
        reference_positions,
        rtol=0.0,
        atol=POSITION_ABSOLUTE_TOLERANCE_A,
    ):
        raise ReplayMismatchError(f"{context}: final positions do not reproduce reference")


def _reference_endpoint_comparison(reference: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "relative_to_fire_energy_eV",
        "relative_to_fire_active_rms_A",
        "relative_to_fire_active_max_A",
        "fire_force_evaluations",
    )
    missing = [field for field in fields if field not in reference]
    if missing:
        raise ValueError(f"reference row lacks endpoint comparison fields: {missing}")
    return {field: reference[field] for field in fields}


def _row_payload(
    *,
    system: str,
    task_item: Mapping[str, Any],
    task: ProposalRelaxationTask,
    backend: str,
    replay: TraceReplayResult,
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    result = replay.result
    trace_records = replay.trace_records
    accepted_state_count = sum(bool(record["accepted_state"]) for record in trace_records)
    return {
        "system": system,
        "task_id": task_item["task_id"],
        "seed": task_item.get("seed"),
        "bias_count": len(task.biases),
        "observation_maxiter": task.maxiter,
        "backend": backend,
        "certificate_satisfied": replay.certificate_satisfied,
        "force_evaluations": replay.evaluation_counts.total,
        "purpose_counts": replay.evaluation_counts.as_dict(),
        "wall_time_s": replay.wall_time_s,
        "final_biased_energy_eV": float(result.energy),
        "final_max_active_atom_force_eV_per_A": float(result.gradient_norm),
        "final_positions": result.state.positions.tolist(),
        "telemetry": asdict(result.telemetry),
        "accepted_callback_hashes": list(replay.accepted_callback_hashes),
        "accepted_callback_count": len(replay.accepted_callback_hashes),
        "accepted_state_count": accepted_state_count,
        "non_accepted_evaluation_count": len(trace_records) - accepted_state_count,
        "trace_records": trace_records,
        "reference_endpoint_comparison": _reference_endpoint_comparison(reference),
    }


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _publish_ledger_atomically(
    output_dir: Path,
    system_payloads: Sequence[Mapping[str, Any]],
    summary_payload: Mapping[str, Any],
    *,
    write_json=_write_json_atomic,
) -> None:
    """Publish a complete ledger with one same-filesystem directory rename."""

    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-",
            dir=output_dir.parent,
        )
    )
    try:
        for system_payload in system_payloads:
            system = system_payload.get("system")
            if not isinstance(system, str) or not system:
                raise ValueError("system ledger payload requires a nonempty system name")
            write_json(staging_dir / f"{system}.json", system_payload)
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
        cwd=RUN_ROOT.parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _fixed_replay_source() -> Mapping[str, Any]:
    if not FIXED_REPLAY_DRIVER.is_file():
        raise FileNotFoundError(FIXED_REPLAY_DRIVER)
    helpers = runpy.run_path(str(FIXED_REPLAY_DRIVER))
    source_factory = helpers.get("_source")
    if not callable(source_factory):
        raise RuntimeError("fixed replay driver does not expose _source()")
    source = source_factory()
    if not isinstance(source, Mapping) or not callable(source.get("_calculator")):
        raise RuntimeError("fixed replay/G1 helper does not expose _calculator")
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
    """Hash the model and every source input before any calculator is created."""

    model_path, model_declared_sha256, model_verified_sha256 = _verified_file_provenance(
        source.get("MODEL"),
        source.get("MODEL_SHA256"),
        label="model",
    )
    source_systems = source.get("SYSTEMS")
    if not isinstance(source_systems, Mapping) or not source_systems:
        raise ValueError("source SYSTEMS mapping is required for input provenance")
    system_inputs: dict[str, dict[str, str]] = {}
    for system, specification in source_systems.items():
        if not isinstance(system, str) or not system:
            raise ValueError("source system name must be a nonempty string")
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
    model_provenance = _verified_model_provenance(source)
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch with CUDA support is required for this replay") from error

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False")
    return {
        "device": "cuda",
        "cuda_device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        **model_provenance,
    }


def _selected_backend_order(task_index: int) -> tuple[str, ...]:
    rotated = (
        REFERENCE_BACKEND_ORDER[task_index % len(REFERENCE_BACKEND_ORDER) :]
        + REFERENCE_BACKEND_ORDER[: task_index % len(REFERENCE_BACKEND_ORDER)]
    )
    return tuple(backend for backend in rotated if backend in BACKENDS)


def run(
    *,
    source_summary_path: Path,
    reference_summary_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Validate, replay, compare, and atomically write the complete trace ledger."""

    if output_dir.exists():
        raise FileExistsError(output_dir)
    source_summary, source_sha256 = _validated_summary(
        source_summary_path,
        label="source summary",
        expected_sha256=EXPECTED_SOURCE_SUMMARY_SHA256,
    )
    reference_summary, reference_sha256 = _validated_summary(
        reference_summary_path,
        label="reference summary",
        expected_sha256=EXPECTED_REFERENCE_SUMMARY_SHA256,
    )
    if reference_summary.get("base_summary_sha256") != source_sha256:
        raise ValueError("reference summary is not tied to the validated source summary")
    if reference_summary.get("observation_maxiter") != OBSERVATION_MAXITER:
        raise ValueError("reference summary is not the shared-cap-400 ledger")
    tasks_by_system = _source_tasks(source_summary)
    reference_by_key = _reference_rows(reference_summary)
    source = _fixed_replay_source()
    cuda_provenance = _cuda_provenance(source)
    calculator_factory = source["_calculator"]
    started = perf_counter()
    system_payloads: list[dict[str, Any]] = []

    for system in SYSTEMS:
        task_items = tasks_by_system[system]
        calculators = {backend: calculator_factory() for backend in BACKENDS}
        for calculator in calculators.values():
            calculator.evaluate(task_items[0][1].initial_state)
        rows: list[dict[str, Any]] = []
        for task_index, (task_item, task) in enumerate(task_items):
            task_id = str(task_item["task_id"])
            for backend in _selected_backend_order(task_index):
                reference = reference_by_key.get((system, task_id, backend))
                if reference is None:
                    raise ValueError(f"reference summary lacks {system}/{task_id}/{backend}")
                replay = replay_task_with_trace(task, calculators[backend], backend)
                if not _trace_values_are_finite(replay.trace_records):
                    raise RuntimeError(f"non-finite trace value: {system}/{task_id}/{backend}")
                if len(replay.trace_records) != replay.evaluation_counts.total:
                    raise RuntimeError(f"recorder/counter mismatch: {system}/{task_id}/{backend}")
                if len(replay.trace_records) != replay.result.telemetry.evaluator_calls:
                    raise RuntimeError(f"recorder/telemetry mismatch: {system}/{task_id}/{backend}")
                _require_reference_match(
                    system=system,
                    task_id=task_id,
                    backend=backend,
                    replay=replay,
                    reference=reference,
                )
                rows.append(
                    _row_payload(
                        system=system,
                        task_item=task_item,
                        task=task,
                        backend=backend,
                        replay=replay,
                        reference=reference,
                    )
                )
        if len(rows) != 16:
            raise RuntimeError(f"unexpected replay row count for {system}: {len(rows)}")
        system_payloads.append(
            {
                "system": system,
                "task_ids": [item["task_id"] for item, _ in task_items],
                "rows": rows,
            }
        )

    payload = {
        "schema_version": 1,
        "source_summary": str(source_summary_path),
        "source_summary_sha256": source_sha256,
        "reference_summary": str(reference_summary_path),
        "reference_summary_sha256": reference_sha256,
        "current_git_commit": _current_commit(),
        "observation_maxiter": OBSERVATION_MAXITER,
        "float_reproduction_tolerances": {
            "final_biased_energy_absolute_eV": ENERGY_ABSOLUTE_TOLERANCE_EV,
            "final_positions_absolute_A": POSITION_ABSOLUTE_TOLERANCE_A,
        },
        "cuda_model_provenance": cuda_provenance,
        "systems": system_payloads,
        "wall_time_total_s": perf_counter() - started,
    }
    _publish_ledger_atomically(output_dir, system_payloads, payload)
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--reference-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    run(
        source_summary_path=args.source_summary,
        reference_summary_path=args.reference_summary,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
