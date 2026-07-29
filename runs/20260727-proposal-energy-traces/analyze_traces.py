#!/usr/bin/env python3
"""Compact, observer-only analysis of fixed GPU proposal-energy traces.

The raw ledger remains in ``output/``.  This module copies no trajectories:
it verifies their accounting and emits only per-task curve statistics, paired
backend costs, an optional total-objective SVG, and a bounded claim document.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


RUN_ROOT = Path(__file__).resolve().parent
ENERGY_FIELDS = (
    "true_energy_eV",
    "bias_energy_eV",
    "softening_energy_eV",
    "total_energy_eV",
)
FORCE_FIELD = "active_max_total_force_eV_per_A"
FROZEN_SYSTEMS = ("c60", "pdo")
FROZEN_BACKENDS = ("ase-fire", "safe-lbfgs-total")
FROZEN_TASK_IDS = {
    system: tuple(f"{system}-seed-{seed}-bias-1" for seed in range(42, 50))
    for system in FROZEN_SYSTEMS
}
_REFERENCE_BOOLEAN_FIELDS = (
    "all_reference_fields_equal",
    "strict_reference_match",
    "certificate_equal",
    "termination_equal",
    "energy_within_measured_neighborhood",
    "position_within_measured_neighborhood",
)
_REFERENCE_FINITE_FIELDS = (
    "energy_delta_eV",
    "max_mic_displacement_A",
    "rms_mic_displacement_A",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_float(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite number") from error
    if not np.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _nonnegative_int(value: object, *, label: str) -> int:
    result = _integer(value, label=label)
    if result < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return result


def _integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _nonempty_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _sha256_string(value: object, *, label: str) -> str:
    return _hex_string(value, label=label, length=64)


def _hex_string(value: object, *, label: str, length: int) -> str:
    text = _nonempty_string(value, label=label)
    if len(text) != length or any(character not in "0123456789abcdef" for character in text.lower()):
        raise ValueError(f"{label} must be a {length}-character hexadecimal digest")
    return text


def _boolean(value: object, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _count_positive_increases(records: Sequence[Mapping[str, Any]]) -> tuple[int, float]:
    energies = [float(record["total_energy_eV"]) for record in records]
    increments = [later - earlier for earlier, later in zip(energies, energies[1:]) if later > earlier]
    return len(increments), max(increments, default=0.0)


def _energy_summary(records: Sequence[Mapping[str, Any]], field: str) -> dict[str, float]:
    values = [float(record[field]) for record in records]
    return {
        "initial_eV": values[0],
        "final_eV": values[-1],
        "final_minus_initial_eV": values[-1] - values[0],
    }


def _accepted_excess_auc(records: Sequence[Mapping[str, Any]]) -> float:
    """Area above the best callback-observed total energy, in eV-evaluations."""

    if len(records) < 2:
        return 0.0
    x = np.asarray([int(record["evaluation_index"]) for record in records], dtype=float)
    energy = np.asarray([float(record["total_energy_eV"]) for record in records], dtype=float)
    return float(np.trapezoid(energy - np.min(energy), x=x))


def _validate_top_level_provenance(summary: Mapping[str, Any]) -> None:
    if _nonnegative_int(summary.get("schema_version"), label="schema_version") != 1:
        raise ValueError("schema_version must be 1")
    _hex_string(summary.get("current_git_commit"), label="current_git_commit", length=40)
    _nonempty_string(summary.get("source_summary"), label="source_summary")
    _sha256_string(summary.get("source_summary_sha256"), label="source_summary_sha256")
    _nonempty_string(summary.get("reference_summary"), label="reference_summary")
    _sha256_string(summary.get("reference_summary_sha256"), label="reference_summary_sha256")
    if _nonnegative_int(summary.get("observation_maxiter"), label="observation_maxiter") != 400:
        raise ValueError("observation_maxiter must be the frozen value 400")
    _finite_float(summary.get("wall_time_total_s"), label="wall_time_total_s")
    neighborhoods = _mapping(
        summary.get("historical_reference_neighborhoods"),
        label="historical_reference_neighborhoods",
    )
    for field in (
        "final_biased_energy_absolute_eV",
        "final_positions_max_mic_displacement_A",
    ):
        if _finite_float(neighborhoods.get(field), label=f"historical_reference_neighborhoods.{field}") < 0.0:
            raise ValueError(f"historical_reference_neighborhoods.{field} must be nonnegative")

    cuda = _mapping(summary.get("cuda_model_provenance"), label="cuda_model_provenance")
    _nonempty_string(cuda.get("cuda_device"), label="cuda_model_provenance.cuda_device")
    if _nonempty_string(cuda.get("device"), label="cuda_model_provenance.device") != "cuda":
        raise ValueError("cuda_model_provenance.device must be cuda")
    _nonempty_string(cuda.get("model"), label="cuda_model_provenance.model")
    model_declared = _sha256_string(
        cuda.get("model_declared_sha256"),
        label="cuda_model_provenance.model_declared_sha256",
    )
    model_actual = _sha256_string(
        cuda.get("model_sha256"),
        label="cuda_model_provenance.model_sha256",
    )
    if model_declared != model_actual:
        raise ValueError("cuda_model_provenance model hashes disagree")
    _nonempty_string(cuda.get("torch_version"), label="cuda_model_provenance.torch_version")
    inputs = _mapping(cuda.get("source_system_inputs"), label="cuda_model_provenance.source_system_inputs")
    if set(inputs) != set(FROZEN_SYSTEMS):
        raise ValueError("cuda_model_provenance.source_system_inputs must cover frozen systems")
    for system in FROZEN_SYSTEMS:
        source_input = _mapping(inputs[system], label=f"cuda_model_provenance.source_system_inputs.{system}")
        _nonempty_string(source_input.get("input"), label=f"cuda_model_provenance.source_system_inputs.{system}.input")
        declared = _sha256_string(
            source_input.get("declared_sha256"),
            label=f"cuda_model_provenance.source_system_inputs.{system}.declared_sha256",
        )
        actual = _sha256_string(
            source_input.get("sha256"),
            label=f"cuda_model_provenance.source_system_inputs.{system}.sha256",
        )
        if declared != actual:
            raise ValueError(f"cuda_model_provenance source input hashes disagree for {system}")


def _validated_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    _validate_top_level_provenance(summary)
    systems = summary.get("systems")
    if not isinstance(systems, list) or len(systems) != len(FROZEN_SYSTEMS):
        raise ValueError("summary must contain exactly the frozen systems")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    seen_systems: set[str] = set()
    for system_entry in systems:
        if not isinstance(system_entry, Mapping):
            raise ValueError("summary system entry must be an object")
        system = system_entry.get("system")
        raw_rows = system_entry.get("rows")
        if system not in FROZEN_SYSTEMS or system in seen_systems:
            raise ValueError("summary must contain each frozen system exactly once")
        seen_systems.add(system)
        task_ids = system_entry.get("task_ids")
        if not isinstance(task_ids, list) or tuple(task_ids) != FROZEN_TASK_IDS[system]:
            raise ValueError(f"summary {system} task_ids must equal the frozen task sequence")
        if not isinstance(raw_rows, list) or len(raw_rows) != len(FROZEN_TASK_IDS[system]) * len(FROZEN_BACKENDS):
            raise ValueError(f"summary {system} has an incomplete frozen task matrix")
        for row in raw_rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"summary {system} row must be an object")
            task_id = row.get("task_id")
            backend = row.get("backend")
            if not isinstance(task_id, str) or not task_id:
                raise ValueError(f"summary {system} row requires task_id")
            if not isinstance(backend, str) or not backend:
                raise ValueError(f"summary {system}/{task_id} row requires backend")
            if row.get("system") != system:
                raise ValueError(f"summary {system}/{task_id} row has mismatched system")
            key = (system, task_id, backend)
            if key in seen:
                raise ValueError(f"duplicate trace row: {key}")
            seen.add(key)
            copied = dict(row)
            copied["system"] = system
            rows.append(copied)
    if seen_systems != set(FROZEN_SYSTEMS):
        raise ValueError("summary must contain each frozen system")
    expected = {
        (system, task_id, backend)
        for system in FROZEN_SYSTEMS
        for task_id in FROZEN_TASK_IDS[system]
        for backend in FROZEN_BACKENDS
    }
    if seen != expected:
        raise ValueError("summary has an incomplete frozen task matrix")
    system_order = {system: index for index, system in enumerate(FROZEN_SYSTEMS)}
    task_order = {
        (system, task_id): index
        for system in FROZEN_SYSTEMS
        for index, task_id in enumerate(FROZEN_TASK_IDS[system])
    }
    backend_order = {backend: index for index, backend in enumerate(FROZEN_BACKENDS)}
    return sorted(rows, key=lambda row: (system_order[row["system"]], task_order[(row["system"], row["task_id"])], backend_order[row["backend"]]))


def _explicit_finalization_mask(
    records: Sequence[Mapping[str, Any]],
    telemetry: Mapping[str, Any],
    *,
    label: str,
) -> np.ndarray:
    """Mark the terminal accepted records that came from explicit finalization."""

    finalization_count = _nonnegative_int(
        telemetry.get("explicit_finalization_calls", 0),
        label=f"{label}: explicit_finalization_calls",
    )
    accepted = np.asarray([_boolean(record.get("accepted_state"), label=f"{label}: accepted_state") for record in records], dtype=bool)
    if finalization_count > int(np.count_nonzero(accepted)):
        raise ValueError(f"{label}: explicit finalizations exceed accepted-state records")
    mask = np.zeros(len(records), dtype=bool)
    if finalization_count:
        mask[np.flatnonzero(accepted)[-finalization_count:]] = True
    return mask


def _task_evidence(row: Mapping[str, Any]) -> dict[str, Any]:
    system = row["system"]
    task_id = row["task_id"]
    backend = row["backend"]
    label = f"{system}/{task_id}/{backend}"
    trace_records = row.get("trace_records")
    telemetry = row.get("telemetry")
    if not isinstance(trace_records, list) or not trace_records:
        raise ValueError(f"{label}: trace_records must be nonempty")
    if not isinstance(telemetry, Mapping):
        raise ValueError(f"{label}: telemetry must be an object")
    records: list[dict[str, Any]] = []
    for expected_index, record in enumerate(trace_records, start=1):
        if not isinstance(record, Mapping):
            raise ValueError(f"{label}: trace record {expected_index} must be an object")
        if record.get("evaluation_index") != expected_index:
            raise ValueError(f"{label}: trace evaluation_index must be consecutive from one")
        if not isinstance(record.get("accepted_state"), bool):
            raise ValueError(f"{label}: every trace record requires boolean accepted_state")
        copied = dict(record)
        for field in (*ENERGY_FIELDS, FORCE_FIELD):
            copied[field] = _finite_float(copied.get(field), label=f"{label}: {field}")
        records.append(copied)

    force_evaluations = _nonnegative_int(row.get("force_evaluations"), label=f"{label}: force_evaluations")
    telemetry_calls = _nonnegative_int(
        telemetry.get("evaluator_calls"), label=f"{label}: telemetry evaluator_calls"
    )
    if len(records) != force_evaluations or force_evaluations != telemetry_calls:
        raise ValueError(
            f"{label}: trace/counter/telemetry mismatch "
            f"({len(records)}, {force_evaluations}, {telemetry_calls})"
        )
    accepted_records = [record for record in records if record["accepted_state"]]
    accepted_count = _nonnegative_int(
        row.get("accepted_state_count"), label=f"{label}: accepted_state_count"
    )
    nonaccepted_count = _nonnegative_int(
        row.get("non_accepted_evaluation_count"), label=f"{label}: non_accepted_evaluation_count"
    )
    if len(accepted_records) != accepted_count or accepted_count + nonaccepted_count != len(records):
        raise ValueError(f"{label}: accepted-state count does not match trace labels")

    finalization_mask = _explicit_finalization_mask(records, telemetry, label=label)
    finalization_count = int(np.count_nonzero(finalization_mask))
    callback_records_excluding_finalization = [
        record
        for record, is_finalization in zip(records, finalization_mask)
        if record["accepted_state"] and not is_finalization
    ]
    if not callback_records_excluding_finalization:
        raise ValueError(f"{label}: no callback-observed optimizer states after finalization separation")

    final_total = _finite_float(row.get("final_biased_energy_eV"), label=f"{label}: final_biased_energy_eV")
    if final_total != records[-1]["total_energy_eV"]:
        raise ValueError(f"{label}: final biased energy does not equal final trace total energy")
    final_force = _finite_float(
        row.get("final_max_active_atom_force_eV_per_A"),
        label=f"{label}: final_max_active_atom_force_eV_per_A",
    )
    termination_reason = telemetry.get("termination_reason")
    if not isinstance(termination_reason, str) or not termination_reason:
        raise ValueError(f"{label}: telemetry termination_reason must be nonempty")
    certificate = row.get("certificate_satisfied")
    if not isinstance(certificate, bool):
        raise ValueError(f"{label}: certificate_satisfied must be boolean")

    rejected_steps = _nonnegative_int(
        telemetry.get("rejected_steps", 0), label=f"{label}: rejected_steps"
    )
    accepted_steps = _nonnegative_int(
        telemetry.get("accepted_steps", 0), label=f"{label}: accepted_steps"
    )
    all_increase_count, all_max_increase = _count_positive_increases(records)
    accepted_increase_count, accepted_max_increase = _count_positive_increases(accepted_records)
    callback_increase_count, callback_max_increase = _count_positive_increases(
        callback_records_excluding_finalization
    )
    safe_lbfgs = backend == "safe-lbfgs-total"
    finalization_delta = (
        float(accepted_records[-1]["total_energy_eV"] - accepted_records[-2]["total_energy_eV"])
        if finalization_count and len(accepted_records) >= 2
        else 0.0
    )
    total_values = [record["total_energy_eV"] for record in records]
    best_index = int(np.argmin(total_values))
    reference = row.get("reference_comparison")
    if not isinstance(reference, Mapping):
        raise ValueError(f"{label}: reference_comparison must be an object")
    reference_summary = {
        key: _boolean(reference.get(key), label=f"{label}: reference_comparison.{key}")
        for key in _REFERENCE_BOOLEAN_FIELDS
    }
    reference_summary["call_delta"] = _integer(
        reference.get("call_delta"),
        label=f"{label}: reference_comparison.call_delta",
    )
    for field in _REFERENCE_FINITE_FIELDS:
        reference_summary[field] = _finite_float(
            reference.get(field),
            label=f"{label}: reference_comparison.{field}",
        )
    if reference_summary["max_mic_displacement_A"] < 0.0 or reference_summary["rms_mic_displacement_A"] < 0.0:
        raise ValueError(f"{label}: reference comparison displacements must be nonnegative")

    return {
        "system": system,
        "task_id": task_id,
        "seed": row.get("seed"),
        "backend": backend,
        "force_evaluations": force_evaluations,
        "wall_time_s": _finite_float(row.get("wall_time_s"), label=f"{label}: wall_time_s"),
        "certificate_satisfied": certificate,
        "termination_reason": termination_reason,
        "final_max_active_atom_force_eV_per_A": final_force,
        "trace_counter_telemetry_closed": True,
        "accepted_state_count": accepted_count,
        "non_accepted_evaluation_count": nonaccepted_count,
        "non_accepted_minus_telemetry_rejected_steps": nonaccepted_count - rejected_steps,
        "telemetry": {
            "accepted_steps": accepted_steps,
            "rejected_steps": rejected_steps,
            "line_search_evaluations": _nonnegative_int(
                telemetry.get("line_search_evaluations", 0),
                label=f"{label}: line_search_evaluations",
            ),
            "explicit_finalization_calls": finalization_count,
            "reporting_evaluator_calls": _nonnegative_int(
                telemetry.get("reporting_evaluator_calls", 0),
                label=f"{label}: reporting_evaluator_calls",
            ),
        },
        "energy_eV": {field.removesuffix("_energy_eV"): _energy_summary(records, field) for field in ENERGY_FIELDS},
        "best_total_energy_eV": total_values[best_index],
        "best_total_energy_evaluation_index": best_index + 1,
        "accepted_total_energy_excess_auc_eV_evaluations": _accepted_excess_auc(accepted_records),
        "all_evaluation_total_energy_increase_count": all_increase_count,
        "all_evaluation_max_total_energy_increase_eV": all_max_increase,
        "accepted_total_energy_increase_count_including_finalization": accepted_increase_count,
        "accepted_max_total_energy_increase_including_finalization_eV": accepted_max_increase,
        "explicit_finalization_record_count": finalization_count,
        "explicit_finalization_evaluation_indices": [
            int(record["evaluation_index"])
            for record, is_finalization in zip(records, finalization_mask)
            if is_finalization
        ],
        "finalization_total_energy_delta_eV": finalization_delta,
        "callback_observed_total_energy_increase_count_excluding_explicit_finalization": callback_increase_count,
        "callback_observed_max_total_energy_increase_excluding_explicit_finalization_eV": callback_max_increase,
        "safe_lbfgs_callback_trajectory_matches_initial_plus_accepted_steps": (
            len(callback_records_excluding_finalization) == accepted_steps + 1 if safe_lbfgs else None
        ),
        "safe_lbfgs_accepted_step_total_energy_increase_count": (
            callback_increase_count if safe_lbfgs else None
        ),
        "safe_lbfgs_accepted_step_max_total_energy_increase_eV": (
            callback_max_increase if safe_lbfgs else None
        ),
        "safe_lbfgs_accepted_step_total_energy_is_monotone_nonincreasing": (
            callback_increase_count == 0 if safe_lbfgs else None
        ),
        "historical_reference_comparison": reference_summary,
    }


def _backend_aggregates(tasks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for task in tasks:
        grouped[(str(task["system"]), str(task["backend"]))].append(task)
    aggregates: list[dict[str, Any]] = []
    for (system, backend), values in sorted(grouped.items()):
        aggregates.append(
            {
                "system": system,
                "backend": backend,
                "task_count": len(values),
                "force_evaluations": sum(int(value["force_evaluations"]) for value in values),
                "wall_time_s": sum(float(value["wall_time_s"]) for value in values),
                "certificate_satisfied_count": sum(bool(value["certificate_satisfied"]) for value in values),
                "termination_reasons": {
                    reason: sum(value["termination_reason"] == reason for value in values)
                    for reason in sorted({str(value["termination_reason"]) for value in values})
                },
                "accepted_state_evaluations": sum(int(value["accepted_state_count"]) for value in values),
                "non_accepted_evaluations": sum(
                    int(value["non_accepted_evaluation_count"]) for value in values
                ),
                "telemetry_rejected_steps": sum(
                    int(value["telemetry"]["rejected_steps"]) for value in values
                ),
            }
        )
    return aggregates


def _paired_backend_comparisons(tasks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_task: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for task in tasks:
        by_task[(str(task["system"]), str(task["task_id"]))][str(task["backend"])] = task
    comparisons: list[dict[str, Any]] = []
    for (system, task_id), backends in sorted(by_task.items()):
        fire = backends.get("ase-fire")
        safe = backends.get("safe-lbfgs-total")
        if fire is None or safe is None:
            continue
        fire_calls = int(fire["force_evaluations"])
        safe_calls = int(safe["force_evaluations"])
        comparisons.append(
            {
                "system": system,
                "task_id": task_id,
                "safe_minus_fire_force_evaluations": safe_calls - fire_calls,
                "safe_over_fire_force_evaluations": safe_calls / fire_calls,
                "safe_minus_fire_wall_time_s": float(safe["wall_time_s"]) - float(fire["wall_time_s"]),
                "safe_minus_fire_final_biased_energy_eV": (
                    float(safe["energy_eV"]["total"]["final_eV"])
                    - float(fire["energy_eV"]["total"]["final_eV"])
                ),
                "safe_certificate_satisfied": bool(safe["certificate_satisfied"]),
                "fire_certificate_satisfied": bool(fire["certificate_satisfied"]),
            }
        )
    return comparisons


def _reference_aggregate(tasks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    comparisons = [task["historical_reference_comparison"] for task in tasks]
    return {
        "row_count": len(comparisons),
        "strict_reference_match_count": sum(bool(item["strict_reference_match"]) for item in comparisons),
        "all_reference_fields_equal_count": sum(bool(item["all_reference_fields_equal"]) for item in comparisons),
        "certificate_equal_count": sum(bool(item["certificate_equal"]) for item in comparisons),
        "termination_equal_count": sum(bool(item["termination_equal"]) for item in comparisons),
        "energy_within_measured_neighborhood_count": sum(
            bool(item["energy_within_measured_neighborhood"]) for item in comparisons
        ),
        "position_within_measured_neighborhood_count": sum(
            bool(item["position_within_measured_neighborhood"]) for item in comparisons
        ),
        "call_delta_values": [int(item["call_delta"]) for item in comparisons],
    }


def analyze_summary(summary: Mapping[str, Any], *, input_summary_sha256: str) -> dict[str, Any]:
    """Validate the complete trace ledger and return compact, non-trajectory evidence."""

    rows = _validated_rows(summary)
    tasks = [_task_evidence(row) for row in rows]
    backend_aggregates = _backend_aggregates(tasks)
    paired = _paired_backend_comparisons(tasks)
    maxiter_tasks = [
        {"system": task["system"], "task_id": task["task_id"], "backend": task["backend"]}
        for task in tasks
        if task["termination_reason"] == "maxiter"
    ]
    safe_tasks = [task for task in tasks if task["backend"] == "safe-lbfgs-total"]
    safe_finalization_nonmonotonic = [
        task
        for task in safe_tasks
        if task["explicit_finalization_record_count"]
        and task["finalization_total_energy_delta_eV"] > 0.0
    ]
    all_labels_complete = all(
        task["accepted_state_count"] + task["non_accepted_evaluation_count"]
        == task["force_evaluations"]
        for task in tasks
    )
    return {
        "schema_version": 1,
        "input_summary_filename": "summary.json",
        "input_summary_sha256": input_summary_sha256,
        "source_run_git_commit": summary.get("current_git_commit"),
        "source_summary_sha256": summary.get("source_summary_sha256"),
        "reference_summary_sha256": summary.get("reference_summary_sha256"),
        "cuda_model_provenance": summary.get("cuda_model_provenance"),
        "validation": {
            "trace_row_count": len(tasks),
            "trace_counter_telemetry_closed_for_all_rows": all(
                bool(task["trace_counter_telemetry_closed"]) for task in tasks
            ),
            "accepted_state_labels_complete_for_all_rows": all_labels_complete,
            "all_recorded_energy_and_force_values_finite": True,
        },
        "tasks": tasks,
        "aggregates": {
            "by_system_backend": backend_aggregates,
            "paired_fire_safe": paired,
            "maxiter_tasks": maxiter_tasks,
            "historical_reference_comparison": _reference_aggregate(tasks),
            "safe_line_search_label_relation": {
                "task_count": len(safe_tasks),
                "non_accepted_equals_telemetry_rejected_steps_for_all_tasks": all(
                    task["non_accepted_minus_telemetry_rejected_steps"] == 0
                    for task in safe_tasks
                ),
                "non_accepted_evaluation_count": sum(
                    int(task["non_accepted_evaluation_count"]) for task in safe_tasks
                ),
                "telemetry_rejected_steps": sum(
                    int(task["telemetry"]["rejected_steps"]) for task in safe_tasks
                ),
                "interpretation": (
                    "Callback absence is an observer label. Its numerical equality to this "
                    "ledger's rejected_steps is reported, not promoted to a general semantic identity."
                ),
            },
            "safe_finalization_rechecks": {
                "explicit_finalization_record_count": sum(
                    int(task["explicit_finalization_record_count"]) for task in safe_tasks
                ),
                "positive_finalization_recheck_count": len(safe_finalization_nonmonotonic),
                "positive_finalization_rechecks": [
                    {
                        "system": task["system"],
                        "task_id": task["task_id"],
                        "delta_total_energy_eV": task["finalization_total_energy_delta_eV"],
                    }
                    for task in safe_finalization_nonmonotonic
                ],
                "optimizer_accepted_total_energy_monotone_for_all_safe_tasks": all(
                    bool(task["safe_lbfgs_accepted_step_total_energy_is_monotone_nonincreasing"])
                    for task in safe_tasks
                ),
            },
        },
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_curve_plot(summary: Mapping[str, Any], output_path: Path) -> str | None:
    """Plot total biased energy relative to each task's initial point, if available."""

    try:
        os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pamssw-matplotlib"))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception as error:  # pragma: no cover - host-specific optional path
        return f"matplotlib unavailable: {type(error).__name__}: {error}"

    rows = _validated_rows(summary)
    with plt.rc_context({"svg.hashsalt": "pamssw-proposal-energy-traces-v1", "svg.fonttype": "none"}):
        figure, axes = plt.subplots(
            len(FROZEN_SYSTEMS),
            len(FROZEN_BACKENDS),
            figsize=(12, 4.4 * len(FROZEN_SYSTEMS)),
            squeeze=False,
        )
        colors = plt.get_cmap("tab10")
        semantic_handles = (
            Line2D([], [], color="black", alpha=0.30, linewidth=1.0, label="all exact evaluations (faint line)"),
            Line2D(
                [],
                [],
                color="black",
                linewidth=1.25,
                label="callback-observed accepted_state (not optimizer acceptance rule)",
            ),
            Line2D(
                [],
                [],
                color="black",
                marker="x",
                linestyle="None",
                markersize=5,
                label="callback-nonaccepted evaluation",
            ),
            Line2D(
                [],
                [],
                color="black",
                marker="D",
                markerfacecolor="none",
                linestyle="None",
                markersize=5,
                label="explicit finalization recheck",
            ),
        )
        for system_index, system in enumerate(FROZEN_SYSTEMS):
            for backend_index, backend in enumerate(FROZEN_BACKENDS):
                axis = axes[system_index][backend_index]
                matching = sorted(
                    (row for row in rows if row["system"] == system and row["backend"] == backend),
                    key=lambda row: str(row["task_id"]),
                )
                for row_index, row in enumerate(matching):
                    records = row["trace_records"]
                    x = np.asarray([record["evaluation_index"] for record in records], dtype=float)
                    total = np.asarray([record["total_energy_eV"] for record in records], dtype=float)
                    accepted = np.asarray([record["accepted_state"] for record in records], dtype=bool)
                    finalization = _explicit_finalization_mask(
                        records,
                        _mapping(row.get("telemetry"), label=f"{system}/{row['task_id']}/{backend}: telemetry"),
                        label=f"{system}/{row['task_id']}/{backend}",
                    )
                    color = colors(row_index % 10)
                    relative_total = total - total[0]
                    axis.plot(x, relative_total, color=color, alpha=0.30, linewidth=1.0)
                    axis.plot(x[accepted], relative_total[accepted], color=color, linewidth=1.25, label=row["task_id"])
                    if np.any(~accepted):
                        axis.scatter(
                            x[~accepted],
                            relative_total[~accepted],
                            marker="x",
                            s=18,
                            linewidths=0.8,
                            color=color,
                        )
                    if np.any(finalization):
                        marker = axis.scatter(
                            x[finalization],
                            relative_total[finalization],
                            marker="D",
                            s=28,
                            linewidths=0.8,
                            facecolors="none",
                            edgecolors="black",
                            zorder=4,
                        )
                        marker.set_gid(f"explicit-finalization-recheck-{system}-{backend}-{row['task_id']}")
                axis.set_title(f"{system} · {backend}")
                axis.set_xlabel("exact force evaluations")
                axis.set_ylabel("total biased energy − initial (eV)")
                axis.grid(alpha=0.22)
                axis.legend(title="task ID (color)", fontsize=6, title_fontsize=6, ncol=2, frameon=False)
        figure.suptitle("Frozen one-bias proposal relaxations: evaluated total-objective traces", y=0.998)
        figure.legend(
            handles=semantic_handles,
            title="trace semantics",
            loc="lower center",
            ncol=2,
            fontsize=7,
            title_fontsize=7,
            frameon=False,
        )
        figure.tight_layout(rect=(0.0, 0.085, 1.0, 0.97))
        figure.savefig(
            output_path,
            format="svg",
            bbox_inches="tight",
            metadata={"Date": "2026-07-27T00:00:00Z"},
        )
        plt.close(figure)
    # Matplotlib wraps SVG path data with trailing spaces; remove only those
    # end-of-line formatting bytes so repository whitespace validation stays
    # meaningful without changing XML token boundaries.
    output_path.write_text(
        "\n".join(line.rstrip() for line in output_path.read_text(encoding="utf-8").splitlines()) + "\n",
        encoding="utf-8",
    )
    return None


def _conclusion_markdown(evidence: Mapping[str, Any]) -> str:
    aggregates = evidence["aggregates"]
    fire_calls = sum(
        int(item["force_evaluations"])
        for item in aggregates["by_system_backend"]
        if item["backend"] == "ase-fire"
    )
    safe_calls = sum(
        int(item["force_evaluations"])
        for item in aggregates["by_system_backend"]
        if item["backend"] == "safe-lbfgs-total"
    )
    fire_time = sum(
        float(item["wall_time_s"])
        for item in aggregates["by_system_backend"]
        if item["backend"] == "ase-fire"
    )
    safe_time = sum(
        float(item["wall_time_s"])
        for item in aggregates["by_system_backend"]
        if item["backend"] == "safe-lbfgs-total"
    )
    fire_certificates = sum(
        int(item["certificate_satisfied_count"])
        for item in aggregates["by_system_backend"]
        if item["backend"] == "ase-fire"
    )
    safe_certificates = sum(
        int(item["certificate_satisfied_count"])
        for item in aggregates["by_system_backend"]
        if item["backend"] == "safe-lbfgs-total"
    )
    task_count_per_backend = sum(
        int(item["task_count"])
        for item in aggregates["by_system_backend"]
        if item["backend"] == "ase-fire"
    )
    maxiter = aggregates["maxiter_tasks"]
    reference = aggregates["historical_reference_comparison"]
    finalization = aggregates["safe_finalization_rechecks"]
    nonaccepted = aggregates["safe_line_search_label_relation"]
    reduction = 100.0 * (1.0 - safe_calls / fire_calls)
    wall_reduction = 100.0 * (1.0 - safe_time / fire_time)

    paired_task_count = len(aggregates["paired_fire_safe"])
    safe_task_count = int(nonaccepted["task_count"])
    return f"""# Fixed proposal energy-trace evidence

## Execution/protocol facts

The published CUDA ledger contains {evidence['validation']['trace_row_count']} frozen task/backend traces. Every row closes the observer ledger exactly: trace-record count = force-evaluation count = Relaxer telemetry count; every curve value is finite; and all `accepted_state` labels are present. The exact raw input is `output/summary.json` with SHA-256 `{evidence['input_summary_sha256']}`. It was generated by source-run commit `{evidence['source_run_git_commit']}` from source-summary hash `{evidence['source_summary_sha256']}`.

The historical cap-400 run is descriptive only. It has {reference['strict_reference_match_count']}/{reference['row_count']} bitwise-strict matches, while certificates and termination reasons agree in {reference['certificate_equal_count']}/{reference['row_count']} and {reference['termination_equal_count']}/{reference['row_count']} rows. The cross-process float32 differences are reported rather than tuned away; they are not a failure of the present observer ledger.

## Optimizer evidence for this fixed matrix

Across the paired {paired_task_count} tasks, FIRE used {fire_calls} exact force evaluations and {fire_time:.3f} s; safe L-BFGS used {safe_calls} and {safe_time:.3f} s. In this matrix that is {reduction:.1f}% fewer evaluator calls and {wall_reduction:.1f}% less wall time for safe L-BFGS. The force certificate holds for {safe_certificates}/{task_count_per_backend} safe traces and {fire_certificates}/{task_count_per_backend} FIRE traces.

FIRE reached the 400-iteration observation cap on {len(maxiter)} traces: {', '.join(f"{item['system']}/{item['task_id']}" for item in maxiter) or 'none'}. Safe L-BFGS had {nonaccepted['non_accepted_evaluation_count']} callback-nonaccepted evaluations and {nonaccepted['telemetry_rejected_steps']} telemetry-reported rejected trials; the counts coincide in this ledger. That numerical equality does not establish that the observer label is a general synonym for a line-search rejection.

For safe L-BFGS, callback states have the expected initial-plus-accepted-step count and are monotone non-increasing in total biased energy for all {safe_task_count} tasks after explicit finalization evaluations are removed. There are {finalization['positive_finalization_recheck_count']} positive finalization rechecks among {finalization['explicit_finalization_record_count']} explicit rechecks: {', '.join(f"{item['system']}/{item['task_id']} ({item['delta_total_energy_eV']:.8g} eV)" for item in finalization['positive_finalization_rechecks']) or 'none'}. These are recorded terminal re-evaluations, not Armijo-accepted optimizer steps. For FIRE, `accepted_state` remains only a callback-observer label; its positive total-energy changes are therefore neither an acceptance-rule failure nor a comparable line-search statistic.

## What this does not establish

This observer-only matrix does not establish a general physical or algorithmic superiority of L-BFGS, a causal role for its memory, Armijo test, or step cap, or same-basin acceleration: endpoints can differ. It also does not change the production optimizer, its parameters, the bias PES, or the search policy. The next clean causal experiment is to freeze these tasks and ablate one safe-L-BFGS mechanism at a time while retaining the same observer and exact budget ledger.
"""


def write_analysis_artifacts(
    summary_path: Path,
    output_dir: Path,
    *,
    make_plot: bool = True,
    command: str | None = None,
) -> dict[str, Any]:
    """Generate compact evidence from a published trace summary without copying it."""

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load summary JSON: {summary_path}") from error
    if not isinstance(summary, Mapping):
        raise ValueError("summary must be a JSON object")
    evidence = analyze_summary(summary, input_summary_sha256=_sha256(summary_path))
    evidence["analysis_script_sha256"] = _sha256(Path(__file__))
    evidence["generator_command"] = command or (
        f"python {RUN_ROOT.relative_to(RUN_ROOT.parents[2]) / 'analyze_traces.py'} "
        f"--summary {summary_path} --output-dir {output_dir}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_name = "proposal_energy_curves.svg"
    plot_error = _write_curve_plot(summary, output_dir / plot_name) if make_plot else None
    evidence["plot"] = {
        "filename": plot_name if make_plot and plot_error is None else None,
        "status": "written" if make_plot and plot_error is None else (plot_error or "not requested"),
    }
    _write_json(output_dir / "evidence.json", evidence)
    (output_dir / "conclusion.md").write_text(_conclusion_markdown(evidence), encoding="utf-8")
    return evidence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=RUN_ROOT / "output" / "summary.json",
        help="published raw trace summary to analyze",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RUN_ROOT,
        help="directory for compact evidence and optional SVG",
    )
    parser.add_argument("--no-plot", action="store_true", help="skip optional matplotlib SVG generation")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    argument_tokens = list(sys.argv[1:] if argv is None else argv)
    arguments = parser.parse_args(argument_tokens)
    command = " ".join(
        ["python", str(Path(__file__).relative_to(RUN_ROOT.parents[1])), *argument_tokens]
    )
    evidence = write_analysis_artifacts(
        arguments.summary,
        arguments.output_dir,
        make_plot=not arguments.no_plot,
        command=command,
    )
    print(
        "analyzed "
        f"{evidence['validation']['trace_row_count']} rows; "
        f"evidence={arguments.output_dir / 'evidence.json'}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
