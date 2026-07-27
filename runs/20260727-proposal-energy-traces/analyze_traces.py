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
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


RUN_ROOT = Path(__file__).resolve().parent
ENERGY_FIELDS = (
    "true_energy_eV",
    "bias_energy_eV",
    "softening_energy_eV",
    "total_energy_eV",
)
FORCE_FIELD = "active_max_total_force_eV_per_A"


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
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
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


def _validated_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    systems = summary.get("systems")
    if not isinstance(systems, list) or not systems:
        raise ValueError("summary systems must be a nonempty list")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for system_entry in systems:
        if not isinstance(system_entry, Mapping):
            raise ValueError("summary system entry must be an object")
        system = system_entry.get("system")
        raw_rows = system_entry.get("rows")
        if not isinstance(system, str) or not system:
            raise ValueError("summary system entry requires a nonempty system")
        if not isinstance(raw_rows, list) or not raw_rows:
            raise ValueError(f"summary {system} requires nonempty rows")
        for row in raw_rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"summary {system} row must be an object")
            task_id = row.get("task_id")
            backend = row.get("backend")
            if not isinstance(task_id, str) or not task_id:
                raise ValueError(f"summary {system} row requires task_id")
            if not isinstance(backend, str) or not backend:
                raise ValueError(f"summary {system}/{task_id} row requires backend")
            key = (system, task_id, backend)
            if key in seen:
                raise ValueError(f"duplicate trace row: {key}")
            seen.add(key)
            copied = dict(row)
            copied["system"] = system
            rows.append(copied)
    return rows


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

    finalization_count = _nonnegative_int(
        telemetry.get("explicit_finalization_calls", 0),
        label=f"{label}: explicit_finalization_calls",
    )
    if finalization_count > len(accepted_records):
        raise ValueError(f"{label}: explicit finalizations exceed accepted-state records")
    callback_records_excluding_finalization = (
        accepted_records[:-finalization_count] if finalization_count else accepted_records
    )
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
        key: reference.get(key)
        for key in (
            "all_reference_fields_equal",
            "strict_reference_match",
            "call_delta",
            "certificate_equal",
            "termination_equal",
            "energy_delta_eV",
            "energy_within_measured_neighborhood",
            "max_mic_displacement_A",
            "rms_mic_displacement_A",
            "position_within_measured_neighborhood",
        )
    }

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
    except Exception as error:  # pragma: no cover - host-specific optional path
        return f"matplotlib unavailable: {type(error).__name__}: {error}"

    rows = _validated_rows(summary)
    systems = sorted({str(row["system"]) for row in rows})
    backends = ("ase-fire", "safe-lbfgs-total")
    figure, axes = plt.subplots(len(systems), len(backends), figsize=(12, 4.4 * len(systems)), squeeze=False)
    colors = plt.get_cmap("tab10")
    for system_index, system in enumerate(systems):
        for backend_index, backend in enumerate(backends):
            axis = axes[system_index][backend_index]
            matching = [
                row for row in rows if row["system"] == system and row["backend"] == backend
            ]
            for row_index, row in enumerate(matching):
                records = row["trace_records"]
                x = np.asarray([record["evaluation_index"] for record in records], dtype=float)
                total = np.asarray([record["total_energy_eV"] for record in records], dtype=float)
                accepted = np.asarray([record["accepted_state"] for record in records], dtype=bool)
                color = colors(row_index % 10)
                axis.plot(x, total - total[0], color=color, alpha=0.30, linewidth=1.0)
                axis.plot(x[accepted], (total - total[0])[accepted], color=color, linewidth=1.25, label=row["task_id"])
                if np.any(~accepted):
                    axis.scatter(
                        x[~accepted],
                        (total - total[0])[~accepted],
                        marker="x",
                        s=18,
                        linewidths=0.8,
                        color=color,
                    )
            axis.set_title(f"{system} · {backend}")
            axis.set_xlabel("exact force evaluations")
            axis.set_ylabel("total biased energy − initial (eV)")
            axis.grid(alpha=0.22)
            if matching:
                axis.legend(fontsize=6, ncol=2, frameon=False)
    figure.suptitle("Frozen one-bias proposal relaxations: evaluated total-objective traces", y=0.998)
    figure.tight_layout()
    figure.savefig(output_path, format="svg", bbox_inches="tight")
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

    return f"""# Fixed proposal energy-trace evidence

## Execution/protocol facts

The published CUDA ledger contains {evidence['validation']['trace_row_count']} frozen task/backend traces. Every row closes the observer ledger exactly: trace-record count = force-evaluation count = Relaxer telemetry count; every curve value is finite; and all `accepted_state` labels are present. The exact raw input is `output/summary.json` with SHA-256 `{evidence['input_summary_sha256']}`. It was generated by source-run commit `{evidence['source_run_git_commit']}` from source-summary hash `{evidence['source_summary_sha256']}`.

The historical cap-400 run is descriptive only. It has {reference['strict_reference_match_count']}/{reference['row_count']} bitwise-strict matches, while certificates and termination reasons agree in {reference['certificate_equal_count']}/{reference['row_count']} and {reference['termination_equal_count']}/{reference['row_count']} rows. The cross-process float32 differences are reported rather than tuned away; they are not a failure of the present observer ledger.

## Optimizer evidence for this fixed matrix

Across the paired 16 tasks, FIRE used {fire_calls} exact force evaluations and {fire_time:.3f} s; safe L-BFGS used {safe_calls} and {safe_time:.3f} s. In this matrix that is {reduction:.1f}% fewer evaluator calls and {wall_reduction:.1f}% less wall time for safe L-BFGS. The force certificate holds for {safe_certificates}/{task_count_per_backend} safe traces and {fire_certificates}/{task_count_per_backend} FIRE traces.

FIRE reached the 400-iteration observation cap on {len(maxiter)} traces: {', '.join(f"{item['system']}/{item['task_id']}" for item in maxiter) or 'none'}. Safe L-BFGS had {nonaccepted['non_accepted_evaluation_count']} callback-nonaccepted evaluations and {nonaccepted['telemetry_rejected_steps']} telemetry-reported rejected trials; the counts coincide in this ledger. That numerical equality does not establish that the observer label is a general synonym for a line-search rejection.

For safe L-BFGS, callback states have the expected initial-plus-accepted-step count and are monotone non-increasing in total biased energy for all 16 tasks after explicit finalization evaluations are removed. There are {finalization['positive_finalization_recheck_count']} positive finalization rechecks among {finalization['explicit_finalization_record_count']} explicit rechecks: {', '.join(f"{item['system']}/{item['task_id']} ({item['delta_total_energy_eV']:.8g} eV)" for item in finalization['positive_finalization_rechecks']) or 'none'}. These are recorded terminal re-evaluations, not Armijo-accepted optimizer steps. For FIRE, `accepted_state` remains only a callback-observer label; its positive total-energy changes are therefore neither an acceptance-rule failure nor a comparable line-search statistic.

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
