#!/usr/bin/env python3
"""Validate and summarize the fixed safe-L-BFGS history-capacity ledger.

This intentionally produces a compact, deterministic evidence bundle.  It is
not a replayer and does not infer claims beyond the pinned fixed-task ledger.
"""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pamssw.pbc import mic_displacement


SYSTEMS = ("c60", "pdo")
SEEDS = tuple(range(42, 50))
MAXITER = 400
ARMS = (
    ("safe-total-gradient-history10", 10),
    ("safe-total-gradient-history0", 0),
)
ARM_LIMITS = dict(ARMS)
SHA256_RE = re.compile(r"[0-9a-f]{64}")
GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")

TELEMETRY_INTEGER_FIELDS = (
    "accepted_secants",
    "accepted_steps",
    "backend_evaluations",
    "evaluator_calls",
    "explicit_finalization_calls",
    "finalization_requests",
    "line_search_evaluations",
    "mic_branch_resets",
    "rejected_secants",
    "rejected_steps",
    "reporting_cache_hits",
    "reporting_evaluator_calls",
)
TRACE_FLOAT_FIELDS = (
    "active_max_total_force_eV_per_A",
    "bias_energy_eV",
    "softening_energy_eV",
    "total_energy_eV",
    "true_energy_eV",
)


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load {label} JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _require_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _require_int(value: object, label: str, *, minimum: int = 0) -> int:
    if not _is_int(value) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _require_finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} is non-finite")
    return number


def _require_sha256(value: object, label: str) -> str:
    digest = _require_string(value, label)
    if not SHA256_RE.fullmatch(digest):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return digest


def _require_git_sha(value: object, label: str) -> str:
    commit = _require_string(value, label)
    if not GIT_SHA_RE.fullmatch(commit):
        raise ValueError(f"{label} must be a 40-character lowercase Git SHA")
    return commit


def _require_finite_array(value: object, label: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a numeric array") from error
    if shape is not None and array.shape != shape:
        raise ValueError(f"{label} must have shape {shape}, got {array.shape}")
    if array.ndim == 0 or not np.isfinite(array).all():
        raise ValueError(f"{label} contains non-finite values")
    return array


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def _validate_provenance(summary: Mapping[str, Any]) -> tuple[str, dict[tuple[str, str], tuple[np.ndarray, tuple[bool, bool, bool]]]]:
    """Validate historical provenance and recover PBC state from pinned source."""

    if summary.get("schema_version") != 1:
        raise ValueError("summary schema_version must be exactly 1")
    if tuple(summary.get("systems", ())) != SYSTEMS:
        raise ValueError("summary systems must be exactly c60, pdo")
    if summary.get("row_count") != 32 or summary.get("task_count") != 16:
        raise ValueError("summary must declare exactly row_count=32 and task_count=16")

    arms = summary.get("arms")
    expected_arms = [
        {"arm_id": arm_id, "history_limit": limit, "kernel": "safe-lbfgs-total"}
        for arm_id, limit in ARMS
    ]
    if arms != expected_arms:
        raise ValueError("summary arms must be the exact fixed history10/history0 matrix")

    source_path = Path(_require_string(summary.get("source_summary"), "source_summary"))
    source_digest = _require_sha256(summary.get("source_summary_sha256"), "source_summary_sha256")
    if not source_path.is_file():
        raise ValueError("pinned source summary is unavailable")
    if _sha256(source_path) != source_digest:
        raise ValueError("pinned source summary SHA256 mismatch")
    source_summary = _load_json_object(source_path, label="pinned source summary")

    descriptor = _require_mapping(summary.get("safe_kernel_descriptor"), "safe_kernel_descriptor")
    if descriptor.get("optimizer") != "safe-lbfgs-total" or descriptor.get("safe_lbfgs_memory") != 10:
        raise ValueError("safe kernel descriptor is not the reviewed safe-lbfgs-total memory-10 kernel")
    constants = _require_mapping(descriptor.get("kernel_constants"), "safe kernel descriptor kernel_constants")
    if not constants:
        raise ValueError("safe kernel descriptor kernel_constants must not be empty")
    for name, value in constants.items():
        _require_string(name, "safe kernel descriptor constant name")
        _require_finite(value, f"safe kernel descriptor kernel_constants.{name}")
    descriptor_json = json.dumps(descriptor, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if sha256(descriptor_json.encode("utf-8")).hexdigest() != _require_sha256(
        summary.get("safe_kernel_descriptor_sha256"), "safe_kernel_descriptor_sha256"
    ):
        raise ValueError("safe kernel descriptor SHA256 mismatch")

    helpers = _require_mapping(summary.get("runner_helper_provenance"), "runner helper provenance")
    for helper_name in ("fixed_replay_driver", "g1_driver", "trace_recorder"):
        helper = _require_mapping(helpers.get(helper_name), f"runner helper provenance {helper_name}")
        _require_string(helper.get("path"), f"runner helper provenance {helper_name}.path")
        _require_sha256(helper.get("sha256"), f"runner helper provenance {helper_name}.sha256")

    pamssw = _require_mapping(summary.get("pamssw_source_provenance"), "pamssw source provenance")
    _require_string(pamssw.get("source_root"), "pamssw source provenance source_root")
    _require_sha256(pamssw.get("bundle_sha256"), "pamssw source provenance bundle_sha256")
    for field in ("imported_module_paths", "imported_symbol_paths"):
        mapping = _require_mapping(pamssw.get(field), f"pamssw source provenance {field}")
        if not mapping or not all(isinstance(path, str) and path for path in mapping.values()):
            raise ValueError(f"pamssw source provenance {field} must contain non-empty paths")

    cuda = _require_mapping(summary.get("cuda_model_input_provenance"), "CUDA/model/input provenance")
    for field in ("cuda_device", "cuda_version", "device", "model", "torch_version"):
        _require_string(cuda.get(field), f"CUDA/model/input provenance {field}")
    model_digest = _require_sha256(cuda.get("model_sha256"), "CUDA/model/input provenance model_sha256")
    if _require_sha256(cuda.get("model_declared_sha256"), "CUDA/model/input provenance model_declared_sha256") != model_digest:
        raise ValueError("CUDA/model/input provenance model SHA256 values differ")
    source_inputs = _require_mapping(cuda.get("source_system_inputs"), "CUDA/model/input provenance source_system_inputs")
    if set(source_inputs) != set(SYSTEMS):
        raise ValueError("CUDA/model/input provenance must contain exactly c60 and pdo inputs")
    for system in SYSTEMS:
        item = _require_mapping(source_inputs[system], f"CUDA/model/input provenance {system}")
        _require_string(item.get("input"), f"CUDA/model/input provenance {system}.input")
        recorded = _require_sha256(item.get("sha256"), f"CUDA/model/input provenance {system}.sha256")
        if _require_sha256(item.get("declared_sha256"), f"CUDA/model/input provenance {system}.declared_sha256") != recorded:
            raise ValueError(f"CUDA/model/input provenance {system} SHA256 values differ")

    current_commit = _require_git_sha(summary.get("current_git_commit"), "current_git_commit")
    git = _require_mapping(summary.get("git_provenance"), "git provenance")
    if _require_git_sha(git.get("actual_git_commit"), "git provenance actual_git_commit") != current_commit:
        raise ValueError("git provenance actual commit differs from execution commit")
    if _require_git_sha(git.get("expected_git_commit"), "git provenance expected_git_commit") != current_commit:
        raise ValueError("git provenance expected commit differs from execution commit")
    _require_string(git.get("repo_root"), "git provenance repo_root")
    _require_bool(git.get("worktree_clean"), "git provenance worktree_clean")

    states = _source_task_states(source_summary)
    return current_commit, states


def _source_task_states(source_summary: Mapping[str, Any]) -> dict[tuple[str, str], tuple[np.ndarray, tuple[bool, bool, bool]]]:
    entries = source_summary.get("systems")
    if not isinstance(entries, list) or tuple(
        entry.get("system") if isinstance(entry, Mapping) else None for entry in entries
    ) != SYSTEMS:
        raise ValueError("pinned source summary must contain ordered c60 and pdo task states")

    states: dict[tuple[str, str], tuple[np.ndarray, tuple[bool, bool, bool]]] = {}
    for system, entry in zip(SYSTEMS, entries, strict=True):
        assert isinstance(entry, Mapping)
        tasks = entry.get("tasks")
        if not isinstance(tasks, list) or len(tasks) != len(SEEDS):
            raise ValueError(f"pinned source summary {system} must contain exactly 8 tasks")
        for seed, task_entry in zip(SEEDS, tasks, strict=True):
            task = _require_mapping(task_entry, f"pinned source summary {system} task")
            task_id = f"{system}-seed-{seed}-bias-1"
            if task.get("task_id") != task_id or task.get("seed") != seed:
                raise ValueError(f"pinned source summary task identity mismatch for {task_id}")
            payload = _require_mapping(task.get("task"), f"pinned source summary task payload {task_id}")
            state = _require_mapping(payload.get("initial_state"), f"pinned source initial_state {task_id}")
            cell = _require_finite_array(state.get("cell"), f"pinned source cell {task_id}", (3, 3))
            pbc_value = state.get("pbc")
            if not isinstance(pbc_value, list) or len(pbc_value) != 3 or not all(
                isinstance(value, bool) for value in pbc_value
            ):
                raise ValueError(f"pinned source pbc {task_id} must be three booleans")
            pbc = tuple(pbc_value)
            if any(pbc) and abs(float(np.linalg.det(cell))) < 1.0e-12:
                raise ValueError(f"pinned source periodic cell {task_id} is singular")
            states[(system, task_id)] = (cell, pbc)
    return states


def _validate_trace_records(row: Mapping[str, Any], label: str) -> tuple[int, int]:
    trace = row.get("trace_records")
    calls = _require_int(row.get("force_evaluations"), f"{label}.force_evaluations", minimum=1)
    telemetry = _require_mapping(row.get("telemetry"), f"{label}.telemetry")
    evaluator_calls = _require_int(telemetry.get("evaluator_calls"), f"{label}.telemetry.evaluator_calls", minimum=1)
    if not isinstance(trace, list) or len(trace) != calls or calls != evaluator_calls:
        raise ValueError(f"{label} trace_records length must equal force_evaluations and telemetry evaluator_calls")
    accepted = 0
    for index, record in enumerate(trace, start=1):
        item = _require_mapping(record, f"{label}.trace_records[{index}]")
        if item.get("evaluation_index") != index:
            raise ValueError(f"{label}.trace_records evaluation_index must be contiguous from one")
        _require_bool(item.get("accepted_state"), f"{label}.trace_records[{index}].accepted_state")
        accepted += int(item["accepted_state"])
        _require_sha256(item.get("positions_sha256"), f"{label}.trace_records[{index}].positions_sha256")
        for field in TRACE_FLOAT_FIELDS:
            _require_finite(item.get(field), f"{label}.trace_records[{index}].{field}")
    return accepted, len(trace) - accepted


def _validate_row(
    row: object,
    *,
    expected_system: str,
    expected_key_set: set[str],
    states: Mapping[tuple[str, str], tuple[np.ndarray, tuple[bool, bool, bool]]],
    descriptor: Mapping[str, Any],
) -> dict[str, Any]:
    record = _require_mapping(row, "ledger row")
    label = f"{expected_system}/{record.get('task_id', '<missing>')}"
    if set(record) != expected_key_set:
        missing = sorted(expected_key_set - set(record))
        extra = sorted(set(record) - expected_key_set)
        raise ValueError(f"{label} ledger row keys differ; missing={missing}, extra={extra}")
    if record.get("system") != expected_system:
        raise ValueError(f"{label} system does not match its ledger file")
    task_id = _require_string(record.get("task_id"), f"{label}.task_id")
    seed = _require_int(record.get("seed"), f"{label}.seed", minimum=0)
    if seed not in SEEDS or task_id != f"{expected_system}-seed-{seed}-bias-1":
        raise ValueError(f"{label} is not a reviewed fixed task")
    if (expected_system, task_id) not in states:
        raise ValueError(f"{label} lacks a pinned source task state")
    arm_id = _require_string(record.get("arm_id"), f"{label}.arm_id")
    if arm_id not in ARM_LIMITS:
        raise ValueError(f"{label} arm is outside the exact fixed matrix")
    if record.get("kernel") != "safe-lbfgs-total" or record.get("history_limit") != ARM_LIMITS[arm_id]:
        raise ValueError(f"{label} kernel/history limit does not match its fixed arm")
    if record.get("objective_descriptor") != descriptor:
        raise ValueError(f"{label} objective descriptor differs from the reviewed kernel")
    _require_sha256(record.get("task_sha256"), f"{label}.task_sha256")
    if record.get("replay_maxiter") != MAXITER:
        raise ValueError(f"{label}.replay_maxiter must be exactly {MAXITER}")
    _require_int(record.get("source_task_maxiter"), f"{label}.source_task_maxiter", minimum=1)
    certificate = _require_bool(record.get("certificate_satisfied"), f"{label}.certificate_satisfied")
    reason = _require_string(record.get("termination_reason"), f"{label}.termination_reason")
    telemetry = _require_mapping(record.get("telemetry"), f"{label}.telemetry")
    for field in TELEMETRY_INTEGER_FIELDS:
        _require_int(telemetry.get(field), f"{label}.telemetry.{field}")
    _require_finite(telemetry.get("bias_secant_curvature_sum"), f"{label}.telemetry.bias_secant_curvature_sum")
    _require_string(telemetry.get("backend"), f"{label}.telemetry.backend")
    _require_string(telemetry.get("gradient_measure"), f"{label}.telemetry.gradient_measure")
    if telemetry.get("termination_reason") != reason:
        raise ValueError(f"{label} telemetry termination reason differs from row")
    converged = _require_bool(telemetry.get("converged"), f"{label}.telemetry.converged")
    optimizer_success = _require_bool(telemetry.get("optimizer_success"), f"{label}.telemetry.optimizer_success")
    if reason not in {"converged", "maxiter"}:
        raise ValueError(f"{label} has an unsupported termination reason")
    if reason == "converged" and (not certificate or not converged or not optimizer_success):
        raise ValueError(f"{label} converged outcome must satisfy its certificate")
    if reason == "maxiter" and (certificate or converged or optimizer_success):
        raise ValueError(f"{label} finite maxiter outcome must be an unsatisfied certificate")

    purpose_counts = _require_mapping(record.get("purpose_counts"), f"{label}.purpose_counts")
    if not purpose_counts:
        raise ValueError(f"{label}.purpose_counts must not be empty")
    for purpose, count in purpose_counts.items():
        _require_string(purpose, f"{label}.purpose_counts key")
        _require_int(count, f"{label}.purpose_counts[{purpose}]")
    if sum(purpose_counts.values()) != record["force_evaluations"]:
        raise ValueError(f"{label}.purpose_counts must sum to force_evaluations")
    callbacks = record.get("accepted_callback_hashes")
    if not isinstance(callbacks, list) or not all(
        isinstance(value, str) and SHA256_RE.fullmatch(value) for value in callbacks
    ):
        raise ValueError(f"{label}.accepted_callback_hashes must be SHA256 strings")

    accepted_records, nonaccepted_records = _validate_trace_records(record, label)
    endpoint = _require_mapping(record.get("endpoint"), f"{label}.endpoint")
    if set(endpoint) != {"biased_energy_eV", "max_active_atom_force_eV_per_A", "positions"}:
        raise ValueError(f"{label}.endpoint must contain exactly energy, force residual, and positions")
    endpoint_positions = _require_finite_array(endpoint.get("positions"), f"{label}.endpoint.positions")
    if endpoint_positions.ndim != 2 or endpoint_positions.shape[1] != 3:
        raise ValueError(f"{label}.endpoint.positions must have shape (n_atoms, 3)")
    _require_finite(endpoint.get("biased_energy_eV"), f"{label}.endpoint.biased_energy_eV")
    _require_finite(endpoint.get("max_active_atom_force_eV_per_A"), f"{label}.endpoint.max_active_atom_force_eV_per_A")
    _require_finite(record.get("wall_time_s"), f"{label}.wall_time_s")

    normalized = dict(record)
    normalized["_accepted_trace_records"] = accepted_records
    normalized["_nonaccepted_trace_records"] = nonaccepted_records
    normalized["_endpoint_positions"] = endpoint_positions
    return normalized


def _validated_rows(
    ledger_dir: Path,
    summary: Mapping[str, Any],
    states: Mapping[tuple[str, str], tuple[np.ndarray, tuple[bool, bool, bool]]],
) -> list[dict[str, Any]]:
    expected_keys: set[str] | None = None
    rows: list[dict[str, Any]] = []
    descriptor = _require_mapping(summary.get("safe_kernel_descriptor"), "safe_kernel_descriptor")
    for system in SYSTEMS:
        payload = _load_json_object(ledger_dir / f"{system}.json", label=f"{system} ledger")
        expected_task_ids = [f"{system}-seed-{seed}-bias-1" for seed in SEEDS]
        if (
            set(payload) != {"rows", "system", "task_ids"}
            or payload.get("system") != system
            or payload.get("task_ids") != expected_task_ids
            or not isinstance(payload["rows"], list)
            or len(payload["rows"]) != 16
        ):
            raise ValueError(f"{system} ledger violates the fixed task-arm matrix: expected exactly 16 rows")
        for raw_row in payload["rows"]:
            if expected_keys is None:
                candidate = _require_mapping(raw_row, "ledger row")
                expected_keys = set(candidate)
            rows.append(
                _validate_row(
                    raw_row,
                    expected_system=system,
                    expected_key_set=expected_keys,
                    states=states,
                    descriptor=descriptor,
                )
            )
    if len(rows) != 32:
        raise ValueError("ledger violates the fixed task-arm matrix: expected exactly 32 rows")
    expected_matrix = {
        (system, f"{system}-seed-{seed}-bias-1", arm_id)
        for system in SYSTEMS
        for seed in SEEDS
        for arm_id, _ in ARMS
    }
    actual_matrix = {(row["system"], row["task_id"], row["arm_id"]) for row in rows}
    if actual_matrix != expected_matrix or len(actual_matrix) != len(rows):
        raise ValueError("ledger violates the exact fixed task-arm matrix")
    return rows


def _validate_summary_outcomes(summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
    satisfied = sum(bool(row["certificate_satisfied"]) for row in rows)
    if _require_bool(summary.get("certificate_all_satisfied"), "certificate_all_satisfied") != (satisfied == len(rows)):
        raise ValueError("summary certificate_all_satisfied disagrees with ledger rows")
    if _require_int(summary.get("certificate_unsatisfied_count"), "certificate_unsatisfied_count") != len(rows) - satisfied:
        raise ValueError("summary certificate_unsatisfied_count disagrees with ledger rows")
    reasons = Counter(str(row["termination_reason"]) for row in rows)
    if summary.get("termination_reason_counts") != dict(sorted(reasons.items())):
        raise ValueError("summary termination_reason_counts disagrees with ledger rows")
    # The runner's total includes orchestration overhead.  Per-row wall time is
    # the only arm-comparable cost reported below, so retain the summary total
    # only as a finite execution-provenance field rather than forcing equality.
    _require_finite(summary.get("wall_time_total_s"), "wall_time_total_s")
    _require_string(summary.get("claim_ceiling"), "claim_ceiling")


def _sum(rows: Sequence[Mapping[str, Any]], path: Sequence[str]) -> int:
    total = 0
    for row in rows:
        value: Any = row
        for field in path:
            value = value[field]
        total += int(value)
    return total


def _by_system_arm(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for system in SYSTEMS:
        grouped[system] = {}
        for arm_id, _ in ARMS:
            selected = [row for row in rows if row["system"] == system and row["arm_id"] == arm_id]
            reason_counts = Counter(str(row["termination_reason"]) for row in selected)
            satisfied = sum(bool(row["certificate_satisfied"]) for row in selected)
            grouped[system][arm_id] = {
                "count": len(selected),
                "force_evaluations": {"sum": _sum(selected, ("force_evaluations",))},
                "wall_time_s": {"sum": sum(float(row["wall_time_s"]) for row in selected)},
                "certificate": {
                    "satisfied_count": satisfied,
                    "unsatisfied_count": len(selected) - satisfied,
                },
                "termination_reasons": dict(sorted(reason_counts.items())),
                "telemetry": {
                    "evaluator_calls": _sum(selected, ("telemetry", "evaluator_calls")),
                    "accepted_steps": _sum(selected, ("telemetry", "accepted_steps")),
                    "rejected_steps": _sum(selected, ("telemetry", "rejected_steps")),
                    "line_search_evaluations": _sum(selected, ("telemetry", "line_search_evaluations")),
                    "accepted_secants": _sum(selected, ("telemetry", "accepted_secants")),
                    "rejected_secants": _sum(selected, ("telemetry", "rejected_secants")),
                    "explicit_finalization_calls": _sum(selected, ("telemetry", "explicit_finalization_calls")),
                    "finalization_requests": _sum(selected, ("telemetry", "finalization_requests")),
                    "accepted_state_records": _sum(selected, ("_accepted_trace_records",)),
                    "nonaccepted_state_records": _sum(selected, ("_nonaccepted_trace_records",)),
                },
                "mic": {"branch_resets": _sum(selected, ("telemetry", "mic_branch_resets"))},
            }
    return grouped


def _paired_tasks(
    rows: Sequence[Mapping[str, Any]],
    states: Mapping[tuple[str, str], tuple[np.ndarray, tuple[bool, bool, bool]]],
) -> list[dict[str, Any]]:
    by_key = {(row["system"], row["task_id"], row["arm_id"]): row for row in rows}
    pairs: list[dict[str, Any]] = []
    for system in SYSTEMS:
        for seed in SEEDS:
            task_id = f"{system}-seed-{seed}-bias-1"
            history10 = by_key[(system, task_id, "safe-total-gradient-history10")]
            history0 = by_key[(system, task_id, "safe-total-gradient-history0")]
            if history10["task_sha256"] != history0["task_sha256"]:
                raise ValueError(f"{system}/{task_id} task SHA256 must match across the paired arms")
            cell, pbc = states[(system, task_id)]
            positions10 = np.asarray(history10["_endpoint_positions"], dtype=float)
            positions0 = np.asarray(history0["_endpoint_positions"], dtype=float)
            if positions10.shape != positions0.shape:
                raise ValueError(f"{system}/{task_id} paired endpoint atom counts differ")
            displacement = mic_displacement(positions0, positions10, cell, pbc)
            norms = np.linalg.norm(displacement, axis=1)
            pair = {
                "system": system,
                "task_id": task_id,
                "seed": seed,
                "task_sha256": history10["task_sha256"],
                "history10": _pair_arm(history10),
                "history0": _pair_arm(history0),
                "endpoint_delta": {
                    "max_mic_displacement_A": float(np.max(norms)),
                    "rms_mic_displacement_A": float(np.sqrt(np.mean(np.square(norms)))),
                    "pbc": list(pbc),
                },
                "force_residual": {
                    "history10_eV_per_A": float(history10["endpoint"]["max_active_atom_force_eV_per_A"]),
                    "history0_eV_per_A": float(history0["endpoint"]["max_active_atom_force_eV_per_A"]),
                    "history0_minus_history10_eV_per_A": float(
                        history0["endpoint"]["max_active_atom_force_eV_per_A"]
                        - history10["endpoint"]["max_active_atom_force_eV_per_A"]
                    ),
                },
            }
            pairs.append(pair)
    return pairs


def _pair_arm(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "calls": int(row["force_evaluations"]),
        "wall_time_s": float(row["wall_time_s"]),
        "endpoint_biased_energy_eV": float(row["endpoint"]["biased_energy_eV"]),
        "certificate_satisfied": bool(row["certificate_satisfied"]),
        "termination_reason": str(row["termination_reason"]),
    }


def build_evidence(ledger_dir: Path) -> dict[str, Any]:
    ledger_dir = ledger_dir.resolve()
    summary_path = ledger_dir / "summary.json"
    summary = _load_json_object(summary_path, label="summary")
    execution_commit, states = _validate_provenance(summary)
    rows = _validated_rows(ledger_dir, summary, states)
    _validate_summary_outcomes(summary, rows)
    reason_counts = Counter(str(row["termination_reason"]) for row in rows)
    satisfied = sum(bool(row["certificate_satisfied"]) for row in rows)
    raw_files = {
        name: {"sha256": _sha256(ledger_dir / name)}
        for name in ("summary.json", "c60.json", "pdo.json")
    }
    return {
        "schema_version": 1,
        "raw_files": raw_files,
        "ledger": {"row_count": len(rows), "task_count": 16},
        "execution": {"git_commit": execution_commit},
        "provenance": {
            "source_summary": summary["source_summary"],
            "source_summary_sha256": summary["source_summary_sha256"],
            "safe_kernel_descriptor": summary["safe_kernel_descriptor"],
            "safe_kernel_descriptor_sha256": summary["safe_kernel_descriptor_sha256"],
            "runner_helper_provenance": summary["runner_helper_provenance"],
            "pamssw_source_provenance": summary["pamssw_source_provenance"],
            "cuda_model_input_provenance": summary["cuda_model_input_provenance"],
            "git_provenance": summary["git_provenance"],
        },
        "certificate": {
            "all_satisfied": satisfied == len(rows),
            "satisfied_count": satisfied,
            "unsatisfied_count": len(rows) - satisfied,
            "by_termination_reason": {
                reason: {
                    "count": count,
                    "certificate_satisfied_count": sum(
                        bool(row["certificate_satisfied"])
                        for row in rows
                        if row["termination_reason"] == reason
                    ),
                }
                for reason, count in sorted(reason_counts.items())
            },
        },
        "aggregates": {"by_system_arm": _by_system_arm(rows)},
        "paired_tasks": _paired_tasks(rows, states),
    }


def _conclusion(evidence: Mapping[str, Any]) -> str:
    certificate = _require_mapping(evidence["certificate"], "evidence certificate")
    aggregate = _require_mapping(_require_mapping(evidence["aggregates"], "evidence aggregates")["by_system_arm"], "by_system_arm")
    history0 = "safe-total-gradient-history0"
    history10 = "safe-total-gradient-history10"
    history0_calls = sum(int(aggregate[system][history0]["force_evaluations"]["sum"]) for system in SYSTEMS)
    history10_calls = sum(int(aggregate[system][history10]["force_evaluations"]["sum"]) for system in SYSTEMS)
    history0_wall = sum(float(aggregate[system][history0]["wall_time_s"]["sum"]) for system in SYSTEMS)
    history10_wall = sum(float(aggregate[system][history10]["wall_time_s"]["sum"]) for system in SYSTEMS)
    history0_secants = sum(int(aggregate[system][history0]["telemetry"]["accepted_secants"]) for system in SYSTEMS)
    lines = [
        "# Safe L-BFGS history-capacity ablation — evidence-limited conclusion",
        "",
        "## Certificate outcome",
        "",
        f"The frozen 16-task C60/PdO matrix produced {certificate['satisfied_count']}/32 certificate-satisfied rows and {certificate['unsatisfied_count']}/32 finite `maxiter` rows.  A `certificate_satisfied: false` / `maxiter` record is retained as a valid, incomplete outcome; it is not converted into convergence.",
        "",
        "## Cost outcome",
        "",
        f"Across this fixed matrix, history 10 used {history10_calls} evaluator calls and {history10_wall:.6f} s, while history 0 used {history0_calls} calls and {history0_wall:.6f} s.  These are recorded replay costs, not an optimization score.",
        "",
        "## Fixed-matrix interpretation ceiling",
        "",
        "The positive retained-history contribution is limited to this fixed matrix: the history-10 arm preserves the certificate-qualified outcomes and incurs fewer recorded evaluator calls than the history-0 arm.  This is not a generic result, a same-basin equivalence result, a full-SSW performance result, or an inferential claim.",
        "",
        f"The history-0 arm recorded {history0_secants} accepted secants, but those secants are not retained history.  Accepted-state and explicit-finalization records are reported only as accounting statistics; no trace curve or endpoint-equivalence inference is made.",
        "",
        "The evidence remains limited to the pinned source tasks, CUDA/model/input provenance, and reviewed safe-L-BFGS kernel recorded in `evidence.json`.",
        "",
    ]
    return "\n".join(lines)


def write_artifacts(evidence: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evidence.json").write_text(_canonical_json(evidence), encoding="utf-8")
    (output_dir / "conclusion.md").write_text(_conclusion(evidence), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    try:
        evidence = build_evidence(arguments.ledger_dir)
        write_artifacts(evidence, arguments.output_dir)
    except (OSError, TypeError, ValueError, KeyError) as error:
        print(f"analysis failed: {error}", file=sys.stderr)
        return 2
    print(
        f"wrote evidence={arguments.output_dir / 'evidence.json'} "
        f"conclusion={arguments.output_dir / 'conclusion.md'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
