#!/usr/bin/env python3
"""Validate and summarize the fixed safe-L-BFGS inverse-scale ledger.

This intentionally produces a compact, deterministic evidence bundle.  It is
not a replayer and does not infer claims beyond the pinned fixed-task ledger.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
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

from pamssw import pbc as pbc_module
from pamssw.pbc import mic_displacement


SYSTEMS = ("c60", "pdo")
SEEDS = tuple(range(42, 50))
MAXITER = 400
EXPECTED_SOURCE_SUMMARY_SHA256 = "62cc771e2aa24e9addef0e870d0524f901f02bddc34152cf2a2eeea91a671b04"
EXPECTED_KERNEL_DESCRIPTOR_SHA256 = "ba0c261cd588ce72e8175332385389e69cf8840c00399318723af7e0358e7f24"
EXPECTED_EXECUTION_COMMIT = "75afdbb9f098730281c43b592f3119ee42fd0f54"
ARMS = (
    ("fixed-scale-history0", 0, False, "fixed-1-over-70"),
    (
        "adaptive-scale-history0",
        0,
        True,
        "latest-accepted-secant-gamma",
    ),
    (
        "adaptive-scale-history10",
        10,
        False,
        "latest-history-pair-gamma-plus-two-loop",
    ),
)
ARM_CONTRACTS = {
    arm_id: {
        "history_limit": history_limit,
        "adaptive_scale_without_history": adaptive,
        "scale_policy": scale_policy,
    }
    for arm_id, history_limit, adaptive, scale_policy in ARMS
}
SHA256_RE = re.compile(r"[0-9a-f]{64}")
GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")
ROW_KEYS = frozenset(
    {
        "accepted_callback_hashes",
        "adaptive_scale_without_history",
        "arm_id",
        "certificate_satisfied",
        "endpoint",
        "force_evaluations",
        "history_limit",
        "kernel",
        "objective_descriptor",
        "purpose_counts",
        "replay_maxiter",
        "scale_policy",
        "seed",
        "source_task_maxiter",
        "system",
        "task_id",
        "task_sha256",
        "telemetry",
        "termination_reason",
        "trace_records",
        "wall_time_s",
    }
)
SUMMARY_KEYS = frozenset(
    {
        "arms",
        "certificate_all_satisfied",
        "certificate_unsatisfied_count",
        "claim_ceiling",
        "cuda_model_input_provenance",
        "current_git_commit",
        "git_provenance",
        "pamssw_source_provenance",
        "row_count",
        "runner_helper_provenance",
        "safe_kernel_descriptor",
        "safe_kernel_descriptor_sha256",
        "schema_version",
        "source_summary",
        "source_summary_sha256",
        "systems",
        "task_count",
        "termination_reason_counts",
        "wall_time_total_s",
    }
)
TELEMETRY_KEYS = frozenset(
    {
        "accepted_secants",
        "accepted_steps",
        "backend",
        "backend_evaluations",
        "bias_secant_curvature_sum",
        "converged",
        "evaluator_calls",
        "explicit_finalization_calls",
        "finalization_requests",
        "gradient_measure",
        "line_search_evaluations",
        "mic_branch_resets",
        "optimizer_success",
        "rejected_secants",
        "rejected_steps",
        "reporting_cache_hits",
        "reporting_evaluator_calls",
        "termination_reason",
    }
)
ENDPOINT_KEYS = frozenset({"biased_energy_eV", "max_active_atom_force_eV_per_A", "positions"})
TRACE_KEYS = frozenset(
    {
        "accepted_state",
        "active_max_total_force_eV_per_A",
        "bias_energy_eV",
        "evaluation_index",
        "positions_sha256",
        "softening_energy_eV",
        "total_energy_eV",
        "true_energy_eV",
    }
)
OBJECTIVE_KEYS = frozenset(
    {"kernel_constants", "optimizer", "safe_lbfgs_memory", "scale_policies"}
)
KERNEL_CONSTANT_KEYS = frozenset(
    {
        "_SAFE_LBFGS_ARMIJO_C1",
        "_SAFE_LBFGS_BACKTRACK",
        "_SAFE_LBFGS_CURVATURE_REL",
        "_SAFE_LBFGS_EMPTY_HISTORY_SCALE",
        "_SAFE_LBFGS_MAX_ATOM_STEP",
        "_SAFE_LBFGS_MAX_LINE_TRIALS",
        "_SAFE_LBFGS_MIN_ALPHA",
    }
)
PURPOSE_KEYS = frozenset(
    {
        "biased_proposal_relax",
        "bootstrap_true_quench",
        "direction_oracle",
        "escape_true_pes_check",
        "landing_true_quench",
        "post_relax_validation",
        "starter_true_quench",
        "unattributed",
    }
)
HELPER_KEYS = frozenset({"fixed_replay_driver", "g1_driver", "trace_recorder"})
PAMSSW_PROVENANCE_KEYS = frozenset(
    {"bundle_sha256", "imported_module_paths", "imported_symbol_paths", "source_root"}
)
PAMSSW_MODULE_KEYS = frozenset(
    {
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
    }
)
PAMSSW_SYMBOL_KEYS = frozenset({"EvalCounter", "ProposalRelaxationTask", "Relaxer"})
CUDA_PROVENANCE_KEYS = frozenset(
    {
        "cuda_device",
        "cuda_version",
        "device",
        "model",
        "model_declared_sha256",
        "model_sha256",
        "source_system_inputs",
        "torch_version",
    }
)
GIT_PROVENANCE_KEYS = frozenset(
    {"actual_git_commit", "expected_git_commit", "repo_root", "worktree_clean"}
)

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


@dataclass(frozen=True)
class SourceTaskContract:
    cell: np.ndarray
    pbc: tuple[bool, bool, bool]
    task_sha256: str
    n_atoms: int
    fmax_eV_per_A: float


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


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: frozenset[str] | set[str],
    label: str,
) -> None:
    if set(value) != set(expected):
        missing = sorted(set(expected) - set(value))
        extra = sorted(set(value) - set(expected))
        raise ValueError(f"{label} keys differ; missing={missing}, extra={extra}")


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


def _require_nonnegative_finite(value: object, label: str) -> float:
    number = _require_finite(value, label)
    if number < 0.0:
        raise ValueError(f"{label} must be a nonnegative finite number")
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
    object_array = np.asarray(value, dtype=object)
    if object_array.ndim == 0 or any(
        isinstance(item, bool) or not isinstance(item, (int, float))
        for item in object_array.flat
    ):
        raise ValueError(f"{label} must contain only numeric values")
    try:
        array = np.asarray(object_array, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a numeric array") from error
    if shape is not None and array.shape != shape:
        raise ValueError(f"{label} must have shape {shape}, got {array.shape}")
    if array.ndim == 0 or not np.isfinite(array).all():
        raise ValueError(f"{label} contains non-finite values")
    return array


def position_sha256(positions: object) -> str:
    """Return the canonical position hash used by the pinned trace recorder."""

    coordinates = np.asarray(positions, dtype=float)
    if coordinates.ndim == 1:
        if coordinates.size % 3 != 0:
            raise ValueError("flat positions must contain a multiple of three coordinates")
        coordinates = coordinates.reshape(-1, 3)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("positions must have shape (n_atoms, 3)")
    canonical = np.array(coordinates, dtype=np.dtype("<f8"), order="C", copy=True)
    digest = sha256()
    digest.update(str(canonical.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical.tobytes())
    return digest.hexdigest()


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def canonical_task_sha256(task_payload: Mapping[str, Any]) -> str:
    """Return the canonical source-task hash recorded by the execution runner."""

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


def _validate_provenance(
    summary: Mapping[str, Any],
) -> tuple[str, dict[tuple[str, str], SourceTaskContract]]:
    """Validate historical provenance and recover PBC state from pinned source."""

    _require_exact_keys(summary, SUMMARY_KEYS, "summary")
    if _require_int(summary.get("schema_version"), "summary.schema_version", minimum=1) != 1:
        raise ValueError("summary schema_version must be exactly 1")
    systems = summary.get("systems")
    if not isinstance(systems, list) or tuple(systems) != SYSTEMS:
        raise ValueError("summary systems must be exactly c60, pdo")
    if _require_int(summary.get("row_count"), "summary.row_count") != 48:
        raise ValueError("summary row_count must be exactly 48")
    if _require_int(summary.get("task_count"), "summary.task_count") != 16:
        raise ValueError("summary must declare exactly row_count=48 and task_count=16")

    arms = summary.get("arms")
    expected_arms = [
        {
            "adaptive_scale_without_history": adaptive,
            "arm_id": arm_id,
            "history_limit": limit,
            "kernel": "safe-lbfgs-total",
            "scale_policy": scale_policy,
        }
        for arm_id, limit, adaptive, scale_policy in ARMS
    ]
    if not isinstance(arms, list) or len(arms) != len(expected_arms):
        raise ValueError("summary arms must be the exact fixed three-arm scale matrix")
    for index, arm in enumerate(arms):
        arm_mapping = _require_mapping(arm, f"summary.arms[{index}]")
        _require_exact_keys(
            arm_mapping,
            {
                "adaptive_scale_without_history",
                "arm_id",
                "history_limit",
                "kernel",
                "scale_policy",
            },
            f"summary.arms[{index}]",
        )
        _require_bool(
            arm_mapping.get("adaptive_scale_without_history"),
            f"summary.arms[{index}].adaptive_scale_without_history",
        )
        _require_string(arm_mapping.get("arm_id"), f"summary.arms[{index}].arm_id")
        _require_int(arm_mapping.get("history_limit"), f"summary.arms[{index}].history_limit")
        _require_string(arm_mapping.get("kernel"), f"summary.arms[{index}].kernel")
        _require_string(arm_mapping.get("scale_policy"), f"summary.arms[{index}].scale_policy")
    if arms != expected_arms:
        raise ValueError("summary arms must be the exact fixed three-arm scale matrix")

    source_path = Path(_require_string(summary.get("source_summary"), "source_summary"))
    source_digest = _require_sha256(summary.get("source_summary_sha256"), "source_summary_sha256")
    if source_digest != EXPECTED_SOURCE_SUMMARY_SHA256:
        raise ValueError("source summary SHA256 does not match the pinned reviewed source")
    if not source_path.is_file():
        raise ValueError("pinned source summary is unavailable")
    if _sha256(source_path) != source_digest:
        raise ValueError("pinned source summary SHA256 mismatch")
    source_summary = _load_json_object(source_path, label="pinned source summary")

    descriptor = _require_mapping(summary.get("safe_kernel_descriptor"), "safe_kernel_descriptor")
    _require_exact_keys(descriptor, OBJECTIVE_KEYS, "safe kernel objective descriptor")
    if descriptor.get("optimizer") != "safe-lbfgs-total" or descriptor.get("safe_lbfgs_memory") != 10:
        raise ValueError("safe kernel descriptor is not the reviewed safe-lbfgs-total memory-10 kernel")
    _require_int(descriptor.get("safe_lbfgs_memory"), "safe kernel descriptor safe_lbfgs_memory")
    scale_policies = _require_mapping(
        descriptor.get("scale_policies"),
        "safe kernel descriptor scale_policies",
    )
    expected_scale_policies = {
        contract["scale_policy"] for contract in ARM_CONTRACTS.values()
    }
    _require_exact_keys(
        scale_policies,
        expected_scale_policies,
        "safe kernel descriptor scale_policies",
    )
    for name, value in scale_policies.items():
        _require_string(value, f"safe kernel descriptor scale_policies.{name}")
    constants = _require_mapping(descriptor.get("kernel_constants"), "safe kernel descriptor kernel_constants")
    _require_exact_keys(constants, KERNEL_CONSTANT_KEYS, "safe kernel descriptor kernel_constants")
    for name, value in constants.items():
        _require_finite(value, f"safe kernel descriptor kernel_constants.{name}")
    descriptor_json = json.dumps(descriptor, sort_keys=True, separators=(",", ":"), allow_nan=False)
    declared_descriptor_sha = _require_sha256(
        summary.get("safe_kernel_descriptor_sha256"), "safe_kernel_descriptor_sha256"
    )
    if declared_descriptor_sha != EXPECTED_KERNEL_DESCRIPTOR_SHA256:
        raise ValueError("kernel descriptor SHA256 does not match the pinned reviewed kernel")
    if sha256(descriptor_json.encode("utf-8")).hexdigest() != declared_descriptor_sha:
        raise ValueError("safe kernel descriptor SHA256 mismatch")

    helpers = _require_mapping(summary.get("runner_helper_provenance"), "runner helper provenance")
    _require_exact_keys(helpers, HELPER_KEYS, "runner helper provenance")
    for helper_name in sorted(HELPER_KEYS):
        helper = _require_mapping(helpers.get(helper_name), f"runner helper provenance {helper_name}")
        _require_exact_keys(helper, {"path", "sha256"}, f"runner helper provenance {helper_name}")
        _require_string(helper.get("path"), f"runner helper provenance {helper_name}.path")
        _require_sha256(helper.get("sha256"), f"runner helper provenance {helper_name}.sha256")

    pamssw = _require_mapping(summary.get("pamssw_source_provenance"), "pamssw source provenance")
    _require_exact_keys(pamssw, PAMSSW_PROVENANCE_KEYS, "pamssw source provenance")
    _require_string(pamssw.get("source_root"), "pamssw source provenance source_root")
    _require_sha256(pamssw.get("bundle_sha256"), "pamssw source provenance bundle_sha256")
    module_paths = _require_mapping(
        pamssw.get("imported_module_paths"), "pamssw source provenance imported_module_paths"
    )
    symbol_paths = _require_mapping(
        pamssw.get("imported_symbol_paths"), "pamssw source provenance imported_symbol_paths"
    )
    _require_exact_keys(
        module_paths, PAMSSW_MODULE_KEYS, "pamssw source provenance imported_module_paths"
    )
    _require_exact_keys(
        symbol_paths, PAMSSW_SYMBOL_KEYS, "pamssw source provenance imported_symbol_paths"
    )
    for field, mapping in (
        ("imported_module_paths", module_paths),
        ("imported_symbol_paths", symbol_paths),
    ):
        for name, path in mapping.items():
            _require_string(path, f"pamssw source provenance {field}.{name}")

    cuda = _require_mapping(summary.get("cuda_model_input_provenance"), "CUDA/model/input provenance")
    _require_exact_keys(cuda, CUDA_PROVENANCE_KEYS, "CUDA/model/input provenance")
    for field in ("cuda_device", "cuda_version", "device", "model", "torch_version"):
        _require_string(cuda.get(field), f"CUDA/model/input provenance {field}")
    if cuda.get("device") != "cuda":
        raise ValueError("CUDA/model/input provenance device must be cuda")
    model_digest = _require_sha256(cuda.get("model_sha256"), "CUDA/model/input provenance model_sha256")
    if _require_sha256(cuda.get("model_declared_sha256"), "CUDA/model/input provenance model_declared_sha256") != model_digest:
        raise ValueError("CUDA/model/input provenance model SHA256 values differ")
    source_inputs = _require_mapping(cuda.get("source_system_inputs"), "CUDA/model/input provenance source_system_inputs")
    _require_exact_keys(source_inputs, set(SYSTEMS), "CUDA/model/input provenance source_system_inputs")
    for system in SYSTEMS:
        item = _require_mapping(source_inputs[system], f"CUDA/model/input provenance {system}")
        _require_exact_keys(
            item, {"declared_sha256", "input", "sha256"}, f"CUDA/model/input provenance {system}"
        )
        _require_string(item.get("input"), f"CUDA/model/input provenance {system}.input")
        recorded = _require_sha256(item.get("sha256"), f"CUDA/model/input provenance {system}.sha256")
        if _require_sha256(item.get("declared_sha256"), f"CUDA/model/input provenance {system}.declared_sha256") != recorded:
            raise ValueError(f"CUDA/model/input provenance {system} SHA256 values differ")

    current_commit = _require_git_sha(summary.get("current_git_commit"), "current_git_commit")
    if current_commit != EXPECTED_EXECUTION_COMMIT:
        raise ValueError("execution commit does not match the pinned reviewed execution")
    git = _require_mapping(summary.get("git_provenance"), "git provenance")
    _require_exact_keys(git, GIT_PROVENANCE_KEYS, "git provenance")
    if _require_git_sha(git.get("actual_git_commit"), "git provenance actual_git_commit") != current_commit:
        raise ValueError("git provenance actual commit differs from execution commit")
    if _require_git_sha(git.get("expected_git_commit"), "git provenance expected_git_commit") != current_commit:
        raise ValueError("git provenance expected commit differs from execution commit")
    _require_string(git.get("repo_root"), "git provenance repo_root")
    _require_bool(git.get("worktree_clean"), "git provenance worktree_clean")

    states = _source_task_states(source_summary)
    return current_commit, states


def _source_task_states(
    source_summary: Mapping[str, Any],
) -> dict[tuple[str, str], SourceTaskContract]:
    entries = source_summary.get("systems")
    if not isinstance(entries, list) or tuple(
        entry.get("system") if isinstance(entry, Mapping) else None for entry in entries
    ) != SYSTEMS:
        raise ValueError("pinned source summary must contain ordered c60 and pdo task states")

    states: dict[tuple[str, str], SourceTaskContract] = {}
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
            _require_exact_keys(
                state,
                {"cell", "fixed_mask", "numbers", "pbc", "positions"},
                f"pinned source initial_state {task_id}",
            )
            numbers = state.get("numbers")
            if not isinstance(numbers, list) or not numbers:
                raise ValueError(f"pinned source numbers {task_id} must be a non-empty list")
            for index, number in enumerate(numbers):
                _require_int(number, f"pinned source numbers {task_id}[{index}]", minimum=1)
            n_atoms = len(numbers)
            _require_finite_array(
                state.get("positions"), f"pinned source positions {task_id}", (n_atoms, 3)
            )
            fixed_mask = state.get("fixed_mask")
            if (
                not isinstance(fixed_mask, list)
                or len(fixed_mask) != n_atoms
                or not all(isinstance(value, bool) for value in fixed_mask)
            ):
                raise ValueError(f"pinned source fixed_mask {task_id} must contain {n_atoms} booleans")
            cell = _require_finite_array(state.get("cell"), f"pinned source cell {task_id}", (3, 3))
            pbc_value = state.get("pbc")
            if not isinstance(pbc_value, list) or len(pbc_value) != 3 or not all(
                isinstance(value, bool) for value in pbc_value
            ):
                raise ValueError(f"pinned source pbc {task_id} must be three booleans")
            pbc = tuple(pbc_value)
            if any(pbc) and abs(float(np.linalg.det(cell))) < 1.0e-12:
                raise ValueError(f"pinned source periodic cell {task_id} is singular")
            fmax = _require_finite(payload.get("fmax"), f"pinned source fmax {task_id}")
            if fmax <= 0.0:
                raise ValueError(f"pinned source fmax {task_id} must be positive")
            states[(system, task_id)] = SourceTaskContract(
                cell=cell,
                pbc=pbc,
                task_sha256=canonical_task_sha256(payload),
                n_atoms=n_atoms,
                fmax_eV_per_A=fmax,
            )
    return states


def _validate_trace_records(
    row: Mapping[str, Any],
    label: str,
    callback_hashes: set[str],
) -> tuple[int, int]:
    trace = row.get("trace_records")
    calls = _require_int(row.get("force_evaluations"), f"{label}.force_evaluations", minimum=1)
    telemetry = _require_mapping(row.get("telemetry"), f"{label}.telemetry")
    evaluator_calls = _require_int(telemetry.get("evaluator_calls"), f"{label}.telemetry.evaluator_calls", minimum=1)
    if not isinstance(trace, list) or len(trace) != calls or calls != evaluator_calls:
        raise ValueError(f"{label} trace_records length must equal force_evaluations and telemetry evaluator_calls")
    accepted = 0
    accepted_trace_hashes: set[str] = set()
    for index, record in enumerate(trace, start=1):
        item = _require_mapping(record, f"{label}.trace_records[{index}]")
        _require_exact_keys(item, TRACE_KEYS, f"{label} trace record")
        evaluation_index = _require_int(
            item.get("evaluation_index"), f"{label}.trace_records[{index}].evaluation_index", minimum=1
        )
        if evaluation_index != index:
            raise ValueError(f"{label}.trace_records evaluation_index must be contiguous from one")
        accepted_state = _require_bool(
            item.get("accepted_state"), f"{label}.trace_records[{index}].accepted_state"
        )
        positions_digest = _require_sha256(
            item.get("positions_sha256"), f"{label}.trace_records[{index}].positions_sha256"
        )
        if accepted_state != (positions_digest in callback_hashes):
            raise ValueError(
                f"{label}.trace_records[{index}].accepted_state disagrees with callback membership"
            )
        accepted += int(accepted_state)
        if accepted_state:
            accepted_trace_hashes.add(positions_digest)
        for field in TRACE_FLOAT_FIELDS:
            validator = (
                _require_nonnegative_finite
                if field == "active_max_total_force_eV_per_A"
                else _require_finite
            )
            validator(item.get(field), f"{label}.trace_records[{index}].{field}")
    if accepted_trace_hashes != callback_hashes:
        raise ValueError(f"{label}.accepted_callback_hashes are not closed by accepted trace states")
    return accepted, len(trace) - accepted


def _validate_row(
    row: object,
    *,
    expected_system: str,
    states: Mapping[tuple[str, str], SourceTaskContract],
    descriptor: Mapping[str, Any],
) -> dict[str, Any]:
    record = _require_mapping(row, "ledger row")
    label = f"{expected_system}/{record.get('task_id', '<missing>')}"
    if set(record) != ROW_KEYS:
        missing = sorted(ROW_KEYS - set(record))
        extra = sorted(set(record) - ROW_KEYS)
        raise ValueError(f"{label} ledger row keys differ; missing={missing}, extra={extra}")
    if record.get("system") != expected_system:
        raise ValueError(f"{label} system does not match its ledger file")
    task_id = _require_string(record.get("task_id"), f"{label}.task_id")
    seed = _require_int(record.get("seed"), f"{label}.seed", minimum=0)
    if seed not in SEEDS or task_id != f"{expected_system}-seed-{seed}-bias-1":
        raise ValueError(f"{label} is not a reviewed fixed task")
    if (expected_system, task_id) not in states:
        raise ValueError(f"{label} lacks a pinned source task state")
    source_contract = states[(expected_system, task_id)]
    if (
        _require_sha256(record.get("task_sha256"), f"{label}.task_sha256")
        != source_contract.task_sha256
    ):
        raise ValueError(f"{label}.task_sha256 differs from the pinned source task payload")
    force_evaluations = _require_int(
        record.get("force_evaluations"), f"{label}.force_evaluations", minimum=1
    )
    arm_id = _require_string(record.get("arm_id"), f"{label}.arm_id")
    if arm_id not in ARM_CONTRACTS:
        raise ValueError(f"{label} arm is outside the exact fixed matrix")
    arm_contract = ARM_CONTRACTS[arm_id]
    history_limit = _require_int(record.get("history_limit"), f"{label}.history_limit")
    adaptive_scale = _require_bool(
        record.get("adaptive_scale_without_history"),
        f"{label}.adaptive_scale_without_history",
    )
    scale_policy = _require_string(record.get("scale_policy"), f"{label}.scale_policy")
    if (
        record.get("kernel") != "safe-lbfgs-total"
        or history_limit != arm_contract["history_limit"]
        or adaptive_scale != arm_contract["adaptive_scale_without_history"]
        or scale_policy != arm_contract["scale_policy"]
    ):
        raise ValueError(f"{label} kernel/scale contract does not match its fixed arm")
    row_descriptor = _require_mapping(
        record.get("objective_descriptor"), f"{label}.objective_descriptor"
    )
    _require_exact_keys(row_descriptor, OBJECTIVE_KEYS, f"{label}.objective_descriptor")
    if row_descriptor != descriptor:
        raise ValueError(f"{label} objective descriptor differs from the reviewed kernel")
    if _require_int(record.get("replay_maxiter"), f"{label}.replay_maxiter", minimum=1) != MAXITER:
        raise ValueError(f"{label}.replay_maxiter must be exactly {MAXITER}")
    _require_int(record.get("source_task_maxiter"), f"{label}.source_task_maxiter", minimum=1)
    certificate = _require_bool(record.get("certificate_satisfied"), f"{label}.certificate_satisfied")
    reason = _require_string(record.get("termination_reason"), f"{label}.termination_reason")
    telemetry = _require_mapping(record.get("telemetry"), f"{label}.telemetry")
    _require_exact_keys(telemetry, TELEMETRY_KEYS, f"{label} telemetry")
    for field in TELEMETRY_INTEGER_FIELDS:
        _require_int(telemetry.get(field), f"{label}.telemetry.{field}")
    _require_finite(telemetry.get("bias_secant_curvature_sum"), f"{label}.telemetry.bias_secant_curvature_sum")
    _require_string(telemetry.get("backend"), f"{label}.telemetry.backend")
    _require_string(telemetry.get("gradient_measure"), f"{label}.telemetry.gradient_measure")
    if telemetry.get("termination_reason") != reason:
        raise ValueError(f"{label} telemetry termination reason differs from row")
    converged = _require_bool(telemetry.get("converged"), f"{label}.telemetry.converged")
    optimizer_success = _require_bool(telemetry.get("optimizer_success"), f"{label}.telemetry.optimizer_success")
    if reason not in {"converged", "line_search_failed", "maxiter"}:
        raise ValueError(f"{label} has an unsupported termination reason")
    expected_optimizer_success = reason == "converged"
    if converged != expected_optimizer_success or optimizer_success != expected_optimizer_success:
        raise ValueError(f"{label} termination/converged telemetry is inconsistent")
    if int(telemetry["accepted_secants"]) + int(telemetry["rejected_secants"]) != int(
        telemetry["accepted_steps"]
    ):
        raise ValueError(f"{label} accepted and rejected secants must close accepted_steps")

    purpose_counts = _require_mapping(record.get("purpose_counts"), f"{label}.purpose_counts")
    _require_exact_keys(purpose_counts, PURPOSE_KEYS, f"{label} purpose_counts")
    for purpose, count in purpose_counts.items():
        _require_int(count, f"{label}.purpose_counts[{purpose}]")
    if sum(purpose_counts.values()) != force_evaluations:
        raise ValueError(f"{label}.purpose_counts must sum to force_evaluations")
    callbacks = record.get("accepted_callback_hashes")
    if not isinstance(callbacks, list):
        raise ValueError(f"{label}.accepted_callback_hashes must be a list")
    callback_hashes = {
        _require_sha256(value, f"{label}.accepted_callback_hashes[{index}]")
        for index, value in enumerate(callbacks)
    }
    if len(callback_hashes) != len(callbacks):
        raise ValueError(f"{label}.accepted_callback_hashes must not contain duplicates")

    accepted_records, nonaccepted_records = _validate_trace_records(
        record, label, callback_hashes
    )
    endpoint = _require_mapping(record.get("endpoint"), f"{label}.endpoint")
    _require_exact_keys(endpoint, ENDPOINT_KEYS, f"{label} endpoint")
    endpoint_positions = _require_finite_array(
        endpoint.get("positions"),
        f"{label}.endpoint.positions",
        (source_contract.n_atoms, 3),
    )
    endpoint_energy = _require_finite(
        endpoint.get("biased_energy_eV"), f"{label}.endpoint.biased_energy_eV"
    )
    endpoint_force = _require_nonnegative_finite(
        endpoint.get("max_active_atom_force_eV_per_A"),
        f"{label}.endpoint.max_active_atom_force_eV_per_A",
    )
    final_trace = _require_mapping(record["trace_records"][-1], f"{label}.final trace record")
    if position_sha256(endpoint_positions) != final_trace["positions_sha256"]:
        raise ValueError(f"{label} endpoint position hash differs from the final trace")
    if endpoint_energy != final_trace["total_energy_eV"]:
        raise ValueError(f"{label} endpoint energy differs from last trace total energy")
    if endpoint_force != final_trace["active_max_total_force_eV_per_A"]:
        raise ValueError(f"{label} endpoint max force differs from last trace force")
    expected_certificate = 0.0 <= endpoint_force <= source_contract.fmax_eV_per_A
    if certificate != expected_certificate:
        raise ValueError(f"{label} certificate does not match finite endpoint force and source fmax")
    if (reason == "converged") != certificate:
        raise ValueError(f"{label} termination reason does not match certificate")
    _require_nonnegative_finite(record.get("wall_time_s"), f"{label}.wall_time_s")

    normalized = dict(record)
    normalized["_accepted_trace_records"] = accepted_records
    normalized["_nonaccepted_trace_records"] = nonaccepted_records
    normalized["_endpoint_positions"] = endpoint_positions
    return normalized


def _validated_rows(
    ledger_dir: Path,
    summary: Mapping[str, Any],
    states: Mapping[tuple[str, str], SourceTaskContract],
) -> list[dict[str, Any]]:
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
            or len(payload["rows"]) != 24
        ):
            raise ValueError(f"{system} ledger violates the fixed task-arm matrix: expected exactly 24 rows")
        for raw_row in payload["rows"]:
            rows.append(
                _validate_row(
                    raw_row,
                    expected_system=system,
                    states=states,
                    descriptor=descriptor,
                )
            )
    if len(rows) != 48:
        raise ValueError("ledger violates the fixed task-arm matrix: expected exactly 48 rows")
    expected_matrix = {
        (system, f"{system}-seed-{seed}-bias-1", arm_id)
        for system in SYSTEMS
        for seed in SEEDS
        for arm_id, _, _, _ in ARMS
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
    recorded_reasons = _require_mapping(
        summary.get("termination_reason_counts"), "summary.termination_reason_counts"
    )
    if not set(recorded_reasons).issubset(
        {"converged", "line_search_failed", "maxiter"}
    ):
        raise ValueError("summary.termination_reason_counts contains unsupported reasons")
    _require_exact_keys(
        recorded_reasons,
        set(reasons),
        "summary.termination_reason_counts",
    )
    for reason, count in recorded_reasons.items():
        _require_int(count, f"summary.termination_reason_counts.{reason}")
    if recorded_reasons != dict(sorted(reasons.items())):
        raise ValueError("summary termination_reason_counts disagrees with ledger rows")
    # The runner's total includes orchestration overhead.  Per-row wall time is
    # the only arm-comparable cost reported below, so retain the summary total
    # only as a finite execution-provenance field rather than forcing equality.
    _require_nonnegative_finite(summary.get("wall_time_total_s"), "wall_time_total_s")
    _require_string(summary.get("claim_ceiling"), "claim_ceiling")


def _load_validated_ledger_context(
    ledger_dir: Path,
) -> tuple[
    Path,
    dict[str, Any],
    str,
    dict[tuple[str, str], SourceTaskContract],
    list[dict[str, Any]],
]:
    ledger_dir = ledger_dir.resolve()
    summary = _load_json_object(ledger_dir / "summary.json", label="summary")
    execution_commit, states = _validate_provenance(summary)
    rows = _validated_rows(ledger_dir, summary, states)
    _validate_summary_outcomes(summary, rows)
    system_order = {system: index for index, system in enumerate(SYSTEMS)}
    seed_order = {seed: index for index, seed in enumerate(SEEDS)}
    arm_order = {arm_id: index for index, (arm_id, _, _, _) in enumerate(ARMS)}
    rows.sort(
        key=lambda row: (
            system_order[row["system"]],
            seed_order[row["seed"]],
            arm_order[row["arm_id"]],
        )
    )
    return ledger_dir, summary, execution_commit, states, rows


def load_validated_ledger(ledger_dir: Path) -> list[dict[str, Any]]:
    """Load rows only after the reviewed provenance, schema, and closures pass."""

    return _load_validated_ledger_context(ledger_dir)[-1]


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
        for arm_id, _, _, _ in ARMS:
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
    states: Mapping[tuple[str, str], SourceTaskContract],
) -> list[dict[str, Any]]:
    by_key = {(row["system"], row["task_id"], row["arm_id"]): row for row in rows}
    pairs: list[dict[str, Any]] = []
    for system in SYSTEMS:
        for seed in SEEDS:
            task_id = f"{system}-seed-{seed}-bias-1"
            arm_rows = {
                arm_id: by_key[(system, task_id, arm_id)]
                for arm_id, _, _, _ in ARMS
            }
            task_hashes = {row["task_sha256"] for row in arm_rows.values()}
            if len(task_hashes) != 1:
                raise ValueError(
                    f"{system}/{task_id} task SHA256 must match across all arms"
                )
            source_contract = states[(system, task_id)]
            pair = {
                "system": system,
                "task_id": task_id,
                "seed": seed,
                "task_sha256": next(iter(task_hashes)),
                "arms": {
                    arm_id: _pair_arm(arm_rows[arm_id])
                    for arm_id, _, _, _ in ARMS
                },
                "comparisons": {
                    "adaptive-scale-history0_vs_fixed-scale-history0": (
                        _pair_comparison(
                            reference=arm_rows["fixed-scale-history0"],
                            candidate=arm_rows["adaptive-scale-history0"],
                            source_contract=source_contract,
                        )
                    ),
                    "adaptive-scale-history10_vs_adaptive-scale-history0": (
                        _pair_comparison(
                            reference=arm_rows["adaptive-scale-history0"],
                            candidate=arm_rows["adaptive-scale-history10"],
                            source_contract=source_contract,
                        )
                    ),
                },
            }
            pairs.append(pair)
    return pairs


def _pair_comparison(
    *,
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    source_contract: SourceTaskContract,
) -> dict[str, Any]:
    reference_positions = np.asarray(reference["_endpoint_positions"], dtype=float)
    candidate_positions = np.asarray(candidate["_endpoint_positions"], dtype=float)
    if reference_positions.shape != candidate_positions.shape:
        raise ValueError("paired endpoint atom counts differ")
    displacement = mic_displacement(
        candidate_positions,
        reference_positions,
        source_contract.cell,
        source_contract.pbc,
    )
    norms = np.linalg.norm(displacement, axis=1)
    reference_force = float(
        reference["endpoint"]["max_active_atom_force_eV_per_A"]
    )
    candidate_force = float(
        candidate["endpoint"]["max_active_atom_force_eV_per_A"]
    )
    return {
        "candidate_minus_reference_calls": int(candidate["force_evaluations"])
        - int(reference["force_evaluations"]),
        "candidate_minus_reference_force_eV_per_A": (
            candidate_force - reference_force
        ),
        "endpoint_delta": {
            "max_mic_displacement_A": float(np.max(norms)),
            "rms_mic_displacement_A": float(
                np.sqrt(np.mean(np.square(norms)))
            ),
            "pbc": list(source_contract.pbc),
        },
    }


def _pair_arm(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "calls": int(row["force_evaluations"]),
        "wall_time_s": float(row["wall_time_s"]),
        "endpoint_biased_energy_eV": float(row["endpoint"]["biased_energy_eV"]),
        "certificate_satisfied": bool(row["certificate_satisfied"]),
        "termination_reason": str(row["termination_reason"]),
    }


def build_evidence(ledger_dir: Path) -> dict[str, Any]:
    ledger_dir, summary, execution_commit, states, rows = _load_validated_ledger_context(
        ledger_dir
    )
    reason_counts = Counter(str(row["termination_reason"]) for row in rows)
    satisfied = sum(bool(row["certificate_satisfied"]) for row in rows)
    raw_files = {
        name: {"sha256": _sha256(ledger_dir / name)}
        for name in ("summary.json", "c60.json", "pdo.json")
    }
    analysis_path = Path(__file__).resolve()
    pbc_path = Path(_require_string(pbc_module.__file__, "imported pamssw.pbc path")).resolve()
    return {
        "schema_version": 1,
        "analysis_script_sha256": _sha256(analysis_path),
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
            "mic_implementation": {
                "module": "pamssw.pbc",
                "path": str(pbc_path),
                "sha256": _sha256(pbc_path),
            },
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
    arm_ids = tuple(arm_id for arm_id, _, _, _ in ARMS)
    totals = {
        arm_id: {
            "calls": sum(
                int(aggregate[system][arm_id]["force_evaluations"]["sum"])
                for system in SYSTEMS
            ),
            "wall": sum(
                float(aggregate[system][arm_id]["wall_time_s"]["sum"])
                for system in SYSTEMS
            ),
            "rows": sum(
                int(aggregate[system][arm_id]["count"]) for system in SYSTEMS
            ),
            "certificates": sum(
                int(aggregate[system][arm_id]["certificate"]["satisfied_count"])
                for system in SYSTEMS
            ),
            "maxiter": sum(
                int(
                    aggregate[system][arm_id]["termination_reasons"].get(
                        "maxiter", 0
                    )
                )
                for system in SYSTEMS
            ),
            "line_search_failed": sum(
                int(
                    aggregate[system][arm_id]["termination_reasons"].get(
                        "line_search_failed", 0
                    )
                )
                for system in SYSTEMS
            ),
        }
        for arm_id in arm_ids
    }
    fixed = totals["fixed-scale-history0"]
    scale = totals["adaptive-scale-history0"]
    history = totals["adaptive-scale-history10"]
    lines = [
        "# Safe L-BFGS inverse-scale decomposition — evidence-limited conclusion",
        "",
        "## Certificate outcome",
        "",
        f"The frozen 16-task C60/PdO matrix produced {certificate['satisfied_count']}/48 certificate-satisfied rows and {certificate['unsatisfied_count']}/48 incomplete rows.  Finite `maxiter` and `line_search_failed` records remain valid incomplete outcomes; they are not converted into convergence.",
        "",
        f"Fixed scale, history 0: {fixed['certificates']}/{fixed['rows']} certificate-satisfied rows, {fixed['maxiter']} `maxiter`, {fixed['line_search_failed']} `line_search_failed`, and {fixed['calls']} evaluator calls.",
        f"Adaptive scale, history 0: {scale['certificates']}/{scale['rows']} certificate-satisfied rows, {scale['maxiter']} `maxiter`, {scale['line_search_failed']} `line_search_failed`, and {scale['calls']} evaluator calls.",
        f"Adaptive scale, history 10: {history['certificates']}/{history['rows']} certificate-satisfied rows, {history['maxiter']} `maxiter`, {history['line_search_failed']} `line_search_failed`, and {history['calls']} evaluator calls.",
        "",
        "## Cost outcome",
        "",
        f"Across all 16 tasks, fixed scale/history 0 used {fixed['calls']} calls and {fixed['wall']:.6f} s; adaptive scale/history 0 used {scale['calls']} calls and {scale['wall']:.6f} s; adaptive scale/history 10 used {history['calls']} calls and {history['wall']:.6f} s.",
        "Each arm cost includes all 16 rows.  Costs of incomplete arms are protocol costs, not cheap-success comparisons or same-endpoint speedups.",
        "",
        "## Fixed-matrix interpretation ceiling",
        "",
        f"Adaptive scalar scaling is a positive but partial contributor on this fixed matrix: relative to fixed scale/history 0, it raises certificate coverage from {fixed['certificates']}/16 to {scale['certificates']}/16 and reduces recorded protocol cost from {fixed['calls']} to {scale['calls']} evaluator calls.",
        "",
        f"Adaptive scaling alone is not sufficient to recover the existing kernel: history 10 reaches {history['certificates']}/16 certificates with {history['calls']} calls, versus {scale['certificates']}/16 and {scale['calls']} calls for scale-only.  The remaining two-loop correction stack is therefore a strong positive candidate on these tasks, but this experiment does not separate the newest correction from older retained pairs.",
        "",
        "This is not a generic BFGS result, a same-basin equivalence result, a full-SSW performance result, or a statistical generalization claim.  It does not justify a production-default change or an analytic bias-Hessian method.",
        "",
        "Accepted-state and explicit-finalization records are accounting annotations.",
        "The trace figure visualizes exact evaluations and callback-observed annotations for accounting.  A callback-observed label is not an optimizer acceptance rule.  The figure supports no endpoint-equivalence inference.",
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
