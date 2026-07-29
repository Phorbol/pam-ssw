#!/usr/bin/env python3
"""Fail-closed analysis of the fixed-budget proposal-fmax GPU matrix.

This module deliberately accepts only a published final matrix.  It does not
inspect a staging/partial directory and it does not run any calculator.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import fsum, isclose, isfinite
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
DEFAULT_OUTPUT_ROOT = RUN_ROOT / "output-seeds42-45-budget6000"
DEFAULT_EVIDENCE_PATH = RUN_ROOT / "ablation_evidence.json"
DEFAULT_CONCLUSION_PATH = RUN_ROOT / "ablation_conclusion.md"

EXPECTED_EXECUTION_COMMIT = "4ca77298d074429eb98a0238e6172d5e91ce9fb1"
POSTERIOR_HARNESS_PATH = (
    RUN_ROOT.parent / "20260728-posterior-starter-policy-gpu-ablation" / "run_ablation.py"
)
EXPECTED_FROZEN_HARNESS_PATH = str(POSTERIOR_HARNESS_PATH)
EXPECTED_FROZEN_HARNESS_SHA256 = (
    "5168438843af122ee47c2fd9948cffcc6fca0ec15dbf2ad6ba1e9c8441665c41"
)
EXPECTED_PAMSSW_BUNDLE_SHA256 = (
    "9ccab5f4ad38f368153448a13b277ed4786661fea9b5c15eb0091fff492d16d0"
)
EXPECTED_INPUTS = {
    "c60": {
        "path": (
            "/mnt/d/download/trae-research-code/ssw/runs/"
            "20260428-c60-mace-production/prerelaxed_c60.xyz"
        ),
        "sha256": "c63788c18cbed305963213b47eabd9fdc4d06dac118da6a1a9e16621d5e32bf9",
    },
    "pdo": {
        "path": "/mnt/d/download/trae-research-code/ssw/PdO.xyz",
        "sha256": "68243ceb7c0fbb6ba7a9454d680287eb98c4e5210efbd9ebb63517ba79aaa8b0",
    },
}
EXPECTED_MODEL = {
    "path": "/root/.cache/mace/mace-omat-0-small.model",
    "sha256": "0abfde07862cf1e93b8b4d03cb702f29ce9c344ff2fc4de2ec0d7166d6c113a5",
}
SYSTEMS = ("c60", "pdo")
MASTER_SEEDS = (42, 43, 44, 45)
ARMS = (
    ("proposal-fmax-0.05", 0.05),
    ("proposal-fmax-0.10", 0.10),
)
POLICY_NAME = "uniform"
ACTION_FORCE_BUDGET = 1000
TOTAL_FORCE_BUDGET = 6000
PURPOSES = (
    "bootstrap_true_quench",
    "starter_true_quench",
    "direction_oracle",
    "escape_true_pes_check",
    "biased_proposal_relax",
    "landing_true_quench",
    "post_relax_validation",
    "unattributed",
)
NONOPERATIVE_OUTPUTS = {
    "accepted_structures_log": None,
    "accepted_structures_dir": None,
    "write_proposal_minima": False,
    "proposal_minima_dir": None,
    "write_relaxation_trajectories": False,
    "relaxation_trajectory_dir": None,
    "direction_diagnostics_enabled": False,
    "direction_diagnostics_path": None,
    "direction_archive_enabled": False,
    "direction_archive_path": None,
}
PROJECTION_IDENTITY_FIELDS = ("arm_index", "arm_id", "proposal_fmax")
SOURCE_CASE_LOCAL_OUTPUT_SUFFIXES = {
    "accepted_structures_dir": "accepted_minima",
    "accepted_structures_log": "accepted_structures.jsonl",
    "direction_diagnostics_path": "direction_trace.jsonl",
}


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read valid JSON from {path}") from error


def _read_jsonl(path: Path) -> list[Any]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"cannot read JSONL from {path}") from error
    values: list[Any] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"{path}:{line_number} is blank")
        try:
            values.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number} is invalid JSON") from error
    return values


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _sequence(value: Any, *, label: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return value


def _nonempty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _positive_int(value: Any, *, label: str) -> int:
    result = _nonnegative_int(value, label=label)
    if result == 0:
        raise ValueError(f"{label} must be positive")
    return result


def _finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a JSON number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def _file_sha256(path: Path) -> str:
    digest = sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ValueError(f"cannot hash frozen harness source {path}") from error
    return digest.hexdigest()


def _purpose_counts(value: Any, *, label: str) -> dict[str, int]:
    counts = _mapping(value, label=f"{label} purpose counts")
    if set(counts) != set(PURPOSES):
        raise ValueError(f"{label} purpose ledger keys mismatch")
    return {
        purpose: _nonnegative_int(counts[purpose], label=f"{label} {purpose}")
        for purpose in PURPOSES
    }


def _inside_output(
    value: Any, *, output_root: Path, label: str, kind: str = "file"
) -> Path:
    raw = Path(_nonempty_string(value, label=label))
    path = raw if raw.is_absolute() else REPO_ROOT / raw
    resolved = path.resolve()
    try:
        resolved.relative_to(output_root)
    except ValueError as error:
        raise ValueError(f"{label} escapes final output") from error
    exists = resolved.is_file() if kind == "file" else resolved.is_dir()
    if not exists:
        raise ValueError(f"{label} does not exist inside final output")
    return resolved


def _output_relative_directory(value: Any, *, output_root: Path, label: str) -> Path:
    relative = Path(_nonempty_string(value, label=label))
    if relative.is_absolute():
        raise ValueError(f"{label} must be relative to final output")
    resolved = (output_root / relative).resolve()
    try:
        resolved.relative_to(output_root)
    except ValueError as error:
        raise ValueError(f"{label} escapes final output") from error
    if not resolved.is_dir():
        raise ValueError(f"{label} does not exist inside final output")
    return resolved


def _expected_matrix() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for system in SYSTEMS:
        for master_seed in MASTER_SEEDS:
            for arm_index, (arm_id, proposal_fmax) in enumerate(ARMS):
                rows.append(
                    {
                        "matrix_index": len(rows),
                        "system": system,
                        "master_seed": master_seed,
                        "policy_name": POLICY_NAME,
                        "batch_size": 1,
                        "max_workers": 1,
                        "arm_index": arm_index,
                        "arm_id": arm_id,
                        "proposal_fmax": proposal_fmax,
                        "relative_case_directory": str(
                            Path(system)
                            / f"seed-{master_seed:08d}"
                            / arm_id
                        ),
                    }
                )
    return rows


def _derive_action_seed(master_seed: int, batch_id: int, slot_id: int) -> int:
    """Mirror the execution harness's injective Cantor-pair seed derivation."""

    def pair(left: int, right: int) -> int:
        total = left + right
        return total * (total + 1) // 2 + right

    return pair(pair(master_seed, batch_id), slot_id)


def _validate_frozen_preflight(preflight: Mapping[str, Any]) -> dict[str, Any]:
    if (
        preflight.get("schema_version") != 1
        or preflight.get("execution_commit") != EXPECTED_EXECUTION_COMMIT
        or preflight.get("systems") != list(SYSTEMS)
        or preflight.get("master_seeds") != list(MASTER_SEEDS)
        or preflight.get("policies")
        != ["uniform", "posterior_proportional", "minimal_ucb"]
        or preflight.get("batch_size") != 1
        or preflight.get("max_workers") != 1
        or preflight.get("action_force_budget") != ACTION_FORCE_BUDGET
        or preflight.get("total_force_budget") != TOTAL_FORCE_BUDGET
    ):
        raise ValueError("frozen preflight protocol mismatch")
    bundle_sha = _sha256(
        preflight.get("pamssw_bundle_sha256"), label="PAMSSW bundle SHA"
    )
    if bundle_sha != EXPECTED_PAMSSW_BUNDLE_SHA256:
        raise ValueError("PAMSSW bundle SHA mismatch")
    inputs = _mapping(preflight.get("inputs"), label="input provenance")
    if set(inputs) != set(SYSTEMS):
        raise ValueError("input provenance systems mismatch")
    normalized_inputs: dict[str, dict[str, str]] = {}
    for system in SYSTEMS:
        input_record = _mapping(inputs[system], label=f"{system} input provenance")
        input_path = _nonempty_string(
            input_record.get("path"), label=f"{system} input path"
        )
        input_sha = _sha256(input_record.get("sha256"), label=f"{system} input SHA")
        if input_path != EXPECTED_INPUTS[system]["path"]:
            raise ValueError(f"{system} input path mismatch")
        if input_sha != EXPECTED_INPUTS[system]["sha256"]:
            raise ValueError(f"{system} input SHA mismatch")
        normalized_inputs[system] = {"path": input_path, "sha256": input_sha}
    model = _mapping(preflight.get("model"), label="model provenance")
    model_path = _nonempty_string(model.get("path"), label="model path")
    model_sha = _sha256(model.get("sha256"), label="model SHA")
    if model_path != EXPECTED_MODEL["path"]:
        raise ValueError("model path mismatch")
    if model_sha != EXPECTED_MODEL["sha256"]:
        raise ValueError("model SHA mismatch")
    normalized_model = {"path": model_path, "sha256": model_sha}
    calculator = _mapping(preflight.get("calculator"), label="calculator provenance")
    if calculator != {
        "default_dtype": "float32",
        "device": "cuda",
        "enable_cueq": False,
        "inference_precision": "float32",
    }:
        raise ValueError("calculator provenance mismatch")
    cuda = _mapping(preflight.get("cuda"), label="CUDA provenance")
    if cuda.get("available") is not True or cuda.get("requested_device") != "cuda":
        raise ValueError("CUDA provenance mismatch")
    _nonempty_string(cuda.get("runtime_version"), label="CUDA runtime version")
    _nonempty_string(cuda.get("device_name"), label="CUDA device name")
    runtime_versions = _mapping(
        preflight.get("runtime_versions"), label="runtime version provenance"
    )
    if set(runtime_versions) != {"python", "numpy", "scipy", "ase", "torch", "mace"}:
        raise ValueError("runtime version provenance keys mismatch")
    for package, version in runtime_versions.items():
        _nonempty_string(version, label=f"{package} runtime version")
    calculator_reuse = _mapping(
        preflight.get("calculator_reuse"), label="calculator reuse provenance"
    )
    if calculator_reuse != {
        "bootstrap": "one uncached caller-thread calculator",
        "actions": "one worker-thread-local calculator reused serially",
        "cross_thread_calculator_sharing": False,
        "accounting": "each bootstrap/action is wrapped by its own EvalCounter",
    }:
        raise ValueError("calculator reuse provenance mismatch")
    projections = _mapping(preflight.get("projections"), label="frozen projections")
    if set(projections) != set(SYSTEMS):
        raise ValueError("frozen projection systems mismatch")
    frozen_effective_configs = {
        system: _mapping(
            _mapping(projections[system], label=f"{system} frozen projection").get(
                "effective_ssw_config"
            ),
            label=f"{system} nested frozen preflight config",
        )
        for system in SYSTEMS
    }
    return {
        "pamssw_bundle_sha256": bundle_sha,
        "inputs": normalized_inputs,
        "model": normalized_model,
        "calculator": dict(calculator),
        "cuda": dict(cuda),
        "runtime_versions": dict(runtime_versions),
        "calculator_reuse": dict(calculator_reuse),
        "frozen_effective_configs": frozen_effective_configs,
    }


def _validate_projection(
    projection: Mapping[str, Any],
    *,
    system: str,
    arm_index: int,
    arm_id: str,
    proposal_fmax: float,
    history_limit: int,
    expected_frozen_config: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    label = f"{system}/{arm_id} projection"
    if (
        projection.get("arm_index") != arm_index
        or projection.get("arm_id") != arm_id
        or not isclose(
            _finite_float(projection.get("proposal_fmax"), label=f"{label} fmax"),
            proposal_fmax,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        or projection.get("arm_overrides") != {"proposal_fmax": proposal_fmax}
    ):
        raise ValueError(f"{label} arm identity mismatch")
    config = _mapping(projection.get("effective_ssw_config"), label=f"{label} config")
    frozen_config = _mapping(
        projection.get("frozen_posterior_effective_ssw_config"),
        label=f"{label} frozen posterior config",
    )
    if frozen_config != expected_frozen_config:
        raise ValueError(f"{label} does not match nested frozen preflight config")
    if set(config) != set(frozen_config):
        raise ValueError(f"{label} frozen posterior config keys mismatch")
    frozen_changes = {
        field for field in config if config[field] != frozen_config[field]
    }
    expected_frozen_changes = set() if proposal_fmax == 0.05 else {"proposal_fmax"}
    if frozen_changes != expected_frozen_changes:
        raise ValueError(
            f"{label} effective optimizer config differs from frozen posterior "
            "config beyond proposal_fmax"
        )
    required = {
        "proposal_optimizer": "safe-lbfgs-total",
        "proposal_fmax": proposal_fmax,
        "proposal_pool_size": 1,
        "quench_optimizer": "ase-lbfgs",
        "quench_fallback_optimizer": "ase-fire",
        "quench_fmax": 0.01,
        "quench_maxiter": 400,
    }
    for field, expected in required.items():
        actual = config.get(field)
        if isinstance(expected, float):
            if not isclose(
                _finite_float(actual, label=f"{label} {field}"),
                expected,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise ValueError(f"{label} effective optimizer config mismatch")
        elif actual != expected:
            raise ValueError(f"{label} effective optimizer config mismatch")
    if projection.get("proposal_protocol") != {
        "optimizer": "safe-lbfgs-total",
        "history_limit": history_limit,
        "proposal_pool_size": 1,
        "softening_enabled": False,
    }:
        raise ValueError(f"{label} proposal protocol mismatch")
    if projection.get("true_quench_protocol") != {
        "optimizer": "ase-lbfgs",
        "fallback_optimizer": "ase-fire",
        "fmax_eV_per_A": 0.01,
        "maxiter": 400,
    }:
        raise ValueError(f"{label} true quench protocol mismatch")
    if projection.get("nonoperative_config_outputs") != NONOPERATIVE_OUTPUTS:
        raise ValueError(f"{label} enables an untracked output")
    return config, frozen_config


def _validate_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if manifest.get("execution_commit") != EXPECTED_EXECUTION_COMMIT:
        raise ValueError("manifest execution commit mismatch")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("systems") != list(SYSTEMS)
        or manifest.get("master_seeds") != list(MASTER_SEEDS)
        or manifest.get("policy_name") != POLICY_NAME
        or manifest.get("policy_support") != "full-support random baseline"
        or manifest.get("batch_size") != 1
        or manifest.get("max_workers") != 1
        or manifest.get("action_force_budget") != ACTION_FORCE_BUDGET
        or manifest.get("total_force_budget") != TOTAL_FORCE_BUDGET
        or manifest.get("proposal_optimizer") != "safe-lbfgs-total"
        or manifest.get("softening_enabled") is not False
    ):
        raise ValueError("manifest execution protocol mismatch")
    harness = _mapping(
        manifest.get("frozen_posterior_harness"), label="frozen harness provenance"
    )
    harness_path = _nonempty_string(harness.get("path"), label="frozen harness path")
    harness_sha = _sha256(harness.get("sha256"), label="frozen harness SHA")
    if harness_path != EXPECTED_FROZEN_HARNESS_PATH:
        raise ValueError("frozen harness path mismatch")
    if harness_sha != EXPECTED_FROZEN_HARNESS_SHA256:
        raise ValueError("frozen harness SHA mismatch")
    if _file_sha256(POSTERIOR_HARNESS_PATH) != EXPECTED_FROZEN_HARNESS_SHA256:
        raise ValueError("frozen harness source SHA mismatch")
    preflight = _validate_frozen_preflight(
        _mapping(harness.get("preflight"), label="frozen preflight")
    )
    history_limit = _positive_int(
        manifest.get("safe_lbfgs_history_limit"), label="safe L-BFGS history limit"
    )
    expected_arms = [
        {"arm_index": index, "arm_id": arm_id, "proposal_fmax": proposal_fmax}
        for index, (arm_id, proposal_fmax) in enumerate(ARMS)
    ]
    if manifest.get("arms") != expected_arms:
        raise ValueError("manifest arms mismatch")
    projections = _mapping(manifest.get("projections"), label="manifest projections")
    if set(projections) != set(SYSTEMS):
        raise ValueError("manifest projection systems mismatch")
    normalized_projections: dict[str, list[Mapping[str, Any]]] = {}
    for system in SYSTEMS:
        values = _sequence(projections[system], label=f"{system} arm projections")
        if len(values) != len(ARMS):
            raise ValueError(f"{system} arm projection count mismatch")
        configs: list[Mapping[str, Any]] = []
        frozen_configs: list[Mapping[str, Any]] = []
        normalized: list[Mapping[str, Any]] = []
        for arm_index, ((arm_id, proposal_fmax), value) in enumerate(
            zip(ARMS, values, strict=True)
        ):
            projection = _mapping(value, label=f"{system} projection {arm_index}")
            config, frozen_config = _validate_projection(
                projection,
                system=system,
                arm_index=arm_index,
                arm_id=arm_id,
                proposal_fmax=proposal_fmax,
                history_limit=history_limit,
                expected_frozen_config=preflight["frozen_effective_configs"][system],
            )
            configs.append(config)
            frozen_configs.append(frozen_config)
            normalized.append(projection)
        if frozen_configs[0] != frozen_configs[1]:
            raise ValueError(f"{system} frozen posterior configs differ across arms")
        if set(configs[0]) != set(configs[1]):
            raise ValueError(f"{system} effective config keys differ across arms")
        changed = {
            field
            for field in configs[0]
            if configs[0][field] != configs[1][field]
        }
        if changed != {"proposal_fmax"}:
            raise ValueError(f"{system} arms differ beyond proposal_fmax")
        normalized_projections[system] = normalized
    if manifest.get("matrix") != _expected_matrix():
        raise ValueError("manifest matrix mismatch")
    return {
        "execution_commit": EXPECTED_EXECUTION_COMMIT,
        "frozen_posterior_harness": {"path": harness_path, "sha256": harness_sha},
        **preflight,
        "projections": normalized_projections,
    }


def _validate_uniform_snapshot(
    snapshot: Mapping[str, Any],
    *,
    current_ids: Mapping[int, Sequence[int]],
    label: str,
) -> tuple[list[int], list[float]]:
    eligible_raw = _sequence(
        snapshot.get("eligible_starter_ids"), label=f"{label} eligible starters"
    )
    eligible = [
        _nonnegative_int(value, label=f"{label} eligible starter ID")
        for value in eligible_raw
    ]
    if not eligible or len(set(eligible)) != len(eligible) or eligible != sorted(current_ids):
        raise ValueError(f"{label} eligible starters mismatch archive")
    probabilities_raw = _sequence(
        snapshot.get("probabilities"), label=f"{label} probabilities")
    if len(probabilities_raw) != len(eligible):
        raise ValueError(f"{label} eligible/probability length mismatch")
    probabilities = [
        _finite_float(value, label=f"{label} probability")
        for value in probabilities_raw
    ]
    expected = 1.0 / len(eligible)
    if (
        snapshot.get("support_complete") is not True
        or any(
            not isclose(value, expected, rel_tol=0.0, abs_tol=1.0e-12)
            for value in probabilities
        )
        or not isclose(fsum(probabilities), 1.0, rel_tol=0.0, abs_tol=1.0e-12)
    ):
        raise ValueError(f"{label} uniform probabilities mismatch")
    return eligible, probabilities


def _validate_action_metric(
    value: Mapping[str, Any], *, ordinal: int, label: str
) -> dict[str, Any]:
    if (
        value.get("schema_version") != 1
        or value.get("ordinal") != ordinal
        or value.get("batch_id") != ordinal
        or value.get("policy_name") != POLICY_NAME
        or value.get("status") != "completed"
        or value.get("failure_reason") is not None
        or value.get("posterior_observed") is not True
    ):
        raise ValueError(f"{label} action metric identity/status mismatch")
    action_id = _nonempty_string(value.get("action_id"), label=f"{label} action ID")
    if action_id != f"batch-{ordinal:08d}-slot-0000":
        raise ValueError(f"{label} canonical action identity mismatch")
    force_evaluations = _nonnegative_int(
        value.get("force_evaluations"), label=f"{label} force evaluations"
    )
    counts = _purpose_counts(value.get("evaluation_counts"), label=label)
    if (
        force_evaluations == 0
        or value.get("evaluator_calls") != force_evaluations
        or sum(counts.values()) != force_evaluations
        or counts["bootstrap_true_quench"] != 0
        or counts["unattributed"] != 0
        or force_evaluations > ACTION_FORCE_BUDGET
    ):
        raise ValueError(f"{label} action FE ledger does not close")
    wall_time = _finite_float(
        value.get("evaluator_wall_time_s"), label=f"{label} evaluator wall time"
    )
    if wall_time < 0.0:
        raise ValueError(f"{label} evaluator wall time must be nonnegative")
    discovered = value.get("discovered_against_snapshot")
    inserted = value.get("inserted_into_archive")
    if not isinstance(discovered, bool) or not isinstance(inserted, bool):
        raise ValueError(f"{label} action outcome flags must be boolean")
    return {
        "action_id": action_id,
        "counts": counts,
        "force_evaluations": force_evaluations,
        "landing_energy_eV": _finite_float(
            value.get("landing_energy_eV"), label=f"{label} landing energy"
        ),
        "best_landing_energy_eV": _finite_float(
            value.get("best_landing_energy_eV"), label=f"{label} best landing energy"
        ),
        "discovered_against_snapshot": discovered,
        "inserted_into_archive": inserted,
        "metric": value,
    }


def _validate_event_triplet(
    values: Sequence[Any],
    action: Mapping[str, Any],
    *,
    ordinal: int,
    master_seed: int,
    current_counts: Mapping[int, Sequence[int]],
    label: str,
) -> tuple[int, bool, bool, float]:
    if len(values) != 3:
        raise ValueError(f"{label} event replay batch must contain three records")
    snapshot = _mapping(values[0], label=f"{label} policy snapshot")
    attempt = _mapping(values[1], label=f"{label} attempt event")
    commit = _mapping(values[2], label=f"{label} batch commit")
    snapshot_policy_version = _nonnegative_int(
        snapshot.get("policy_version"), label=f"{label} snapshot policy version"
    )
    snapshot_archive_version = _nonnegative_int(
        snapshot.get("archive_version"), label=f"{label} snapshot archive version"
    )
    if (
        snapshot.get("schema_version") != 2
        or snapshot.get("record_type") != "policy_snapshot"
        or snapshot.get("batch_id") != ordinal
        or snapshot.get("policy_name") != POLICY_NAME
        or snapshot_policy_version != ordinal
        or snapshot_archive_version != ordinal
    ):
        raise ValueError(
            f"{label} event/action identity mismatch: policy snapshot identity mismatch"
        )
    eligible, probabilities = _validate_uniform_snapshot(
        snapshot, current_ids=current_counts, label=label
    )
    metric = action["metric"]
    if (
        metric.get("eligible_starter_ids") != eligible
        or metric.get("probabilities") != list(snapshot["probabilities"])
        or metric.get("policy_support_complete") is not True
    ):
        raise ValueError(f"{label} policy snapshot/action metric mismatch")
    starter_id = _nonnegative_int(metric.get("starter_id"), label=f"{label} starter ID")
    if starter_id not in eligible:
        raise ValueError(f"{label} selected starter is ineligible")
    selected_probability = probabilities[eligible.index(starter_id)]
    if not isclose(
        _finite_float(
            metric.get("selection_probability"), label=f"{label} metric selection probability"
        ),
        selected_probability,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(f"{label} metric selection probability mismatch")
    if (
        attempt.get("schema_version") != 2
        or attempt.get("record_type") != "attempt"
        or attempt.get("action_id") != action["action_id"]
        or attempt.get("batch_id") != ordinal
        or attempt.get("policy_name") != POLICY_NAME
        or attempt.get("policy_version") != snapshot_policy_version
        or attempt.get("archive_version") != snapshot_archive_version
        or attempt.get("slot_id") != 0
        or attempt.get("random_seed")
        != _derive_action_seed(master_seed, ordinal, 0)
        or attempt.get("within_batch_collision") is not False
        or attempt.get("starter_id") != starter_id
    ):
        raise ValueError(f"{label} event/action identity mismatch")
    event_discovered = attempt.get("discovered_against_snapshot")
    event_inserted = attempt.get("inserted_into_archive")
    event_landing_energy = _finite_float(
        attempt.get("landing_energy"), label=f"{label} event landing energy"
    )
    if (
        attempt.get("status") != "completed"
        or attempt.get("failure_reason") is not None
        or attempt.get("cost_is_exact") is not True
        or attempt.get("force_budget") != ACTION_FORCE_BUDGET
        or attempt.get("force_evaluations") != action["force_evaluations"]
        or attempt.get("evaluation_counts") != action["counts"]
        or attempt.get("posterior_observed") is not True
        or event_discovered is not action["discovered_against_snapshot"]
        or event_inserted is not action["inserted_into_archive"]
        or not isclose(
            _finite_float(
                attempt.get("selection_probability"),
                label=f"{label} event selection probability",
            ),
            selected_probability,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        or not isclose(
            event_landing_energy,
            action["landing_energy_eV"],
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        raise ValueError(f"{label} event/action outcome mismatch")
    if (
        commit.get("schema_version") != 2
        or commit.get("record_type") != "batch_commit"
        or commit.get("batch_id") != ordinal
        or commit.get("action_ids") != [action["action_id"]]
    ):
        raise ValueError(f"{label} batch commit mismatch")
    landing_entry_id = _nonnegative_int(
        attempt.get("landing_entry_id"), label=f"{label} landing entry ID"
    )
    return starter_id, event_discovered, event_inserted, landing_entry_id


def _validate_diagnostics(
    path: Path, actions: Sequence[Mapping[str, Any]], *, label: str
) -> dict[str, int]:
    payload = _mapping(_read_json(path), label=f"{label} optimizer diagnostics")
    if payload.get("schema_version") != 1:
        raise ValueError(f"{label} optimizer diagnostics schema mismatch")
    attempts = _sequence(payload.get("attempts"), label=f"{label} diagnostics attempts")
    if len(attempts) != len(actions):
        raise ValueError(f"{label} optimizer diagnostic attempt count mismatch")
    fallback_attempts = 0
    fallback_converged = 0
    action_ids: list[str] = []
    for index, value in enumerate(attempts):
        attempt = _mapping(value, label=f"{label} diagnostic {index}")
        action_ids.append(
            _nonempty_string(
                attempt.get("action_id"), label=f"{label} diagnostic action ID"
            )
        )
        stats = _mapping(attempt.get("stats"), label=f"{label} diagnostic stats")
        if (
            stats.get("diagnostic_stage") != "completed"
            or stats.get("budget_exhausted") != 0
            or stats.get("force_evaluations") != actions[index]["force_evaluations"]
            or stats.get("proposal_optimizer") != "safe-lbfgs-total"
            or stats.get("quench_optimizer") != "ase-lbfgs"
            or stats.get("quench_fallback_optimizer") != "ase-fire"
            or stats.get("true_quench_unconverged") != 0
        ):
            raise ValueError(f"{label} optimizer diagnostic protocol mismatch")
        used = _nonnegative_int(
            stats.get("quench_fallback_attempts"), label=f"{label} fallback attempts"
        )
        converged = _nonnegative_int(
            stats.get("quench_fallback_converged"),
            label=f"{label} fallback converged",
        )
        if converged > used:
            raise ValueError(f"{label} fallback convergence count is impossible")
        fallback_attempts += used
        fallback_converged += converged
    if action_ids != [action["action_id"] for action in actions]:
        raise ValueError(f"{label} optimizer diagnostic action IDs mismatch")
    return {
        "quench_fallback_attempts": fallback_attempts,
        "quench_fallback_converged": fallback_converged,
    }


def _validate_campaign(
    *,
    output_root: Path,
    index_entry: Mapping[str, Any],
    matrix_row: Mapping[str, Any],
    manifest_projection: Mapping[str, Any],
) -> dict[str, Any]:
    system = str(matrix_row["system"])
    master_seed = int(matrix_row["master_seed"])
    arm_id = str(matrix_row["arm_id"])
    label = f"{system}/seed-{master_seed:08d}/{arm_id}"
    expected_case = _output_relative_directory(
        matrix_row["relative_case_directory"],
        output_root=output_root,
        label=f"{label} case directory",
    )
    expected_summary = expected_case / "campaign_summary.json"
    summary_path = _inside_output(
        index_entry.get("campaign_summary_path"),
        output_root=output_root,
        label=f"{label} campaign summary path",
    )
    if summary_path != expected_summary:
        raise ValueError(f"{label} campaign summary path mismatch")
    runtime_projection = _mapping(
        index_entry.get("projection"), label=f"{label} runtime projection"
    )
    expected_runtime_keys = set(manifest_projection) - set(PROJECTION_IDENTITY_FIELDS)
    if set(runtime_projection) != expected_runtime_keys:
        raise ValueError(f"{label} runtime projection keys mismatch")
    manifest_scientific = {
        field: value
        for field, value in manifest_projection.items()
        if field not in PROJECTION_IDENTITY_FIELDS
    }
    manifest_source = _mapping(
        manifest_scientific.get("source_config"),
        label=f"{label} manifest source config",
    )
    runtime_source = _mapping(
        runtime_projection.get("source_config"),
        label=f"{label} runtime source config",
    )
    for field, suffix in SOURCE_CASE_LOCAL_OUTPUT_SUFFIXES.items():
        for source_kind, source_config in (
            ("manifest", manifest_source),
            ("runtime", runtime_source),
        ):
            source_path = Path(
                _nonempty_string(
                    source_config.get(field),
                    label=f"{label} {source_kind} source {field}",
                )
            )
            if source_path.name != suffix:
                raise ValueError(f"{label} source path suffix mismatch for {field}")
    normalized_manifest_source = {
        field: value
        for field, value in manifest_source.items()
        if field not in SOURCE_CASE_LOCAL_OUTPUT_SUFFIXES
    }
    normalized_runtime_source = {
        field: value
        for field, value in runtime_source.items()
        if field not in SOURCE_CASE_LOCAL_OUTPUT_SUFFIXES
    }
    normalized_manifest = dict(manifest_scientific)
    normalized_manifest["source_config"] = normalized_manifest_source
    normalized_runtime = dict(runtime_projection)
    normalized_runtime["source_config"] = normalized_runtime_source
    if normalized_runtime != normalized_manifest:
        raise ValueError(f"{label} runtime projection scientific mismatch")
    summary = _mapping(_read_json(summary_path), label=f"{label} campaign summary")
    if (
        summary.get("schema_version") != 1
        or summary.get("policy_name") != POLICY_NAME
        or summary.get("master_seed") != master_seed
        or summary.get("batch_size") != 1
        or summary.get("max_workers") != 1
        or summary.get("action_force_budget") != ACTION_FORCE_BUDGET
        or summary.get("total_force_budget") != TOTAL_FORCE_BUDGET
    ):
        raise ValueError(f"{label} summary protocol mismatch")
    if (
        summary.get("benchmark_eligible") is not True
        or summary.get("benchmark_ineligibility_reasons") != []
        or summary.get("event_log_replayed") is not True
        or summary.get("failed_attempts") != 0
    ):
        raise ValueError(f"{label} is not a completed benchmark-eligible campaign")
    expected_action_path = expected_case / "action_metrics.jsonl"
    expected_event_path = expected_case / "events.jsonl"
    action_path = _inside_output(
        summary.get("action_metrics_path"),
        output_root=output_root,
        label=f"{label} action metrics path",
    )
    event_path = _inside_output(
        summary.get("event_log_path"),
        output_root=output_root,
        label=f"{label} event log path",
    )
    if action_path != expected_action_path or event_path != expected_event_path:
        raise ValueError(f"{label} summary artifact paths mismatch")
    diagnostic_path = _inside_output(
        str(expected_case / "optimizer_diagnostics.json"),
        output_root=output_root,
        label=f"{label} optimizer diagnostics path",
    )
    metrics = _read_jsonl(action_path)
    bootstrap_energy = _finite_float(
        summary.get("bootstrap_energy_eV"), label=f"{label} bootstrap energy"
    )
    completed_attempts = _positive_int(
        summary.get("completed_attempts"), label=f"{label} completed attempts"
    )
    if (
        len(metrics) != completed_attempts
        or summary.get("completed_batches") != completed_attempts
        or summary.get("posterior_observed_attempts") != completed_attempts
    ):
        raise ValueError(f"{label} action metric row count mismatch")
    actions: list[dict[str, Any]] = []
    cumulative_force = 0
    best_landing_energy = bootstrap_energy
    for ordinal, value in enumerate(metrics):
        action = _validate_action_metric(
            _mapping(value, label=f"{label} action {ordinal}"),
            ordinal=ordinal,
            label=f"{label} action {ordinal}",
        )
        cumulative_force += action["force_evaluations"]
        if action["metric"].get("cumulative_action_force_evaluations") != cumulative_force:
            raise ValueError(f"{label} cumulative action FE ledger mismatch")
        best_landing_energy = min(best_landing_energy, action["landing_energy_eV"])
        if not isclose(
            action["best_landing_energy_eV"],
            best_landing_energy,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(f"{label} best landing energy running-min mismatch")
        actions.append(action)
    if len({action["action_id"] for action in actions}) != len(actions):
        raise ValueError(f"{label} action IDs are not unique")
    event_values = _read_jsonl(event_path)
    if len(event_values) != 3 * completed_attempts:
        raise ValueError(f"{label} event replay row count mismatch")
    current_counts: dict[int, list[int]] = {0: [0, 0]}
    reconstructed_energies: dict[int, float] = {0: bootstrap_energy}
    for ordinal, action in enumerate(actions):
        metric = action["metric"]
        starter_id, discovered, inserted, landing_entry_id = _validate_event_triplet(
            event_values[3 * ordinal : 3 * ordinal + 3],
            action,
            ordinal=ordinal,
            master_seed=master_seed,
            current_counts=current_counts,
            label=f"{label} batch {ordinal}",
        )
        if starter_id not in current_counts:
            raise ValueError(f"{label} selected unknown starter")
        successes, failures = current_counts[starter_id]
        posterior_before = _mapping(
            metric.get("posterior_before"), label=f"{label} action {ordinal} posterior"
        )
        if posterior_before != {
            "successes": successes,
            "failures": failures,
            "mean": (1.0 + successes) / (2.0 + successes + failures),
        }:
            raise ValueError(f"{label} posterior before-state mismatch")
        current_counts[starter_id][0 if discovered else 1] += 1
        if inserted:
            if landing_entry_id in current_counts:
                raise ValueError(f"{label} duplicated inserted archive entry")
            current_counts[landing_entry_id] = [0, 0]
            reconstructed_energies[landing_entry_id] = action["landing_energy_eV"]
        elif landing_entry_id not in reconstructed_energies:
            raise ValueError(f"{label} unknown noninserted landing entry")
    action_evaluations = _nonnegative_int(
        summary.get("action_evaluations"), label=f"{label} action evaluations"
    )
    bootstrap_evaluations = _nonnegative_int(
        summary.get("bootstrap_evaluations"), label=f"{label} bootstrap evaluations"
    )
    total_evaluations = _nonnegative_int(
        summary.get("total_evaluations"), label=f"{label} total evaluations"
    )
    unused_force_budget = _nonnegative_int(
        summary.get("unused_force_budget"), label=f"{label} unused budget"
    )
    if (
        action_evaluations != cumulative_force
        or bootstrap_evaluations + action_evaluations != total_evaluations
        or total_evaluations > TOTAL_FORCE_BUDGET
        or unused_force_budget != TOTAL_FORCE_BUDGET - total_evaluations
    ):
        raise ValueError(f"{label} total FE ledger does not close")
    purpose_counts = _purpose_counts(summary.get("purpose_counts"), label=label)
    if sum(purpose_counts.values()) != total_evaluations or purpose_counts["unattributed"] != 0:
        raise ValueError(f"{label} total FE purpose ledger does not close")
    action_purposes = {
        purpose: sum(action["counts"][purpose] for action in actions)
        for purpose in PURPOSES
    }
    residual = {
        purpose: purpose_counts[purpose] - action_purposes[purpose]
        for purpose in PURPOSES
    }
    if (
        any(value < 0 for value in residual.values())
        or sum(residual.values()) != bootstrap_evaluations
        or residual["bootstrap_true_quench"] != bootstrap_evaluations - 1
        or residual["post_relax_validation"] != 1
        or any(
            residual[purpose] != 0
            for purpose in PURPOSES
            if purpose not in {"bootstrap_true_quench", "post_relax_validation"}
        )
    ):
        raise ValueError(f"{label} bootstrap/action purpose ledger does not close")
    factory = _mapping(summary.get("calculator_factory"), label=f"{label} calculator factory")
    if factory != {
        "bootstrap_instances": 1,
        "action_instances": 1,
        "action_thread_count": 1,
    }:
        raise ValueError(f"{label} calculator factory/thread protocol mismatch")
    archive_entries = _nonnegative_int(
        summary.get("archive_entries"), label=f"{label} archive entries"
    )
    inserted_count = sum(bool(action["inserted_into_archive"]) for action in actions)
    if archive_entries != inserted_count + 1:
        raise ValueError(f"{label} archive insertion ledger mismatch")
    duplicate_rate = _finite_float(
        summary.get("duplicate_rate"), label=f"{label} duplicate rate"
    )
    expected_duplicate = 1.0 - archive_entries / (completed_attempts + 1)
    if not isclose(duplicate_rate, expected_duplicate, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(f"{label} duplicate rate mismatch")
    final_posterior = _sequence(
        summary.get("final_posterior"), label=f"{label} final posterior"
    )
    reported: dict[int, tuple[int, int, float, float]] = {}
    for position, value in enumerate(final_posterior):
        row = _mapping(value, label=f"{label} final posterior row {position}")
        entry_id = _nonnegative_int(row.get("starter_id"), label=f"{label} posterior ID")
        if entry_id in reported:
            raise ValueError(f"{label} duplicate final posterior entry")
        reported[entry_id] = (
            _nonnegative_int(row.get("successes"), label=f"{label} posterior successes"),
            _nonnegative_int(row.get("failures"), label=f"{label} posterior failures"),
            _finite_float(row.get("mean"), label=f"{label} posterior mean"),
            _finite_float(row.get("energy_eV"), label=f"{label} posterior energy"),
        )
    if set(reported) != set(reconstructed_energies):
        raise ValueError(f"{label} final posterior starter set mismatch")
    for entry_id, energy in reconstructed_energies.items():
        successes, failures = current_counts[entry_id]
        expected = (
            successes,
            failures,
            (1.0 + successes) / (2.0 + successes + failures),
            energy,
        )
        if reported[entry_id] != expected:
            raise ValueError(f"{label} final posterior reconstruction mismatch")
    best_energy = _finite_float(
        summary.get("best_archive_energy_eV"), label=f"{label} best archive energy"
    )
    expected_best = min([bootstrap_energy, *(action["landing_energy_eV"] for action in actions)])
    if not isclose(best_energy, expected_best, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(f"{label} best archive energy mismatch")
    campaign_wall = _finite_float(
        summary.get("campaign_wall_time_s"), label=f"{label} campaign wall time"
    )
    bootstrap_wall = _finite_float(
        summary.get("bootstrap_evaluator_wall_time_s"), label=f"{label} bootstrap wall time"
    )
    if campaign_wall < 0.0 or bootstrap_wall < 0.0:
        raise ValueError(f"{label} wall time must be nonnegative")
    fallback = _validate_diagnostics(diagnostic_path, actions, label=label)
    return {
        "master_seed": master_seed,
        "proposal_fmax": matrix_row["proposal_fmax"],
        "completed_attempts": completed_attempts,
        "failed_attempts": 0,
        "total_force_evaluations": total_evaluations,
        "bootstrap_force_evaluations": bootstrap_evaluations,
        "action_force_evaluations": action_evaluations,
        "unused_force_budget": unused_force_budget,
        "campaign_wall_time_s": campaign_wall,
        "bootstrap_evaluator_wall_time_s": bootstrap_wall,
        "archive_entries": archive_entries,
        "duplicate_rate": duplicate_rate,
        "bootstrap_energy_eV": bootstrap_energy,
        "best_archive_energy_eV": best_energy,
        "best_energy_drop_eV": bootstrap_energy - best_energy,
        "purpose_counts": purpose_counts,
        "quench_fallback_attempts": fallback["quench_fallback_attempts"],
        "quench_fallback_converged": fallback["quench_fallback_converged"],
        "true_quench_unconverged": 0,
        "benchmark_eligible": True,
    }


def _describe(values: Sequence[float], *, sign_counts: bool = False) -> dict[str, Any]:
    if not values:
        raise ValueError("cannot describe an empty sample")
    result: dict[str, Any] = {
        "n": len(values),
        "mean": fsum(values) / len(values),
        "median": float(median(values)),
    }
    if sign_counts:
        result["sign_counts"] = {
            "negative": sum(value < 0.0 for value in values),
            "zero": sum(value == 0.0 for value in values),
            "positive": sum(value > 0.0 for value in values),
        }
    return result


CORE_FIELDS = (
    "completed_attempts",
    "failed_attempts",
    "total_force_evaluations",
    "bootstrap_force_evaluations",
    "action_force_evaluations",
    "unused_force_budget",
    "campaign_wall_time_s",
    "archive_entries",
    "duplicate_rate",
    "best_energy_drop_eV",
    "quench_fallback_attempts",
    "quench_fallback_converged",
    "true_quench_unconverged",
)


def _aggregate_campaigns(campaigns: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        **{
            field: _describe([float(campaign[field]) for campaign in campaigns])
            for field in CORE_FIELDS
        },
        "purpose_counts": {
            purpose: _describe(
                [float(campaign["purpose_counts"][purpose]) for campaign in campaigns]
            )
            for purpose in PURPOSES
        },
    }


def _aggregate(
    campaigns: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Any]]]]
) -> dict[str, Any]:
    by_system_arm: dict[str, dict[str, Any]] = {}
    paired: dict[str, Any] = {}
    strict_arm, loose_arm = (arm_id for arm_id, _ in ARMS)
    for system in SYSTEMS:
        by_system_arm[system] = {}
        for arm_id, proposal_fmax in ARMS:
            rows = [campaigns[system][str(seed)][arm_id] for seed in MASTER_SEEDS]
            by_system_arm[system][arm_id] = {
                "proposal_fmax": proposal_fmax,
                "per_seed": rows,
                "descriptive": _aggregate_campaigns(rows),
            }
        per_seed: list[dict[str, Any]] = []
        deltas: dict[str, list[float]] = {field: [] for field in CORE_FIELDS}
        purpose_deltas: dict[str, list[float]] = {purpose: [] for purpose in PURPOSES}
        for seed in MASTER_SEEDS:
            strict = campaigns[system][str(seed)][strict_arm]
            loose = campaigns[system][str(seed)][loose_arm]
            delta = {field: float(loose[field]) - float(strict[field]) for field in CORE_FIELDS}
            delta_purposes = {
                purpose: float(loose["purpose_counts"][purpose])
                - float(strict["purpose_counts"][purpose])
                for purpose in PURPOSES
            }
            per_seed.append(
                {
                    "master_seed": seed,
                    "loose_minus_strict": {**delta, "purpose_counts": delta_purposes},
                }
            )
            for field, value in delta.items():
                deltas[field].append(value)
            for purpose, value in delta_purposes.items():
                purpose_deltas[purpose].append(value)
        paired[system] = {
            "strict_arm": strict_arm,
            "loose_arm": loose_arm,
            "seeds": list(MASTER_SEEDS),
            "per_seed": per_seed,
            "descriptive": {
                **{field: _describe(values, sign_counts=True) for field, values in deltas.items()},
                "purpose_counts": {
                    purpose: _describe(values, sign_counts=True)
                    for purpose, values in purpose_deltas.items()
                },
            },
        }
    return {
        "by_system_arm": by_system_arm,
        "paired_loose_minus_strict": paired,
    }


def analyze(*, output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    """Validate exactly the published 2 x 4 x 2 matrix and describe it."""
    requested_output_root = Path(output_root)
    if "partial" in requested_output_root.name:
        raise ValueError("refusing to analyze a partial output root")
    output_root = requested_output_root.resolve()
    if any("partial" in component for component in output_root.parts):
        raise ValueError("refusing to analyze a partial output root")
    if not output_root.is_dir():
        raise ValueError(f"final output root does not exist: {output_root}")
    manifest = _mapping(_read_json(output_root / "manifest.json"), label="manifest")
    provenance = _validate_manifest(manifest)
    index = _mapping(_read_json(output_root / "index.json"), label="index")
    if index.get("schema_version") != 1:
        raise ValueError("index schema mismatch")
    if _mapping(index.get("manifest"), label="index manifest") != manifest:
        raise ValueError("index/manifest exact mismatch")
    index_campaigns = _sequence(index.get("campaigns"), label="campaign index")
    matrix = _expected_matrix()
    if len(index_campaigns) != len(matrix):
        raise ValueError("campaign matrix is incomplete or unexpected")
    campaigns: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
        system: {str(seed): {} for seed in MASTER_SEEDS} for system in SYSTEMS
    }
    for matrix_row, value in zip(matrix, index_campaigns, strict=True):
        entry = _mapping(value, label=f"campaign index {matrix_row['matrix_index']}")
        if any(
            entry.get(field) != matrix_row[field]
            for field in PROJECTION_IDENTITY_FIELDS
        ):
            raise ValueError("campaign index arm identity mismatch")
        expected_keys = set(matrix_row) | {"projection", "campaign_summary_path"}
        if set(entry) != expected_keys or any(
            entry[field] != expected for field, expected in matrix_row.items()
        ):
            raise ValueError("campaign matrix is incomplete or unexpected")
        system = str(matrix_row["system"])
        seed = int(matrix_row["master_seed"])
        arm_index = int(matrix_row["arm_index"])
        arm_id = str(matrix_row["arm_id"])
        campaign = _validate_campaign(
            output_root=output_root,
            index_entry=entry,
            matrix_row=matrix_row,
            manifest_projection=provenance["projections"][system][arm_index],
        )
        campaigns[system][str(seed)][arm_id] = campaign
    for system in SYSTEMS:
        for seed in MASTER_SEEDS:
            if set(campaigns[system][str(seed)]) != {arm_id for arm_id, _ in ARMS}:
                raise ValueError("campaign matrix has a missing paired arm")
    all_campaigns = [
        campaigns[system][str(seed)][arm_id]
        for system in SYSTEMS
        for seed in MASTER_SEEDS
        for arm_id, _ in ARMS
    ]
    aggregates = _aggregate(campaigns)
    overall_purposes = {
        purpose: sum(campaign["purpose_counts"][purpose] for campaign in all_campaigns)
        for purpose in PURPOSES
    }
    total_force = sum(campaign["total_force_evaluations"] for campaign in all_campaigns)
    return {
        "schema_version": 1,
        "validation": {
            "status": "passed",
            "campaign_count": len(all_campaigns),
            "execution_commit": EXPECTED_EXECUTION_COMMIT,
            "matrix": {
                "systems": list(SYSTEMS),
                "master_seeds": list(MASTER_SEEDS),
                "arms": [arm_id for arm_id, _ in ARMS],
            },
            "protocol": {
                "policy_name": POLICY_NAME,
                "action_force_budget": ACTION_FORCE_BUDGET,
                "total_force_budget": TOTAL_FORCE_BUDGET,
                "batch_size": 1,
                "max_workers": 1,
                "proposal_optimizer": "safe-lbfgs-total",
                "true_quench_optimizer": "ase-lbfgs",
                "true_quench_fallback_optimizer": "ase-fire",
                "true_quench_fmax_eV_per_A": 0.01,
                "true_quench_maxiter": 400,
            },
            "provenance": provenance,
            "index_manifest_exact": "passed",
            "published_artifact_paths": "passed",
            "event_action_summary_fe_closure": "passed",
            "uniform_probability_reconstruction": "passed",
            "optimizer_diagnostics": "passed",
        },
        "campaigns": campaigns,
        "aggregates": aggregates,
        "overall": {
            "completed_campaigns": len(all_campaigns),
            "completed_attempts": sum(campaign["completed_attempts"] for campaign in all_campaigns),
            "failed_attempts": 0,
            "total_force_evaluations": total_force,
            "total_wall_time_s": sum(campaign["campaign_wall_time_s"] for campaign in all_campaigns),
            "purpose_counts": overall_purposes,
            "purpose_fractions": {
                purpose: overall_purposes[purpose] / total_force for purpose in PURPOSES
            },
            "quench_fallback_attempts": sum(
                campaign["quench_fallback_attempts"] for campaign in all_campaigns
            ),
            "quench_fallback_converged": sum(
                campaign["quench_fallback_converged"] for campaign in all_campaigns
            ),
            "true_quench_unconverged": 0,
        },
        "diagnostic_scope": {
            "optimizer_diagnostics": "action-scoped and final-backend-only",
            "true_quench_cost": "purpose_ledger_not_final_only_optimizer_diagnostics",
            "reason": (
                "Fallback replaces primary-attempt optimizer telemetry, so its "
                "diagnostic evaluator-call field is not an additive FE ledger."
            ),
        },
        "claim_boundary": {
            "comparison": "paired descriptive loose-minus-strict summaries only",
            "sample_size": "four paired seeds per system",
            "gpu_nondeterminism": "not controlled away and may amplify through PES search",
            "inference": "no inferential p-values are calculated",
            "ranking": "not established",
            "default_decision": "not established",
        },
    }


def render_conclusion(evidence: Mapping[str, Any]) -> str:
    """Render a compact exact table without turning descriptive data into a decision."""
    validation = _mapping(evidence.get("validation"), label="evidence validation")
    aggregates = _mapping(evidence.get("aggregates"), label="evidence aggregates")
    by_system_arm = _mapping(aggregates.get("by_system_arm"), label="arm aggregates")
    paired = _mapping(
        aggregates.get("paired_loose_minus_strict"), label="paired aggregates"
    )
    overall = _mapping(evidence.get("overall"), label="evidence overall")
    lines = [
        "# Proposal-fmax fixed-budget GPU ablation",
        "",
        "## Verified execution facts",
        "",
        (
            f"- {validation['campaign_count']}/16 published campaigns passed exact "
            "manifest/index, output-path, event/action/summary FE-ledger, uniform "
            "snapshot, and action-scoped optimizer-diagnostic validation."
        ),
        (
            "- Both arms use `safe-lbfgs-total`; true quench is `ase-lbfgs` with "
            "one `ase-fire` fallback, `fmax=0.01 eV/A`, and `maxiter=400`."
        ),
        (
            f"- {overall['completed_attempts']} actions completed; failed actions "
            f"are {overall['failed_attempts']}; diagnostics report "
            f"{overall['quench_fallback_converged']}/"
            f"{overall['quench_fallback_attempts']} converged fallback uses."
        ),
        "",
        "| system | arm | n | attempts mean/median | total FE mean/median | unused FE mean/median | wall s mean/median | archive mean/median | duplicate mean/median | best drop eV mean/median | fallback mean/median |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for system in SYSTEMS:
        system_arms = _mapping(by_system_arm[system], label=f"{system} arm aggregates")
        for arm_id, _ in ARMS:
            aggregate = _mapping(system_arms[arm_id], label=f"{system}/{arm_id} aggregate")
            descriptive = _mapping(aggregate["descriptive"], label=f"{system}/{arm_id} descriptive")
            def pair(field: str) -> str:
                stats = _mapping(descriptive[field], label=f"{system}/{arm_id} {field}")
                return f"{stats['mean']:.6f}/{stats['median']:.6f}"
            lines.append(
                "| "
                + " | ".join(
                    (
                        system,
                        arm_id,
                        str(_mapping(descriptive["completed_attempts"], label="attempt count")["n"]),
                        pair("completed_attempts"),
                        pair("total_force_evaluations"),
                        pair("unused_force_budget"),
                        pair("campaign_wall_time_s"),
                        pair("archive_entries"),
                        pair("duplicate_rate"),
                        pair("best_energy_drop_eV"),
                        pair("quench_fallback_attempts"),
                    )
                )
                + " |"
            )
    lines.extend(
        (
            "",
            "## Purpose-ledger FE mean/median",
            "",
            "| system | arm | bootstrap true quench | starter true quench | direction oracle | escape true PES | biased proposal relax | landing true quench | post-relax validation | unattributed |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        )
    )
    for system in SYSTEMS:
        system_arms = _mapping(by_system_arm[system], label=f"{system} purpose aggregates")
        for arm_id, _ in ARMS:
            aggregate = _mapping(system_arms[arm_id], label=f"{system}/{arm_id} purpose aggregate")
            descriptive = _mapping(aggregate["descriptive"], label=f"{system}/{arm_id} purpose descriptive")
            purpose_stats = _mapping(descriptive["purpose_counts"], label=f"{system}/{arm_id} purposes")
            values = []
            for purpose in PURPOSES:
                stats = _mapping(purpose_stats[purpose], label=f"{system}/{arm_id} {purpose}")
                values.append(f"{stats['mean']:.6f}/{stats['median']:.6f}")
            lines.append(f"| {system} | {arm_id} | " + " | ".join(values) + " |")
    lines.extend(
        (
            "",
            "## Paired loose-minus-strict (0.10 - 0.05)",
            "",
            "| system | n | d total FE mean/median | signs -/0/+ | d best drop eV mean/median | signs -/0/+ | d wall s mean/median | signs -/0/+ |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        )
    )
    for system in SYSTEMS:
        entry = _mapping(paired[system], label=f"{system} paired aggregate")
        descriptive = _mapping(entry["descriptive"], label=f"{system} paired descriptive")
        def delta(field: str) -> tuple[str, str]:
            stats = _mapping(descriptive[field], label=f"{system} paired {field}")
            signs = _mapping(stats["sign_counts"], label=f"{system} paired signs")
            return (
                f"{stats['mean']:.6f}/{stats['median']:.6f}",
                f"{signs['negative']}/{signs['zero']}/{signs['positive']}",
            )
        total, total_signs = delta("total_force_evaluations")
        drop, drop_signs = delta("best_energy_drop_eV")
        wall, wall_signs = delta("campaign_wall_time_s")
        lines.append(
            f"| {system} | {len(entry['seeds'])} | {total} | {total_signs} | "
            f"{drop} | {drop_signs} | {wall} | {wall_signs} |"
        )
    lines.extend(
        (
            "",
            "## Claim boundary",
            "",
            "- These are small-n (four paired seeds per system) descriptive summaries. GPU nondeterminism is not controlled away and can be amplified by PES exploration.",
            "- No inferential p-value is calculated; this analysis establishes neither an arm ranking nor a default decision.",
            "- FE totals come from the closed purpose ledger. Optimizer diagnostics are action-scoped final-backend telemetry and are not an additive cost ledger when fallback occurs.",
            "",
        )
    )
    return "\n".join(lines)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE_PATH)
    parser.add_argument("--conclusion", type=Path, default=DEFAULT_CONCLUSION_PATH)
    args = parser.parse_args()
    evidence = analyze(output_root=args.output_root)
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    args.conclusion.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.evidence, evidence)
    args.conclusion.write_text(render_conclusion(evidence), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": evidence["validation"]["status"],
                "campaigns": evidence["validation"]["campaign_count"],
                "evidence": str(args.evidence),
                "conclusion": str(args.conclusion),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
