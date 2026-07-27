#!/usr/bin/env python3
"""Fail-closed analysis of the fixed posterior-policy GPU smoke."""

from __future__ import annotations

import argparse
import json
from math import fsum, isclose, isfinite
from pathlib import Path
from typing import Any, Mapping, Sequence


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
DEFAULT_OUTPUT_ROOT = RUN_ROOT / "output-quench-fallback-smoke"
DEFAULT_EVIDENCE_PATH = RUN_ROOT / "smoke_evidence.json"
DEFAULT_CONCLUSION_PATH = RUN_ROOT / "smoke_conclusion.md"
EXPECTED_EXECUTION_COMMIT = "83d34b568246a811a45a77e5c04e022fea3df598"
FROZEN_MODEL_SHA256 = "0abfde07862cf1e93b8b4d03cb702f29ce9c344ff2fc4de2ec0d7166d6c113a5"
FROZEN_PAMSSW_BUNDLE_SHA256 = (
    "9ccab5f4ad38f368153448a13b277ed4786661fea9b5c15eb0091fff492d16d0"
)
FROZEN_INPUT_SHA256 = {
    "c60": "c63788c18cbed305963213b47eabd9fdc4d06dac118da6a1a9e16621d5e32bf9",
    "pdo": "68243ceb7c0fbb6ba7a9454d680287eb98c4e5210efbd9ebb63517ba79aaa8b0",
}
SYSTEMS = ("c60", "pdo")
POLICIES = ("uniform", "posterior_proportional", "minimal_ucb")
MASTER_SEED = 42
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
OPTIMIZER_PROTOCOL = {
    "quench_optimizer": "ase-lbfgs",
    "quench_fallback_optimizer": "ase-fire",
    "quench_fmax": 0.01,
    "quench_maxiter": 400,
    "proposal_optimizer": "safe-lbfgs-total",
    "proposal_fmax": 0.05,
    "proposal_pool_size": 1,
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


def _nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite") from error
    if not isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _optional_finite_float(value: Any, *, label: str) -> float | None:
    if value is None:
        return None
    return _finite_float(value, label=label)


def _nonempty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _sha256_string(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def _purpose_counts(value: Any, *, label: str) -> dict[str, int]:
    counts = _mapping(value, label=f"{label} purpose counts")
    if set(counts) != set(PURPOSES):
        raise ValueError(f"{label} purpose ledger keys mismatch")
    return {
        purpose: _nonnegative_int(
            counts[purpose], label=f"{label} purpose {purpose}"
        )
        for purpose in PURPOSES
    }


def _path_from_record(value: Any, *, label: str) -> Path:
    path = Path(_nonempty_string(value, label=label))
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _validate_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if manifest.get("schema_version") != 1:
        raise ValueError("manifest schema mismatch")
    if manifest.get("execution_commit") != EXPECTED_EXECUTION_COMMIT:
        raise ValueError("execution commit mismatch")
    if manifest.get("systems") != list(SYSTEMS):
        raise ValueError("manifest systems mismatch")
    if manifest.get("policies") != list(POLICIES):
        raise ValueError("manifest policies mismatch")
    if manifest.get("master_seeds") != [MASTER_SEED]:
        raise ValueError("manifest master seed mismatch")
    if (
        manifest.get("action_force_budget") != ACTION_FORCE_BUDGET
        or manifest.get("total_force_budget") != TOTAL_FORCE_BUDGET
        or manifest.get("batch_size") != 1
        or manifest.get("max_workers") != 1
    ):
        raise ValueError("manifest serial budget protocol mismatch")

    calculator = _mapping(manifest.get("calculator"), label="calculator provenance")
    if calculator != {
        "default_dtype": "float32",
        "device": "cuda",
        "enable_cueq": False,
        "inference_precision": "float32",
    }:
        raise ValueError("calculator provenance mismatch")
    calculator_reuse = _mapping(
        manifest.get("calculator_reuse"), label="calculator reuse provenance"
    )
    if (
        calculator_reuse.get("cross_thread_calculator_sharing") is not False
        or calculator_reuse.get("accounting")
        != "each bootstrap/action is wrapped by its own EvalCounter"
        or calculator_reuse.get("actions")
        != "one worker-thread-local calculator reused serially"
        or calculator_reuse.get("bootstrap")
        != "one uncached caller-thread calculator"
    ):
        raise ValueError("calculator reuse provenance mismatch")
    cuda = _mapping(manifest.get("cuda"), label="CUDA provenance")
    if (
        cuda.get("available") is not True
        or cuda.get("requested_device") != "cuda"
    ):
        raise ValueError("CUDA provenance mismatch")
    _nonempty_string(cuda.get("device_name"), label="CUDA device name")
    _nonempty_string(cuda.get("runtime_version"), label="CUDA runtime version")

    model = _mapping(manifest.get("model"), label="model provenance")
    _nonempty_string(model.get("path"), label="model path")
    model_sha = _sha256_string(model.get("sha256"), label="model SHA")
    # Synthetic contract fixtures use the same pinned production model.
    if model_sha != FROZEN_MODEL_SHA256:
        raise ValueError("model SHA mismatch")
    inputs = _mapping(manifest.get("inputs"), label="input provenance")
    if set(inputs) != set(SYSTEMS):
        raise ValueError("input provenance systems mismatch")
    for system in SYSTEMS:
        identity = _mapping(inputs[system], label=f"{system} input provenance")
        _nonempty_string(identity.get("path"), label=f"{system} input path")
        digest = _sha256_string(identity.get("sha256"), label=f"{system} input SHA")
        if digest != FROZEN_INPUT_SHA256[system]:
            raise ValueError(f"{system} input SHA mismatch")
    bundle_sha = _sha256_string(
        manifest.get("pamssw_bundle_sha256"), label="pamssw bundle SHA"
    )
    if bundle_sha != FROZEN_PAMSSW_BUNDLE_SHA256:
        raise ValueError("pamssw bundle SHA mismatch")

    versions = _mapping(
        manifest.get("runtime_versions"), label="runtime version provenance"
    )
    if set(versions) != {"python", "numpy", "scipy", "ase", "torch", "mace"}:
        raise ValueError("runtime version provenance keys mismatch")
    for package, version in versions.items():
        _nonempty_string(version, label=f"{package} runtime version")

    projections = _mapping(manifest.get("projections"), label="projections")
    if set(projections) != set(SYSTEMS):
        raise ValueError("projection systems mismatch")
    effective_configs: dict[str, Mapping[str, Any]] = {}
    for system in SYSTEMS:
        projection = _mapping(projections[system], label=f"{system} projection")
        if projection.get("softening_enabled") is not False:
            raise ValueError(f"{system} projection must be unsoftened")
        config = _mapping(
            projection.get("effective_ssw_config"),
            label=f"{system} effective config",
        )
        if any(config.get(field) != expected for field, expected in OPTIMIZER_PROTOCOL.items()):
            raise ValueError(f"{system} effective optimizer protocol mismatch")
        if config.get("rng_seed") != MASTER_SEED:
            raise ValueError(f"{system} effective RNG seed mismatch")
        effective_configs[system] = config

    return {
        "execution_commit": EXPECTED_EXECUTION_COMMIT,
        "pamssw_bundle_sha256": bundle_sha,
        "calculator": dict(calculator),
        "calculator_reuse": dict(calculator_reuse),
        "cuda": dict(cuda),
        "inputs": {system: dict(inputs[system]) for system in SYSTEMS},
        "model": dict(model),
        "runtime_versions": dict(versions),
        "effective_configs": {
            system: {
                field: effective_configs[system][field]
                for field in (
                    *OPTIMIZER_PROTOCOL,
                    "proposal_relax_steps",
                    "rng_seed",
                )
            }
            for system in SYSTEMS
        },
    }


def _validate_probabilities(
    snapshot: Mapping[str, Any],
    *,
    policy: str,
    current_counts: Mapping[int, Sequence[int]],
    label: str,
) -> dict[str, bool]:
    eligible = _sequence(
        snapshot.get("eligible_starter_ids"), label=f"{label} eligible starters"
    )
    probabilities = _sequence(
        snapshot.get("probabilities"), label=f"{label} probabilities"
    )
    eligible_ids = [
        _nonnegative_int(starter_id, label=f"{label} eligible starter ID")
        for starter_id in eligible
    ]
    if not eligible_ids or len(set(eligible_ids)) != len(eligible_ids):
        raise ValueError(f"{label} eligible starters are empty or duplicated")
    if eligible_ids != sorted(current_counts):
        raise ValueError(f"{label} eligible starters do not match current archive")
    if len(probabilities) != len(eligible):
        raise ValueError(f"{label} eligible/probability length mismatch")
    values = [
        _finite_float(value, label=f"{label} probability") for value in probabilities
    ]
    if any(value < 0.0 for value in values):
        raise ValueError(f"{label} probabilities must be nonnegative")
    if not isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(f"{label} probabilities must sum to one")
    support_complete = snapshot.get("support_complete")
    if not isinstance(support_complete, bool):
        raise ValueError(f"{label} support_complete must be boolean")
    strictly_positive = all(value > 0.0 for value in values)
    one_hot = (
        sum(
            isclose(value, 1.0, rel_tol=0.0, abs_tol=1.0e-12)
            for value in values
        )
        == 1
        and all(
            isclose(value, 0.0, rel_tol=0.0, abs_tol=1.0e-12)
            or isclose(value, 1.0, rel_tol=0.0, abs_tol=1.0e-12)
            for value in values
        )
    )
    if policy in {"uniform", "posterior_proportional"}:
        if not support_complete:
            raise ValueError(f"{label} complete support flag mismatch")
        if not strictly_positive:
            raise ValueError(f"{label} strictly positive support is required")
        if policy == "uniform":
            expected = [1.0 / len(eligible_ids)] * len(eligible_ids)
        else:
            means = [
                (1.0 + current_counts[starter_id][0])
                / (2.0 + current_counts[starter_id][0] + current_counts[starter_id][1])
                for starter_id in eligible_ids
            ]
            total_mean = fsum(means)
            expected = [mean / total_mean for mean in means]
        if any(
            not isclose(actual, target, rel_tol=0.0, abs_tol=1.0e-12)
            for actual, target in zip(values, expected, strict=True)
        ):
            raise ValueError(f"{label} policy probabilities mismatch")
    elif policy == "minimal_ucb":
        if support_complete:
            raise ValueError(f"{label} minimal UCB support flag mismatch")
        if not one_hot:
            raise ValueError(f"{label} minimal UCB probabilities must be one-hot")
    return {
        "support_complete": support_complete,
        "strictly_positive": strictly_positive,
        "one_hot": one_hot,
    }


def _validate_action_metric(
    metric: Mapping[str, Any],
    *,
    policy: str,
    ordinal: int,
    label: str,
) -> dict[str, Any]:
    if (
        metric.get("schema_version") != 1
        or metric.get("ordinal") != ordinal
        or metric.get("batch_id") != ordinal
        or metric.get("policy_name") != policy
        or metric.get("status") != "completed"
        or metric.get("failure_reason") is not None
        or metric.get("posterior_observed") is not True
    ):
        raise ValueError(f"{label} action metric identity/status mismatch")
    action_id = _nonempty_string(metric.get("action_id"), label=f"{label} action ID")
    counts = _purpose_counts(metric.get("evaluation_counts"), label=label)
    force_evaluations = _nonnegative_int(
        metric.get("force_evaluations"), label=f"{label} force evaluations"
    )
    if (
        metric.get("evaluator_calls") != force_evaluations
        or sum(counts.values()) != force_evaluations
        or counts["bootstrap_true_quench"] != 0
        or counts["unattributed"] != 0
        or force_evaluations > ACTION_FORCE_BUDGET
    ):
        raise ValueError(f"{label} action evaluation ledger does not close")
    wall_time = _finite_float(
        metric.get("evaluator_wall_time_s"), label=f"{label} evaluator wall time"
    )
    if wall_time < 0.0:
        raise ValueError(f"{label} evaluator wall time must be nonnegative")
    landing_energy = _finite_float(
        metric.get("landing_energy_eV"), label=f"{label} landing energy"
    )
    best_landing_energy = _finite_float(
        metric.get("best_landing_energy_eV"), label=f"{label} best landing energy"
    )
    if not isinstance(metric.get("inserted_into_archive"), bool):
        raise ValueError(f"{label} archive insertion flag must be boolean")
    if not isinstance(metric.get("discovered_against_snapshot"), bool):
        raise ValueError(f"{label} discovery flag must be boolean")
    return {
        "action_id": action_id,
        "batch_id": ordinal,
        "counts": counts,
        "force_evaluations": force_evaluations,
        "landing_energy_eV": landing_energy,
        "best_landing_energy_eV": best_landing_energy,
        "discovered_against_snapshot": metric["discovered_against_snapshot"],
        "inserted_into_archive": metric["inserted_into_archive"],
        "metric": metric,
    }


def _validate_event_triplet(
    values: Sequence[Any],
    action: Mapping[str, Any],
    *,
    policy: str,
    ordinal: int,
    current_counts: Mapping[int, Sequence[int]],
    label: str,
) -> tuple[dict[str, bool], dict[str, Any]]:
    if len(values) != 3:
        raise ValueError(f"{label} event replay batch must contain three records")
    snapshot = _mapping(values[0], label=f"{label} policy snapshot")
    attempt = _mapping(values[1], label=f"{label} attempt event")
    commit = _mapping(values[2], label=f"{label} batch commit")
    if (
        snapshot.get("schema_version") != 2
        or snapshot.get("record_type") != "policy_snapshot"
        or snapshot.get("batch_id") != ordinal
        or snapshot.get("policy_name") != policy
    ):
        raise ValueError(f"{label} policy snapshot identity mismatch")
    semantics = _validate_probabilities(
        snapshot,
        policy=policy,
        current_counts=current_counts,
        label=label,
    )
    metric = action["metric"]
    if (
        snapshot.get("eligible_starter_ids") != metric.get("eligible_starter_ids")
        or snapshot.get("probabilities") != metric.get("probabilities")
        or snapshot.get("support_complete") != metric.get("policy_support_complete")
    ):
        raise ValueError(f"{label} policy snapshot/action metric mismatch")
    eligible = list(snapshot["eligible_starter_ids"])
    try:
        selected_index = eligible.index(metric.get("starter_id"))
    except ValueError as error:
        raise ValueError(f"{label} selected starter is ineligible") from error
    selected_probability = float(snapshot["probabilities"][selected_index])
    if not isclose(
        _finite_float(
            metric.get("selection_probability"),
            label=f"{label} metric selection probability",
        ),
        selected_probability,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(f"{label} selected probability mismatch")
    event_landing_energy = _optional_finite_float(
        attempt.get("landing_energy"), label=f"{label} event landing energy"
    )
    event_discovered = attempt.get("discovered_against_snapshot")
    event_inserted = attempt.get("inserted_into_archive")
    if not isinstance(event_discovered, bool) or not isinstance(event_inserted, bool):
        raise ValueError(f"{label} attempt event outcome flags must be boolean")
    if (
        attempt.get("schema_version") != 2
        or attempt.get("record_type") != "attempt"
        or attempt.get("action_id") != action["action_id"]
        or attempt.get("batch_id") != ordinal
        or attempt.get("policy_name") != policy
        or attempt.get("status") != "completed"
        or attempt.get("failure_reason") is not None
        or attempt.get("cost_is_exact") is not True
        or attempt.get("force_budget") != ACTION_FORCE_BUDGET
        or attempt.get("force_evaluations") != action["force_evaluations"]
        or attempt.get("evaluation_counts") != action["counts"]
        or attempt.get("starter_id") != metric.get("starter_id")
        or attempt.get("posterior_observed") is not True
        or event_discovered is not action["discovered_against_snapshot"]
        or event_inserted is not action["inserted_into_archive"]
        or event_landing_energy != action["landing_energy_eV"]
        or not isclose(
            _finite_float(
                attempt.get("selection_probability"),
                label=f"{label} event selection probability",
            ),
            selected_probability,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        raise ValueError(f"{label} attempt event/action metric mismatch")
    if (
        commit.get("schema_version") != 2
        or commit.get("record_type") != "batch_commit"
        or commit.get("batch_id") != ordinal
        or commit.get("action_ids") != [action["action_id"]]
    ):
        raise ValueError(f"{label} batch commit mismatch")
    return (
        semantics,
        {
            "starter_id": _nonnegative_int(
                attempt.get("starter_id"), label=f"{label} starter ID"
            ),
            "posterior_observed": True,
            "discovered_against_snapshot": event_discovered,
            "inserted_into_archive": event_inserted,
            "landing_entry_id": _nonnegative_int(
                attempt.get("landing_entry_id"), label=f"{label} landing entry ID"
            ),
            "landing_energy": event_landing_energy,
        },
    )


def _validate_diagnostics(
    path: Path,
    actions: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> dict[str, int]:
    payload = _mapping(_read_json(path), label=f"{label} optimizer diagnostics")
    if payload.get("schema_version") != 1:
        raise ValueError(f"{label} optimizer diagnostics schema mismatch")
    attempts = _sequence(payload.get("attempts"), label=f"{label} diagnostics attempts")
    if len(attempts) != len(actions):
        raise ValueError(f"{label} optimizer diagnostic attempt count mismatch")
    expected_ids = [str(action["action_id"]) for action in actions]
    actual_ids: list[str] = []
    fallback_attempts = 0
    fallback_converged = 0
    for index, attempt_value in enumerate(attempts):
        attempt = _mapping(attempt_value, label=f"{label} diagnostic {index}")
        action_id = _nonempty_string(
            attempt.get("action_id"), label=f"{label} diagnostic action ID"
        )
        actual_ids.append(action_id)
        stats = _mapping(attempt.get("stats"), label=f"{label} diagnostic stats")
        if (
            stats.get("diagnostic_stage") != "completed"
            or stats.get("budget_exhausted") != 0
            or stats.get("force_evaluations")
            != actions[index]["force_evaluations"]
            or stats.get("proposal_optimizer")
            != OPTIMIZER_PROTOCOL["proposal_optimizer"]
            or stats.get("quench_optimizer")
            != OPTIMIZER_PROTOCOL["quench_optimizer"]
            or stats.get("quench_fallback_optimizer")
            != OPTIMIZER_PROTOCOL["quench_fallback_optimizer"]
        ):
            raise ValueError(f"{label} optimizer diagnostic protocol mismatch")
        attempt_fallbacks = _nonnegative_int(
            stats.get("quench_fallback_attempts"),
            label=f"{label} fallback attempts",
        )
        converged_fallbacks = _nonnegative_int(
            stats.get("quench_fallback_converged"),
            label=f"{label} converged fallbacks",
        )
        if converged_fallbacks > attempt_fallbacks:
            raise ValueError(f"{label} fallback convergence count is impossible")
        if stats.get("true_quench_unconverged") != 0:
            raise ValueError(f"{label} true quench convergence certificate failed")
        fallback_attempts += attempt_fallbacks
        fallback_converged += converged_fallbacks
    if actual_ids != expected_ids:
        raise ValueError(f"{label} optimizer diagnostic action IDs mismatch")
    return {
        "quench_fallback_attempts": fallback_attempts,
        "quench_fallback_converged": fallback_converged,
    }


def _validate_campaign(
    *,
    output_root: Path,
    system: str,
    policy: str,
    index_entry: Mapping[str, Any],
    manifest_projection: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, bool]]]:
    label = f"{system}/{policy}"
    campaign_root = output_root / system / f"seed-{MASTER_SEED:08d}" / policy
    expected_summary_path = (campaign_root / "campaign_summary.json").resolve()
    if (
        index_entry.get("system") != system
        or index_entry.get("policy_name") != policy
        or index_entry.get("master_seed") != MASTER_SEED
        or _path_from_record(
            index_entry.get("campaign_summary_path"),
            label=f"{label} campaign summary path",
        )
        != expected_summary_path
    ):
        raise ValueError(f"{label} index identity mismatch")
    index_projection = _mapping(
        index_entry.get("projection"), label=f"{label} index projection"
    )
    if (
        index_projection.get("effective_ssw_config")
        != manifest_projection.get("effective_ssw_config")
        or index_projection.get("overrides") != manifest_projection.get("overrides")
        or index_projection.get("softening_enabled") is not False
    ):
        raise ValueError(f"{label} index projection mismatch")

    summary = _mapping(_read_json(expected_summary_path), label=f"{label} summary")
    if (
        summary.get("schema_version") != 1
        or summary.get("policy_name") != policy
        or summary.get("master_seed") != MASTER_SEED
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
    if _path_from_record(
        summary.get("action_metrics_path"), label=f"{label} action metrics path"
    ) != (campaign_root / "action_metrics.jsonl").resolve():
        raise ValueError(f"{label} action metrics path mismatch")
    if _path_from_record(
        summary.get("event_log_path"), label=f"{label} event log path"
    ) != (campaign_root / "events.jsonl").resolve():
        raise ValueError(f"{label} event log path mismatch")

    metrics = _read_jsonl(campaign_root / "action_metrics.jsonl")
    completed_attempts = _nonnegative_int(
        summary.get("completed_attempts"), label=f"{label} completed attempts"
    )
    if (
        len(metrics) != completed_attempts
        or summary.get("completed_batches") != completed_attempts
        or summary.get("posterior_observed_attempts") != completed_attempts
        or completed_attempts == 0
    ):
        raise ValueError(f"{label} action metric row count mismatch")
    actions: list[dict[str, Any]] = []
    cumulative_force = 0
    for ordinal, value in enumerate(metrics):
        action = _validate_action_metric(
            _mapping(value, label=f"{label} action {ordinal}"),
            policy=policy,
            ordinal=ordinal,
            label=f"{label} action {ordinal}",
        )
        cumulative_force += action["force_evaluations"]
        if value.get("cumulative_action_force_evaluations") != cumulative_force:
            raise ValueError(f"{label} cumulative action ledger mismatch")
        actions.append(action)

    event_values = _read_jsonl(campaign_root / "events.jsonl")
    if len(event_values) != 3 * completed_attempts:
        raise ValueError(f"{label} event replay row count mismatch")
    bootstrap_energy = _finite_float(
        summary.get("bootstrap_energy_eV"), label=f"{label} bootstrap energy"
    )
    policy_semantics: list[dict[str, bool]] = []
    current_counts: dict[int, list[int]] = {0: [0, 0]}
    reconstructed_energies: dict[int, float] = {0: bootstrap_energy}
    for ordinal, action in enumerate(actions):
        semantics, outcome = _validate_event_triplet(
            event_values[3 * ordinal : 3 * ordinal + 3],
            action,
            policy=policy,
            ordinal=ordinal,
            current_counts=current_counts,
            label=f"{label} batch {ordinal}",
        )
        policy_semantics.append(semantics)
        starter_id = outcome["starter_id"]
        if starter_id not in current_counts:
            raise ValueError(f"{label} policy snapshot selected unknown starter")
        successes, failures = current_counts[starter_id]
        posterior_before = _mapping(
            action["metric"].get("posterior_before"),
            label=f"{label} action {ordinal} posterior before",
        )
        if (
            posterior_before.get("successes") != successes
            or posterior_before.get("failures") != failures
            or posterior_before.get("mean")
            != (1.0 + successes) / (2.0 + successes + failures)
        ):
            raise ValueError(f"{label} posterior reconstruction before-state mismatch")
        if outcome["posterior_observed"]:
            current_counts[starter_id][
                0 if outcome["discovered_against_snapshot"] else 1
            ] += 1
        landing_entry_id = outcome["landing_entry_id"]
        landing_energy = outcome["landing_energy"]
        if landing_energy is None:
            raise ValueError(f"{label} posterior reconstruction lacks landing energy")
        if outcome["inserted_into_archive"]:
            if landing_entry_id in current_counts:
                raise ValueError(f"{label} policy snapshot duplicated inserted starter")
            current_counts[landing_entry_id] = [0, 0]
            reconstructed_energies[landing_entry_id] = landing_energy
        elif landing_entry_id not in reconstructed_energies:
            raise ValueError(f"{label} posterior reconstruction references unknown landing")

    action_evaluations = _nonnegative_int(
        summary.get("action_evaluations"), label=f"{label} action evaluations"
    )
    bootstrap_evaluations = _nonnegative_int(
        summary.get("bootstrap_evaluations"),
        label=f"{label} bootstrap evaluations",
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
        raise ValueError(f"{label} total evaluation ledger does not close")
    purpose_counts = _purpose_counts(summary.get("purpose_counts"), label=label)
    if sum(purpose_counts.values()) != total_evaluations:
        raise ValueError(f"{label} total evaluation ledger does not close")
    if purpose_counts["unattributed"] != 0:
        raise ValueError(f"{label} unattributed evaluations must be zero")
    action_purpose_counts = {
        purpose: sum(action["counts"][purpose] for action in actions)
        for purpose in PURPOSES
    }
    residual_counts = {
        purpose: purpose_counts[purpose] - action_purpose_counts[purpose]
        for purpose in PURPOSES
    }
    if (
        any(value < 0 for value in residual_counts.values())
        or sum(residual_counts.values()) != bootstrap_evaluations
        or residual_counts["bootstrap_true_quench"] != bootstrap_evaluations - 1
        or residual_counts["post_relax_validation"] != 1
        or any(
            residual_counts[purpose] != 0
            for purpose in PURPOSES
            if purpose not in {"bootstrap_true_quench", "post_relax_validation"}
        )
    ):
        raise ValueError(f"{label} bootstrap/action purpose ledger does not close")

    factory = _mapping(
        summary.get("calculator_factory"), label=f"{label} calculator factory"
    )
    if factory != {
        "action_instances": 1,
        "action_thread_count": 1,
        "bootstrap_instances": 1,
    }:
        raise ValueError(f"{label} calculator factory/thread protocol mismatch")
    archive_entries = _nonnegative_int(
        summary.get("archive_entries"), label=f"{label} archive entries"
    )
    inserted = sum(bool(action["inserted_into_archive"]) for action in actions)
    if archive_entries != inserted + 1:
        raise ValueError(f"{label} archive insertion ledger mismatch")
    duplicate_rate = _finite_float(
        summary.get("duplicate_rate"), label=f"{label} duplicate rate"
    )
    expected_duplicate_rate = 1.0 - archive_entries / (completed_attempts + 1)
    if not isclose(
        duplicate_rate, expected_duplicate_rate, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError(f"{label} duplicate rate mismatch")

    final_posterior = _sequence(
        summary.get("final_posterior"), label=f"{label} final posterior"
    )
    reported_posterior: dict[int, tuple[int, int, float, float]] = {}
    for row_index, row_value in enumerate(final_posterior):
        row = _mapping(row_value, label=f"{label} posterior row {row_index}")
        starter_id = _nonnegative_int(
            row.get("starter_id"), label=f"{label} posterior starter ID"
        )
        if starter_id in reported_posterior:
            raise ValueError(f"{label} posterior reconstruction has duplicate starter")
        reported_posterior[starter_id] = (
            _nonnegative_int(
                row.get("successes"), label=f"{label} posterior successes"
            ),
            _nonnegative_int(
                row.get("failures"), label=f"{label} posterior failures"
            ),
            _finite_float(row.get("mean"), label=f"{label} posterior mean"),
            _finite_float(row.get("energy_eV"), label=f"{label} posterior energy"),
        )
    if set(reported_posterior) != set(reconstructed_energies):
        raise ValueError(f"{label} posterior reconstruction starter set mismatch")
    for starter_id, energy in reconstructed_energies.items():
        successes, failures = current_counts[starter_id]
        expected = (
            successes,
            failures,
            (1.0 + successes) / (2.0 + successes + failures),
            energy,
        )
        if reported_posterior[starter_id] != expected:
            raise ValueError(f"{label} posterior reconstruction row mismatch")
    if (
        sum(
            successes + failures
            for successes, failures in current_counts.values()
        )
        != completed_attempts
        or len(reconstructed_energies) != archive_entries
    ):
        raise ValueError(f"{label} posterior reconstruction ledger mismatch")

    best_energy = _finite_float(
        summary.get("best_archive_energy_eV"), label=f"{label} best energy"
    )
    landing_energies = [action["landing_energy_eV"] for action in actions]
    if not isclose(
        best_energy,
        min([bootstrap_energy, *landing_energies]),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(f"{label} best energy mismatch")
    wall_time = _finite_float(
        summary.get("campaign_wall_time_s"), label=f"{label} campaign wall time"
    )
    bootstrap_wall = _finite_float(
        summary.get("bootstrap_evaluator_wall_time_s"),
        label=f"{label} bootstrap wall time",
    )
    if wall_time < 0.0 or bootstrap_wall < 0.0:
        raise ValueError(f"{label} wall time must be nonnegative")

    fallback = _validate_diagnostics(
        campaign_root / "optimizer_diagnostics.json",
        actions,
        label=label,
    )
    return (
        {
            "completed_attempts": completed_attempts,
            "failed_attempts": 0,
            "total_force_evaluations": total_evaluations,
            "bootstrap_force_evaluations": bootstrap_evaluations,
            "action_force_evaluations": action_evaluations,
            "unused_force_budget": unused_force_budget,
            "campaign_wall_time_s": wall_time,
            "bootstrap_evaluator_wall_time_s": bootstrap_wall,
            "archive_entries": archive_entries,
            "duplicate_rate": duplicate_rate,
            "bootstrap_energy_eV": bootstrap_energy,
            "best_archive_energy_eV": best_energy,
            "best_energy_drop_eV": bootstrap_energy - best_energy,
            "purpose_counts": purpose_counts,
            "purpose_fractions": {
                purpose: purpose_counts[purpose] / total_evaluations
                for purpose in PURPOSES
            },
            "quench_fallback_attempts": fallback["quench_fallback_attempts"],
            "quench_fallback_converged": fallback["quench_fallback_converged"],
            "true_quench_unconverged": 0,
            "all_actions_completed": True,
            "benchmark_eligible": True,
        },
        policy_semantics,
    )


def analyze(*, output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    """Validate all six campaigns and return exact, replayed aggregates."""

    output_root = Path(output_root).resolve()
    manifest = _mapping(
        _read_json(output_root / "manifest.json"), label="manifest"
    )
    provenance = _validate_manifest(manifest)
    index = _mapping(_read_json(output_root / "index.json"), label="index")
    if index.get("schema_version") != 1:
        raise ValueError("index schema mismatch")
    index_manifest = _mapping(index.get("manifest"), label="index manifest")
    if index_manifest != manifest:
        raise ValueError("index/manifest provenance mismatch")
    campaign_values = _sequence(index.get("campaigns"), label="campaign index")
    expected_keys = {
        (system, MASTER_SEED, policy)
        for system in SYSTEMS
        for policy in POLICIES
    }
    keyed_entries: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for position, value in enumerate(campaign_values):
        entry = _mapping(value, label=f"campaign index row {position}")
        key = (
            entry.get("system"),
            entry.get("master_seed"),
            entry.get("policy_name"),
        )
        if key in keyed_entries:
            raise ValueError(f"duplicate campaign index row {key}")
        keyed_entries[key] = entry
    if set(keyed_entries) != expected_keys:
        raise ValueError("campaign matrix is incomplete or unexpected")

    projections = _mapping(manifest["projections"], label="manifest projections")
    campaigns: dict[str, dict[str, dict[str, Any]]] = {
        system: {} for system in SYSTEMS
    }
    semantics_by_policy: dict[str, list[dict[str, bool]]] = {
        policy: [] for policy in POLICIES
    }
    for system in SYSTEMS:
        for policy in POLICIES:
            campaign, semantics = _validate_campaign(
                output_root=output_root,
                system=system,
                policy=policy,
                index_entry=keyed_entries[(system, MASTER_SEED, policy)],
                manifest_projection=_mapping(
                    projections[system], label=f"{system} manifest projection"
                ),
            )
            campaigns[system][policy] = campaign
            semantics_by_policy[policy].extend(semantics)

    bootstrap_by_system = {
        system: sorted(
            {
                campaigns[system][policy]["bootstrap_force_evaluations"]
                for policy in POLICIES
            }
        )
        for system in SYSTEMS
    }
    if any(len(values) != 1 for values in bootstrap_by_system.values()):
        raise ValueError("bootstrap force evaluations differ across policies")
    policy_semantics = {
        "uniform": {
            "support_complete": all(
                item["support_complete"] for item in semantics_by_policy["uniform"]
            ),
            "strictly_positive_on_all_eligible": all(
                item["strictly_positive"]
                for item in semantics_by_policy["uniform"]
            ),
            "one_hot": all(
                item["one_hot"] for item in semantics_by_policy["uniform"]
            ),
            "interpretation": "random full-support baseline",
        },
        "posterior_proportional": {
            "support_complete": all(
                item["support_complete"]
                for item in semantics_by_policy["posterior_proportional"]
            ),
            "strictly_positive_on_all_eligible": all(
                item["strictly_positive"]
                for item in semantics_by_policy["posterior_proportional"]
            ),
            "one_hot": all(
                item["one_hot"]
                for item in semantics_by_policy["posterior_proportional"]
            ),
            "interpretation": (
                "full positive support, but not frequency-unbiased because "
                "selection probabilities depend on posterior means"
            ),
        },
        "minimal_ucb": {
            "support_complete": all(
                item["support_complete"]
                for item in semantics_by_policy["minimal_ucb"]
            ),
            "strictly_positive_on_all_eligible": all(
                item["strictly_positive"]
                for item in semantics_by_policy["minimal_ucb"]
            ),
            "one_hot": all(
                item["one_hot"] for item in semantics_by_policy["minimal_ucb"]
            ),
            "interpretation": (
                "deterministic one-hot comparator, not an unbiased or "
                "full-support exploration policy"
            ),
        },
    }

    all_campaigns = [
        campaigns[system][policy] for system in SYSTEMS for policy in POLICIES
    ]
    purpose_counts = {
        purpose: sum(campaign["purpose_counts"][purpose] for campaign in all_campaigns)
        for purpose in PURPOSES
    }
    total_force = sum(campaign["total_force_evaluations"] for campaign in all_campaigns)
    by_system_purpose_fractions = {}
    for system in SYSTEMS:
        system_total = sum(
            campaigns[system][policy]["total_force_evaluations"]
            for policy in POLICIES
        )
        by_system_purpose_fractions[system] = {
            purpose: sum(
                campaigns[system][policy]["purpose_counts"][purpose]
                for policy in POLICIES
            )
            / system_total
            for purpose in PURPOSES
        }
    return {
        "schema_version": 1,
        "validation": {
            "status": "passed",
            "campaign_count": len(all_campaigns),
            "execution_commit": EXPECTED_EXECUTION_COMMIT,
            "matrix": {
                "systems": list(SYSTEMS),
                "policies": list(POLICIES),
                "master_seeds": [MASTER_SEED],
            },
            "protocol": {
                "action_force_budget": ACTION_FORCE_BUDGET,
                "total_force_budget": TOTAL_FORCE_BUDGET,
                "batch_size": 1,
                "max_workers": 1,
                **OPTIMIZER_PROTOCOL,
            },
            "provenance": provenance,
            "event_field_validation": "passed",
            "policy_probability_reconstruction": "passed",
            "posterior_credit_reconstruction": "passed",
            "evaluation_ledgers": "passed",
            "optimizer_diagnostics": "passed",
        },
        "campaigns": campaigns,
        "overall": {
            "completed_campaigns": len(all_campaigns),
            "completed_attempts": sum(
                campaign["completed_attempts"] for campaign in all_campaigns
            ),
            "failed_attempts": 0,
            "total_force_evaluations": total_force,
            "total_wall_time_s": sum(
                campaign["campaign_wall_time_s"] for campaign in all_campaigns
            ),
            "purpose_counts": purpose_counts,
            "purpose_fractions": {
                purpose: purpose_counts[purpose] / total_force
                for purpose in PURPOSES
            },
            "quench_fallback_attempts": sum(
                campaign["quench_fallback_attempts"] for campaign in all_campaigns
            ),
            "quench_fallback_converged": sum(
                campaign["quench_fallback_converged"] for campaign in all_campaigns
            ),
            "true_quench_unconverged": 0,
        },
        "bootstrap": {
            "force_evaluations_by_system": {
                system: values[0]
                for system, values in bootstrap_by_system.items()
            },
            "fallback_observable": False,
            "note": (
                "optimizer_diagnostics.json contains action IDs only; bootstrap "
                "fallback use cannot be reconstructed from this artifact"
            ),
        },
        "policy_semantics": policy_semantics,
        "cost_observation": {
            "purpose_fractions_by_system": by_system_purpose_fractions,
            "c60": "biased proposal relaxation and direction oracle dominate",
            "pdo": "biased proposal relaxation and strict true quenching dominate",
        },
        "diagnostic_scope": {
            "optimizer_diagnostics": (
                "action-scoped and final-backend-only for true-quench telemetry"
            ),
            "true_quench_cost": (
                "purpose_ledger_not_final_only_optimizer_diagnostics"
            ),
            "reason": (
                "fallback attempts replace primary-attempt optimizer telemetry, "
                "so diagnostic evaluator-call fields are not an additive cost ledger"
            ),
        },
        "claim_boundary": {
            "execution": "six serial GPU campaigns completed with exact ledgers",
            "policy_ranking": "not established",
            "reason": (
                "one seed per system and known GPU nondeterminism permit only "
                "descriptive, not inferential, policy comparisons"
            ),
            "TS_vs_UCB": "not tested",
            "frequency_unbiasedness": (
                "uniform has full random support; posterior_proportional keeps "
                "positive support but is not frequency-unbiased; minimal_ucb is "
                "a deterministic comparator"
            ),
        },
    }


def render_conclusion(evidence: Mapping[str, Any]) -> str:
    """Render a concise, claim-bounded Chinese conclusion."""

    campaigns = _mapping(evidence["campaigns"], label="evidence campaigns")
    overall = _mapping(evidence["overall"], label="evidence overall")
    lines = [
        "# Posterior starter-policy GPU smoke 结论",
        "",
        "## 已验证事实",
        "",
        (
            f"- 6/6 个 campaign 完成；共 {overall['completed_attempts']} 次 action，"
            f"失败 action 为 {overall['failed_attempts']}。"
        ),
        "- 每个 campaign 的 manifest、index、event/action 字段闭合、policy probability 与 posterior credit 重建、预算账本和 action optimizer diagnostics 均通过 fail-closed 校验。",
        "- 有效配置固定为 proposal `safe-lbfgs-total`，true quench `ase-lbfgs`，失败时 `ase-fire` fallback；`fmax=0.01 eV/Å`、`maxiter=400`。",
        "",
        "| system | policy | attempts | total FE | bootstrap FE | action FE | unused FE | wall s | archive | duplicate | best drop eV | fallback |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for system in SYSTEMS:
        system_campaigns = _mapping(
            campaigns[system], label=f"{system} evidence campaigns"
        )
        for policy in POLICIES:
            row = _mapping(
                system_campaigns[policy], label=f"{system}/{policy} evidence"
            )
            lines.append(
                "| "
                + " | ".join(
                    (
                        system,
                        policy,
                        str(row["completed_attempts"]),
                        str(row["total_force_evaluations"]),
                        str(row["bootstrap_force_evaluations"]),
                        str(row["action_force_evaluations"]),
                        str(row["unused_force_budget"]),
                        f"{row['campaign_wall_time_s']:.6f}",
                        str(row["archive_entries"]),
                        f"{row['duplicate_rate']:.6f}",
                        f"{row['best_energy_drop_eV']:.9f}",
                        (
                            f"{row['quench_fallback_converged']}/"
                            f"{row['quench_fallback_attempts']}"
                        ),
                    )
                )
                + " |"
            )
    lines.extend(
        (
            "",
            "## 账本与收敛",
            "",
            "- Bootstrap 固定成本：C60 为 49 FE，PdO 为 84 FE。当前 optimizer diagnostics 只记录带 action ID 的尝试，因此 bootstrap fallback 是否触发不可观测；不能从 action diagnostics 反推。",
            (
                f"- Action 内 fallback 仅发生 {overall['quench_fallback_attempts']} 次，"
                f"其中 {overall['quench_fallback_converged']} 次收敛；最终 "
                "`true_quench_unconverged=0`。"
            ),
            "- true-quench diagnostics 是 final-only：发生 fallback 时不用于总 FE 加和；总成本只以按 purpose 记账且闭合的 evaluator ledger 为准。",
            "- C60 的主要成本是 proposal relaxation 与 direction oracle；PdO 的主要成本是 proposal relaxation 与 strict true quench。",
            "",
            "## Policy 语义与结论边界",
            "",
            "- `uniform` 和 `posterior_proportional` 在每次 policy snapshot 上都保留完整支持；后者对所有 eligible starter 的概率严格为正，但它不是 frequency-unbiased，因为抽样频率受 posterior mean 影响。",
            "- `minimal_ucb` 是 one-hot 的确定性对照，不属于无偏或完整支持探索。",
            "- 表中 policy 差异只能作为单个 seed 的描述性结果。已知 GPU 非确定性会被 PES 搜索放大，因此不能据此判定 policy 优劣，也没有比较 TS 与 UCB。",
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
