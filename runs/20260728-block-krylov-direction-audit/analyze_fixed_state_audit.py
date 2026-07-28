#!/usr/bin/env python3
"""Fail-closed projection of the block-Krylov fixed-state direction audit."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


ALLOCATIONS = {
    "variational_breadth": {"block_krylov_blocks": 6, "block_krylov_depth": 1},
    "shallow_refinement": {"block_krylov_blocks": 3, "block_krylov_depth": 2},
    "balanced_refinement": {"block_krylov_blocks": 2, "block_krylov_depth": 3},
    "deep_refinement": {"block_krylov_blocks": 1, "block_krylov_depth": 6},
}
HVP_EPSILON = 1.0e-3
MAX_HVPS = 12
FIXED_SYSTEMS = ("c60", "pdo")
FIXED_STATE_PROTOCOL = {
    "bootstrap_quenched": ("bootstrap", "reconstruct_frozen_starter_true_quench"),
    "intermediate_accepted": ("intermediate", "locked_accepted_structure"),
    "plateau_accepted": ("plateau", "locked_accepted_structure"),
}
STRICT_OUTPUT_CONFIG_FIELDS = frozenset(
    {
        "accepted_structures_dir",
        "accepted_structures_log",
        "direction_diagnostics_path",
    }
)
CURRENT_CONFIG_SCHEMA_ADDITIONS = {
    "block_krylov_blocks": 2,
    "block_krylov_depth": 3,
}
REQUIRED_DIAGNOSTICS = {
    "krylov_blocks",
    "krylov_selected_block",
    "krylov_hvp_count",
    "krylov_dimensions",
    "krylov_initial_ranks",
    "krylov_residual_norm",
    "krylov_initial_span_overlap",
    "krylov_antisymmetry",
    "krylov_termination",
    "direction_participation_ratio",
}
REQUIRED_RAW_FIELDS = {
    "schema_version",
    "git_commit",
    "dirty",
    "operator",
    "hvp_epsilon",
    "max_hvps",
    "allocations",
    "runtime",
    "rows",
    "fixed_state_registry",
    "wall_seconds",
}
CLAIM_CEILING = (
    "This direction-only audit supports only block-Krylov algebra, central-FD "
    "operator diagnostics, and purpose-resolved force-accounting claims. It does "
    "not rank allocations by terminal energy or establish basin-discovery, "
    "chemical, optimizer, or production-search superiority."
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be an object")
    return value


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{label} must be a finite number")
    return result


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"{label} must be a non-negative integer")
    return int(value)


def _sha256_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise RuntimeError(f"{label} must be a SHA-256 digest")
    return value


def _required_string(mapping: Mapping[str, Any], key: str, label: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label}.{key} must be a non-empty string")
    return value


def _historical_config_projection(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in config.items()
        if key not in STRICT_OUTPUT_CONFIG_FIELDS and key not in CURRENT_CONFIG_SCHEMA_ADDITIONS
    }


def _validate_row(row: Mapping[str, Any], index: int) -> dict[str, Any]:
    label = f"rows[{index}]"
    for key in (
        "case",
        "kind",
        "arm",
        "intent_seed",
        "force_evaluations",
        "krylov_hvp_count",
        "purpose_counts",
        "diagnostics",
        "selection",
        "state_sha256",
        "wall_seconds",
    ):
        if key not in row:
            raise RuntimeError(f"{label} is missing {key}")
    if not isinstance(row["case"], str) or not row["case"]:
        raise RuntimeError(f"{label}.case must be a non-empty string")
    if row["kind"] not in {"analytic", "fixed_state"}:
        raise RuntimeError(f"{label}.kind must be analytic or fixed_state")
    arm = row["arm"]
    if arm not in ALLOCATIONS:
        raise RuntimeError(f"{label}.arm is not preregistered: {arm!r}")
    intent_seed = _nonnegative_int(row["intent_seed"], f"{label}.intent_seed")
    force_evaluations = _nonnegative_int(row["force_evaluations"], f"{label}.force_evaluations")
    hvp_count = _nonnegative_int(row["krylov_hvp_count"], f"{label}.krylov_hvp_count")
    if force_evaluations != 2 * hvp_count:
        raise RuntimeError(f"{label} violates central-FD force accounting")
    if hvp_count > MAX_HVPS:
        raise RuntimeError(f"{label} exceeds the preregistered HVP budget")

    purpose_counts = _mapping(row["purpose_counts"], f"{label}.purpose_counts")
    purpose_total = sum(
        _nonnegative_int(value, f"{label}.purpose_counts[{name!r}]")
        for name, value in purpose_counts.items()
    )
    if purpose_total != force_evaluations:
        raise RuntimeError(f"{label} purpose ledger does not close")
    if purpose_counts.get("direction_oracle") != force_evaluations:
        raise RuntimeError(f"{label} has direction evaluations outside direction_oracle")
    if purpose_counts.get("unattributed") != 0:
        raise RuntimeError(f"{label} has unattributed evaluations")

    diagnostics = _mapping(row["diagnostics"], f"{label}.diagnostics")
    missing = REQUIRED_DIAGNOSTICS - diagnostics.keys()
    if missing:
        raise RuntimeError(f"{label}.diagnostics is missing {sorted(missing)}")
    if diagnostics["krylov_hvp_count"] != hvp_count:
        raise RuntimeError(f"{label} diagnostics HVP count does not close")
    blocks = _nonnegative_int(diagnostics["krylov_blocks"], f"{label}.diagnostics.krylov_blocks")
    expected_blocks = ALLOCATIONS[arm]["block_krylov_blocks"]
    if blocks != expected_blocks:
        raise RuntimeError(f"{label} has the wrong preregistered block count")
    selected_block = _nonnegative_int(
        diagnostics["krylov_selected_block"], f"{label}.diagnostics.krylov_selected_block"
    )
    if selected_block >= blocks:
        raise RuntimeError(f"{label} selected an out-of-range Krylov block")
    for list_key in ("krylov_dimensions", "krylov_initial_ranks"):
        values = diagnostics[list_key]
        if not isinstance(values, list) or len(values) != blocks:
            raise RuntimeError(f"{label}.diagnostics.{list_key} does not close to blocks")
        if any(_nonnegative_int(value, f"{label}.{list_key}") == 0 for value in values):
            raise RuntimeError(f"{label}.diagnostics.{list_key} contains a zero rank")
    for numeric_key in (
        "krylov_residual_norm",
        "krylov_initial_span_overlap",
        "krylov_antisymmetry",
        "direction_participation_ratio",
    ):
        value = _finite_number(diagnostics[numeric_key], f"{label}.diagnostics.{numeric_key}")
        if value < 0.0:
            raise RuntimeError(f"{label}.diagnostics.{numeric_key} must be non-negative")
    if not isinstance(diagnostics["krylov_termination"], str) or not diagnostics["krylov_termination"]:
        raise RuntimeError(f"{label}.diagnostics.krylov_termination must be a non-empty string")

    selection = _mapping(row["selection"], f"{label}.selection")
    for key in ("curvature", "true_curvature"):
        if key not in selection:
            raise RuntimeError(f"{label}.selection is missing {key}")
        _finite_number(selection[key], f"{label}.selection.{key}")
    if not isinstance(row["state_sha256"], str) or len(row["state_sha256"]) != 64:
        raise RuntimeError(f"{label}.state_sha256 must be a SHA-256 digest")
    if _finite_number(row["wall_seconds"], f"{label}.wall_seconds") < 0.0:
        raise RuntimeError(f"{label}.wall_seconds must be non-negative")
    return {
        "case": row["case"],
        "kind": row["kind"],
        "system": row.get("system", "analytic"),
        "state_id": row.get("state_id", row["case"]),
        "arm": arm,
        "intent_seed": intent_seed,
        "force_evaluations": force_evaluations,
        "krylov_hvp_count": hvp_count,
        "curvature": selection.get("curvature"),
        "true_curvature": selection.get("true_curvature"),
        "residual_norm": diagnostics["krylov_residual_norm"],
        "initial_span_overlap": diagnostics["krylov_initial_span_overlap"],
        "antisymmetry": diagnostics["krylov_antisymmetry"],
        "participation_ratio": diagnostics["direction_participation_ratio"],
        "subspace_angle_to_variational_breadth_degrees": row.get(
            "subspace_angle_to_variational_breadth_degrees"
        ),
    }


def _fixed_registry_lookup(
    registry: Mapping[str, Any], runtime: Mapping[str, Any]
) -> dict[tuple[str, str], Mapping[str, Any]]:
    if set(registry) != set(FIXED_SYSTEMS):
        raise RuntimeError("fixed-state registry must contain exactly c60 and pdo")
    for key in ("model_path", "model_sha256", "device", "precision", "dtype"):
        _required_string(runtime, key, "raw.runtime")
    lookup: dict[tuple[str, str], Mapping[str, Any]] = {}
    for system in FIXED_SYSTEMS:
        entries = registry[system]
        if not isinstance(entries, list) or len(entries) != len(FIXED_STATE_PROTOCOL):
            raise RuntimeError(f"fixed-state registry {system} must contain the exact three-state cohort")
        for entry_index, entry_value in enumerate(entries):
            entry = _mapping(entry_value, f"fixed_state_registry.{system}[{entry_index}]")
            state_id = _required_string(entry, "state_id", f"fixed_state_registry.{system}[{entry_index}]")
            if state_id not in FIXED_STATE_PROTOCOL or (system, state_id) in lookup:
                raise RuntimeError("fixed-state registry has an unknown or duplicate state_id")
            phase, source = FIXED_STATE_PROTOCOL[state_id]
            if entry.get("phase") != phase or entry.get("source") != source:
                raise RuntimeError("fixed-state registry phase/source does not match the locked cohort")
            entry_label = f"fixed_state_registry.{system}.{state_id}"
            for key in ("origin_summary_path", "origin_commit", "model_path"):
                _required_string(entry, key, entry_label)
            for key in ("origin_summary_sha256", "model_sha256"):
                _sha256_digest(entry.get(key), f"{entry_label}.{key}")
            if len(str(entry["origin_commit"])) != 40:
                raise RuntimeError(f"{entry_label}.origin_commit must be a full commit SHA")
            if entry["model_path"] != runtime["model_path"] or entry["model_sha256"] != runtime["model_sha256"]:
                raise RuntimeError(f"{entry_label} model provenance does not match raw runtime")
            _nonnegative_int(entry.get("intent_seed"), f"{entry_label}.intent_seed")
            if source == "locked_accepted_structure":
                _required_string(entry, "structure_path", entry_label)
                _sha256_digest(entry.get("structure_sha256"), f"{entry_label}.structure_sha256")
            else:
                for key in ("raw_input_path", "strict_wrapper_path", "base_runner_path"):
                    _required_string(entry, key, entry_label)
                for key in (
                    "raw_input_sha256",
                    "strict_wrapper_sha256",
                    "base_runner_sha256",
                ):
                    _sha256_digest(entry.get(key), f"{entry_label}.{key}")
                strict_diff = _mapping(entry.get("strict_config_diff"), f"{entry_label}.strict_config_diff")
                if not strict_diff:
                    raise RuntimeError(f"{entry_label}.strict_config_diff must not be empty")
            lookup[(system, state_id)] = entry
    if set(lookup) != {
        (system, state_id) for system in FIXED_SYSTEMS for state_id in FIXED_STATE_PROTOCOL
    }:
        raise RuntimeError("fixed-state registry does not close to the locked cohort")
    return lookup


def _validate_group_fairness(
    rows: Sequence[Mapping[str, Any]],
    fixed_lookup: Mapping[tuple[str, str], Mapping[str, Any]] | None,
) -> None:
    groups: dict[tuple[str, str, str], list[tuple[str, int]]] = {}
    for index, row in enumerate(rows):
        kind = str(row["kind"])
        system = str(row.get("system", "analytic"))
        identifier = str(row.get("state_id", row["case"])) if kind == "fixed_state" else str(row["case"])
        seed = _nonnegative_int(row.get("intent_seed"), f"rows[{index}].intent_seed")
        groups.setdefault((kind, system, identifier), []).append((str(row["arm"]), seed))

    expected_arms = set(ALLOCATIONS)
    for (kind, system, identifier), arm_seeds in groups.items():
        arms = [arm for arm, _ in arm_seeds]
        label = f"{kind}:{system}:{identifier}"
        if len(arms) != len(expected_arms) or set(arms) != expected_arms:
            raise RuntimeError(f"{label} must contain exactly all four arms once")
        seeds = {seed for _, seed in arm_seeds}
        if len(seeds) != 1:
            raise RuntimeError(f"{label} must use one common intent_seed across all arms")
        if kind == "fixed_state":
            if fixed_lookup is None or (system, identifier) not in fixed_lookup:
                raise RuntimeError(f"{label} lacks fixed-state registry provenance")
            expected_seed = _nonnegative_int(
                fixed_lookup[(system, identifier)].get("intent_seed"),
                f"fixed_state_registry.{system}.{identifier}.intent_seed",
            )
            if seeds != {expected_seed}:
                raise RuntimeError(f"{label} does not match registry intent_seed")


def _validate_fixed_provenance(
    row: Mapping[str, Any],
    index: int,
    entry: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    label = f"rows[{index}]"
    system = _required_string(row, "system", label)
    state_id = _required_string(row, "state_id", label)
    phase, source = FIXED_STATE_PROTOCOL[state_id]
    if row.get("case") != f"{system}:{state_id}":
        raise RuntimeError(f"{label}.case does not identify its locked fixed state")
    case_metadata = _mapping(row.get("case_metadata"), f"{label}.case_metadata")
    if case_metadata.get("phase") != phase:
        raise RuntimeError(f"{label}.case_metadata.phase does not match the locked state")
    provenance = _mapping(row.get("state_provenance"), f"{label}.state_provenance")
    if provenance.get("source") != source:
        raise RuntimeError(f"{label}.state_provenance.source does not match the locked state")
    if provenance.get("state_sha256") != row.get("state_sha256"):
        raise RuntimeError(f"{label} state checksum does not close to state provenance")
    for key in ("origin_summary_path", "origin_summary_sha256", "origin_commit", "model_path", "model_sha256"):
        if provenance.get(key) != entry.get(key):
            raise RuntimeError(f"{label}.state_provenance.{key} does not match fixed-state registry")

    calculator = _mapping(row.get("calculator"), f"{label}.calculator")
    for key in ("model_path", "model_sha256", "device", "precision", "dtype"):
        _required_string(calculator, key, f"{label}.calculator")
        if calculator[key] != runtime[key]:
            raise RuntimeError(f"{label}.calculator.{key} does not match raw runtime")
    if calculator["model_path"] != entry["model_path"] or calculator["model_sha256"] != entry["model_sha256"]:
        raise RuntimeError(f"{label}.calculator model provenance does not match fixed-state registry")

    summary: dict[str, Any] = {
        "state_id": state_id,
        "phase": phase,
        "source": source,
        "state_sha256": row["state_sha256"],
        "origin_summary_path": provenance["origin_summary_path"],
        "origin_summary_sha256": provenance["origin_summary_sha256"],
        "origin_commit": provenance["origin_commit"],
        "model_path": provenance["model_path"],
        "model_sha256": provenance["model_sha256"],
    }
    if source == "locked_accepted_structure":
        for key in ("structure_path", "structure_sha256"):
            if provenance.get(key) != entry.get(key):
                raise RuntimeError(f"{label}.state_provenance.{key} does not match fixed-state registry")
            summary[key] = provenance[key]
        return summary

    for key in (
        "raw_input_path",
        "raw_input_sha256",
        "strict_wrapper_path",
        "strict_wrapper_sha256",
        "base_runner_path",
        "base_runner_sha256",
    ):
        if provenance.get(key) != entry.get(key):
            raise RuntimeError(f"{label}.state_provenance.{key} does not match fixed-state registry")
        summary[key] = provenance[key]
    effective_config = _mapping(provenance.get("effective_config"), f"{label}.state_provenance.effective_config")
    origin_effective_config = _mapping(
        provenance.get("origin_effective_config"), f"{label}.state_provenance.origin_effective_config"
    )
    if provenance.get("current_config_schema_additions") != CURRENT_CONFIG_SCHEMA_ADDITIONS:
        raise RuntimeError(f"{label}.state_provenance current config additions drifted")
    if any(key in origin_effective_config for key in CURRENT_CONFIG_SCHEMA_ADDITIONS):
        raise RuntimeError(f"{label}.state_provenance origin strict config contains current-only fields")
    if {key: effective_config.get(key) for key in CURRENT_CONFIG_SCHEMA_ADDITIONS} != CURRENT_CONFIG_SCHEMA_ADDITIONS:
        raise RuntimeError(f"{label}.state_provenance current strict config additions drifted")
    if _historical_config_projection(effective_config) != _historical_config_projection(origin_effective_config):
        raise RuntimeError(f"{label}.state_provenance strict config does not match the origin summary")
    strict_config = {
        "quench_optimizer": "ase-lbfgs",
        "quench_fallback_optimizer": "ase-fire",
        "quench_fmax": 0.01,
    }
    if {key: effective_config.get(key) for key in strict_config} != strict_config:
        raise RuntimeError(f"{label}.state_provenance strict config is not ASE L-BFGS plus FIRE fallback")
    if provenance.get("effective_config_diff") != entry.get("strict_config_diff"):
        raise RuntimeError(f"{label}.state_provenance strict config diff does not match fixed-state registry")
    purpose_counts = _mapping(
        provenance.get("bootstrap_purpose_counts"), f"{label}.state_provenance.bootstrap_purpose_counts"
    )
    purpose_total = sum(
        _nonnegative_int(value, f"{label}.state_provenance.bootstrap_purpose_counts[{name!r}]")
        for name, value in purpose_counts.items()
    )
    bootstrap_force_evaluations = _nonnegative_int(
        provenance.get("bootstrap_force_evaluations"),
        f"{label}.state_provenance.bootstrap_force_evaluations",
    )
    if purpose_total != bootstrap_force_evaluations or purpose_counts.get("unattributed") != 0:
        raise RuntimeError(f"{label}.state_provenance bootstrap purpose ledger does not close")
    if _nonnegative_int(purpose_counts.get("starter_true_quench"), f"{label}.bootstrap starter quench") <= 0:
        raise RuntimeError(f"{label}.state_provenance bootstrap lacks STARTER_TRUE_QUENCH")
    if _finite_number(provenance.get("bootstrap_wall_seconds"), f"{label}.state_provenance.bootstrap_wall_seconds") < 0.0:
        raise RuntimeError(f"{label}.state_provenance bootstrap wall time must be non-negative")
    if provenance.get("resulting_state_sha256") != row.get("state_sha256"):
        raise RuntimeError(f"{label}.state_provenance resulting state checksum does not close")
    summary.update(
        {
            "raw_input_path": provenance["raw_input_path"],
            "raw_input_sha256": provenance["raw_input_sha256"],
            "strict_wrapper_path": provenance["strict_wrapper_path"],
            "strict_wrapper_sha256": provenance["strict_wrapper_sha256"],
            "base_runner_path": provenance["base_runner_path"],
            "base_runner_sha256": provenance["base_runner_sha256"],
            "effective_config": dict(effective_config),
            "effective_config_diff": provenance["effective_config_diff"],
            "bootstrap_purpose_counts": dict(purpose_counts),
            "bootstrap_force_evaluations": bootstrap_force_evaluations,
            "bootstrap_wall_seconds": float(provenance["bootstrap_wall_seconds"]),
            "resulting_state_sha256": provenance["resulting_state_sha256"],
        }
    )
    return summary


def project_evidence(raw: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(raw, "raw audit")
    missing = REQUIRED_RAW_FIELDS - raw.keys()
    if missing:
        raise RuntimeError(f"raw audit is missing {sorted(missing)}")
    if raw["schema_version"] != 1:
        raise RuntimeError("unsupported raw audit schema")
    if raw["dirty"] is not False:
        raise RuntimeError("direction audit must originate from a clean tracked worktree")
    if raw["operator"] != "total_proposal_central_fd":
        raise RuntimeError("unexpected direction operator")
    if _finite_number(raw["hvp_epsilon"], "raw.hvp_epsilon") != HVP_EPSILON:
        raise RuntimeError("unexpected HVP epsilon")
    if raw["max_hvps"] != MAX_HVPS:
        raise RuntimeError("unexpected HVP budget")
    if raw["allocations"] != ALLOCATIONS:
        raise RuntimeError("allocation registry does not match preregistration")
    if not isinstance(raw["git_commit"], str) or len(raw["git_commit"]) != 40:
        raise RuntimeError("raw.git_commit must be a full commit SHA")
    runtime = _mapping(raw["runtime"], "raw.runtime")
    for key in ("mode", "device", "precision"):
        if not isinstance(runtime.get(key), str) or not runtime[key]:
            raise RuntimeError(f"raw.runtime.{key} must be a non-empty string")
    mode = runtime["mode"]
    if mode not in {"analytic_only", "full_fixed_state"}:
        raise RuntimeError("raw.runtime.mode must be analytic_only or full_fixed_state")
    fixed_state_registry = _mapping(raw["fixed_state_registry"], "raw.fixed_state_registry")
    if _finite_number(raw["wall_seconds"], "raw.wall_seconds") < 0.0:
        raise RuntimeError("raw.wall_seconds must be non-negative")
    rows = raw["rows"]
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("raw.rows must be a non-empty list")
    raw_rows = [_mapping(row, f"rows[{index}]") for index, row in enumerate(rows)]
    projected_rows = [_validate_row(row, index) for index, row in enumerate(raw_rows)]
    fixed_rows = [
        (index, row)
        for index, row in enumerate(raw_rows)
        if row.get("kind") == "fixed_state"
    ]
    fixed_provenance: dict[str, list[dict[str, Any]]] = {}
    fixed_lookup: Mapping[tuple[str, str], Mapping[str, Any]] | None = None
    if mode == "analytic_only":
        if fixed_rows:
            raise RuntimeError("analytic_only runtime mode must not contain fixed-state rows")
        if any(row.get("kind") != "analytic" for row in raw_rows):
            raise RuntimeError("analytic_only runtime mode must contain only analytic rows")
    else:
        if not fixed_rows:
            raise RuntimeError("full_fixed_state runtime mode requires the exact fixed-state cohort")
        fixed_lookup = _fixed_registry_lookup(fixed_state_registry, runtime)
        expected_cohort = {
            (system, state_id, arm)
            for system in FIXED_SYSTEMS
            for state_id in FIXED_STATE_PROTOCOL
            for arm in ALLOCATIONS
        }
        observed_cohort: list[tuple[str, str, str]] = []
        by_state: dict[tuple[str, str], dict[str, Any]] = {}
        for index, row in fixed_rows:
            system = _required_string(row, "system", f"rows[{index}]")
            state_id = _required_string(row, "state_id", f"rows[{index}]")
            if (system, state_id) not in fixed_lookup:
                raise RuntimeError(f"rows[{index}] fixed-state system/state_id is not preregistered")
            summary = _validate_fixed_provenance(row, index, fixed_lookup[(system, state_id)], runtime)
            observed_cohort.append((system, state_id, str(row["arm"])))
            existing = by_state.setdefault((system, state_id), summary)
            if existing != summary:
                raise RuntimeError(f"rows[{index}] fixed-state provenance differs across allocation arms")
        if len(observed_cohort) != len(expected_cohort) or set(observed_cohort) != expected_cohort:
            raise RuntimeError("fixed-state cohort has missing, duplicate, or extra system/state/arm rows")
        fixed_provenance = {
            system: [by_state[(system, state_id)] for state_id in FIXED_STATE_PROTOCOL]
            for system in FIXED_SYSTEMS
        }
    _validate_group_fairness(raw_rows, fixed_lookup)

    analytic_cases = sorted({row["case"] for row in projected_rows if row["kind"] == "analytic"})
    c60_states = sorted(
        {row["state_id"] for row in projected_rows if row["kind"] == "fixed_state" and row["system"] == "c60"}
    )
    pdo_states = sorted(
        {row["state_id"] for row in projected_rows if row["kind"] == "fixed_state" and row["system"] == "pdo"}
    )
    return {
        "schema_version": 1,
        "git_commit": raw["git_commit"],
        "dirty": False,
        "operator": raw["operator"],
        "hvp_epsilon": HVP_EPSILON,
        "allocations": ALLOCATIONS,
        "runtime": dict(runtime),
        "wall_seconds": float(raw["wall_seconds"]),
        "analytic_cases": analytic_cases,
        "c60_states": c60_states,
        "pdo_states": pdo_states,
        "fixed_state_provenance": fixed_provenance,
        "accounting_invariants": {
            "all_rows_exact_central_fd": True,
            "all_rows_within_hvp_budget": True,
            "all_rows_purpose_closed": True,
            "all_rows_attributed_to_direction_oracle": True,
        },
        "direction_observations": projected_rows,
        "claim_ceiling": CLAIM_CEILING,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    raw = json.loads(args.input.read_text(encoding="utf-8"))
    evidence = project_evidence(raw)
    _write_json(args.output, evidence)
    print(json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
