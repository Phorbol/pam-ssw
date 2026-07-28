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


def _validate_row(row: Mapping[str, Any], index: int) -> dict[str, Any]:
    label = f"rows[{index}]"
    for key in (
        "case",
        "kind",
        "arm",
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
    for key in ("device", "precision"):
        if not isinstance(runtime.get(key), str) or not runtime[key]:
            raise RuntimeError(f"raw.runtime.{key} must be a non-empty string")
    _mapping(raw["fixed_state_registry"], "raw.fixed_state_registry")
    if _finite_number(raw["wall_seconds"], "raw.wall_seconds") < 0.0:
        raise RuntimeError("raw.wall_seconds must be non-negative")
    rows = raw["rows"]
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("raw.rows must be a non-empty list")
    projected_rows = [_validate_row(_mapping(row, f"rows[{index}]"), index) for index, row in enumerate(rows)]

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
