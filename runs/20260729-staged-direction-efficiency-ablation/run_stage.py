#!/usr/bin/env python3
"""Run preregistered fixed-starter direction-efficiency stages."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping, Sequence


FROZEN_FIELDS = {
    "direction_selection_mode": "discrete",
    "direction_synthesis_mode": "none",
    "direction_type_ucb_enabled": False,
    "archive_escape_momentum_enabled": False,
    "choice_aligned_softening_enabled": False,
    "direction_probe_enabled": False,
    "plateau_evolution_enabled": False,
    "direction_curvature_source": "inner",
    "proposal_optimizer": "safe-lbfgs-total",
    "proposal_fmax": 0.05,
    "quench_fmax": 0.01,
}

BASELINE_FIELDS = {
    "enable_momentum_candidate": True,
    "oracle_candidates": 12,
    "max_steps_per_walk": 8,
    "proposal_relax_steps": 80,
}

COMMON_OVERRIDES = {
    "max_trials": 1,
    "max_force_evals": None,
    "quench_optimizer": "ase-lbfgs",
    "quench_fallback_optimizer": "ase-fire",
    "quench_fmax": 0.01,
    "quench_maxiter": 400,
}


def config_projection(
    *,
    case,
    case_dir: Path,
    base_runner,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[Any]]]:
    source_config = base_runner.build_config("c60", case_dir)
    source = asdict(source_config)
    for field, expected in FROZEN_FIELDS.items():
        if source[field] != expected:
            raise RuntimeError(f"frozen protocol drifted: {field}")
    for field, expected in BASELINE_FIELDS.items():
        if source[field] != expected:
            raise RuntimeError(f"frozen baseline drifted: {field}")
    effective_config = replace(
        source_config,
        **COMMON_OVERRIDES,
        rng_seed=case.seed,
        enable_momentum_candidate=(
            case.settings.enable_momentum_candidate
        ),
        oracle_candidates=case.settings.oracle_candidates,
        max_steps_per_walk=case.settings.max_steps_per_walk,
        proposal_relax_steps=case.settings.proposal_relax_steps,
        direction_diagnostics_enabled=True,
        direction_diagnostics_path=str(
            Path(case_dir) / "direction_trace.jsonl"
        ),
    )
    effective = asdict(effective_config)
    diff = {
        key: [source[key], effective[key]]
        for key in source
        if source[key] != effective[key]
    }
    return source, effective, diff


def validate_direction_trace(
    rows: Sequence[Mapping[str, Any]],
    *,
    oracle_candidates: int,
) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("direction trace is empty")
    candidate_kinds: Counter[str] = Counter()
    selected_kinds: Counter[str] = Counter()
    for index, row in enumerate(rows):
        if int(row["candidate_count"]) != oracle_candidates:
            raise RuntimeError(
                f"direction row {index} candidate count drifted"
            )
        counts = row.get("evaluated_candidate_kind_counts")
        if not isinstance(counts, dict):
            raise RuntimeError(
                f"direction row {index} lacks source counts"
            )
        exact_counts = {
            str(key): int(value) for key, value in counts.items()
        }
        if sum(exact_counts.values()) != oracle_candidates:
            raise RuntimeError(
                f"direction row {index} source counts do not close"
            )
        selected = str(row["selected_kind"])
        if exact_counts.get(selected, 0) <= 0:
            raise RuntimeError(
                f"direction row {index} selected source was not evaluated"
            )
        expected_fe = 2 * oracle_candidates
        for field in (
            "oracle_selection_force_evaluations_delta",
            "oracle_direction_force_evaluations_delta",
        ):
            if int(row[field]) != expected_fe:
                raise RuntimeError(
                    f"direction row {index} {field} does not close"
                )
        candidate_kinds.update(exact_counts)
        selected_kinds[selected] += 1
    return {
        "selection_count": len(rows),
        "candidate_count": len(rows) * oracle_candidates,
        "candidate_kind_counts": dict(sorted(candidate_kinds.items())),
        "selected_kind_counts": dict(sorted(selected_kinds.items())),
        "direction_oracle_force_evaluations": (
            2 * oracle_candidates * len(rows)
        ),
    }
