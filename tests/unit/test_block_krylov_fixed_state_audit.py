"""Fail-closed evidence contracts for the fixed-state direction audit."""

from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path
import sys

import pytest


AUDIT_ROOT = (
    Path(__file__).resolve().parents[2]
    / "runs"
    / "20260728-block-krylov-direction-audit"
)
ANALYZER_PATH = AUDIT_ROOT / "analyze_fixed_state_audit.py"
RUNNER_PATH = AUDIT_ROOT / "run_fixed_state_audit.py"


def _load_analyzer():
    spec = importlib.util.spec_from_file_location("block_krylov_fixed_state_audit", ANALYZER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_runner():
    spec = importlib.util.spec_from_file_location("block_krylov_fixed_state_runner", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _row(name: str, arm: str) -> dict[str, object]:
    blocks, depth = {
        "variational_breadth": (6, 1),
        "shallow_refinement": (3, 2),
        "balanced_refinement": (2, 3),
        "deep_refinement": (1, 6),
    }[arm]
    hvp_count = 2 * blocks * depth
    return {
        "case": name,
        "kind": "analytic",
        "arm": arm,
        "force_evaluations": 2 * hvp_count,
        "krylov_hvp_count": hvp_count,
        "purpose_counts": {"direction_oracle": 2 * hvp_count, "unattributed": 0},
        "selection": {"curvature": 1.0, "true_curvature": 1.0},
        "state_sha256": "c" * 64,
        "wall_seconds": 0.1,
        "diagnostics": {
            "krylov_blocks": blocks,
            "krylov_selected_block": 0,
            "krylov_hvp_count": hvp_count,
            "krylov_dimensions": [2] * blocks,
            "krylov_initial_ranks": [2] * blocks,
            "krylov_residual_norm": 0.0,
            "krylov_initial_span_overlap": 1.0,
            "krylov_antisymmetry": 0.0,
            "krylov_termination": "depth_reached",
            "direction_participation_ratio": 1.0,
        },
    }


def _raw() -> dict[str, object]:
    allocations = {
        "variational_breadth": {"block_krylov_blocks": 6, "block_krylov_depth": 1},
        "shallow_refinement": {"block_krylov_blocks": 3, "block_krylov_depth": 2},
        "balanced_refinement": {"block_krylov_blocks": 2, "block_krylov_depth": 3},
        "deep_refinement": {"block_krylov_blocks": 1, "block_krylov_depth": 6},
    }
    return {
        "schema_version": 1,
        "git_commit": "a" * 40,
        "dirty": False,
        "operator": "total_proposal_central_fd",
        "hvp_epsilon": 1.0e-3,
        "max_hvps": 12,
        "allocations": allocations,
        "runtime": {"device": "cpu", "precision": "float64"},
        "rows": [_row("diagonal", arm) for arm in allocations],
        "fixed_state_registry": {"c60": [], "pdo": []},
        "wall_seconds": 0.4,
    }


def test_analyzer_projects_only_budget_closed_direction_evidence():
    assert ANALYZER_PATH.is_file()
    analyzer = _load_analyzer()

    evidence = analyzer.project_evidence(_raw())

    assert evidence["accounting_invariants"]["all_rows_exact_central_fd"] is True
    assert evidence["accounting_invariants"]["all_rows_within_hvp_budget"] is True
    assert evidence["analytic_cases"] == ["diagonal"]
    assert "does not rank allocations by terminal energy" in evidence["claim_ceiling"]


def test_analyzer_rejects_missing_required_direction_diagnostic():
    assert ANALYZER_PATH.is_file()
    analyzer = _load_analyzer()
    raw = deepcopy(_raw())
    del raw["rows"][0]["diagnostics"]["krylov_antisymmetry"]

    with pytest.raises(RuntimeError, match="krylov_antisymmetry"):
        analyzer.project_evidence(raw)


def test_analyzer_rejects_raw_evidence_without_runtime_provenance():
    analyzer = _load_analyzer()
    raw = deepcopy(_raw())
    del raw["runtime"]

    with pytest.raises(RuntimeError, match="runtime"):
        analyzer.project_evidence(raw)


def test_analytic_diagonal_case_does_not_wrap_central_fd_probes_across_periodic_cell():
    """A positive analytic quadratic must not become negative at a cell boundary."""

    assert RUNNER_PATH.is_file()
    runner = _load_runner()
    case = runner._analytic_cases()[0]
    row, _ = runner._direction_row(
        case=case["case"],
        kind="analytic",
        system="analytic",
        state_id=case["case"],
        state=case["state"],
        state_provenance={"origin": "test"},
        calculator_factory=lambda: runner.AnalyticCalculator(case["potential"]),
        calculator_provenance={"kind": "analytic"},
        intent_seed=case["intent_seed"],
        arm="variational_breadth",
        force_rank_one=False,
        case_metadata=case["metadata"],
    )

    assert row["selection"]["curvature"] > 0.0


def test_full_rank_analytic_diagonal_uses_the_preregistered_twelve_hvps():
    runner = _load_runner()
    case = runner._analytic_cases()[0]
    rows = runner._run_case_allocations(
        case=case["case"],
        kind="analytic",
        system="analytic",
        state_id=case["case"],
        state=case["state"],
        state_provenance={"origin": "test"},
        calculator_factory=lambda: runner.AnalyticCalculator(case["potential"]),
        calculator_provenance={"kind": "analytic"},
        intent_seed=case["intent_seed"],
        case_metadata=case["metadata"],
    )

    assert {row["krylov_hvp_count"] for row in rows} == {12}
