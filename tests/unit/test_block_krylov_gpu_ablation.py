"""Preregistered contracts for the Stage-2 block-Krylov GPU ablation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


RUN_ROOT = (
    Path(__file__).resolve().parents[2]
    / "runs"
    / "20260728-block-krylov-direction-gpu-ablation"
)
RUNNER_PATH = RUN_ROOT / "run_ablation.py"
ANALYZER_PATH = RUN_ROOT / "analyze_evidence.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_stage2_runner_and_analyzer_are_preregistered_before_terminal_execution():
    assert RUNNER_PATH.is_file(), "Stage-2 runner must be committed before GPU execution"
    assert ANALYZER_PATH.is_file(), "Stage-2 analyzer must be committed before GPU execution"


def test_stage2_arms_have_stable_amended_allocation_order():
    runner = _load(RUNNER_PATH, "block_krylov_arm_contract")

    assert tuple(runner.ARMS) == (
        "discrete",
        "variational_breadth",
        "balanced_refinement",
        "deep_refinement",
    )
    assert runner.ARMS["discrete"] == {"direction_selection_mode": "discrete"}
    assert runner.ARMS["variational_breadth"] == {
        "direction_selection_mode": "block_krylov",
        "block_krylov_blocks": 6,
        "block_krylov_depth": 1,
    }
    assert runner.ARMS["balanced_refinement"] == {
        "direction_selection_mode": "block_krylov",
        "block_krylov_blocks": 2,
        "block_krylov_depth": 3,
    }
    assert runner.ARMS["deep_refinement"] == {
        "direction_selection_mode": "block_krylov",
        "block_krylov_blocks": 1,
        "block_krylov_depth": 6,
    }
    assert runner.SEEDS == (42, 43, 44)
    assert runner.TOTAL_FORCE_BUDGET == 6000


def test_block_trace_contract_requires_exact_hvp_and_force_ledger():
    runner = _load(RUNNER_PATH, "block_krylov_trace_contract")
    rows = [
        {
            "selected_kind": "block_ritz",
            "candidate_count": 0,
            "krylov_blocks": 2,
            "krylov_hvp_requested": 6,
            "krylov_hvp_consumed": 6,
            "krylov_hvp_count": 6,
            "oracle_selection_force_evaluations_delta": 12,
        }
    ]

    audit = runner.validate_direction_trace(
        arm="balanced_refinement",
        direction_rows=rows,
    )

    assert audit["selection_count"] == 1
    assert audit["direction_oracle_force_evaluations"] == 12
    assert audit["selected_kind_counts"] == {"block_ritz": 1}
    bad = [dict(rows[0], oracle_selection_force_evaluations_delta=10)]
    with pytest.raises(RuntimeError, match="2 \* krylov_hvp_consumed"):
        runner.validate_direction_trace(arm="balanced_refinement", direction_rows=bad)


def test_analyzer_selects_one_c60_survivor_by_amended_tie_break_and_rejects_manual_pdo(tmp_path):
    analyzer = _load(ANALYZER_PATH, "block_krylov_evidence_contract")
    cases = []
    for seed in (42, 43, 44):
        cases.extend(
            [
                _case("discrete", "c60", seed, best=-10.0, auc=-50000.0, archive=10),
                _case("variational_breadth", "c60", seed, best=-10.2, auc=-51000.0, archive=9),
                _case("balanced_refinement", "c60", seed, best=-10.3, auc=-51050.0, archive=10),
                _case("deep_refinement", "c60", seed, best=-10.3, auc=-51100.0, archive=10),
            ]
        )

    evidence = analyzer.build_evidence_from_cases(cases, system="c60")

    assert evidence["c60_survivors"] == [
        "variational_breadth",
        "balanced_refinement",
        "deep_refinement",
    ]
    assert evidence["selected_pdo_transfer_arm"] == "deep_refinement"
    assert evidence["pdo_status"] == "not_run_pending_selected_c60_survivor"
    with pytest.raises(analyzer.EvidenceError, match="selected_pdo_transfer_arm"):
        analyzer.build_evidence_from_cases(
            [
                _case("discrete", "pdo", 42, best=-1.0, auc=-1.0, archive=1),
                _case("balanced_refinement", "pdo", 42, best=-1.1, auc=-1.1, archive=1),
            ],
            system="pdo",
            selected_pdo_transfer_arm="deep_refinement",
        )


def _case(arm: str, system: str, seed: int, *, best: float, auc: float, archive: int) -> dict[str, object]:
    return {
        "arm": arm,
        "system": system,
        "seed": seed,
        "best_energy_eV": best,
        "best_energy_auc_eV_force_evals": auc,
        "archive_size": archive,
        "budget_closed": True,
        "total_force_evaluations": 6000,
    }
