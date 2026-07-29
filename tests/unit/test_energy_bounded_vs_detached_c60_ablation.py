from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = (
    REPO_ROOT
    / "runs"
    / "20260729-energy-bounded-vs-detached-c60"
)
RUNNER_PATH = RUN_ROOT / "run_ablation.py"
EVIDENCE_PATH = RUN_ROOT / "output" / "evidence.json"


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "_energy_bounded_vs_detached_c60_ablation",
        RUNNER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_energy_bounded_vs_detached_matrix_is_exact_and_hvp_paired() -> None:
    runner = _load_runner()

    assert runner.STATE_IDS == (
        "intermediate_accepted",
        "plateau_accepted",
    )
    assert runner.SEEDS == (42, 43, 44)
    assert runner.ARMS == {
        "detached_ritz": {
            "direction_selection_mode": "block_krylov",
            "block_krylov_blocks": 1,
            "block_krylov_depth": 6,
        },
        "energy_bounded_anchor": {
            "direction_selection_mode": "energy_bounded_anchor",
            "block_krylov_blocks": 1,
            "block_krylov_depth": 12,
        },
    }
    assert runner.EXPECTED_HVP_PER_SELECTION == {
        "detached_ritz": 12,
        "energy_bounded_anchor": 12,
    }
    assert len(runner.case_matrix()) == 12
    assert len(
        {
            (row["state_id"], row["seed"], row["arm"])
            for row in runner.case_matrix()
        }
    ) == 12


def test_detached_direction_overlap_summary_is_sign_invariant() -> None:
    runner = _load_runner()

    assert runner._median_absolute_anchor_overlap(
        [
            {"anchor_cosine": -0.8},
            {"anchor_cosine": 0.4},
            {"anchor_cosine": -0.2},
        ]
    ) == pytest.approx(0.4)


def test_energy_bounded_vs_detached_evidence_contract_after_execution() -> None:
    if not EVIDENCE_PATH.is_file():
        pytest.skip("locked GPU evidence has not been generated yet")
    runner = _load_runner()
    evidence = runner.load_and_validate_evidence(EVIDENCE_PATH)

    assert evidence["cohort"]["completed_cases"] == 12
    assert evidence["certificate_count"] == 12
    assert evidence["production_default_changed"] is False
    assert evidence["totals"]["purpose_counts"][
        "bootstrap_true_quench"
    ] == 0
    assert evidence["totals"]["purpose_counts"][
        "starter_true_quench"
    ] == 0
    assert evidence["totals"]["purpose_counts"]["unattributed"] == 0
