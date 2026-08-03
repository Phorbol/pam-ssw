from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = REPO_ROOT / "runs" / "20260801-fixed-step-target-gate"
PROTOCOL_PATH = RUN_ROOT / "protocol.py"
RUNNER_PATH = RUN_ROOT / "run_gate.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _protocol():
    return _load(PROTOCOL_PATH, "_fixed_step_target_protocol_test")


def _runner():
    return _load(RUNNER_PATH, "_fixed_step_target_runner_test")


def test_ut1_matrix_has_six_cases_and_two_target_modes():
    protocol = _protocol()

    cases = protocol.case_matrix(systems=("c60", "pdo", "cuo"), seeds=(46,))

    assert len(cases) == 6
    assert {row["target_mode"] for row in cases} == {
        "archive_scaled",
        "fixed_reference",
    }


def test_gain_auc_integrates_best_gain_over_complete_budget():
    protocol = _protocol()

    auc = protocol.gain_auc(
        initial_energy_eV=-10.0,
        bootstrap_force_evaluations=10,
        accepted_rows=[
            {"force_evaluations": 20, "best_energy": -11.0},
            {"force_evaluations": 60, "best_energy": -13.0},
        ],
        total_force_budget=100,
    )

    assert auc == pytest.approx((40.0 * 1.0 + 40.0 * 3.0) / 100.0)


def _decision_rows():
    values = {
        "c60": {"archive_scaled": 2.0, "fixed_reference": 3.0},
        "pdo": {"archive_scaled": 3.0, "fixed_reference": 4.0},
        "cuo": {"archive_scaled": 4.0, "fixed_reference": 2.0},
    }
    return [
        {
            "system": system,
            "seed": 46,
            "target_mode": mode,
            "gain_auc_eV": auc,
        }
        for system, modes in values.items()
        for mode, auc in modes.items()
    ]


def test_ut1_decision_requires_strict_fixed_win_in_two_systems():
    protocol = _protocol()
    rows = _decision_rows()

    assert protocol.cohort_decision(rows) == {
        "decision": "ADMIT_U_T2",
        "fixed_winning_system_count": 2,
        "fixed_winning_systems": ["c60", "pdo"],
    }

    tied = deepcopy(rows)
    next(
        row
        for row in tied
        if row["system"] == "pdo" and row["target_mode"] == "fixed_reference"
    )["gain_auc_eV"] = 3.0
    assert protocol.cohort_decision(tied)["decision"] == "DO_NOT_ADMIT_U_T2"


def test_partial_smoke_does_not_receive_scientific_decision():
    protocol = _protocol()
    rows = [row for row in _decision_rows() if row["system"] == "c60"]

    assert protocol.cohort_decision(rows) == {
        "decision": "NOT_EVALUATED_PARTIAL_COHORT",
        "fixed_winning_system_count": 0,
        "fixed_winning_systems": [],
    }


class _FakeController:
    def __init__(self):
        self.recorded = 0
        self.requests = 0

    def target(self, archive=None):
        self.requests += 1
        return 0.25

    def record_trial(self, **kwargs):
        self.recorded += 1

    def stats(self):
        return {"adaptive_step_target": 0.25, "adaptive_step_multiplier": 1.0}


def test_run_local_target_wrapper_is_fixed_or_delegated_and_keeps_bookkeeping():
    runner = _runner()
    scaled_base = _FakeController()
    scaled = runner.TargetModeController(scaled_base, mode="archive_scaled", reference_eV=0.8)
    fixed_base = _FakeController()
    fixed = runner.TargetModeController(fixed_base, mode="fixed_reference", reference_eV=0.8)

    assert scaled.target(object()) == pytest.approx(0.25)
    assert fixed.target(object()) == pytest.approx(0.8)
    scaled.record_trial(escaped=True, damaged=False)
    fixed.record_trial(escaped=True, damaged=False)

    assert scaled_base.requests == 1
    assert fixed_base.requests == 0
    assert scaled_base.recorded == fixed_base.recorded == 1
    assert scaled.history_eV == pytest.approx([0.25])
    assert fixed.history_eV == pytest.approx([0.8])
    assert scaled.stats()["step_target_mode"] == "archive_scaled"
    assert fixed.stats()["step_target_mode"] == "fixed_reference"
    assert fixed.stats()["adaptive_step_target"] == pytest.approx(0.8)


def test_runner_uses_metropolis_and_changes_no_production_target_config(tmp_path):
    runner = _runner()

    for system in runner.SYSTEMS:
        configs = {
            mode: runner.build_config(
                system,
                tmp_path / system / mode,
                seed=46,
                target_mode=mode,
                force_budget=20_000,
            )
            for mode in runner.TARGET_MODES
        }
        scaled = configs["archive_scaled"]
        fixed = configs["fixed_reference"]
        differing = {
            name
            for name in scaled.__dataclass_fields__
            if getattr(scaled, name) != getattr(fixed, name)
        }
        assert differing == {"accepted_structures_log"}
        assert scaled.seed_selection_mode == "metropolis_chain"
        assert scaled.target_uphill_energy == pytest.approx(0.8)
        if system == "cuo":
            assert scaled.local_softening_scope == "oracle"


def test_validate_evidence_requires_shared_bootstrap_and_closed_ledger():
    runner = _runner()
    purpose = {"biased_proposal_relax": 19_950, "bootstrap_true_quench": 50, "unattributed": 0}
    cases = []
    for row in _decision_rows():
        cases.append(
            {
                **row,
                "bootstrap_state_sha256": f"state-{row['system']}",
                "bootstrap_energy_eV": -1.0,
                "shared_bootstrap_force_evaluations": 50,
                "force_evaluations": 20_000,
                "campaign_force_budget": 20_000,
                "budget_exhausted": True,
                "effective_config": {"oracle_candidates": 8},
                "purpose_counts": purpose,
            }
        )
    evidence = {
        "cases": cases,
        "cohort": {
            "systems": ["c60", "pdo", "cuo"],
            "seeds": [46],
            "target_modes": ["archive_scaled", "fixed_reference"],
            "force_budget_per_case": 20_000,
        },
        "decision": runner.protocol.cohort_decision(cases),
    }

    checked = runner.validate_evidence(evidence)
    assert checked["case_count"] == 6
    assert checked["shared_bootstrap_block_count"] == 3
    assert checked["unattributed"] == 0

    broken = deepcopy(evidence)
    broken["cases"][1]["bootstrap_energy_eV"] = -2.0
    with pytest.raises(ValueError, match="shared bootstrap"):
        runner.validate_evidence(broken)


def test_validate_evidence_allows_only_atomic_batch_budget_residual():
    runner = _runner()
    purpose = {
        "biased_proposal_relax": 19_940,
        "bootstrap_true_quench": 50,
        "direction_oracle": 8,
        "unattributed": 0,
    }
    cases = []
    for row in _decision_rows():
        cases.append(
            {
                **row,
                "bootstrap_state_sha256": f"state-{row['system']}",
                "bootstrap_energy_eV": -1.0,
                "shared_bootstrap_force_evaluations": 50,
                "force_evaluations": 19_998,
                "campaign_force_budget": 20_000,
                "budget_exhausted": True,
                "effective_config": {"oracle_candidates": 8},
                "purpose_counts": purpose,
            }
        )
    evidence = {
        "cases": cases,
        "cohort": {
            "systems": ["c60", "pdo", "cuo"],
            "seeds": [46],
            "target_modes": ["archive_scaled", "fixed_reference"],
            "force_budget_per_case": 20_000,
        },
        "decision": runner.protocol.cohort_decision(cases),
    }

    checked = runner.validate_evidence(evidence)
    assert checked["unused_force_evaluations"] == 12
    assert checked["maximum_case_budget_residual"] == 2

    not_exhausted = deepcopy(evidence)
    not_exhausted["cases"][0]["budget_exhausted"] = False
    with pytest.raises(ValueError, match="prematurely"):
        runner.validate_evidence(not_exhausted)

    too_large = deepcopy(evidence)
    too_large["cases"][0]["force_evaluations"] = 19_984
    too_large["cases"][0]["purpose_counts"] = {
        "biased_proposal_relax": 19_926,
        "bootstrap_true_quench": 50,
        "direction_oracle": 8,
        "unattributed": 0,
    }
    with pytest.raises(ValueError, match="atomic batch"):
        runner.validate_evidence(too_large)


def test_ut2_decision_requires_six_of_nine_and_positive_median():
    protocol = _protocol()
    rows = []
    deltas = [1.0, 0.5, -0.1, 0.4, -0.2, 0.3, 0.2, -0.3, 0.1]
    for (system, seed), delta in zip(
        (
            (system, seed)
            for system in protocol.SYSTEMS
            for seed in (46, 47, 48)
        ),
        deltas,
    ):
        rows.extend(
            [
                {
                    "system": system,
                    "seed": seed,
                    "target_mode": "archive_scaled",
                    "gain_auc_eV": 2.0,
                },
                {
                    "system": system,
                    "seed": seed,
                    "target_mode": "fixed_reference",
                    "gain_auc_eV": 2.0 + delta,
                },
            ]
        )

    admitted = protocol.ut2_decision(rows)
    assert admitted["decision"] == "ADMIT_FIXED_REFERENCE_FOR_PRODUCTION_REVIEW"
    assert admitted["fixed_winning_block_count"] == 6
    assert admitted["median_fixed_minus_scaled_gain_auc_eV"] == pytest.approx(0.2)

    rows[-1]["gain_auc_eV"] = 1.0
    rejected = protocol.ut2_decision(rows)
    assert rejected["decision"] == "RETAIN_ARCHIVE_SCALED_DEFAULT"
    assert rejected["fixed_winning_block_count"] == 5


def test_ut2_decision_rejects_incomplete_or_duplicate_matrix():
    protocol = _protocol()
    rows = [
        {
            "system": system,
            "seed": seed,
            "target_mode": mode,
            "gain_auc_eV": 1.0,
        }
        for system in protocol.SYSTEMS
        for seed in (46, 47, 48)
        for mode in protocol.TARGET_MODES
    ]
    with pytest.raises(ValueError, match="required matrix"):
        protocol.ut2_decision(rows[:-1])
    with pytest.raises(ValueError, match="required matrix"):
        protocol.ut2_decision(rows + [rows[0]])
