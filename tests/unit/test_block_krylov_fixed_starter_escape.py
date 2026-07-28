from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from pamssw.state import State


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    REPO_ROOT
    / "runs"
    / "20260728-block-krylov-fixed-starter-escape"
    / "run_ablation.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("_fixed_starter_escape_ablation", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _row(module, *, state_id: str, seed: int, arm: str, landing_delta: float):
    return {
        "state_id": state_id,
        "seed": seed,
        "arm": arm,
        "status": "completed",
        "starter_energy_eV": -10.0,
        "escape_energy_eV": -9.0,
        "landing_energy_eV": -10.0 + landing_delta,
        "landing_delta_eV": landing_delta,
        "is_new_basin": True,
        "descriptor_delta": 0.2,
        "force_evaluations": 100,
        "purpose_counts": {
            "direction_oracle": 24,
            "biased_proposal_relax": 50,
            "landing_true_quench": 24,
            "escape_true_pes_check": 1,
            "post_relax_validation": 1,
            "starter_true_quench": 0,
            "bootstrap_true_quench": 0,
            "unattributed": 0,
        },
        "direction_selection_count": 1,
        "direction_hvp_count": 12,
        "direction_trace_valid": True,
        "quench_converged": True,
        "wall_time_s": 1.0,
    }


def test_preregistered_case_matrix_is_exact_and_stable():
    module = _load_module()
    matrix = module.case_matrix()

    assert len(matrix) == 36
    assert matrix[0] == {
        "state_id": "bootstrap_quenched",
        "seed": 42,
        "arm": "discrete",
    }
    assert matrix[-1] == {
        "state_id": "plateau_accepted",
        "seed": 44,
        "arm": "deep_refinement",
    }
    assert len({(row["state_id"], row["seed"], row["arm"]) for row in matrix}) == 36


def test_evidence_is_descriptive_and_uses_paired_landing_deltas():
    module = _load_module()
    arm_offsets = {
        "discrete": 0.0,
        "variational_breadth": -0.3,
        "balanced_refinement": 0.2,
        "deep_refinement": -0.1,
    }
    rows = [
        _row(
            module,
            state_id=case["state_id"],
            seed=case["seed"],
            arm=case["arm"],
            landing_delta=arm_offsets[case["arm"]],
        )
        for case in module.case_matrix()
    ]

    evidence = module.build_evidence(rows)

    assert evidence["cohort"]["completed_cases"] == 36
    assert evidence["production_default_changed"] is False
    assert evidence["arm_results"]["variational_breadth"]["paired_better_count"] == 9
    assert evidence["arm_results"]["balanced_refinement"]["paired_better_count"] == 0
    assert evidence["arm_results"]["deep_refinement"]["median_paired_landing_delta_eV"] == pytest.approx(-0.1)


def test_evidence_rejects_incomplete_or_unclosed_cases():
    module = _load_module()
    rows = [
        _row(
            module,
            state_id=case["state_id"],
            seed=case["seed"],
            arm=case["arm"],
            landing_delta=0.0,
        )
        for case in module.case_matrix()
    ]

    with pytest.raises(ValueError, match="exactly 36"):
        module.build_evidence(rows[:-1])

    rows[0]["purpose_counts"]["direction_oracle"] = 23
    with pytest.raises(ValueError, match="purpose ledger"):
        module.build_evidence(rows)


def test_strict_refine_evidence_requires_certificates_and_closed_quench_ledgers():
    module = _load_module()
    rows = [
        {
            "state_id": case["state_id"],
            "seed": case["seed"],
            "arm": case["arm"],
            "status": "completed",
            "certificate": True,
            "exact_starter_reference": True,
            "starter_energy_eV": -10.0,
            "landing_energy_eV": -11.0,
            "landing_delta_eV": -1.0,
            "is_new_basin": True,
            "fallback_used": False,
            "force_evaluations": 21,
            "purpose_counts": {
                "direction_oracle": 0,
                "biased_proposal_relax": 0,
                "landing_true_quench": 20,
                "escape_true_pes_check": 0,
                "post_relax_validation": 1,
                "starter_true_quench": 0,
                "bootstrap_true_quench": 0,
                "unattributed": 0,
            },
        }
        for case in module.case_matrix()
    ]

    evidence = module.build_strict_evidence(rows)

    assert evidence["cohort"]["completed_cases"] == 36
    assert evidence["certificate_count"] == 36
    assert evidence["exact_starter_reference_count"] == 36
    assert evidence["arm_results"]["deep_refinement"]["new_basin_count"] == 9

    rows[0]["certificate"] = False
    evidence = module.build_strict_evidence(rows)
    assert evidence["certificate_count"] == 35

    rows[0]["purpose_counts"]["unattributed"] = 1
    with pytest.raises(ValueError, match="purpose ledger"):
        module.build_strict_evidence(rows)


def test_starter_snapshot_round_trips_the_exact_state(tmp_path):
    module = _load_module()
    state = State(
        numbers=np.array([6, 8]),
        positions=np.array(
            [
                [0.1234567890123456, -0.25, 1.0],
                [2.0, 3.0, 4.0],
            ]
        ),
        cell=np.diag([5.0, 6.0, 7.0]),
        pbc=(True, False, True),
        fixed_mask=np.array([True, False]),
        metadata={"ignored": "snapshot identity is structural"},
    )
    path = tmp_path / "starter.json"

    module._write_state_snapshot(path, state)
    restored = module._read_state_snapshot(path)

    assert module._state_sha256(restored) == module._state_sha256(state)
