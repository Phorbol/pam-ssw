from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from pamssw.softening import LocalSofteningModel
from pamssw.state import State


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = (
    REPO_ROOT / "runs" / "20260731-ls-four-operator-gate" / "protocol.py"
)
RUNNER_PATH = (
    REPO_ROOT / "runs" / "20260731-ls-four-operator-gate" / "run_gate.py"
)


def _load_protocol():
    spec = importlib.util.spec_from_file_location(
        "_ls_four_operator_protocol",
        PROTOCOL_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {PROTOCOL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "_ls_four_operator_runner",
        RUNNER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_pair_hessian_splits_radial_and_transverse_curvature():
    protocol = _load_protocol()
    state = State(
        numbers=np.array([1, 1]),
        positions=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
    )
    model = LocalSofteningModel.from_state(
        state,
        pairs=[(0, 1)],
        strength=0.6,
        mode="manual",
        penalty="buckingham_repulsive",
        xi=0.2,
        reference_scaled_xi=True,
        cutoff=None,
    )
    direction = np.array([0.0, 0.0, 0.0, 0.3, 0.4, 0.0])

    action = protocol.pair_operator_action(model, state, direction)

    decay_length = 0.2 * 2.0
    first_derivative = -0.6 / decay_length
    second_derivative = 0.6 / decay_length**2
    expected_radial = second_derivative * 0.3**2
    expected_transverse = first_derivative / 2.0 * 0.4**2
    assert action.radial_curvature == pytest.approx(expected_radial)
    assert action.transverse_curvature == pytest.approx(expected_transverse)
    assert float(direction @ action.hvp) == pytest.approx(
        expected_radial + expected_transverse
    )
    assert action.pair_expressivity_numerator == pytest.approx(0.3**2)


@pytest.mark.parametrize("penalty", ["buckingham_repulsive", "gaussian_well"])
def test_pair_hessian_matches_finite_difference_gradient(penalty):
    protocol = _load_protocol()
    reference = State(
        numbers=np.array([1, 1]),
        positions=np.array([[0.0, 0.0, 0.0], [1.7, 0.0, 0.0]]),
    )
    model = LocalSofteningModel.from_state(
        reference,
        pairs=[(0, 1)],
        strength=0.4,
        mode="manual",
        penalty=penalty,
        xi=0.3,
        cutoff=None,
    )
    evaluation_state = reference.with_flat_positions(
        np.array([0.0, 0.0, 0.0, 1.8, 0.2, 0.0])
    )
    direction = np.array([0.03, -0.02, 0.01, -0.04, 0.05, -0.03])
    direction /= np.linalg.norm(direction)

    action = protocol.pair_operator_action(model, evaluation_state, direction)
    epsilon = 1.0e-5
    flat = evaluation_state.flatten_positions()
    _, gradient_plus = model.evaluate(flat + epsilon * direction)
    _, gradient_minus = model.evaluate(flat - epsilon * direction)
    finite_difference = (gradient_plus - gradient_minus) / (2.0 * epsilon)

    np.testing.assert_allclose(action.hvp, finite_difference, rtol=1.0e-6, atol=1.0e-7)


def test_alignment_transports_reference_direction_to_rotated_prestrained_frame():
    protocol = _load_protocol()
    reference = State(
        numbers=np.array([6, 6, 6, 6]),
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0],
            ]
        ),
    )
    angle = 0.37
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    transformed = State(
        numbers=reference.numbers,
        positions=reference.positions @ rotation.T + np.array([2.0, -1.0, 0.5]),
    )
    direction = np.arange(12, dtype=float).reshape(-1, 3)

    alignment = protocol.align_prestrained_to_reference(reference, transformed)

    np.testing.assert_allclose(
        alignment.positions_in_reference_frame,
        reference.positions,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        alignment.direction_to_prestrained(direction.reshape(-1)),
        (direction @ rotation.T).reshape(-1),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        alignment.direction_to_reference((direction @ rotation.T).reshape(-1)),
        direction.reshape(-1),
        atol=1.0e-12,
    )


def test_four_operator_row_separates_operator_prestrain_and_interaction():
    protocol = _load_protocol()
    direction_x0 = np.array([1.0, 0.0, 0.0])
    direction_xr = np.array([0.0, 1.0, 0.0])
    true_hvp_x0 = np.array([2.0, 0.0, 0.0])
    true_hvp_xr = np.array([0.0, 3.0, 0.0])
    ls_x0 = protocol.PairOperatorAction(
        hvp=np.array([-0.5, 0.0, 0.0]),
        radial_curvature=-0.7,
        transverse_curvature=0.2,
        pair_expressivity_numerator=0.4,
    )
    ls_xr = protocol.PairOperatorAction(
        hvp=np.array([0.0, -0.2, 0.0]),
        radial_curvature=-0.4,
        transverse_curvature=0.2,
        pair_expressivity_numerator=0.6,
    )

    row = protocol.build_operator_row(
        direction_x0=direction_x0,
        direction_xr=direction_xr,
        true_hvp_x0=true_hvp_x0,
        true_hvp_xr=true_hvp_xr,
        ls_action_x0=ls_x0,
        ls_action_xr=ls_xr,
    )

    assert row["kappa_a"] == pytest.approx(2.0)
    assert row["kappa_b"] == pytest.approx(1.5)
    assert row["kappa_c"] == pytest.approx(3.0)
    assert row["kappa_d"] == pytest.approx(2.8)
    assert row["operator_effect_x0"] == pytest.approx(-0.5)
    assert row["prestrain_effect"] == pytest.approx(1.0)
    assert row["operator_geometry_interaction"] == pytest.approx(0.3)
    assert row["pair_expressivity_x0"] == pytest.approx(0.4)
    assert row["pair_expressivity_xr"] == pytest.approx(0.6)
    assert row["ls_radial_curvature_x0"] == pytest.approx(-0.7)
    assert row["ls_transverse_curvature_x0"] == pytest.approx(0.2)


def _synthetic_operator_cases():
    cases = []
    for state_id in ("bootstrap", "mid", "late"):
        for seed in (42, 43, 44):
            offset = float(seed - 42)
            candidate_rows = []
            for candidate_index in range(2):
                base = float(candidate_index) + offset
                candidate_rows.append(
                    {
                        "candidate_index": candidate_index,
                        "candidate_kind": "random" if candidate_index == 0 else "bond",
                        "kappa_a": base + 1.0,
                        "kappa_b": base + 0.8,
                        "kappa_c": base + 0.5,
                        "kappa_d": base + 0.4,
                        "operator_effect_x0": -0.2,
                        "prestrain_effect": -0.5,
                        "operator_effect_xr": -0.1,
                        "operator_geometry_interaction": 0.1,
                        "ls_radial_curvature_x0": 0.3,
                        "ls_transverse_curvature_x0": -0.5,
                        "ls_radial_curvature_xr": 0.2,
                        "ls_transverse_curvature_xr": -0.3,
                        "pair_expressivity_x0": 0.4,
                        "pair_expressivity_xr": 0.5,
                        "direction_norm_x0": 1.0,
                        "direction_norm_xr": 1.0,
                    }
                )
            cases.append(
                {
                    "state_id": state_id,
                    "seed": seed,
                    "candidate_pool_sha256": f"{state_id}-{seed}",
                    "candidate_rows": candidate_rows,
                    "selected_candidate": {"a": 0, "b": 1, "c": 0, "d": 1},
                    "force_evaluations": 20,
                    "purpose_counts": {
                        "direction_oracle": 16,
                        "local_softening_pre_relax": 4,
                        "unattributed": 0,
                    },
                }
            )
    return cases


def test_build_evidence_requires_complete_closed_nine_block_cohort():
    protocol = _load_protocol()
    cases = _synthetic_operator_cases()

    evidence = protocol.build_evidence(cases)

    assert evidence["cohort"] == {
        "states": ["bootstrap", "mid", "late"],
        "seeds": [42, 43, 44],
        "blocks": 9,
        "candidates_per_block": 2,
    }
    assert evidence["force_accounting"] == {
        "total": 180,
        "unattributed": 0,
    }
    assert evidence["selection_changes"] == {
        "b_vs_a": 9,
        "c_vs_a": 0,
        "d_vs_a": 9,
    }
    assert evidence["median_candidate_effects"]["operator_effect_x0"] == pytest.approx(-0.2)


@pytest.mark.parametrize("mutation", ["missing", "unattributed", "pool_drift"])
def test_build_evidence_rejects_incomplete_or_unclosed_cases(mutation):
    protocol = _load_protocol()
    cases = _synthetic_operator_cases()
    if mutation == "missing":
        cases.pop()
    elif mutation == "unattributed":
        cases[0]["purpose_counts"]["unattributed"] = 1
        cases[0]["force_evaluations"] += 1
    else:
        cases[0]["candidate_rows"][1]["candidate_index"] = 3

    with pytest.raises(ValueError):
        protocol.build_evidence(cases)


def test_runner_contract_reuses_two_true_hvp_stencils_and_forbids_proposal_ls():
    runner = _load_runner()

    contract = runner.execution_contract(candidate_count=4)

    assert contract == {
        "true_pes_hvp_geometries": 2,
        "central_fd_force_evaluations": 16,
        "operator_a_b_share_true_hvp": True,
        "operator_c_d_share_true_hvp": True,
        "analytic_ls_hvp_force_evaluations": 0,
        "proposal_side_ls": False,
    }


def test_runner_selects_each_operator_from_the_same_candidate_rows():
    runner = _load_runner()
    rows = [
        {
            "candidate_index": 0,
            "score_a": 2.0,
            "score_b": 1.0,
            "score_c": 0.0,
            "score_d": 3.0,
        },
        {
            "candidate_index": 1,
            "score_a": 1.0,
            "score_b": 2.0,
            "score_c": 3.0,
            "score_d": 0.0,
        },
    ]

    selected = runner.select_candidate_indices(rows)

    assert selected == {"a": 0, "b": 1, "c": 1, "d": 0}
