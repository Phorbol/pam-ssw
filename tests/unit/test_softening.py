import numpy as np
import pytest

from pamssw.softening import LocalSofteningModel, automatic_neighbor_pairs
from pamssw.state import State


def test_automatic_neighbor_pairs_use_covalent_cutoff():
    state = State(
        numbers=np.array([6, 1, 1]),
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.09, 0.0, 0.0],
                [3.00, 0.0, 0.0],
            ]
        ),
    )

    pairs = automatic_neighbor_pairs(state, cutoff_scale=1.25)

    assert pairs == [(0, 1)]


def test_automatic_neighbor_pairs_use_mic_for_periodic_slab_axes():
    state = State(
        numbers=np.array([1, 1]),
        positions=np.array([[0.1, 0.0, 0.0], [9.8, 0.0, 0.0]]),
        cell=np.diag([10.0, 10.0, 20.0]),
        pbc=(True, True, False),
    )

    pairs = automatic_neighbor_pairs(state, cutoff_scale=1.25)

    assert pairs == [(0, 1)]


def test_automatic_neighbor_pairs_can_filter_to_active_atoms():
    state = State(
        numbers=np.array([6, 1, 6, 1]),
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.09, 0.0, 0.0],
                [5.0, 0.0, 0.0],
                [6.09, 0.0, 0.0],
            ]
        ),
    )

    pairs = automatic_neighbor_pairs(state, cutoff_scale=1.25, active_indices=np.array([2]))

    assert pairs == [(2, 3)]


def test_local_softening_model_from_neighbor_auto_builds_terms():
    state = State(
        numbers=np.array([6, 1]),
        positions=np.array([[0.0, 0.0, 0.0], [1.09, 0.0, 0.0]]),
    )

    model = LocalSofteningModel.from_state(
        state,
        pairs=None,
        strength=0.6,
        mode="neighbor_auto",
        cutoff_scale=1.25,
    )

    assert len(model.terms) == 1
    assert model.terms[0].atom_i == 0
    assert model.terms[0].atom_j == 1
    assert np.isclose(model.terms[0].reference_distance, 1.09)


def test_local_softening_model_manual_mode_preserves_explicit_pairs():
    state = State(
        numbers=np.array([6, 1, 1]),
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.09, 0.0, 0.0],
                [3.00, 0.0, 0.0],
            ]
        ),
    )

    model = LocalSofteningModel.from_state(
        state,
        pairs=[(0, 2)],
        strength=0.6,
        mode="manual",
    )

    assert [(term.atom_i, term.atom_j) for term in model.terms] == [(0, 2)]
    assert np.isclose(model.terms[0].reference_distance, 3.0)


def test_local_softening_model_rejects_negative_manual_pair_index():
    state = State(
        numbers=np.array([6, 1]),
        positions=np.array([[0.0, 0.0, 0.0], [1.09, 0.0, 0.0]]),
    )

    with pytest.raises(ValueError, match="pair|atom"):
        LocalSofteningModel.from_state(
            state,
            pairs=[(-1, 0)],
            strength=0.6,
            mode="manual",
        )


def test_local_softening_model_rejects_invalid_pbc_length():
    with pytest.raises(ValueError, match="pbc"):
        LocalSofteningModel([], pbc=(True, False))


def test_local_softening_model_reference_distance_uses_mic():
    state = State(
        numbers=np.array([1, 1]),
        positions=np.array([[0.1, 0.0, 0.0], [9.8, 0.0, 0.0]]),
        cell=np.diag([10.0, 10.0, 20.0]),
        pbc=(True, True, False),
    )

    model = LocalSofteningModel.from_state(
        state,
        pairs=[(0, 1)],
        strength=0.6,
        mode="manual",
    )

    assert np.isclose(model.terms[0].reference_distance, 0.3)


def test_local_softening_model_evaluate_uses_mic_for_periodic_coordinates():
    state = State(
        numbers=np.array([1, 1]),
        positions=np.array([[0.1, 0.0, 0.0], [9.8, 0.0, 0.0]]),
        cell=np.diag([10.0, 10.0, 20.0]),
        pbc=(True, True, False),
    )

    model = LocalSofteningModel.from_state(
        state,
        pairs=[(0, 1)],
        strength=0.6,
        mode="manual",
    )

    energy, gradient = model.evaluate(state.flatten_positions())

    assert energy > 0.59
    assert np.isfinite(gradient).all()
    assert gradient.shape == state.flatten_positions().shape


def test_local_softening_model_mic_gradient_matches_finite_difference():
    state = State(
        numbers=np.array([1, 1]),
        positions=np.array([[0.1, 0.0, 0.0], [9.8, 0.0, 0.0]]),
        cell=np.diag([10.0, 10.0, 20.0]),
        pbc=(True, True, False),
    )
    model = LocalSofteningModel.from_state(
        state,
        pairs=[(0, 1)],
        strength=0.6,
        mode="manual",
    )
    flat = state.flatten_positions()
    flat[3] = 9.7
    step = 1.0e-6

    _, gradient = model.evaluate(flat)
    plus = flat.copy()
    minus = flat.copy()
    plus[3] += step
    minus[3] -= step
    energy_plus, _ = model.evaluate(plus)
    energy_minus, _ = model.evaluate(minus)
    finite_difference = (energy_plus - energy_minus) / (2.0 * step)

    assert np.isclose(gradient[3], finite_difference, rtol=1.0e-5, atol=1.0e-7)


def test_local_softening_model_rejects_invalid_mode():
    state = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))

    with pytest.raises(ValueError, match="mode"):
        LocalSofteningModel.from_state(
            state,
            pairs=None,
            strength=0.6,
            mode="unknown",
        )
