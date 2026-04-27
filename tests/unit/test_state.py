import numpy as np
import pytest

from pamssw.state import State


def test_state_accepts_cluster_and_slab_shapes():
    cluster = State(
        numbers=np.array([1, 1]),
        positions=np.array([[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]]),
    )
    slab = State(
        numbers=np.array([29, 29]),
        positions=np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 1.8]]),
        cell=np.diag([5.0, 5.0, 15.0]),
        pbc=(True, True, False),
        fixed_mask=np.array([True, False]),
    )

    assert cluster.positions.shape == (2, 3)
    assert slab.cell.shape == (3, 3)
    assert slab.fixed_mask.tolist() == [True, False]


def test_state_rejects_mismatched_atom_counts():
    with pytest.raises(ValueError):
        State(
            numbers=np.array([1, 1]),
            positions=np.array([[0.0, 0.0, 0.0]]),
        )
