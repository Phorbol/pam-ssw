import numpy as np
import pytest

from pamssw.metric import AtomCellBlockMetric, EuclideanMetric, MassWeightedMetric, MetricKind


def test_metric_kinds_match_documented_names():
    assert MetricKind.EUCLIDEAN.value == "euclidean"
    assert MetricKind.MASS_WEIGHTED.value == "mass_weighted"
    assert MetricKind.ATOM_CELL_BLOCK.value == "atom_cell_block"


def test_euclidean_metric_norm_and_dot():
    metric = EuclideanMetric()
    vector = np.array([3.0, 4.0])

    assert metric.dot(vector, vector) == 25.0
    assert metric.norm(vector) == 5.0


def test_mass_weighted_metric_repeats_atomic_masses_per_coordinate():
    metric = MassWeightedMetric(np.array([1.0, 4.0]))
    vector = np.ones(6)

    assert metric.dot(vector, vector) == 15.0


def test_atom_cell_block_metric_is_explicitly_unavailable():
    with pytest.raises(NotImplementedError):
        AtomCellBlockMetric()
