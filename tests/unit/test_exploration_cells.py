from __future__ import annotations

import numpy as np

from pamssw.exploration.cells import build_fps_cell_partition


def test_fps_cells_give_equal_cell_mass_and_uniform_mass_within_each_cell():
    partition = build_fps_cell_partition(
        starter_ids=(10, 20, 30, 40),
        features=np.array([[0.0], [1.0], [2.0], [10.0]]),
        max_cells=2,
    )

    assert partition.center_ids == (10, 40)
    assert partition.members_by_center == ((10, 20, 30), (40,))
    assert partition.cell_probability_for(10) == 0.5
    assert partition.conditional_probability_for(10) == 1.0 / 3.0
    assert partition.probability_for(10) == 1.0 / 6.0
    assert partition.probability_for(20) == 1.0 / 6.0
    assert partition.probability_for(30) == 1.0 / 6.0
    assert partition.probability_for(40) == 0.5
    assert sum(partition.probabilities) == 1.0
    assert all(probability > 0.0 for probability in partition.probabilities)


def test_fps_partition_is_invariant_to_input_order_and_returns_sorted_starters():
    first = build_fps_cell_partition(
        starter_ids=(7, 2, 9, 4),
        features=np.array([[5.0], [0.0], [9.0], [1.0]]),
        max_cells=2,
    )
    second = build_fps_cell_partition(
        starter_ids=(4, 9, 2, 7),
        features=np.array([[1.0], [9.0], [0.0], [5.0]]),
        max_cells=2,
    )

    assert first == second
    assert first.starter_ids == (2, 4, 7, 9)


def test_cell_capacity_at_least_archive_size_reduces_exactly_to_node_uniform():
    partition = build_fps_cell_partition(
        starter_ids=(3, 1, 2),
        features=np.array([[3.0], [1.0], [2.0]]),
        max_cells=8,
    )

    assert partition.center_ids == (1, 3, 2)
    assert partition.members_by_center == ((1,), (3,), (2,))
    assert partition.probabilities == (1.0 / 3.0,) * 3


def test_exactly_duplicate_features_do_not_create_empty_cells():
    partition = build_fps_cell_partition(
        starter_ids=(1, 2, 3),
        features=np.array([[0.0], [0.0], [1.0]]),
        max_cells=3,
    )

    assert partition.center_ids == (1, 3)
    assert partition.members_by_center == ((1, 2), (3,))
    assert partition.probabilities == (0.25, 0.25, 0.5)


def test_cell_partition_builds_a_full_support_policy_snapshot_with_exact_marginals():
    partition = build_fps_cell_partition(
        starter_ids=(10, 20, 30, 40),
        features=np.array([[0.0], [1.0], [2.0], [10.0]]),
        max_cells=2,
    )

    snapshot = partition.policy_snapshot(version=7, archive_version=11)

    assert snapshot.policy_name == "fps_cell_uniform"
    assert snapshot.version == 7
    assert snapshot.archive_version == 11
    assert snapshot.eligible_starter_ids == partition.starter_ids
    assert snapshot.probabilities == partition.probabilities
    assert snapshot.support_complete is True
