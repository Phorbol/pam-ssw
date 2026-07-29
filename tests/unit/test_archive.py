import copy

import numpy as np

from pamssw.archive import MinimaArchive
from pamssw.state import State


def _state(x):
    return State(
        numbers=np.array([1]),
        positions=np.array([[x, 0.0, 0.0]]),
    )


def _pair_state(distance: float) -> State:
    return State(
        numbers=np.array([1, 1]),
        positions=np.array([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]]),
    )


def test_archive_deduplicates_nearby_structures():
    archive = MinimaArchive(energy_tol=1e-3, rmsd_tol=0.05)

    first = archive.add(_state(-1.0), -1.0, parent_id=None)
    second = archive.add(_state(-1.02), -1.0005, parent_id=first.entry_id)
    third = archive.add(_state(1.0), -0.8, parent_id=first.entry_id)

    assert first.entry_id == 0
    assert second.entry_id == 0
    assert third.entry_id == 1
    assert len(archive.entries) == 2


def test_archive_deduplicates_rigidly_moved_cluster():
    base = State(
        numbers=np.array([18, 18, 18, 18]),
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 0.8660254, 0.0],
                [0.5, 0.2886751, 0.8164966],
            ]
        ),
    )
    angle = np.deg2rad(37.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    moved = State(
        numbers=base.numbers.copy(),
        positions=base.positions @ rotation.T + np.array([4.0, -2.5, 1.7]),
    )

    archive = MinimaArchive(energy_tol=1e-4, rmsd_tol=1e-2)
    first = archive.add(base, -6.0, parent_id=None)
    second = archive.add(moved, -6.0 + 5e-5, parent_id=first.entry_id)

    assert second.entry_id == first.entry_id
    assert len(archive.entries) == 1


def test_archive_deduplicates_nonperiodic_cluster_with_visualization_cell():
    base = State(
        numbers=np.array([18, 18, 18, 18]),
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 0.8660254, 0.0],
                [0.5, 0.2886751, 0.8164966],
            ]
        ),
        cell=np.diag([20.0, 20.0, 20.0]),
        pbc=(False, False, False),
    )
    angle = np.deg2rad(37.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    moved = State(
        numbers=base.numbers.copy(),
        positions=base.positions @ rotation.T + np.array([4.0, -2.5, 1.7]),
        cell=base.cell,
        pbc=base.pbc,
    )

    archive = MinimaArchive(energy_tol=1e-4, rmsd_tol=1e-2)
    first = archive.add(base, -6.0, parent_id=None)
    second = archive.add(moved, -6.0 + 5e-5, parent_id=first.entry_id)

    assert second.entry_id == first.entry_id
    assert len(archive.entries) == 1


def test_archive_rmsd_uses_mic_for_periodic_duplicates():
    archive = MinimaArchive(energy_tol=1e-6, rmsd_tol=0.25)
    cell = np.diag([5.0, 5.0, 5.0])
    first = State(
        numbers=np.array([18, 18]),
        positions=np.array([[0.2, 0.0, 0.0], [1.2, 0.0, 0.0]]),
        cell=cell,
        pbc=(True, True, True),
    )
    wrapped_copy = State(
        numbers=np.array([18, 18]),
        positions=np.array([[5.2, 0.0, 0.0], [1.2, 0.0, 0.0]]),
        cell=cell,
        pbc=(True, True, True),
    )

    entry = archive.add(first, -1.0, parent_id=None)
    duplicate = archive.add(wrapped_copy, -1.0, parent_id=None)

    assert duplicate.entry_id == entry.entry_id
    assert len(archive.entries) == 1


def test_archive_keeps_distinct_single_particle_positions():
    archive = MinimaArchive(energy_tol=1e-4, rmsd_tol=1e-2)
    first = archive.add(_state(-1.0), 0.0, parent_id=None)
    second = archive.add(_state(1.0), 0.0, parent_id=first.entry_id)

    assert second.entry_id != first.entry_id
    assert len(archive.entries) == 2


def test_archive_prototype_set_is_bounded_independently_of_entries():
    archive = MinimaArchive(energy_tol=1e-8, rmsd_tol=1e-8, max_prototypes=3)

    for index in range(8):
        archive.add(_pair_state(0.8 + 0.2 * index), float(index), parent_id=None)

    assert len(archive.entries) == 8
    assert len(archive.prototypes) == 3
    assert archive.prototype_occupancy()["n_prototypes"] == 3


def test_archive_density_uses_weighted_prototypes_for_occupancy():
    archive = MinimaArchive(energy_tol=1e-8, rmsd_tol=1e-8, max_prototypes=4)
    crowded = archive.add(_pair_state(1.0), 0.0, parent_id=None)
    for index in range(5):
        archive.add(_pair_state(1.0 + 0.01 * (index + 1)), float(index + 1), parent_id=crowded.entry_id)
    sparse = archive.add(_pair_state(3.0), 10.0, parent_id=crowded.entry_id)

    assert archive.descriptor_density(crowded) > archive.descriptor_density(sparse)


def test_frontier_value_comes_from_observable_low_visit_sparse_node():
    archive = MinimaArchive(energy_tol=1e-8, rmsd_tol=1e-8)
    crowded = archive.add(_pair_state(1.0), -10.0, parent_id=None)
    sparse = archive.add(_pair_state(3.0), -9.9, parent_id=None)
    for index in range(4):
        archive.add(_pair_state(1.0 + 0.01 * (index + 1)), -9.8 + index, parent_id=crowded.entry_id)
    crowded.node_trials = 8
    crowded.node_successes = 0
    sparse.node_trials = 0
    sparse.node_successes = 0

    archive.refresh_frontier_status()

    assert sparse.frontier_score > crowded.frontier_score
    assert sparse.is_frontier


def test_dead_node_status_comes_from_duplicate_and_failed_trial_statistics():
    archive = MinimaArchive(energy_tol=1e-8, rmsd_tol=1e-8)
    entry = archive.add(_pair_state(1.0), -10.0, parent_id=None)
    entry.node_trials = 12
    entry.node_successes = 0
    entry.node_duplicate_failures = 10

    archive.refresh_frontier_status()

    assert entry.is_dead
    assert entry.frontier_score == 0.0


def test_duplicate_hits_on_target_basin_do_not_make_that_target_a_dead_seed():
    archive = MinimaArchive(energy_tol=1e-8, rmsd_tol=1e-8)
    target = archive.add(_pair_state(1.0), -10.0, parent_id=None)
    target.node_trials = 12
    target.node_successes = 0
    target.duplicate_hits = 10

    archive.refresh_frontier_status()

    assert not target.is_dead


def test_find_match_is_read_only():
    archive = MinimaArchive(energy_tol=1e-3, rmsd_tol=0.05)
    entry = archive.add(_state(-1.0), -1.0, parent_id=None)
    archive.add(_state(1.0), -0.8, parent_id=entry.entry_id)
    entries_before = copy.deepcopy(archive.entries)
    prototypes_before = copy.deepcopy(archive.prototypes)

    match = archive.find_match(_state(-1.02), -1.0005)

    assert match is entry
    assert entry.visits == 1
    assert entry.duplicate_hits == 0
    assert len(archive.entries) == len(entries_before)
    for after, before in zip(archive.entries, entries_before, strict=True):
        assert after.entry_id == before.entry_id
        assert after.energy == before.energy
        assert after.parent_id == before.parent_id
        assert after.visits == before.visits
        assert after.node_trials == before.node_trials
        assert after.node_successes == before.node_successes
        assert after.frontier_value == before.frontier_value
        assert after.duplicate_hits == before.duplicate_hits
        assert after.node_duplicate_failures == before.node_duplicate_failures
        assert after.frontier_score == before.frontier_score
        assert after.is_frontier == before.is_frontier
        assert after.is_dead == before.is_dead
        np.testing.assert_array_equal(after.state.numbers, before.state.numbers)
        np.testing.assert_array_equal(after.state.positions, before.state.positions)
        assert after.state.pbc == before.state.pbc
        assert after.state.metadata == before.state.metadata
        np.testing.assert_array_equal(after.state.fixed_mask, before.state.fixed_mask)
        assert (after.state.cell is None) == (before.state.cell is None)
        if after.state.cell is not None:
            np.testing.assert_array_equal(after.state.cell, before.state.cell)
        assert (after.descriptor is None) == (before.descriptor is None)
        if after.descriptor is not None:
            np.testing.assert_array_equal(after.descriptor, before.descriptor)

    assert len(archive.prototypes) == len(prototypes_before)
    for after, before in zip(archive.prototypes, prototypes_before, strict=True):
        np.testing.assert_array_equal(after.descriptor, before.descriptor)
        assert after.representative_entry_id == before.representative_entry_id
        assert after.weight == before.weight


def test_archive_clone_is_independent_of_source_mutations():
    archive = MinimaArchive(energy_tol=1e-3, rmsd_tol=0.05)
    archive.add(_state(-1.0), -1.0, parent_id=None)
    cloned = archive.clone()

    cloned.add(_state(1.0), -0.8, parent_id=0)
    cloned.entries[0].node_trials = 7

    assert len(archive.entries) == 1
    assert archive.entries[0].node_trials == 0
    assert len(cloned.entries) == 2


def test_archive_clone_does_not_share_prototype_list_or_descriptor_arrays():
    archive = MinimaArchive(energy_tol=1e-3, rmsd_tol=0.05)
    archive.add(_state(-1.0), -1.0, parent_id=None)
    source_prototype = copy.deepcopy(archive.prototypes[0])
    cloned = archive.clone()

    cloned.prototypes[0].descriptor[0] += 1.0
    cloned.prototypes.append(copy.deepcopy(cloned.prototypes[0]))

    assert len(archive.prototypes) == 1
    np.testing.assert_array_equal(archive.prototypes[0].descriptor, source_prototype.descriptor)
    assert archive.prototypes[0].representative_entry_id == source_prototype.representative_entry_id
    assert archive.prototypes[0].weight == source_prototype.weight
    assert len(cloned.prototypes) == 2
