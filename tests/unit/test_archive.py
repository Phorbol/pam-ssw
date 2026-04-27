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
