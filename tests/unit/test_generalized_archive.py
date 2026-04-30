from __future__ import annotations

import numpy as np

from pamssw.archive import MinimaArchive
from pamssw.fingerprint import cell_descriptor, variable_cell_structural_descriptor
from pamssw.state import State


def make_state(cell_scale: float = 1.0) -> State:
    cell = np.eye(3) * 5.0 * cell_scale
    return State(
        numbers=np.array([6, 6]),
        positions=np.array([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5]]) * cell_scale,
        cell=cell,
        pbc=(True, True, True),
    )


def test_cell_descriptor_tracks_volume_per_atom() -> None:
    base = cell_descriptor(make_state(1.0))
    expanded = cell_descriptor(make_state(1.2))
    assert expanded[0] > base[0]


def test_variable_cell_descriptor_appends_lattice_terms() -> None:
    descriptor = variable_cell_structural_descriptor(make_state())
    assert descriptor.shape == (27,)


def test_archive_does_not_dedup_different_cells() -> None:
    archive = MinimaArchive(energy_tol=1.0, rmsd_tol=0.1, variable_cell=True, cell_tol=0.1)
    first = archive.add(make_state(1.0), -1.0, None)
    second = archive.add(make_state(1.2), -1.0, None)
    assert first.entry_id != second.entry_id
    assert len(archive.entries) == 2


def test_archive_dedups_same_cell_same_atoms() -> None:
    archive = MinimaArchive(energy_tol=1.0, rmsd_tol=0.1, variable_cell=True, cell_tol=0.1)
    first = archive.add(make_state(1.0), -1.0, None)
    second = archive.add(make_state(1.0), -1.0, None)
    assert first.entry_id == second.entry_id
    assert len(archive.entries) == 1


def test_archive_descriptor_for_exposes_variable_cell_descriptor() -> None:
    archive = MinimaArchive(energy_tol=1.0, rmsd_tol=0.1, variable_cell=True, cell_tol=0.1)
    descriptor = archive.descriptor_for(make_state())
    assert descriptor.shape == (27,)
