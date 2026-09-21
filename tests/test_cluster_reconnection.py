import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from pamssw.standalone.cluster_reconnection import (
    ClusterReconnectionResult,
    reconnect_clusters,
)
from research.ga_ssw.native_vapor_reference import vapor_reference


def test_two_components_preserves_input_and_internal_distances():
    atoms = Atoms('C4', positions=[[0, 0, 0], [.5, 0, 0], [5, 0, 0], [5.5, 0, 0]])
    original = atoms.positions.copy()
    before = atoms.get_all_distances(mic=False)
    result = reconnect_clusters(atoms, 1.1, repair=True)
    assert isinstance(result, ClusterReconnectionResult)
    assert np.array_equal(atoms.positions, original)
    assert np.array_equal(result.atoms.get_chemical_symbols(), atoms.get_chemical_symbols())
    assert np.allclose(result.atoms.get_all_distances(mic=False)[:2, :2], before[:2, :2])
    assert np.allclose(result.atoms.get_all_distances(mic=False)[2:, 2:], before[2:, 2:])
    assert result.last_separation == pytest.approx(4.5)
    assert len(result.moves) == 1


def test_last_attachment_is_not_global_pair_minimum():
    atoms = Atoms('C6', positions=[[0, 0, 0], [.5, 0, 0], [5, 0, 0], [5.5, 0, 0], [20, 0, 0], [20.5, 0, 0]])
    result = reconnect_clusters(atoms, 1.1, repair=False)
    assert result.last_separation == pytest.approx(4.5)
    assert result.last_separation > np.min(atoms.get_all_distances(mic=False)[np.triu_indices(6, 1)])


@pytest.mark.parametrize('atoms', [Atoms('C2', positions=[[0, 0, 0], [1, 0, 0]], pbc=True),
                                    Atoms('C2', positions=[[0, 0, 0], [1, 0, 0]], constraint=[([0])])])
def test_rejects_periodic_or_constrained_atoms(atoms):
    with pytest.raises(ValueError):
        reconnect_clusters(atoms, 1.1)


def test_rejects_invalid_size_and_distance():
    with pytest.raises(ValueError):
        reconnect_clusters(Atoms('C'), 1.1)
    with pytest.raises(ValueError):
        reconnect_clusters(Atoms('C2', positions=[[0, 0, 0], [1, 0, 0]]), 0)
    with pytest.raises(ValueError):
        reconnect_clusters(Atoms('C2', positions=[[0, 0, 0], [1, 0, 0]]), 'bad')


def test_existing_54_reference_inputs_match():
    rows = json.loads(Path('research/ga_ssw/evidence/native-vapor-reference/comparison.json').read_text())
    for row in rows:
        atoms = Atoms('C' * row['n'], positions=row['positions'])
        got = reconnect_clusters(atoms, 1.7, repair=bool(row['mode']))
        ref = vapor_reference(row['positions'], 1.7, repair=bool(row['mode']))
        assert got.medoid == ref['medoid'] == row['medoid']
        assert got.last_separation == pytest.approx(ref['scalar'], abs=1e-12)
        assert np.allclose(got.atoms.positions, ref['positions'], atol=1e-12)
