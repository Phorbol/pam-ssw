import importlib
import importlib.util

import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms


def _native_bond_groups():
    spec = importlib.util.find_spec("pamssw.standalone.native_bond_groups")
    assert spec is not None, "native bond-group implementation is missing"
    return importlib.import_module(spec.name).native_bond_groups


def test_native_bond_groups_builds_two_disconnected_endpoint_components():
    atoms = Atoms(
        "CHCH",
        positions=[[0, 0, 0], [1.0, 0, 0], [6, 0, 0], [7.0, 0, 0]],
    )

    result = _native_bond_groups()(atoms, (0, 2))

    assert result.status == "separate_groups"
    assert np.array_equal(result.first_group, [0, 1, 0, 0])
    assert np.array_equal(result.second_group, [0, 0, 0, 1])


def test_native_bond_groups_reports_connected_pair_fallback():
    atoms = Atoms("CHH", positions=[[0, 0, 0], [1.0, 0, 0], [4, 0, 0]])

    result = _native_bond_groups()(atoms, (0, 1))

    assert result.status == "connected_pair_fallback"
    assert np.array_equal(result.first_group, [0, 1, 0])
    assert np.array_equal(result.second_group, [0, 0, 0])


def test_native_bond_groups_uses_strict_fastbond_cutoff():
    # Native C-H fastbond is 1.0839999914169312 A; the route multiplies by 1.3.
    threshold = 1.0839999914169312 * 1.3
    below = Atoms("CHH", positions=[[0, 0, 0], [threshold - 1e-9, 0, 0], [8, 0, 0]])
    exact = Atoms("CHH", positions=[[0, 0, 0], [threshold, 0, 0], [8, 0, 0]])

    assert _native_bond_groups()(below, (0, 2)).first_group.tolist() == [0, 1, 0]
    assert _native_bond_groups()(exact, (0, 2)).first_group.tolist() == [0, 0, 0]


def test_native_bond_groups_uses_native_radius_fallback_for_cu_au():
    # fastbond(Cu, Au)=0.0, so the native 1.25+1.25 A radii are used.
    atoms = Atoms("CuAu", positions=[[0, 0, 0], [3.0, 0, 0]])

    result = _native_bond_groups()(atoms, (0, 1))

    assert result.first_group.tolist() == [0, 1]
    assert result.status == "connected_pair_fallback"
    assert result.cutoff_sources == ("radius_fallback",)


def test_native_bond_groups_rejects_unverified_pairs_periodicity_and_constraints():
    unsupported = Atoms("XC", positions=[[0, 0, 0], [2, 0, 0]])
    with pytest.raises(ValueError, match="physical atomic numbers"):
        _native_bond_groups()(unsupported, (0, 1))
    periodic = Atoms("CH", positions=[[0, 0, 0], [1, 0, 0]], cell=[20, 20, 20], pbc=True)
    with pytest.raises(ValueError, match="nonperiodic"):
        _native_bond_groups()(periodic, (0, 1))
    constrained = Atoms("CH", positions=[[0, 0, 0], [1, 0, 0]])
    constrained.set_constraint(FixAtoms(indices=[0]))
    with pytest.raises(ValueError, match="constraints"):
        _native_bond_groups()(constrained, (0, 1))


def test_full_native_table_supports_metal_radius_fallback():
    atoms=Atoms('Cu3',positions=[[0,0,0],[2.5,0,0],[8,1,0]])
    result=_native_bond_groups()(atoms,(0,2))
    assert result.first_group.tolist()==[0,1,0]
    assert result.second_group.tolist()==[0,0,0]
    assert result.cutoff_sources==('radius_fallback',)
