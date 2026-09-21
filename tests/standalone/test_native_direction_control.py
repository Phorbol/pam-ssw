"""Controlled native outputs for the Q-off coefficient branch."""
import numpy as np
import pytest
from pamssw.standalone.native_direction_control import select_local_coefficients


def test_pair_group_route_and_native_draw_consumption():
    rng=iter([.3,.2,.7,.1,.9,.81])
    got=select_local_coefficients([0,1,1],rng,ratio_local=50,
                                  local_probability=.5,group_threshold=.5)
    np.testing.assert_allclose(got.coefficients,[0,1,0,0,1.6,0,0,0,0,0],atol=1e-15,rtol=0)
    assert got.group_marker == -1
    assert next(rng)==.81


def test_group_nonempty_and_strict_probability_boundary():
    got=select_local_coefficients([0,1,1],iter([.3,.2,.7,.5,.9]),ratio_local=50,
                                  local_probability=.5,group_threshold=.5)
    np.testing.assert_allclose(got.coefficients,[0,1,0,0,0,0,1.6,0,0,0],atol=1e-15,rtol=0)
    assert got.group_marker is None  # c6 leaves the caller's marker unchanged.


def test_empty_group_uses_pair_and_equal_marker_threshold_is_false():
    got=select_local_coefficients([0,0,0],iter([.3,.2,.7,.9,.5]),ratio_local=50,
                                  local_probability=.5,group_threshold=.5)
    assert got.coefficients[4] == 1.6
    assert got.group_marker == 0


def test_complete_c1_c6_generation_matches_native_geometry_fixtures():
    import json
    from pathlib import Path
    from ase.build import molecule
    from pamssw.standalone.native_direction_control import generate_local_direction
    evidence=json.loads((Path(__file__).parent/'fixtures/native_random_group_generator.json').read_text())
    for row in evidence['cases']:
        atoms=molecule(row['name']);atoms.positions=row['positions']
        coeff=np.zeros(10);coeff[1]=1;coeff[6]=row['weight']
        got=generate_local_direction(atoms,np.zeros_like(atoms.positions),coeff,
                                    row['pair'],row['group'],iter([row['seed']]),group_marker=0)
        np.testing.assert_allclose(got.direction,row['output'],atol=3e-14,rtol=0)
        assert not got.release_all
        assert got.local_route=='torsion'


def test_c4_initial_and_displacement_update_match_native():
    import json
    from pathlib import Path
    from ase import Atoms
    from pamssw.standalone.native_direction_control import generate_local_direction
    evidence=json.loads((Path(__file__).parent/'fixtures/native_pair_generator.json').read_text())
    for row in evidence['cases']:
        atoms=Atoms('Cu'*row['n'],positions=row['positions'])
        got=generate_local_direction(atoms,row['input_seed'],row['coefficients'],
                                    row['pair'],np.zeros(len(atoms),int),lambda:row['uniform'],group_marker=0)
        np.testing.assert_allclose(got.direction,row['output'],atol=3e-14,rtol=0)
        assert got.local_route=='pair'
        assert not got.release_all


def test_forbidden_pair_skips_local_motion_and_zero_requests_true_relaxation():
    # Native gate oracle: native-pair-generator-forbidden-20260917.json.
    from ase.build import molecule
    from pamssw.standalone.native_direction_control import generate_local_direction
    atoms=molecule('C6H6');atoms.positions+=15
    coefficients=np.zeros(10);coefficients[4]=.6;coefficients[9]=.72
    got=generate_local_direction(atoms,np.zeros_like(atoms.positions),coefficients,
                                (8,3),np.zeros(len(atoms),int),iter([.17]),group_marker=0)
    assert got.local_route=='forbidden'
    assert got.release_all
    assert not np.any(got.direction)


def test_c4_connected_fallback_and_separate_groups_match_native():
    import json
    from pathlib import Path
    from ase import Atoms
    from pamssw.standalone.cluster_frame import ClusterFrame
    from pamssw.standalone.native_direction_control import generate_local_direction
    evidence = json.loads((Path(__file__).parent / 'fixtures/native_c4_group_generator.json').read_text())
    for row in evidence['cases']:
        atoms = Atoms(numbers=row['numbers'], positions=row['positions'])
        seed = ClusterFrame(atoms).project(np.arange(len(atoms)*3).reshape(-1,3)*.013)
        seed /= np.linalg.norm(seed)
        coefficients = np.zeros(10)
        coefficients[4], coefficients[9] = .6, .72
        got = generate_local_direction(atoms, seed, coefficients, row['pair'],
            np.zeros(len(atoms), int), lambda: .17, group_marker=-1)
        np.testing.assert_allclose(got.direction, row['output'], atol=3e-14, rtol=0)
        connected = row['reference_status'] == 'connected_pair_fallback'
        assert got.local_route == ('pair_fallback' if connected else 'pair_group')
        assert got.group_marker == (0 if connected else -1)
        assert not got.release_all


def test_stage_state_uses_saved_center_not_previous_direction():
    import json
    from pathlib import Path
    from ase import Atoms
    from pamssw.standalone.native_direction_control import LocalDirectionState
    rows = json.loads((Path(__file__).parent/'fixtures/native_pair_generator.json').read_text())['cases']
    for row in rows:
        if row['coefficients'][9] == 0:
            continue
        atoms = Atoms('Cu'*row['n'], positions=row['positions'])
        center = atoms.positions - .05*np.asarray(row['input_seed'])
        coefficients = np.zeros(10); coefficients[1] = 1; coefficients[4] = .6
        state = LocalDirectionState(row['pair'], np.zeros(len(atoms), int), coefficients,
                                    group_marker=0)
        state.save_gaussian_center(center)
        center[:] = 0  # State must own its snapshot.
        got = state.update(atoms, lambda: row['uniform'])
        np.testing.assert_allclose(got.direction, row['output'], atol=3e-13, rtol=0)
        assert got.local_route == 'pair'
        np.testing.assert_array_equal(state.coefficients, coefficients)


def test_state_zero_displacement_still_generates_local_move():
    from ase import Atoms
    from pamssw.standalone.native_direction_control import LocalDirectionState
    atoms = Atoms('CHCH', positions=[[14.7,15.1,15],[13.9,14.8,15.2],
                                   [17.2,15.8,14.8],[18,16.3,15.1]])
    coefficients = np.zeros(10); coefficients[4] = .6
    state = LocalDirectionState((0,2), np.zeros(4,int), coefficients, group_marker=-1)
    state.save_gaussian_center(atoms.positions)
    got = state.update(atoms, lambda: .17)
    assert not got.release_all
    assert got.local_route == 'pair_group'
    assert np.isclose(np.linalg.norm(got.direction), 1.)


def test_c1_default_restricted_still_rejects_non_all_near_domain():
    from ase import Atoms
    from pamssw.standalone.native_direction_control import generate_local_direction
    atoms = Atoms('C4', positions=[[0, 0, 0], [1.4, 0, 0], [0, 1.4, 0], [20., 0, 0]])
    coeff = np.zeros(10); coeff[1] = 1.
    with pytest.raises(NotImplementedError, match='all-near'):
        generate_local_direction(atoms, np.zeros((4, 3)), coeff, (0, 1),
                                 np.zeros(4, int), iter([.17]), group_marker=0)


def test_c1_per_atom_matches_restricted_when_all_atoms_are_near():
    from ase import Atoms
    from pamssw.standalone.native_direction_control import generate_local_direction
    atoms = Atoms('C4', positions=[[0, 0, 0], [1.4, 0, 0], [0, 1.4, 0], [1.4, 1.4, 0]])
    coeff = np.zeros(10); coeff[1] = 1.
    kwargs = dict(atoms=atoms, seed=np.zeros((4, 3)), coefficients=coeff,
                  pair=(0, 1), group=np.zeros(4, int), group_marker=0)
    restricted = generate_local_direction(**kwargs, rng=iter([.17]), c1_radius_policy='restricted')
    per_atom = generate_local_direction(**kwargs, rng=iter([.17]), c1_radius_policy='per_atom')
    np.testing.assert_allclose(per_atom.direction, restricted.direction, atol=1e-14, rtol=0)


def test_c1_per_atom_handles_outlier_with_rigid_projected_normalized_direction():
    from ase import Atoms
    from pamssw.standalone.cluster_frame import ClusterFrame
    from pamssw.standalone.native_direction_control import generate_local_direction
    atoms = Atoms('C4', positions=[[0, 0, 0], [1.4, 0, 0], [0, 1.4, 0], [20., 0, 0]])
    coeff = np.zeros(10); coeff[1] = 1.
    got = generate_local_direction(atoms, np.zeros((4, 3)), coeff, (0, 1),
                                   np.zeros(4, int), iter([.17]), group_marker=0,
                                   c1_radius_policy='per_atom')
    assert np.isfinite(got.direction).all()
    assert np.isclose(np.linalg.norm(got.direction), 1.)
    projected = ClusterFrame(atoms).project(got.direction)
    np.testing.assert_allclose(projected, got.direction, atol=2e-14, rtol=0)


def test_c1_per_atom_passes_three_coordinate_mask_to_native_random(monkeypatch):
    from ase import Atoms
    import pamssw.standalone.native_random as random_module
    import pamssw.standalone.native_direction_control as module
    atoms = Atoms('C4', positions=[[0, 0, 0], [1.4, 0, 0], [0, 1.4, 0], [20., 0, 0]])
    coeff = np.zeros(10); coeff[1] = 1.
    captured = {}
    def capture(initial, mask, seed):
        captured['mask'] = np.array(mask, copy=True)
        return np.zeros_like(initial)
    monkeypatch.setattr(random_module, 'native_vmb2', capture)
    module.generate_local_direction(atoms, np.zeros((4, 3)), coeff, (0, 1),
                                    np.zeros(4, int), iter([.17]), group_marker=0,
                                    c1_radius_policy='per_atom')
    np.testing.assert_array_equal(captured['mask'],
                                  [[True, True, True], [True, True, True],
                                   [True, True, True], [False, False, False]])


def test_c1_per_atom_accepts_atom_permutation_without_claiming_rng_invariance():
    from ase import Atoms
    from pamssw.standalone.native_direction_control import generate_local_direction
    atoms = Atoms('C4', positions=[[20., 0, 0], [0, 0, 0], [1.4, 0, 0], [0, 1.4, 0]])
    coeff = np.zeros(10); coeff[1] = 1.
    got = generate_local_direction(atoms, np.zeros((4, 3)), coeff, (1, 2),
                                   np.zeros(4, int), iter([.17]), group_marker=0,
                                   c1_radius_policy='per_atom')
    assert np.isfinite(got.direction).all()
    assert np.isclose(np.linalg.norm(got.direction), 1.)
