"""Compare independent arithmetic against frozen original-instruction oracles.

C60/C4H6 geometries verify initialization only, not LS search effectiveness.
No native binary or physical energy calculator is executed by these tests.
"""
import json
from pathlib import Path
import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.constraints import FixAtoms
from pamssw.standalone.native_ls import (
    initialize_native_ls,freeze_native_ls,effective_amplitude,
    response_mev_per_atom,normal_update_due,update_native_table,
    HC_BOND_ENERGIES,HC_BOND_LENGTHS,TIO_BOND_ENERGIES,TIO_BOND_LENGTHS)

EVIDENCE=Path(__file__).resolve().parents[2]/'research/ga_ssw/evidence'


def test_tio2_native_initialization_matches_instruction_fixture():
    evidence=json.loads((EVIDENCE/'native-ls-tio-init-20260917/result.json').read_text())
    row=evidence['input']
    atoms=Atoms(numbers=row['numbers'],positions=row['positions_A'],cell=row['cell_A'],pbc=row['pbc'])
    result=initialize_native_ls(atoms,bond_energies=TIO_BOND_ENERGIES,
                                bond_lengths=TIO_BOND_LENGTHS,
                                scale=evidence['formula']['scale'])
    assert result.bond_count==evidence['native']['bond_count']==2
    np.testing.assert_allclose(
        [[result.table[tuple(sorted((a,b)))] for b in evidence['scope']['elements'].values()]
         for a in evidence['scope']['elements'].values()],
        evidence['native']['energy_matrix_B'],atol=0.,rtol=0.)
    np.testing.assert_allclose(
        [[result.lengths[tuple(sorted((a,b)))] for b in evidence['scope']['elements'].values()]
         for a in evidence['scope']['elements'].values()],
        evidence['native']['length_matrix_L_A'],atol=0.,rtol=0.)


def test_frozen_native_arithmetic_oracles():
    evidence=json.loads((EVIDENCE/'native-ls-response/result.json').read_text())
    for row in evidence['rows']:
        if row['block']=='effective_amplitude':
            actual=effective_amplitude(row['B'],amp_c=row['amp_c'],filter_i=row['filter_i'],filter_j=row['filter_j'])
            assert actual==row['actual']
        elif row['block']=='matrix_update':
            result=update_native_table({(6,6):row['B']},natoms=row['N'],old_bond_count=row['nb_old'],new_bond_count=row['nb_new'],response_mev_per_atom=row['response'],target_mev_per_atom=row['target'],eta=row['step'])
            assert result.q==pytest.approx(row['Q_supplied'],abs=1e-14)
            assert result.table[(6,6)]==pytest.approx(row['actual'],abs=1e-15)
        elif row['block']=='saved_response':
            assert response_mev_per_atom(row['energy_before'],row['energy_after'],row['N'])==row['actual']


def test_real_c60_c4h6_geometry_native_initialization_and_frozen_force():
    evidence=json.loads((EVIDENCE/'native-ls-initialization/result.json').read_text())
    for row in evidence['rows']:
        atoms=Atoms(numbers=row['numbers'],positions=row['positions'],cell=[50.]*3,pbc=True)
        original=atoms.copy()
        result=initialize_native_ls(atoms,bond_energies=HC_BOND_ENERGIES,bond_lengths=HC_BOND_LENGTHS,scale=row['bond_ener_scale'])
        assert result.bond_count==row['bond_count']
        for i,a in enumerate(row['elements']):
            for j,b in enumerate(row['elements']):
                assert result.table[tuple(sorted((a,b)))]==pytest.approx(row['energy_matrix'][i][j],abs=1e-16)
                assert result.lengths[tuple(sorted((a,b)))]==row['length_matrix'][i][j]
        energy,forces=result.potential.evaluate(atoms)
        assert energy==pytest.approx(sum(result.potential.strengths))
        np.testing.assert_allclose(forces.sum(axis=0),0.,atol=1e-14)
        # Genuine pair force/energy consistency on the supplied real geometry.
        shifted=atoms.copy();h=1e-6;shifted.positions[0,0]+=h
        plus=result.potential.evaluate(shifted)[0];shifted.positions[0,0]-=2*h
        minus=result.potential.evaluate(shifted)[0]
        assert -(plus-minus)/(2*h)==pytest.approx(forces[0,0],abs=1e-8)
        filters=np.ones(len(atoms));filters[0]=0.
        filtered=freeze_native_ls(atoms,result.table,result.lengths,atom_filter=filters)
        assert len(filtered.pairs)==result.bond_count
        for pair,strength in zip(filtered.pairs,filtered.strengths):
            if 0 in pair:assert strength==0.
        np.testing.assert_array_equal(atoms.positions,original.positions)


def test_normal_schedule_and_unsupported_branches_are_explicit():
    assert not normal_update_due(0)
    assert normal_update_due(1) and normal_update_due(100)
    assert not normal_update_due(101) and normal_update_due(110)
    assert not normal_update_due(1,presteps=0)
    table={(6,6):.1,(1,6):.2}
    result=update_native_table(table,natoms=10,old_bond_count=9,new_bond_count=8,response_mev_per_atom=1000.)
    assert result.q>1 and table[(6,6)]==.1
    with pytest.raises(NotImplementedError,match='save/restore'):
        update_native_table(table,natoms=10,old_bond_count=9,new_bond_count=8,response_mev_per_atom=20.,branch='periodic_restore')
    a=Atoms('CC',positions=[[0,0,0],[1.4,0,0]])
    a.set_constraint(FixAtoms(indices=[0]))
    with pytest.raises(NotImplementedError,match='frozen-atom'):
        initialize_native_ls(a,bond_energies=HC_BOND_ENERGIES,bond_lengths=HC_BOND_LENGTHS)
    with pytest.raises(ValueError,match='zero bond'):
        initialize_native_ls(Atoms('CC',positions=[[0,0,0],[10.,0,0]]),bond_energies=HC_BOND_ENERGIES,bond_lengths=HC_BOND_LENGTHS)


def test_cycle_state_static_transition_fixtures():
    from pamssw.standalone.native_ls import NativeLSCycleState
    fixtures=json.loads((Path(__file__).parent/'fixtures/native_ls_cycle_static.json').read_text())
    for adaptive,key in [(True,'adaptive'),(False,'nonadaptive')]:
        state=NativeLSCycleState({(6,6):.1},old_bond_count=8,natoms=10,
            cycle=10,ratio=.4,frequency=2,presteps=2,lselfadapt=adaptive)
        for row in fixtures[key]:
            result=state.advance(row['step'],measured_response_mev_per_atom=row['response'],new_bond_count=row['new_count'])
            assert list(result['actions'])==row['actions']
            assert result['table'][(6,6)]==pytest.approx(row['B'],abs=1e-15)
            assert result['old_bond_count']==row['old_count']
            if 'stored_response' in row:assert result['response']==row['stored_response']
        if adaptive:
            assert state.note_response_integer==19 and state.note_bond_count==8
            assert state.note_table=={(6,6):.1}


def test_default_cycle_inactive_and_normal_oracle_through_controller():
    from pamssw.standalone.native_ls import NativeLSCycleState
    evidence=json.loads((EVIDENCE/'native-ls-response/result.json').read_text())
    for row in evidence['rows']:
        if row['block']!='matrix_update':continue
        state=NativeLSCycleState({(6,6):row['B']},old_bond_count=row['nb_old'],natoms=row['N'])
        assert state.nsoftstep==110
        result=state.advance(110,measured_response_mev_per_atom=row['response'],new_bond_count=row['nb_new'])
        assert result['phase'] is None and result['actions']==('normal_update',)
        assert result['table'][(6,6)]==pytest.approx(row['actual'],abs=1e-15)
    # Fortran nearest-integer behavior differs from Python round at half ties.
    assert NativeLSCycleState({(6,6):.1},8,10,cycle=10,ratio=.25).nsoftstep==3


def test_cycle_failure_is_transactional_and_negative_response_note_truncates():
    from copy import deepcopy
    from pamssw.standalone.native_ls import NativeLSCycleState
    state=NativeLSCycleState({(6,6):.1},8,10,cycle=10,ratio=.4,frequency=2,presteps=2)
    old=deepcopy(vars(state))
    with pytest.raises(ValueError,match='saved note'):
        state.advance(12,measured_response_mev_per_atom=10.,new_bond_count=9)
    assert vars(state)==old
    state.advance(6,measured_response_mev_per_atom=-19.9,new_bond_count=9)
    assert state.note_response_integer==-19
    old=deepcopy(vars(state))
    with pytest.raises(ValueError,match='finite'):
        state.advance(7,measured_response_mev_per_atom=np.nan,new_bond_count=9)
    assert vars(state)==old


def test_periodic_image_geometry_counts_and_extensive_scale():
    table={(29,29):3.}; lengths={(29,29):3.}
    systems=(bulk('Cu','fcc',a=3.6), bulk('Cu','fcc',a=3.6,cubic=True),
             bulk('Cu','fcc',a=3.6,cubic=True).repeat((2,1,1)))
    results=[initialize_native_ls(a,bond_energies=table,bond_lengths=lengths,
                                  bond_geometry='periodic-images') for a in systems]
    assert [r.bond_count for r in results]==[6,24,48]
    assert all(type(r.potential).__name__=='FrozenPeriodicBondSoftening' for r in results)
    per_atom=[sum(r.potential.strengths)/len(a) for r,a in zip(results,systems)]
    assert per_atom[0]==pytest.approx(per_atom[1])==pytest.approx(per_atom[2])
    assert np.array_equal(results[0].potential.pairs[0],(0,0))
    np.testing.assert_allclose(results[0].potential.evaluate(systems[0])[1],0.,atol=1e-14)


def test_periodic_image_force_fd_and_frozen_crossing_are_smooth():
    atoms=bulk('Cu','fcc',a=3.6,cubic=True)
    result=initialize_native_ls(atoms,bond_energies={(29,29):3.},bond_lengths={(29,29):3.},
                                bond_geometry='periodic-images')
    atoms.positions[0] += [.07,-.03,.02]
    force=result.potential.evaluate(atoms)[1]
    h=1e-6; plus=atoms.copy(); plus.positions[0,0]+=h
    minus=atoms.copy(); minus.positions[0,0]-=h
    assert force[0,0]==pytest.approx(-(result.potential.evaluate(plus)[0]-result.potential.evaluate(minus)[0])/(2*h),abs=1e-8)
    # The frozen image labels do not switch at a cell boundary.
    left=atoms.copy(); right=atoms.copy(); left.positions[0,0]=atoms.cell[0,0]-1e-7; right.positions[0,0]=atoms.cell[0,0]+1e-7
    assert np.isfinite(result.potential.evaluate(left)[0]) and np.isfinite(result.potential.evaluate(right)[0])
    assert abs(result.potential.evaluate(left)[0]-result.potential.evaluate(right)[0])<1e-3


def test_periodic_image_perfect_cu4_cancellation_and_repeat_equivariance():
    base=bulk('Cu','fcc',a=3.6,cubic=True)
    pristine=initialize_native_ls(base,bond_energies={(29,29):3.},bond_lengths={(29,29):3.},
                                   bond_geometry='periodic-images')
    _,f0=pristine.potential.evaluate(base)
    np.testing.assert_allclose(f0,0.,atol=1e-12)
    base.positions[0] += [.07,-.03,.02]
    one=initialize_native_ls(base,bond_energies={(29,29):3.},bond_lengths={(29,29):3.},
                             bond_geometry='periodic-images')
    repeated=base.repeat((2,1,1))
    two=initialize_native_ls(repeated,bond_energies={(29,29):3.},bond_lengths={(29,29):3.},
                             bond_geometry='periodic-images')
    f_one=one.potential.evaluate(base)[1]; f_two=two.potential.evaluate(repeated)[1]
    np.testing.assert_allclose(f_two[:4],f_one,atol=1e-10)
    assert sum(one.potential.strengths)/len(base)==pytest.approx(sum(two.potential.strengths)/len(repeated))


def test_periodic_images_nonperiodic_and_invalid_mode_contracts():
    atoms=Atoms('CC',positions=[[0,0,0],[1.4,0,0]])
    kwargs=dict(bond_energies=HC_BOND_ENERGIES,bond_lengths=HC_BOND_LENGTHS)
    old=initialize_native_ls(atoms,**kwargs)
    explicit=initialize_native_ls(atoms,**kwargs,bond_geometry='periodic-images')
    assert old.bond_count==explicit.bond_count and old.potential==explicit.potential
    with pytest.raises(ValueError,match='bond_geometry'):
        initialize_native_ls(atoms,**kwargs,bond_geometry='bad-mode')
    with pytest.raises(ValueError,match='bond_geometry'):
        freeze_native_ls(atoms,{(6,6):1.},{(6,6):2.},bond_geometry='bad-mode')
