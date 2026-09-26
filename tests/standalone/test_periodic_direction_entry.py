"""Periodic full-direction public contracts; not scientific effect tests."""
import copy
from dataclasses import replace
import numpy as np
import pytest
from ase.build import bulk
from ase.constraints import FixAtoms
from pamssw.standalone.paper_reference import SSWConfig, run_ssw, load_ssw_checkpoint
from pamssw.standalone.constrained_reference import ConstrainedSSWConfig, run_constrained_ssw
from pamssw.standalone.recovered_direction import RecoveredDirectionSettings

class Flat:
    requests=0
    def evaluate(self, atoms):
        self.requests+=1
        return 0.,np.zeros_like(atoms.positions)

def settings():
    return RecoveredDirectionSettings(50,.5,.5,1,1,.2,.02,'euclidean',8,geometry='periodic_local')

def atoms():
    return bulk('Cu',cubic=True).repeat((2,2,2))

def config():
    return SSWConfig(width=.1,rotation_bias=1.,max_gaussians=2,temperature_K=0.,
        fmax=.03,relax_steps=2,fd_step=.001,rotation_hvp=2,rotation_tol=.02,
        direction_sampling='global',cluster_frame='translation_only')

def test_periodic_initial_checkpoint_and_corrupted_geometry(tmp_path):
    a=atoms(); path=tmp_path/'state.pkl'
    run_ssw(a,Flat(),steps=0,config=config(),rng=np.random.default_rng(41),
            recovered_direction=settings(),checkpoint_path=path)
    cp=load_ssw_checkpoint(path)
    run_ssw(a,Flat(),steps=0,config=config(),rng=np.random.default_rng(99),checkpoint=cp)
    object.__setattr__(cp.recovered_direction_state,"geometry_cell",None)
    surface=Flat();rng=np.random.default_rng(99);state=copy.deepcopy(rng.bit_generator.state)
    with pytest.raises(ValueError,match='geometry|cell|periodic'):
        run_ssw(a,surface,steps=0,config=config(),rng=rng,checkpoint=cp)
    assert surface.requests==0 and rng.bit_generator.state==state

def test_periodic_fixed_atoms_checkpoint_preserves_mask_and_identity():
    a=atoms();a.pbc=(True,True,False);a.set_constraint(FixAtoms(indices=[0,1]))
    cfg=ConstrainedSSWConfig(width=.1,rotation_bias=1.,max_gaussians=2,
        fmax=.03,gradient_tol=.1,relax_steps=2,fd_step=.001)
    result=run_constrained_ssw(a,Flat(),steps=0,config=cfg,rng=np.random.default_rng(41),
                              recovered_direction=settings())
    assert result.status=='completed'
    cp=result.checkpoint
    assert cp.recovered_direction_state.geometry_pbc==(True,True,False)
    assert not cp.recovered_direction_state.active_mask[:2].any()
    result2=run_constrained_ssw(a,Flat(),steps=0,config=cfg,rng=np.random.default_rng(99),
                              recovered_direction=settings(),checkpoint=cp)
    np.testing.assert_array_equal(result2.current.atoms.positions,a.positions)

def test_periodic_option_rejects_nonperiodic_before_pes():
    a=atoms();a.pbc=False;surface=Flat()
    with pytest.raises(ValueError,match='periodic'):
        run_ssw(a,surface,steps=0,config=config(),rng=np.random.default_rng(41),
                recovered_direction=settings())
    assert surface.requests==0


def test_fixed_reference_pair_still_reports_consumed_selection_draw():
    from ase import Atoms
    from pamssw.standalone.periodic_direction import select_periodic_direction_group
    a=Atoms('Cu2',positions=[[0,0,0],[4,0,0]],cell=[12,12,12],pbc=True)
    rng=iter([.4,.7])
    selected=select_periodic_direction_group(a.positions,a,rng,active_mask=np.array([True,False]))
    assert selected.pair==(0,1)
    assert not selected.group_mask.any()
    assert selected.draw_count==1
    assert next(rng)==.7


def test_constrained_periodic_stage_reports_actual_route():
    a=atoms();a.set_constraint(FixAtoms(indices=[0]))
    cfg=ConstrainedSSWConfig(width=.1,rotation_bias=1.,max_gaussians=1,
        fmax=.03,gradient_tol=10.,relax_steps=2,fd_step=.001,
        rotation_exit_policy='force_or_budget')
    result=run_constrained_ssw(a,Flat(),steps=1,config=cfg,rng=np.random.default_rng(41),
                              recovered_direction=settings())
    stage=result.records[1]['climb'][0]
    assert stage['recovered_direction']['route'] in {'pair','pair_fallback','pair_group','torsion','forbidden','none'}


def test_active_score_references_remain_active_in_large_cell_limit():
    from ase import Atoms
    from pamssw.standalone.periodic_direction import select_periodic_direction_group
    from pamssw.standalone.recovered_direction import _select_active_direction_group
    a=Atoms('Cu4',positions=[[8,0,0],[1,1,0],[3,3,0],[7,1,0]],cell=[40,40,40],pbc=True)
    reference=a.positions.copy();reference[1:,2]-=[.5,1.,1.5]
    active=np.array([False,True,True,True])
    periodic=select_periodic_direction_group(reference,a,iter([.4]),active_mask=active)
    isolated=a.copy();isolated.pbc=False
    expected=_select_active_direction_group(reference,isolated,iter([.4]),active)
    assert periodic.pair==expected.pair
    np.testing.assert_array_equal(periodic.group_mask,expected.group_mask)
