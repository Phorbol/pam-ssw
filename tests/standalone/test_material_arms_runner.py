"""Bounded real EMT lifecycle wiring; no cross-material efficacy claim."""
import numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT
from pamssw.standalone.paper_reference import SSWConfig
from pamssw.standalone.block_ssw import BlockSSWConfig
from pamssw.standalone.vc_reference import VCSSWConfig
from pamssw.standalone.vc_geometry import ASEStressSurface
from research.ga_ssw.compare_material_arms import run_material_arm, BudgetExhausted


def configs():
    a=SSWConfig(width=.2,rotation_bias=.5,max_gaussians=1,temperature_K=0.,fmax=.01,
        relax_steps=150,fd_step=1e-4,rotation_hvp=41,rotation_tol=.02,
        direction_sampling='global',rotation_solver='dimer',cluster_frame='translation_only',quench_optimizer='safe-lbfgs-total')
    b=BlockSSWConfig(a,3.6,cell_cycles=1,cell_step_fraction=.03,partial_atom_steps=1)
    v=VCSSWConfig(strain_length=3.6,width=.2,rotation_bias=.5,temperature_K=0.,
                  max_gaussians=1,relax_steps=150,rotation_hvp=41)
    return a,b,v


class Bounded(ASEStressSurface):
    exhausted=False
    def __init__(self,cap=300):super().__init__(EMT());self.cap=cap
    def evaluate(self,a):
        if self.requests>=self.cap:
            self.exhausted=True;raise BudgetExhausted('EMT wiring cap')
        return super().evaluate(a)


def test_all_four_real_cu_lifecycles_schedule_common_start_and_costs():
    a,b,v=configs();atoms=bulk('Cu','fcc',a=3.65,cubic=True)
    original=atoms.copy();runs={}
    for arm in ('fixed','pqc','block','joint'):
        surface=Bounded()
        r=run_material_arm(atoms,surface,arm=arm,atomic_config=a,block_config=b,joint_config=v,steps=2,seed=7)
        runs[arm]=r
        assert r['status']=='completed'
        assert r['requests_reconciled'] and r['requests']==surface.requests
        assert r['requests']==sum(x['after']-x['before'] for x in r['ledger'])
        assert len(r['records'])==3
        for landing in r['landings']:
            assert landing['certificate']['certified']
        np.testing.assert_array_equal(atoms.positions,original.positions)
        np.testing.assert_array_equal(atoms.cell.array,original.cell.array)
    for r in runs.values():
        np.testing.assert_array_equal(r['common_start'].positions,runs['fixed']['common_start'].positions)
        np.testing.assert_array_equal(r['common_start'].cell.array,runs['fixed']['common_start'].cell.array)
        assert r['initial_requests']==runs['fixed']['initial_requests']
    block=runs['block']['records']
    assert not block[1]['atomic_scheduled'] and block[1]['atomic'] is None
    assert block[2]['atomic_scheduled'] and block[2]['atomic'] is not None
    assert runs['block']['valid_proposals']==1
    # The combined branch is exercised but its dimer does not converge at this
    # declared budget. Keep this failure and its cost; do not tune it away.
    assert block[2]['status']=='atomic_rotation_failed'
    assert block[2]['requests']>0 and not block[2]['accepted']
    for record in runs['pqc']['records'][1:]:
        assert record['atomic'] is not None and record['landing'] is not None
        assert record['landing'].converged
        assert record['requests']==record['atomic'].requests+record['landing'].requests
    for landing in runs['fixed']['landings']:
        np.testing.assert_array_equal(landing['atoms'].cell.array,runs['fixed']['common_start'].cell.array)
    # With T=0, higher-energy candidates remain observable even if MC rejects.
    for r in runs.values():
        assert len(r['landings'])==1+r['valid_proposals']
        for record in r['records'][1:]:
            if 'delta' in record and record['delta']>0:assert not record['accepted']
    assert sum(r['requests'] for r in runs.values())<=1200


def test_censor_keeps_initial_and_failure_cost_for_all_arms():
    a,b,v=configs();atoms=bulk('Cu','fcc',a=3.65,cubic=True)
    for arm in ('fixed','pqc','block','joint'):
        surface=Bounded(cap=1)
        r=run_material_arm(atoms,surface,arm=arm,atomic_config=a,block_config=b,joint_config=v,steps=2,seed=7)
        assert r['status']=='censored' and r['requests']==1
        assert not r['landings'] and r['valid_proposals']==0
        assert r['requests_reconciled']
        assert any('error' in e for e in r['ledger'])
