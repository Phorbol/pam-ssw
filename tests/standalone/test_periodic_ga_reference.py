"""Periodic GA stage wiring; identity tolerances here are not phase validation."""
import numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT
from pamssw.standalone.vc_geometry import ASEStressSurface
from pamssw.standalone.vc_reference import VCSSWConfig
from pamssw.standalone.periodic_ga_reference import PeriodicGAConfig,run_periodic_ga


def settings():
    return PeriodicGAConfig(quick_steps=0,generations=1,generation_steps=0,fine_steps=1,
        regions=3,fine_regions=1,min_ga=1,max_batches=1,max_cut_attempts=30,
        max_pair_attempts=50,partition_max_draws=100,slots_per_parent=1,cuts_per_slot=1)


def kwargs(seeds):
    return dict(config=settings(),walker_config=VCSSWConfig(strain_length=3.6,width=.2,
        rotation_bias=.5,max_gaussians=1,rotation_hvp=41,relax_steps=150),
        rng=np.random.default_rng(7),descriptor_basis=seeds,bond_lengths={(29,29):2.6},
        neighbor_range=1.1,projection_weights=[.3,.2,.2,.1,.1,.1],bond_limits={(29,29):.5},
        # Deliberately exact representation matcher to isolate controller wiring.
        # It is NOT proposed as a crystal identity algorithm or unique-minimum metric.
        matcher=lambda a,b:np.array_equal(a.positions,b.positions) and np.array_equal(a.cell.array,b.cell.array))


class Bounded(ASEStressSurface):
    exhausted=False
    def __init__(self,cap=1000):super().__init__(EMT());self.cap=cap
    def evaluate(self,a):
        if self.requests>=self.cap:self.exhausted=True;raise RuntimeError('test EFS cap')
        return super().evaluate(a)


def test_real_cu_full_type1_three_phase_pipeline_and_costs():
    seeds=[bulk('Cu','fcc',a=x,cubic=True) for x in (3.55,3.65,3.8)]
    snapshots=[a.copy() for a in seeds];surface=Bounded()
    r=run_periodic_ga(seeds,surface,**kwargs(seeds))
    assert r['status']=='completed' and r['requests_reconciled']
    assert r['requests']==surface.requests
    phases=[x['phase'] for x in r['walks']]
    assert phases.count('quick')==3 and 'offspring_quick' in phases and 'fine' in phases
    assert r['proposals'][0]['result'].candidates
    assert all(o['certificate']['certified'] for o in r['observations'])
    assert all(x['result'].records[0]['stage']=='initial' for x in r['walks'])
    assert r['requests']==sum(x['result'].requests for x in r['walks'])
    for a,old in zip(seeds,snapshots):
        np.testing.assert_array_equal(a.positions,old.positions)
        np.testing.assert_array_equal(a.cell.array,old.cell.array)


def test_duplicate_archive_and_source_restriction_no_fallback():
    seeds=[bulk('Cu','fcc',a=3.6,cubic=True) for _ in range(3)];k=kwargs(seeds)
    # Caller declares these identical; observations must remain, routing dedup is separate.
    k['matcher']=lambda a,b:True
    s=Bounded();r=run_periodic_ga(seeds,s,**k)
    assert len(r['archive'])==1 and len(r['observations'])>=3
    assert r['proposals'][0]['status']=='no_proposal'
    assert 'requires' in r['proposals'][0]['reason']
    assert not any(w['phase']=='offspring_quick' for w in r['walks'])
    assert r['requests_reconciled']


def test_budget_failure_preserves_paid_initial_stage():
    seeds=[bulk('Cu','fcc',a=x,cubic=True) for x in (3.55,3.65,3.8)];s=Bounded(cap=1)
    r=run_periodic_ga(seeds,s,**kwargs(seeds))
    assert r['status']=='censored' and r['requests']==1 and r['requests_reconciled']
    assert len(r['walks'])==1 and not r['archive']


def test_molecular_proposal_hook_retains_group_lineage_and_full_walk(monkeypatch):
    from types import SimpleNamespace
    from pamssw.standalone.ga_operators import GeneticCandidate
    from pamssw.standalone.molecular_periodic_ga_reference import run_molecular_periodic_ga
    import pamssw.standalone.molecular_periodic_ga_reference as module
    seen=[]
    def proposal(parents,energies,groups,rng,**kwargs):
        seen.append(tuple(tuple(g) for g in groups))
        child=GeneticCandidate(parents[0].copy(),tuple(tuple(g) for g in groups),'controlled_molecular_proposal',(0,0),{})
        return SimpleNamespace(candidates=(child,),status='completed')
    monkeypatch.setattr(module,'propose_type2',proposal)
    seeds=[bulk('Cu','fcc',a=x,cubic=True) for x in (3.55,3.65,3.8)];s=Bounded()
    r=run_molecular_periodic_ga(seeds,s,molecules=((0,1),(2,3)),**kwargs(seeds))
    assert seen==[((0,1),(2,3))] and r['requests_reconciled']
    offspring=[w for w in r['walks'] if w['phase']=='offspring_quick']
    assert len(offspring)==1 and len(offspring[0]['parent_ids'])==2
    assert any(w['phase']=='fine' for w in r['walks'])


def test_molecular_topology_rejected_before_any_oracle_request():
    import pytest
    from pamssw.standalone.molecular_periodic_ga_reference import run_molecular_periodic_ga
    seeds=[bulk('Cu','fcc',a=x,cubic=True) for x in (3.55,3.65,3.8)];s=Bounded()
    with pytest.raises(ValueError,match='partition|disjoint|contiguous'):
        run_molecular_periodic_ga(seeds,s,molecules=((0,1),(1,2,3)),**kwargs(seeds))
    assert s.requests==0
