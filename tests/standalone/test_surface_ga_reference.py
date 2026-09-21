"""Controller contracts; physical EMT experiment is a separate bounded artifact."""
from types import SimpleNamespace
import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from pamssw.standalone.surface_ga_reference import SurfaceGAConfig,run_surface_ga
from pamssw.standalone.constrained_reference import ConstrainedSSWConfig
from pamssw.standalone.surface_ga import SurfaceCandidate,SurfaceProposal


def config():return SurfaceGAConfig(0,1,0,0,2,1,4,1,10,10,100,100,100)


def test_lineage_rejected_observations_active_certificate_and_costs():
    seeds=[]
    for x in (3.,1.,2.):
        seeds.append(Atoms('Cu3',positions=[[0,0,0],[x,0,1],[x,1,1]],cell=[10,10,10],pbc=[1,1,0],constraint=FixAtoms(indices=[0])))
    surface=SimpleNamespace(requests=0,exhausted=False)
    def walker(a,surface,**kwargs):
        surface.requests+=2;e=float(a.positions[1,0]);a=a.copy()
        minimum=SimpleNamespace(atoms=a,energy=e,active_fmax=.001,full_raw_fmax=9.,converged=True,certificate={'certified':True})
        # Includes a second valid observation despite a hypothetical MC rejection.
        return SimpleNamespace(status='completed',minima=[minimum,minimum],records=[{'accepted':False}],requests=2)
    def proposals(parents,energies,support,ads,rng,**kwargs):
        # A per-atom parent pair must remap through selected archive row order.
        a=parents[0].copy();a.positions[1,0]=4.
        c=SurfaceCandidate(a,support,ads,(0,2,1),(0,1,2),'test_cross',{})
        return SurfaceProposal((c,),'target_reached',(),17)
    r=run_surface_ga(seeds,surface,config=config(),walker_config=ConstrainedSSWConfig(.2,10.),rng=np.random.default_rng(8),
        substrate_indices=[0],adsorbate_indices=[1,2],routing=lambda a:[a.positions[1,0],0,0],
        matcher=lambda a,b:np.array_equal(a.positions,b.positions),bond_limits={('Cu','Cu'):.5},atomic_radii={29:1.},site_fractional=(.5,.5),
        walker=walker,proposal_factory=proposals)
    selected=r['proposals'][0]['archive_indices'];child=next(w for w in r['walks'] if w['phase']=='offspring_quick')
    assert child['parent_ids']==(selected[0],selected[2],selected[1])
    assert r['requests']==10 and r['requests_reconciled']
    assert r['auxiliary_evaluations']==17 and len(r['observations'])==10
    assert all(o['certificate']['certified'] and o['certificate']['full_raw_fmax']==9. for o in r['observations'])
    assert r['best']['energy']==1.


def test_missing_physical_contract_rejected_before_ef():
    import pytest
    a=Atoms('Cu3',positions=[[0,0,0],[2,0,1],[2,1,1]],cell=[10,10,10],pbc=[1,1,0])
    s=SimpleNamespace(requests=0,exhausted=False)
    with pytest.raises(ValueError,match='missing explicit bond limit'):
        run_surface_ga([a],s,config=config(),walker_config=ConstrainedSSWConfig(.2,10.),rng=np.random.default_rng(1),
            substrate_indices=[0],adsorbate_indices=[1,2],routing=lambda a:[0,0,0],matcher=lambda a,b:False,
            bond_limits={},atomic_radii={29:1.},site_fractional=(.5,.5))
    assert s.requests==0


def test_type4_generation_rejects_small_min_ga_before_walker():
    import pytest
    seeds = [Atoms('Cu3', positions=[[0, 0, 0], [2 + x, 0, 1], [0, 2, 1]],
                   cell=[10, 10, 10], pbc=[True, True, False],
                   constraint=FixAtoms(indices=[0])) for x in (0., .1, .2)]
    cfg = SurfaceGAConfig(1, 1, 1, 1, 2, 1, 1, 1, 10, 10, 10, 10, 10)
    calls = []
    def walker(*args, **kwargs):
        calls.append(1)
        raise AssertionError('walker must not run for invalid native min_ga')
    with pytest.raises(ValueError, match='min_ga.*4'):
        run_surface_ga(seeds, SimpleNamespace(requests=0, exhausted=False), config=cfg,
                       walker_config=ConstrainedSSWConfig(.2, 10.),
                       rng=np.random.default_rng(2), substrate_indices=[0],
                       adsorbate_indices=[1, 2], routing=lambda a: [a.positions[1, 0], 0., 0.],
                       matcher=lambda a, b: False, bond_limits={('Cu', 'Cu'): .1},
                       atomic_radii={29: 1.}, site_fractional=(.5, .5), walker=walker)
    assert calls == []


def test_type4_custom_factory_keeps_min_ga_one_compatibility():
    seeds = [Atoms('Cu3', positions=[[0, 0, 0], [2 + x, 0, 1], [0, 2, 1]],
                   cell=[10, 10, 10], pbc=[True, True, False],
                   constraint=FixAtoms(indices=[0])) for x in (0., .1, .2)]
    cfg = SurfaceGAConfig(1, 1, 1, 1, 2, 1, 1, 1, 10, 10, 10, 10, 10)
    calls = []
    def walker(a, surface, **kwargs):
        calls.append(1); surface.requests += 1
        minimum = SimpleNamespace(atoms=a.copy(), energy=float(a.positions[1, 0]),
                                  active_fmax=.001, full_raw_fmax=.001,
                                  converged=True, certificate={})
        return SimpleNamespace(status='completed', minima=[minimum], records=[])
    def factory(*args, **kwargs):
        return SurfaceProposal((), 'target_reached', (), 0)
    result = run_surface_ga(seeds, SimpleNamespace(requests=0, exhausted=False), config=cfg,
                            walker_config=ConstrainedSSWConfig(.2, 10.),
                            rng=np.random.default_rng(3), substrate_indices=[0],
                            adsorbate_indices=[1, 2], routing=lambda a: [a.positions[1, 0], 0., 0.],
                            matcher=lambda a, b: False, bond_limits={('Cu', 'Cu'): .1},
                            atomic_radii={29: 1.}, site_fractional=(.5, .5),
                            walker=walker, proposal_factory=factory)
    assert calls == [1, 1, 1, 1]
    assert result['proposals'][0]['status'] == 'target_reached'


def test_type4_generation_zero_keeps_min_ga_one_compatibility():
    a = Atoms('Cu3', positions=[[0, 0, 0], [2, 0, 1], [0, 2, 1]],
              cell=[10, 10, 10], pbc=[True, True, False],
              constraint=FixAtoms(indices=[0]))
    cfg = SurfaceGAConfig(1, 0, 1, 1, 2, 1, 1, 1, 10, 10, 10, 10, 10)
    calls = []
    def walker(x, surface, **kwargs):
        calls.append(1); surface.requests += 1
        return SimpleNamespace(status='completed', minima=[], records=[])
    result = run_surface_ga([a], SimpleNamespace(requests=0, exhausted=False), config=cfg,
                            walker_config=ConstrainedSSWConfig(.2, 10.),
                            rng=np.random.default_rng(4), substrate_indices=[0],
                            adsorbate_indices=[1, 2], routing=lambda x: [0., 0., 0.],
                            matcher=lambda x, y: False, bond_limits={}, atomic_radii={29: 1.},
                            site_fractional=(.5, .5), walker=walker)
    assert calls == [1]
    assert result['status'] == 'completed'
