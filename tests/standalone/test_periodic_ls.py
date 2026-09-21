"""Real Cu/EMT integration, explicit synthetic LS strengths; no efficiency claim."""
import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from pamssw.standalone.periodic_softening import FrozenPeriodicBondSoftening
from pamssw.standalone.softening import LSResponseState
from pamssw.standalone.paper_reference import SSWConfig,LSSettings,run_ls_ssw
from pamssw.standalone.surface import ASESurface

TABLE=dict(bond_energies={(29,29):1.},bond_lengths={(29,29):2.9})

def test_periodic_bonds_count_images_and_repeat_extensively():
    a=bulk('Cu','fcc',a=3.6,cubic=True)
    soft=FrozenPeriodicBondSoftening.from_atoms(a,**TABLE)
    assert len(soft.pairs)==24  # four atoms x twelve neighbors / two
    a.positions[0]+=[.03,-.02,.01]
    e,f=soft.evaluate(a)
    # Freeze at the same reference in a doubled cell, then repeat displacement.
    doubled=bulk('Cu','fcc',a=3.6,cubic=True).repeat((2,1,1))
    repeated=FrozenPeriodicBondSoftening.from_atoms(doubled,**TABLE)
    doubled.positions[[0,4]]+=[.03,-.02,.01]
    ee,ff=repeated.evaluate(doubled)
    assert ee==pytest.approx(2*e,abs=1e-12)
    np.testing.assert_allclose(ff,np.tile(f,(2,1)),atol=1e-12)
    for k in range(a.positions.size):
        plus=a.copy();minus=a.copy();plus.positions.flat[k]+=1e-6;minus.positions.flat[k]-=1e-6
        fd=(soft.evaluate(plus)[0]-soft.evaluate(minus)[0])/2e-6
        assert fd==pytest.approx(-f.flat[k],abs=1e-8)
    response=LSResponseState(.001)
    rebuilt=response.update(soft,bulk('Cu','fcc',a=3.6,cubic=True),energy_before=0.,energy_after=.002,**TABLE)
    assert isinstance(rebuilt,FrozenPeriodicBondSoftening)
    assert len(rebuilt.pairs)==24


def test_one_atom_cell_self_images_and_zero_atomic_force():
    a=bulk('Cu','fcc',a=3.6)
    soft=FrozenPeriodicBondSoftening.from_atoms(a,**TABLE)
    assert len(soft.pairs)==6
    assert all(i==j==0 for i,j in soft.pairs)
    np.testing.assert_allclose(soft.evaluate(a)[1],0.,atol=0.)


def test_partial_pbc_has_no_images_on_nonperiodic_axis_and_force_fd():
    a = bulk('Cu', 'fcc', a=3.6, cubic=True)
    a.pbc = [True, True, False]
    soft = FrozenPeriodicBondSoftening.from_atoms(a, **TABLE)
    assert all(shift[2] == 0 for shift in soft.image_shifts)
    for k in range(a.positions.size):
        plus=a.copy();minus=a.copy();plus.positions.flat[k]+=1e-6;minus.positions.flat[k]-=1e-6
        fd=(soft.evaluate(plus)[0]-soft.evaluate(minus)[0])/2e-6
        assert fd==pytest.approx(-soft.evaluate(a)[1].flat[k],abs=1e-8)


def test_one_periodic_axis_retains_pair_image_and_fd():
    a = bulk('Cu', 'fcc', a=3.6, cubic=True).repeat((1, 1, 2))
    a.pbc = [False, False, True]
    soft = FrozenPeriodicBondSoftening.from_atoms(a, **TABLE)
    assert any(shift[2] != 0 for shift in soft.image_shifts)
    q = a.copy(); q.positions[0, 2] += .01
    e, f = soft.evaluate(q)
    plus=q.copy();minus=q.copy();plus.positions[0,2]+=1e-6;minus.positions[0,2]-=1e-6
    assert (soft.evaluate(plus)[0]-soft.evaluate(minus)[0])/2e-6 == pytest.approx(-f[0,2],abs=1e-8)


def test_periodic_ls_complete_lifecycle_true_landings_and_cost():
    a=bulk('Cu','fcc',a=3.6,cubic=True);original=a.copy()
    c=SSWConfig(width=.1,rotation_bias=100,max_gaussians=1,temperature_K=300,
        fmax=.01,relax_steps=150,fd_step=1e-4,rotation_hvp=40,rotation_tol=.02,
        direction_sampling='global',rotation_solver='dimer',cluster_frame='translation_only',
        quench_optimizer='safe-lbfgs-total')
    s=ASESurface(EMT())
    r=run_ls_ssw(a,s,steps=2,config=c,rng=np.random.default_rng(17),
        ls=LSSettings(**TABLE,target_per_atom=.001))
    assert len(r.records)==2 and len(r.minima)>=2
    assert all(x.energy_response is not None for x in r.records)
    assert r.evaluation_requests==s.requests==r.initial.evaluation_requests+sum(x.evaluation_requests for x in r.records)
    for m in r.minima:
        np.testing.assert_array_equal(m.atoms.cell.array,original.cell.array)
        e,f=ASESurface(EMT()).evaluate(m.atoms)
        assert e==pytest.approx(m.energy,abs=1e-10)
        assert np.linalg.norm(f,axis=1).max()<=c.fmax
    np.testing.assert_array_equal(a.positions,original.positions)


def test_ls_prequench_failure_is_not_retried_and_keeps_cost(monkeypatch):
    import pamssw.standalone.paper_reference as module
    calls=[]
    def failed(atoms,surface,**kwargs):
        calls.append(1);surface.evaluate(atoms)
        raise ValueError('injected soft preparation failure')
    monkeypatch.setattr(module,'prepare_ls_step',failed)
    a=bulk('Cu','fcc',a=3.6,cubic=True);s=ASESurface(EMT())
    c=SSWConfig(width=.1,rotation_bias=100,max_gaussians=1,temperature_K=300,
        fmax=.01,relax_steps=150,fd_step=1e-4,rotation_hvp=40,rotation_tol=.02,
        direction_sampling='global',rotation_solver='dimer',cluster_frame='translation_only',quench_optimizer='safe-lbfgs-total')
    r=run_ls_ssw(a,s,steps=3,config=c,rng=np.random.default_rng(17),ls=LSSettings(**TABLE,target_per_atom=.001))
    assert len(calls)==len(r.records)==1 and r.status=='ls_prequench_failed'
    assert len(r.minima)==1
    assert r.evaluation_requests==s.requests==r.initial.evaluation_requests+r.records[0].evaluation_requests
