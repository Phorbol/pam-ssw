import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from pamssw.standalone.atomic_climb import atomic_climb
from pamssw.standalone.paper_reference import SSWConfig
from pamssw.standalone.surface import ASESurface

def cfg(**kw):
    d=dict(width=.2,rotation_bias=.5,max_gaussians=1,temperature_K=0.,fmax=.01,relax_steps=150,fd_step=1e-4,rotation_hvp=41,rotation_tol=.02,direction_sampling='global',cluster_frame='translation_only',quench_optimizer='safe-lbfgs-total')
    d.update(kw); return SSWConfig(**d)

def test_bias_fmax_validation_and_default():
    assert cfg().bias_fmax is None
    assert cfg(bias_fmax=.1).bias_fmax == .1
    with pytest.raises(ValueError): cfg(bias_fmax=0.)
    with pytest.raises(ValueError): cfg(bias_fmax=float('nan'))

def test_real_cu_bias_fmax_is_stage_metadata():
    a=bulk('Cu','fcc',a=3.65,cubic=True); a.positions[0]+=[.03,-.02,.01]
    c=cfg(bias_fmax=.1)
    r=atomic_climb(a,ASESurface(EMT()),reference_energy=-1e9,config=c,rng=np.random.default_rng(7))
    assert r.status=='gaussian_limit' and len(r.climb)==1
    assert r.climb[0]['bias_fmax']==.1
    assert np.isfinite(r.atoms.positions).all()


def test_real_cu_looser_bias_force_does_not_loosen_final_force():
    from pamssw.standalone.paper_reference import run_ssw
    atoms=bulk('Cu','fcc',a=3.65,cubic=True)
    c=cfg(bias_fmax=.1)
    result=run_ssw(atoms,ASESurface(EMT()),steps=1,config=c,rng=np.random.default_rng(7))
    event=result.records[0]
    assert event.landing is not None and event.landing.converged
    stage=event.climb[0]
    assert stage['max_force']<=c.bias_fmax and stage['max_force']>c.fmax
    assert stage['optimizer_telemetry'].converged
    fresh=event.landing.atoms.copy();fresh.calc=EMT()
    assert np.linalg.norm(fresh.get_forces(),axis=1).max()<=c.fmax
    assert fresh.get_potential_energy()==pytest.approx(event.landing.energy,abs=1e-10)
