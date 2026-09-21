import numpy as np
import pytest
from types import SimpleNamespace
from ase.build import bulk
from ase.calculators.emt import EMT
from pamssw.standalone.pam_gaussian import PAMCurvatureGaussian
from pamssw.standalone.gaussian import ProjectedGaussian

def test_height_and_weight_formula_and_history_hessian():
    center=np.zeros((2,3)); d=np.zeros((2,3)); d[0,0]=1.; d[1,0]=1.; d=d/np.linalg.norm(d)
    old=ProjectedGaussian(center.copy(),d,.4,2.)
    shifted=np.array([[.13,0,0],[0,0,0.]])
    old=ProjectedGaussian(center.copy(),d,.4,2.)
    mode=SimpleNamespace(curvature=.2,direction=d); policy=PAMCurvatureGaussian(mode='height_width')
    got=policy.choose(mode=mode,anchor=d,center=center,terms=(old,),base_width=.6,rotation_bias=.5)
    ktrue=.2+.5
    khist=-2./.4**2
    expected_inner=ktrue+khist
    assert got['k_true']==pytest.approx(ktrue)
    assert got['k_inner']==pytest.approx(expected_inner)
    assert got['width']==pytest.approx(np.sqrt(1.2/.7))
    assert got['weight']==pytest.approx(got['width']**2*max(expected_inner+.05,0.))
    p=np.dot((shifted-old.center).reshape(-1),old.direction.reshape(-1)); z=p/old.sigma
    expected_h=old.weight*np.exp(-.5*z*z)*(p*p/old.sigma**4-1/old.sigma**2)*np.dot(old.direction.reshape(-1),d.reshape(-1))**2
    # Same analytic Hessian as GaussianBiasTerm, at nonzero projection.
    from pamssw.bias import GaussianBiasTerm
    term=GaussianBiasTerm(old.center.reshape(-1),old.direction.reshape(-1),old.sigma,old.weight)
    assert expected_h == pytest.approx(term.hvp_contribution(d.reshape(-1),shifted.reshape(-1)).dot(d.reshape(-1)))

def test_zero_weight_is_valid_and_invalid_policy_rejected():
    assert PAMCurvatureGaussian().min_weight == 0.
    with pytest.raises(ValueError): PAMCurvatureGaussian(mode='bad')
    with pytest.raises(ValueError): PAMCurvatureGaussian(min_width=2.,max_width=1.)

def test_real_cu_two_gaussian_checkpoint_replay():
    from pamssw.standalone.atomic_climb import atomic_climb, resume_atomic_climb
    from pamssw.standalone.surface import ASESurface
    class Stateless(ASESurface):
        def evaluate(self, atoms): self.calculator=EMT(); return super().evaluate(atoms)
    from pamssw.standalone.paper_reference import SSWConfig
    a=bulk('Cu','fcc',a=3.65,cubic=True); a.positions[0]+=[.03,-.02,.01]
    c=SSWConfig(width=.2,rotation_bias=.5,max_gaussians=2,temperature_K=0.,fmax=.01,relax_steps=150,fd_step=1e-4,rotation_hvp=41,rotation_tol=.02,direction_sampling='global',cluster_frame='translation_only',quench_optimizer='safe-lbfgs-total',bias_stage_steps=1)
    p=PAMCurvatureGaussian(); full=atomic_climb(a,Stateless(EMT()),reference_energy=-1e9,config=c,rng=np.random.default_rng(7),gaussian_policy=p)
    assert full.status=='gaussian_limit' and len(full.climb)==2
    assert all('gaussian_policy' in e and np.isfinite(e['width']) and np.isfinite(e['weight']) for e in full.climb)
    paused=atomic_climb(a,Stateless(EMT()),reference_energy=-1e9,config=c,rng=np.random.default_rng(7),gaussian_policy=p,max_completed_gaussians=1)
    resumed=resume_atomic_climb(paused.checkpoint,Stateless(EMT()),c)
    assert len(resumed.climb)==len(full.climb)==2
    np.testing.assert_allclose(resumed.climb[1]['center'],full.climb[1]['center'],atol=1e-12)
    assert resumed.climb[1]['gaussian_policy']==full.climb[1]['gaussian_policy']

@pytest.mark.parametrize('mode', ['height_only', 'height_width'])
def test_real_vacancy_cu_strict_combined_and_fresh_certificate(mode):
    from pamssw.standalone.block_ssw import BlockSSWConfig, run_block_ssw
    from pamssw.standalone.paper_reference import SSWConfig
    from pamssw.standalone.vc_geometry import ASEStressSurface
    atoms=bulk('Cu','fcc',a=3.65,cubic=True).repeat((2,1,1));del atoms[0]
    c=SSWConfig(width=.2,rotation_bias=.5,max_gaussians=1,temperature_K=300.,
        fmax=.01,relax_steps=150,fd_step=1e-4,rotation_hvp=41,rotation_tol=.02,
        direction_sampling='global',cluster_frame='translation_only',quench_optimizer='safe-lbfgs-total')
    b=BlockSSWConfig(c,3.6,cell_cycles=1,cell_step_fraction=.03,partial_atom_steps=1,
        atomic_gaussian_policy=PAMCurvatureGaussian(mode=mode))
    surface=ASEStressSurface(EMT())
    result=run_block_ssw(atoms,surface,steps=2,config=b,rng=np.random.default_rng(7))
    event=result.records[2]
    assert event['status']=='valid_landing'
    stage=event['atomic'].climb[0]
    assert stage['gaussian_policy']['parameters']['mode']==mode
    assert stage['optimizer_telemetry'].converged
    if mode=='height_only':assert stage['width']==c.width
    e,f,s=ASEStressSurface(EMT()).evaluate(event['landing'].evaluation.atoms)
    assert np.linalg.norm(f,axis=1).max()<=c.fmax
    assert np.abs(s+b.pressure*np.eye(3)).max()<=b.stress_tol
    assert result.requests==surface.requests==sum(r['requests'] for r in result.records)


def test_height_only_preserves_width_and_nonfinite_inputs_fail():
    policy=PAMCurvatureGaussian(mode='height_only')
    d=np.ones((1,3))/np.sqrt(3.)
    kw=dict(mode=SimpleNamespace(direction=d,curvature=1.),anchor=d,
            center=np.zeros_like(d),terms=(),base_width=2.)
    assert policy.choose(**kw)['width']==2.
    with pytest.raises(ValueError):PAMCurvatureGaussian(min_weight=float('nan'))
    with pytest.raises(ValueError):policy.choose(**dict(kw,anchor=np.zeros_like(d)))
    with pytest.raises(ValueError):policy.choose(**dict(kw,base_width=float('nan')))
