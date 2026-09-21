import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms, Hookean

from pamssw.standalone.constrained_reference import (
    ConstrainedSSWConfig, run_constrained_ssw,
)
from pamssw.standalone.pam_gaussian import PAMCurvatureGaussian
from pamssw.standalone.gaussian import ProjectedGaussian
from types import SimpleNamespace


class AnchoredSurface:
    def __init__(self): self.requests = 0
    @property
    def exhausted(self): return False
    def evaluate(self, atoms):
        self.requests += 1
        x = atoms.positions.copy()
        target = np.zeros_like(x)
        target[1, 0] = 1.2
        d = x - target
        return .5 * float(np.square(d).sum()), -d


def atoms():
    a = Atoms('Cu2', positions=[[0., 0., 0.], [1.2, 0., 0.]], cell=[8., 8., 8.], pbc=True)
    a.set_constraint([FixAtoms(indices=[0]), Hookean(0, 1, k=2., rt=.7)])
    return a


def config():
    return ConstrainedSSWConfig(width=.2, rotation_bias=10., temperature_K=0.,
                                max_gaussians=1, relax_steps=80, rotation_hvp=2)


def test_explicit_pam_policy_uses_active_cartesian_shape_and_records_parameters(tmp_path):
    policy = PAMCurvatureGaussian(mode='height_width')
    result = run_constrained_ssw(atoms(), AnchoredSurface(), steps=1,
        config=config(), rng=np.random.default_rng(12), gaussian_policy=policy,
        checkpoint_path=tmp_path / 'pam.pkl')
    event = result.records[1]
    assert event['climb'][0]['gaussian_policy']['parameters'] == policy.parameters()
    assert result.checkpoint.gaussian_policy.parameters() == policy.parameters()


def test_pam_policy_checkpoint_mismatch_is_rejected_before_pes(tmp_path):
    path = tmp_path / 'pam.pkl'
    policy = PAMCurvatureGaussian(mode='height_width')
    run_constrained_ssw(atoms(), AnchoredSurface(), steps=1, config=config(),
        rng=np.random.default_rng(12), gaussian_policy=policy, checkpoint_path=path)
    from pamssw.standalone.constrained_reference import load_constrained_checkpoint
    cp = load_constrained_checkpoint(path)
    surface = AnchoredSurface()
    changed = PAMCurvatureGaussian(mode='height_only')
    try:
        run_constrained_ssw(atoms(), surface, steps=0, config=config(),
            rng=np.random.default_rng(12), gaussian_policy=changed, checkpoint=cp)
    except ValueError:
        pass
    else:
        raise AssertionError('policy mismatch was accepted')
    assert surface.requests == 0


def test_pam_staged_restricted_policy_lifts_pre_anchor_to_full_active_space():
    a = Atoms('Cu4', positions=[[0., 0., 0.], [1.2, 0., 0.],
                                 [0., 1.2, 0.], [0., 0., 1.2]],
              cell=[8., 8., 8.], pbc=True)
    a.set_constraint(FixAtoms(indices=[0]))
    cfg = ConstrainedSSWConfig(width=.2, rotation_bias=None, temperature_K=0.,
        max_gaussians=1, relax_steps=40, rotation_hvp=6,
        rotation_solver='ritz', pre_rotation_hvp=2)
    result = run_constrained_ssw(a, AnchoredSurface(), steps=1, config=cfg,
        rng=np.random.default_rng(22), direction_fixed_indices=[1],
        gaussian_policy=PAMCurvatureGaussian(mode='height_width'))
    assert result.records[1]['climb'][0]['gaussian_policy']['width'] > 0


def test_pam_history_uses_each_gaussian_width_in_curvature():
    policy = PAMCurvatureGaussian(mode='height_width')
    mode = SimpleNamespace(direction=np.array([[1., 0., 0.]]), curvature=-.2)
    anchor = np.array([[1., 0., 0.]])
    center = np.array([[0., 0., 0.]])
    first = policy.choose(mode=mode, anchor=anchor, center=center, terms=(),
                          base_width=.2, rotation_bias=1.)
    term = ProjectedGaussian(center.copy(), anchor.copy(), first['width'], first['weight'])
    second = policy.choose(mode=mode, anchor=anchor, center=center, terms=(term,),
                           base_width=.2, rotation_bias=1.)
    assert second['k_inner'] != first['k_inner']
    assert second['width'] > 0 and second['weight'] >= 0


import pytest
from dataclasses import replace
from pamssw.standalone import LSSettings, NativeLSSettings
from research.ga_ssw.run_hookean_multicase import serial


@pytest.mark.parametrize('ls', [None,
    LSSettings({(29,29):3.}, {(29,29):1.2}, target_per_atom=.2),
    NativeLSSettings({(29,29):3.}, {(29,29):1.2}, target_mev_per_atom=200.)])
def test_pam_ls_resume_continues_and_matches_two_steps(tmp_path, ls):
    policy=PAMCurvatureGaussian()
    cfg=replace(config(),relax_steps=200)
    rng=np.random.default_rng(31)
    full=run_constrained_ssw(atoms(),AnchoredSurface(),steps=2,config=cfg,
        rng=rng,ls=ls,gaussian_policy=policy,checkpoint_path=tmp_path/'full')
    first_surface=AnchoredSurface()
    first=run_constrained_ssw(atoms(),first_surface,steps=1,config=cfg,
        rng=np.random.default_rng(31),ls=ls,gaussian_policy=policy,checkpoint_path=tmp_path/'first')
    assert first.checkpoint.next_index==1
    resumed_surface=AnchoredSurface();resumed_rng=np.random.default_rng(999)
    resumed=run_constrained_ssw(atoms(),resumed_surface,steps=1,config=cfg,
        rng=resumed_rng,ls=ls,gaussian_policy=policy,checkpoint=first.checkpoint)
    assert resumed.checkpoint.next_index==2
    assert full.requests==first_surface.requests+resumed_surface.requests==resumed.requests
    assert serial(full.records)==serial(resumed.records)
    assert serial(full.checkpoint.ls_state)==serial(resumed.checkpoint.ls_state)
    assert rng.bit_generator.state==resumed_rng.bit_generator.state
    np.testing.assert_array_equal(full.current.atoms.positions,resumed.current.atoms.positions)


def test_default_none_retains_every_evaluation_and_rng():
    class Ledger(AnchoredSurface):
        def __init__(self): super().__init__();self.calls=[]
        def evaluate(self,a):
            e,f=super().evaluate(a);self.calls.append((a.positions.copy(),e,f.copy()));return e,f
    a,b=Ledger(),Ledger();ra,rb=np.random.default_rng(27),np.random.default_rng(27)
    x=run_constrained_ssw(atoms(),a,steps=2,config=config(),rng=ra)
    y=run_constrained_ssw(atoms(),b,steps=2,config=config(),rng=rb,gaussian_policy=None)
    assert x.requests==y.requests and serial(x.records)==serial(y.records)
    assert len(a.calls)==len(b.calls)==x.requests
    for p,q in zip(a.calls,b.calls):
        for u,v in zip(p,q):np.testing.assert_array_equal(u,v)
    assert ra.bit_generator.state==rb.bit_generator.state


def test_driver_retains_mixed_width_history_and_conservative_gradient(monkeypatch):
    import pamssw.standalone.rc_reference as shared
    original=shared.safe_lbfgs
    observed=[]
    def checked(q,evaluate,**kwargs):
        trial=q+np.arange(len(q))*.013
        e,g=evaluate(trial)
        fd=[]
        for i in range(len(q)):
            d=np.eye(len(q))[i]*1e-5
            fd.append((evaluate(trial+d)[0]-evaluate(trial-d)[0])/2e-5)
        np.testing.assert_allclose(g,fd,atol=1e-7,rtol=1e-7)
        observed.append(1)
        return original(q,evaluate,**kwargs)
    class DifferentWidths(PAMCurvatureGaussian):
        def choose(self,**kwargs):
            width=.2 if not kwargs['terms'] else .35
            return dict(width=width,weight=1.)
    monkeypatch.setattr(shared,'safe_lbfgs',checked)
    cfg=replace(config(),max_gaussians=2,relax_steps=200,fmax=1e-8)
    a=atoms();a.set_constraint(FixAtoms(indices=[0]))
    result=run_constrained_ssw(a,AnchoredSurface(),steps=1,config=cfg,
        rng=np.random.default_rng(12),gaussian_policy=DifferentWidths())
    widths=[t['width'] for t in result.records[1]['frozen_gaussians']]
    assert widths==[.2,.35] and len(observed)==2
