"""Driver contract checks; stub optimizers do not establish search efficacy."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms

from pamssw.standalone.paper_reference import SSWConfig, run_ssw
from pamssw.standalone.surface import QuenchResult


def config(**kwargs):
    return SSWConfig(width=.1,rotation_bias=1.,max_gaussians=1,
        temperature_K=0.,fmax=.03,relax_steps=2,fd_step=.001,
        rotation_hvp=2,rotation_tol=.02,direction_sampling='global',
        rotation_solver='dimer',**kwargs)


class Flat:
    requests = 0

    def evaluate(self, atoms):
        self.requests += 1
        return 0., np.zeros_like(atoms.positions)


def execute(monkeypatch, policy, *, reason='budget_exhausted', bad=False,
            landing_converged=True, solver_error=False):
    import pamssw.standalone.paper_reference as driver
    import pamssw.standalone.dimer as dimer
    quench_calls = []

    def quench(atoms, surface, **kwargs):
        quench_calls.append(bool(kwargs.get('terms')))
        valid = len(quench_calls) != 3 or landing_converged
        return QuenchResult(atoms.copy(),0.,0. if valid else .1,valid,0,0,'stub')

    def solve(atoms, anchor, **kwargs):
        if solver_error:
            raise ValueError('invalid endpoint force')
        n = anchor.copy()
        if bad:
            n[0,0] = np.nan
        return SimpleNamespace(direction=n,curvature=-1.,residual_norm=.4,
            force_calls=3,hvp_calls=2,converged=False,
            projected_symmetry_error=0.,stop_reason=reason)

    monkeypatch.setattr(driver,'quench',quench)
    monkeypatch.setattr(dimer,'paper_dimer_direction',solve)
    result = run_ssw(Atoms('H',positions=[[0.,0.,0.]]),Flat(),steps=1,
        config=config(rotation_exit_policy=policy),rng=np.random.default_rng(11))
    return result, quench_calls


def test_policy_is_explicit_and_default_remains_strict():
    assert config().rotation_exit_policy == 'force'
    with pytest.raises(ValueError,match='rotation_exit_policy'):
        config(rotation_exit_policy='anything')


def test_budget_exit_preserves_nonconvergence_and_reaches_true_quench(monkeypatch):
    strict,calls = execute(monkeypatch,'force')
    assert strict.records[0].status == 'rotation_failed'
    assert calls == [False]
    result,calls = execute(monkeypatch,'force_or_budget')
    assert calls == [False,True,False]
    assert len(result.minima) == 2
    event = result.records[0].climb[0]
    assert event['rotation_stop_reason'] == 'budget_exhausted'
    assert event['rotation_converged'] is False
    assert event['rotation_residual'] == .4


@pytest.mark.parametrize('reason',['subspace_exhausted','unspecified'])
def test_nonbudget_failure_is_not_relabelled_or_released(monkeypatch,reason):
    result,calls = execute(monkeypatch,'force_or_budget',reason=reason)
    assert result.records[0].status == 'rotation_failed'
    assert calls == [False]


@pytest.mark.parametrize('kwargs',[{'bad':True},{'solver_error':True}])
def test_invalid_rotation_does_not_create_gaussian(monkeypatch,kwargs):
    result,calls = execute(monkeypatch,'force_or_budget',**kwargs)
    assert result.status == 'evaluation_failed'
    assert calls == [False]
    assert len(result.minima) == 1


def test_unconverged_landing_still_cannot_enter_minima(monkeypatch):
    result,calls = execute(monkeypatch,'force_or_budget',landing_converged=False)
    assert calls == [False,True,False]
    assert result.records[0].status == 'true_quench_failed'
    assert len(result.minima) == 1


def test_old_config_instance_uses_default_exit_policy():
    import pickle
    cfg=config()
    object.__getattribute__(cfg,'__dict__').pop('rotation_exit_policy')
    cfg=pickle.loads(pickle.dumps(cfg))
    assert cfg.rotation_exit_policy == 'force'
    assert replace(cfg).rotation_exit_policy == 'force'
    assert cfg == config()


def test_other_climb_entrypoint_cannot_silently_ignore_policy():
    from pamssw.standalone.atomic_climb import atomic_climb
    surface=Flat()
    cfg=replace(config(rotation_exit_policy='force_or_budget'),
        cluster_frame='translation_only',quench_optimizer='safe-lbfgs-total')
    atoms=Atoms('H2',positions=[[0,0,0],[1,0,0]],cell=[5,5,5],pbc=True)
    with pytest.raises(NotImplementedError,match='rotation_exit_policy'):
        atomic_climb(atoms,surface,reference_energy=0.,config=cfg,
                     rng=np.random.default_rng(0))
    assert surface.requests == 0
