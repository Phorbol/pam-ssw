"""Contract checks only; scientific evidence lives in the multicase campaigns."""
from dataclasses import replace
import numpy as np
import pytest
from ase import Atoms
from pamssw.standalone.paper_reference import SSWConfig, run_ssw
from pamssw.standalone.staged_direction import two_stage_dimer_direction


def cfg(**changes):
    return replace(SSWConfig(width=.1, rotation_bias=None, pre_rotation_hvp=3,
        max_gaussians=1, temperature_K=150., fmax=.01, relax_steps=0,
        fd_step=1e-4, rotation_hvp=12, rotation_tol=.02,
        direction_sampling='global'), **changes)


@pytest.mark.parametrize('changes', [dict(rotation_bias=100.),
    dict(pre_rotation_hvp=None), dict(pre_rotation_hvp=True),
    dict(rotation_hvp=3), dict(pre_rotation_hvp=10)])
def test_invalid_staged_configuration_rejected_before_pes(changes):
    with pytest.raises(ValueError):
        cfg(**changes)


@pytest.mark.parametrize('solver', ['dimer', 'ritz'])
def test_shared_budget_counts_both_centers(solver):
    atoms = Atoms('H', positions=[[0., 0., 0.]])
    diagonal = np.array([[1., 3., 7.]])
    calls = []
    def evaluate(candidate):
        calls.append(candidate.positions.copy())
        return float((diagonal*candidate.positions**2).sum()/2), -diagonal*candidate.positions
    result = two_stage_dimer_direction(atoms, np.ones((1, 3))/np.sqrt(3),
        fd_step=1e-4, max_hvp=12, pre_rotation_hvp=3, tol=.02,
        evaluate=evaluate, main_solver=solver)
    assert len(calls) == result.force_calls <= 13
    assert result.force_calls == result.hvp_calls + 2
    assert result.bias_curvature == max(result.pre.curvature, 0.)
    assert np.array_equal(atoms.positions, np.zeros((1, 3)))


def test_height_receives_curvature_without_rank_one_bias(monkeypatch):
    from pamssw.standalone.native_height_policy import ConservativeNativeHeightPolicy
    from types import SimpleNamespace
    import pamssw.standalone.staged_direction as staged
    n = np.array([[1., 0., 0.]])
    pre = SimpleNamespace(direction=n, curvature=4., residual_norm=.1,
                          hvp_calls=3, force_calls=4, converged=False)
    main = SimpleNamespace(direction=n, curvature=-1., residual_norm=.001,
                           hvp_calls=2, force_calls=3, converged=True)
    mode = SimpleNamespace(**vars(main), pre=pre, main=main, bias_curvature=4.)
    monkeypatch.setattr(staged, 'two_stage_dimer_direction', lambda *a, **kw: mode)
    observed = []
    original = ConservativeNativeHeightPolicy.prepare
    def prepare(self, *args, **kwargs):
        observed.append(kwargs['curvature'])
        return original(self, *args, **kwargs)
    monkeypatch.setattr(ConservativeNativeHeightPolicy, 'prepare', prepare)
    class Surface:
        requests = 0
        def evaluate(self, atoms):
            self.requests += 1
            return float((atoms.positions**2).sum()/2), -atoms.positions.copy()
    policy = ConservativeNativeHeightPolicy(1., 2., 0, 100., 1.1, 1.2)
    run_ssw(Atoms('H', positions=[[0., 0., 0.]]), Surface(), steps=1,
            config=cfg(), rng=np.random.default_rng(11), height_policy=policy)
    assert observed == [3.]


def test_atomic_substage_rejects_staged_config_before_pes():
    from pamssw.standalone.atomic_climb import atomic_climb
    class Forbidden:
        def evaluate(self, atoms):
            raise AssertionError('must reject before PES')
    with pytest.raises(NotImplementedError, match='run_ssw'):
        atomic_climb(Atoms('Cu', cell=[3., 3., 3.], pbc=True), Forbidden(),
            reference_energy=0., config=cfg(), rng=np.random.default_rng(11))
