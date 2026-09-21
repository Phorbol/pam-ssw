"""Adapter contracts, not scientific efficiency claims."""
from dataclasses import replace
import numpy as np
import pytest
from ase import Atoms
from pamssw.standalone import SSWConfig, run_ssw
from pamssw.standalone.pam_gaussian import PAMCurvatureGaussian
from pamssw.standalone.native_height_policy import ConservativeNativeHeightPolicy


def config():
    return SSWConfig(width=.1, rotation_bias=10., max_gaussians=1,
        temperature_K=150., fmax=.01, bias_fmax=.1, relax_steps=0,
        fd_step=1e-4, rotation_hvp=12, rotation_tol=.02,
        direction_sampling='global')


class Surface:
    requests = 0
    def __init__(self):
        self.positions = []
    def evaluate(self, atoms):
        self.requests += 1
        self.positions.append(atoms.positions.copy())
        return float((atoms.positions**2).sum()/2), -atoms.positions.copy()


def test_conflicting_policies_rejected_before_pes():
    surface = Surface()
    with pytest.raises(ValueError):
        run_ssw(Atoms('H'), surface, steps=1, config=config(),
            rng=np.random.default_rng(11), gaussian_policy=PAMCurvatureGaussian(),
            height_policy=ConservativeNativeHeightPolicy(1., 2., 0, 10., 1.1, 1.2))
    assert surface.requests == 0


@pytest.mark.parametrize('staged', [False, True])
def test_selected_width_sets_displacement_and_force_objective(staged):
    cfg = config()
    if staged:
        cfg = replace(cfg, rotation_bias=None, pre_rotation_hvp=3)
    surface = Surface()
    policy = PAMCurvatureGaussian(target_uphill_energy=.08, min_width=.1, max_width=1.)
    result = run_ssw(Atoms('H'), surface, steps=1, config=cfg,
        rng=np.random.default_rng(11), gaussian_policy=policy)
    event = result.records[0].climb[0]
    assert event['width'] == pytest.approx(.4)
    assert event['weight'] == pytest.approx(.4**2*1.05)
    direction = np.asarray(event['direction'])
    assert any(np.allclose(x, .4*direction, atol=1e-10) for x in surface.positions)
    assert not any(np.allclose(x, .1*direction, atol=1e-10) for x in surface.positions)
    # At x=sigma*d, Gaussian force is W/sigma*exp(-1/2)*d.
    expected = abs(-.4 + event['weight']/.4*np.exp(-.5))
    assert event['max_force'] == pytest.approx(expected)


def test_adaptive_history_uses_existing_gaussian_hessian():
    cfg = replace(config(), max_gaussians=2, relax_steps=100)
    result = run_ssw(Atoms('H'), Surface(), steps=1, config=cfg,
        rng=np.random.default_rng(11),
        gaussian_policy=PAMCurvatureGaussian(target_uphill_energy=.08, min_width=.1, max_width=1.))
    events = result.records[0].climb
    assert len(events) == 2
    assert 'gaussian_policy' in events[1]
    assert events[1]['gaussian_policy']['k_inner'] != pytest.approx(events[1]['gaussian_policy']['k_true'])
    assert result.records[0].landing is not None
