import numpy as np
import pytest
from types import SimpleNamespace
from ase.build import bulk

from pamssw.standalone.direction import SoftModeResult
from pamssw.standalone.generalized_numerics import generalized_central_ritz
from pamssw.standalone.vc_reference import VCSSWConfig, run_vc_ssw
from pamssw.standalone.rc_vc_reference import RCVCSSWConfig, run_rc_vc_ssw
from pamssw.standalone.vc_geometry import VCEvaluation


H = np.array([[2.0, .3], [.3, 4.0]])


def quadratic(q):
    q = np.asarray(q)
    return float(q @ H @ q), 2.0 * H @ q


def test_public_central_ritz_uses_explicit_force_budget():
    calls = []

    def evaluate(q):
        calls.append(q.copy())
        return quadratic(q)

    result = generalized_central_ritz(
        np.array([1.0, 0.2]), np.array([1.0, 1.0]),
        rotation_bias=0.0, fd_step=1e-4, max_force_calls=8,
        tol=1e-8, evaluate=evaluate,
    )
    assert result.converged
    assert result.force_calls == len(calls) <= 6
    assert result.residual_norm < 1e-8


def test_custom_solver_requires_budget_before_any_vc_pes_call():
    class NoPES:
        requests = 0

        def evaluate(self, atoms):
            self.requests += 1
            raise AssertionError("PES called before solver validation")

    config = VCSSWConfig(strain_length=3.6, width=.2, rotation_bias=.5)
    with pytest.raises(ValueError, match="rotation_force_calls"):
        run_vc_ssw(
            bulk("Cu", "fcc", a=3.6, cubic=True), NoPES(), steps=1,
            config=config, rng=np.random.default_rng(3),
            direction_solver=lambda **kwargs: None,
        )


def test_custom_solver_requires_budget_before_any_rcvc_pes_call():
    class NoPES:
        requests = 0

        def evaluate(self, atoms):
            self.requests += 1
            raise AssertionError("PES called before solver validation")

    config = RCVCSSWConfig(
        rotation_length=2.0, torsion_length=2.0, strain_length=3.6,
        width=.2, rotation_bias=.5,
    )
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    trees = [dict(bodies=[tuple(range(4))], parents=(-1,), joints=(None,))]
    with pytest.raises(ValueError, match="rotation_force_calls"):
        run_rc_vc_ssw(
            atoms, NoPES(), trees=trees, anchor=0, steps=1,
            config=config, rng=np.random.default_rng(3),
            direction_solver=lambda **kwargs: None,
        )


def test_custom_rcvc_solver_receives_force_budget(monkeypatch):
    import pamssw.standalone.rc_vc_reference as rcvc

    class Flat:
        requests = 0

        def evaluate(self, atoms):
            self.requests += 1
            return 0.0, np.zeros_like(atoms.positions), np.zeros((3, 3))

    def full(atoms, surface, **kwargs):
        surface.evaluate(atoms)
        ev = VCEvaluation(0.0, np.zeros(3 * len(atoms) + 6), atoms.copy(),
                           0.0, np.zeros_like(atoms.positions),
                           np.zeros((3, 3)), atoms.get_volume())
        return SimpleNamespace(converged=True, evaluation=ev,
                               certificate={'certified': True}, requests=1)

    seen = []

    def solver(q0, anchor, **kwargs):
        seen.append(kwargs)
        direction = np.asarray(anchor, dtype=float)
        direction /= np.linalg.norm(direction)
        return SoftModeResult(direction, -1.0, 0.0, 1, 1, True, 0.0)

    monkeypatch.setattr(rcvc, 'cell_quench', full)
    atoms = bulk('Cu', 'fcc', a=3.6, cubic=True)
    trees = [dict(bodies=[tuple(range(4))], parents=(-1,), joints=(None,))]
    config = RCVCSSWConfig(rotation_length=2., torsion_length=2.,
        strain_length=3.6, width=.2, rotation_bias=.5,
        max_gaussians=1, relax_steps=2, temperature_K=0.)
    run_rc_vc_ssw(atoms, Flat(), trees=trees, anchor=0, steps=1,
                  config=config, rng=np.random.default_rng(3),
                  direction_solver=solver, rotation_force_calls=9)
    assert [item['max_force_calls'] for item in seen] == [9]


def test_custom_vc_solver_receives_force_budget_and_default_dimer_is_unchanged(monkeypatch):
    import pamssw.standalone.vc_reference as vc

    class Flat:
        requests = 0

        def evaluate(self, atoms):
            self.requests += 1
            return 0.0, np.zeros_like(atoms.positions), np.zeros((3, 3))

    seen = []

    def solver(q0, anchor, **kwargs):
        seen.append(kwargs)
        kwargs["evaluate"](np.asarray(q0))
        direction = np.asarray(anchor, dtype=float)
        direction /= np.linalg.norm(direction)
        return SoftModeResult(direction, -1.0, 0.0, 1, 1, True, 0.0)

    config = VCSSWConfig(
        strain_length=3.6, width=.2, rotation_bias=.5,
        max_gaussians=1, relax_steps=2,
    )
    surface = Flat()
    result = run_vc_ssw(
        bulk("Cu", "fcc", a=3.6, cubic=True), surface, steps=1,
        config=config, rng=np.random.default_rng(7), direction_solver=solver,
        rotation_force_calls=7,
    )
    assert result.records[1]["climb"]
    assert seen
    assert [item["max_force_calls"] for item in seen] == [7]
    assert "max_hvp" not in seen[0]

    called = []
    original = vc.generalized_dimer

    def dimer(*args, **kwargs):
        called.append(kwargs["max_hvp"])
        return original(*args, **kwargs)

    monkeypatch.setattr(vc, "generalized_dimer", dimer)
    default_config = VCSSWConfig(
        strain_length=3.6, width=.2, rotation_bias=.5,
        max_gaussians=1, relax_steps=2, rotation_hvp=5,
    )
    run_vc_ssw(
        bulk("Cu", "fcc", a=3.6, cubic=True), Flat(), steps=1,
        config=default_config, rng=np.random.default_rng(7),
    )
    assert called == [5]
