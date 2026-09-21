import numpy as np
import pytest
from types import SimpleNamespace
from ase.build import molecule, bulk
from ase.calculators.emt import EMT

from pamssw.standalone.generalized_numerics import safe_lbfgs
from pamssw.standalone.cell_relax import relax_cell_coordinates
from pamssw.standalone.vc_reference import VCSSWConfig
from pamssw.standalone.rc_reference import RCSSWConfig
from pamssw.standalone.rc_forest_reference import RCForestSSWConfig
from pamssw.standalone.rc_vc_reference import RCVCSSWConfig
from pamssw.standalone.paper_reference import SSWConfig
from pamssw.standalone.block_ssw import BlockSSWConfig
from pamssw.standalone.surface import ASESurface, QuenchResult
from pamssw.standalone.vc_geometry import ASEStressSurface, VCEvaluation


def _quadratic(q):
    return float(np.dot(q, q)), 2.0 * q


def test_generalized_safe_lbfgs_accepts_explicit_history_and_rejects_invalid_before_evaluation():
    calls = 0

    def counted(q):
        nonlocal calls
        calls += 1
        return _quadratic(q)

    result = safe_lbfgs(np.ones(3), counted, gradient_norm=np.linalg.norm,
                        step_norm=np.linalg.norm, gtol=1e-8, max_step=.2,
                        maxiter=100, lbfgs_memory=400)
    assert result.requests == calls
    assert result.converged

    for value in (0, -1, True, 1.5, "10"):
        calls = 0
        with pytest.raises(ValueError):
            safe_lbfgs(np.ones(3), counted, gradient_norm=np.linalg.norm,
                       step_norm=np.linalg.norm, gtol=1e-8, max_step=.2,
                       maxiter=1, lbfgs_memory=value)
        assert calls == 0


@pytest.mark.parametrize("factory", [
    lambda: VCSSWConfig(strain_length=3.6, width=.2, rotation_bias=.5, lbfgs_memory=400),
    lambda: RCSSWConfig(torsion_length=2., width=.2, rotation_bias=.5, lbfgs_memory=400),
    lambda: RCForestSSWConfig(rotation_length=2., torsion_length=2., width=.2,
                              rotation_bias=.5, lbfgs_memory=400),
    lambda: RCVCSSWConfig(rotation_length=2., torsion_length=2., strain_length=4.,
                          width=.2, rotation_bias=.5, lbfgs_memory=400),
])
def test_generalized_ssw_configs_carry_explicit_history(factory):
    assert factory().lbfgs_memory == 400


def test_generalized_ssw_configs_reject_memory_for_non_safe_backend():
    # The generalized drivers have only Safe-total; this regression documents
    # that the public field remains positive-integer validated at config time.
    with pytest.raises(ValueError):
        VCSSWConfig(strain_length=3.6, width=.2, rotation_bias=.5, lbfgs_memory=0)


def test_rc_and_forest_public_memory_reaches_initial_biased_and_final_quenches(monkeypatch):
    import pamssw.standalone.rc_reference as rc
    import pamssw.standalone.rc_forest_reference as forest
    seen = []
    seen_safe = []

    def spy_quench(atoms, surface, **kwargs):
        seen.append(kwargs.get('lbfgs_memory'))
        surface.evaluate(atoms)
        return QuenchResult(atoms.copy(), 0., 0., True, 0, 1, 'true')

    monkeypatch.setattr(rc, 'quench', spy_quench)
    monkeypatch.setattr(rc, 'safe_lbfgs', lambda *a, **kw: _safe_spy(a, kw, seen_safe))
    # A flat real-coordinate oracle makes the short climb deterministic while
    # the lifecycle still executes the actual reduced-coordinate driver.
    class Flat:
        requests = 0
        def evaluate(self, atoms):
            self.requests += 1
            return 0., np.zeros_like(atoms.positions)
    trees = [dict(bodies=[(0, 1, 4, 6, 7), (0, 1, 2, 10, 11),
                          (1, 2, 3, 5, 8, 9, 12, 13)],
                  parents=(-1, 0, 1), joints=(None, (0, 1), (1, 2)))]
    a = molecule('trans-butane')
    cfg = RCSSWConfig(torsion_length=2., width=.2, rotation_bias=10.,
                      max_gaussians=1, temperature_K=0., lbfgs_memory=400)
    rc.run_rc_ssw(a, Flat(), bodies=trees[0]['bodies'], parents=(-1, 0, 1),
                  joints=(None, (0, 1), (1, 2)), steps=1, config=cfg,
                  rng=np.random.default_rng(3))
    assert seen == [400, 400]
    assert seen_safe and all(value == 400 for value in seen_safe)

    seen.clear()
    seen_safe.clear()
    a = molecule('H2O'); b = a.copy(); b.translate([3., .2, .1]); a += b
    forest_cfg = RCForestSSWConfig(rotation_length=2., torsion_length=2.,
        width=.2, rotation_bias=10., max_gaussians=1, temperature_K=0., lbfgs_memory=400)
    forest.run_rc_forest_ssw(a, Flat(), trees=[
        dict(bodies=[(0, 1, 2)], parents=(-1,), joints=(None,)),
        dict(bodies=[(3, 4, 5)], parents=(-1,), joints=(None,))],
        anchor=0, steps=1, config=forest_cfg, rng=np.random.default_rng(3))
    assert seen == [400, 400]
    assert seen_safe and all(value == 400 for value in seen_safe)


def _safe_spy(args, kwargs, seen):
    seen.append(kwargs.get('lbfgs_memory'))
    return safe_lbfgs(*args, **kwargs)


def test_rcvc_public_memory_reaches_cell_lifecycle_and_biased_quench(monkeypatch):
    import pamssw.standalone.rc_vc_reference as rcvc
    seen_cell, seen_safe = [], []

    class Flat:
        requests = 0
        def evaluate(self, atoms):
            self.requests += 1
            return 0., np.zeros_like(atoms.positions), np.zeros((3, 3))

    def cell_spy(atoms, surface, **kwargs):
        seen_cell.append(kwargs.get('lbfgs_memory'))
        surface.evaluate(atoms)
        ev = VCEvaluation(0., np.zeros(3 * len(atoms) + 6), atoms.copy(), 0.,
                          np.zeros_like(atoms.positions), np.zeros((3, 3)), atoms.get_volume())
        return SimpleNamespace(converged=True, evaluation=ev,
            certificate={'certified': True, 'fmax': 0., 'stress_max': 0.}, requests=1)

    def safe_spy(*args, **kwargs):
        seen_safe.append(kwargs.get('lbfgs_memory'))
        return safe_lbfgs(*args, **kwargs)

    monkeypatch.setattr(rcvc, 'cell_quench', cell_spy)
    monkeypatch.setattr(rcvc, 'safe_lbfgs', safe_spy)
    a = bulk('Cu', 'fcc', a=3.6, cubic=True)
    trees = [dict(bodies=[tuple(range(4))], parents=(-1,), joints=(None,))]
    cfg = RCVCSSWConfig(rotation_length=2., torsion_length=2., strain_length=4.,
        width=.2, rotation_bias=10., max_gaussians=1, temperature_K=0., lbfgs_memory=400)
    result = rcvc.run_rc_vc_ssw(a, Flat(), trees=trees, anchor=0, steps=1,
                                config=cfg, rng=np.random.default_rng(3))
    assert result.records[1]['status'] == 'valid_landing'
    assert seen_cell == [400, 400]
    assert seen_safe and all(value == 400 for value in seen_safe)


def test_vc_real_cu_memory_reaches_joint_initial_biased_and_final_quenches(monkeypatch):
    import pamssw.standalone.vc_reference as vc
    import pamssw.standalone.cell_relax as cell
    seen_vc, seen_cell = [], []
    original_vc, original_cell = vc.safe_lbfgs, cell.safe_lbfgs

    def spy_vc(*args, **kwargs):
        seen_vc.append(kwargs.get('lbfgs_memory'))
        return original_vc(*args, **kwargs)

    def spy_cell(*args, **kwargs):
        seen_cell.append(kwargs.get('lbfgs_memory'))
        return original_cell(*args, **kwargs)

    monkeypatch.setattr(vc, 'safe_lbfgs', spy_vc)
    monkeypatch.setattr(cell, 'safe_lbfgs', spy_cell)
    atoms = bulk('Cu', 'fcc', a=3.65, cubic=True)
    atoms.positions[0] += [.03, -.02, .01]
    config = VCSSWConfig(strain_length=3.6, width=.2, rotation_bias=.5,
        max_gaussians=1, rotation_hvp=41, relax_steps=150,
        lbfgs_memory=400)
    result = vc.run_vc_ssw(atoms, ASEStressSurface(EMT()), steps=1,
                           config=config, rng=np.random.default_rng(7))
    assert result.status == 'completed' and len(result.minima) == 2
    assert result.records[1]['climb'][0]['status'] == 'converged'
    assert seen_cell and all(value == 400 for value in seen_cell)
    assert seen_vc and all(value == 400 for value in seen_vc)


def test_block_uses_atomic_memory_for_cell_partial_and_atomic_scopes(monkeypatch):
    import pamssw.standalone.block_ssw as block
    cell_seen, partial_seen, atomic_seen = [], [], []

    class Surface:
        requests = 0
        def evaluate(self, atoms):
            self.requests += 1
            return 0., np.zeros_like(atoms.positions), np.zeros((3, 3))

    def cell_spy(atoms, surface, **kwargs):
        cell_seen.append(kwargs.get('lbfgs_memory'))
        ev = SimpleNamespace(atoms=atoms.copy(), objective=0.)
        return SimpleNamespace(converged=True, evaluation=ev, certificate={'certified': True},
                               requests=1)

    def partial_spy(*args, **kwargs):
        partial_seen.append(kwargs.get('lbfgs_memory'))
        q = np.asarray(args[0])
        return SimpleNamespace(q=q.copy(), status='maxiter', steps=1, error=None,
                               converged=False)

    def direction_spy(*args, **kwargs):
        return SimpleNamespace(direction=np.ones(9) / 3., force_calls=1)

    def atomic_spy(atoms, surface, **kwargs):
        atomic_seen.append(kwargs['config'].lbfgs_memory)
        return SimpleNamespace(status='gaussian_limit', atoms=atoms.copy(), requests=1)

    monkeypatch.setattr(block, 'cell_quench', cell_spy)
    monkeypatch.setattr(block, 'safe_lbfgs', partial_spy)
    monkeypatch.setattr(block, 'cell_direction', direction_spy)
    monkeypatch.setattr(block, 'atomic_climb', atomic_spy)
    atomic = SSWConfig(width=.2, rotation_bias=.5, max_gaussians=1,
        temperature_K=0., fmax=.01, relax_steps=1, fd_step=1e-4,
        rotation_hvp=2, rotation_tol=.02, direction_sampling='global',
        cluster_frame='translation_only', quench_optimizer='safe-lbfgs-total',
        lbfgs_memory=400)
    config = BlockSSWConfig(atomic, 3.6, cell_cycles=1, atomic_period=2,
                            partial_atom_steps=1)
    result = block.run_block_ssw(bulk('Cu', 'fcc', cubic=True), Surface(),
                                 steps=2, config=config, rng=np.random.default_rng(3))
    assert result.status == 'completed'
    assert cell_seen == [400, 400, 400]
    assert partial_seen == [400, 400]
    assert atomic_seen == [400]
