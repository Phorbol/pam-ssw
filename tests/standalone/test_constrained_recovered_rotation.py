import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms, Hookean
from ase.calculators.emt import EMT

from pamssw.standalone.constrained_reference import (
    ConstrainedSSWConfig, run_constrained_ssw,
)
from pamssw.standalone.recovered_rotation import RecoveredRotationSettings
from pamssw.standalone.surface import ASESurface
from pamssw.standalone.ase_constraints import bind_hookean_surface, normalize_constraints


class Harmonic:
    def __init__(self):
        self.requests = 0

    def evaluate(self, atoms):
        self.requests += 1
        x = np.asarray(atoms.positions, float)
        return float(0.5 * np.sum(x * x)), -x.copy()


def atoms():
    return Atoms('H3', positions=[[0, 0, 0], [1.0, 0.1, 0], [0, 1.1, 0.2]])


def config(**kw):
    value = dict(width=.05, rotation_bias=1.0, max_gaussians=1,
                 relax_steps=20, fmax=2.0, rotation_hvp=3, rotation_tol=10.)
    value.update(kw)
    return ConstrainedSSWConfig(**value)


def recovered(**kw):
    value = dict(pre_rotmax=0, rotmax=0, pre_ftol=1e-12, ftol=1e-12,
                 metric='euclidean', max_force_calls=2)
    value.update(kw)
    return RecoveredRotationSettings(**value)


def test_recovered_settings_are_typed_and_incompatible_with_presweep():
    cfg = config(recovered_rotation=recovered())
    assert cfg.recovered_rotation.max_force_calls == 2
    assert cfg.rotation_exit_policy == 'force'
    with pytest.raises(ValueError, match='pre_rotation_hvp|recovered_rotation'):
        config(recovered_rotation=recovered(), pre_rotation_hvp=2,
               rotation_solver='ritz', rotation_bias=None)


def test_pam_gaussian_is_rejected_before_any_surface_request():
    from pamssw.standalone.pam_gaussian import PAMCurvatureGaussian
    surface = Harmonic()
    with pytest.raises((ValueError, NotImplementedError), match='recovered|PAM|anchor'):
        run_constrained_ssw(atoms(), surface, steps=0,
            config=config(recovered_rotation=recovered()),
            rng=np.random.default_rng(1),
            gaussian_policy=PAMCurvatureGaussian())
    assert surface.requests == 0


def test_force_or_budget_records_unconverged_recovered_rotation(monkeypatch):
    # The constrained runner must use the same explicit release contract as
    # paper_reference, while preserving mode.converged=False in the event.
    from types import SimpleNamespace
    import pamssw.standalone.constrained_reference as module
    def callback(work, anchor, **kwargs):
        return SimpleNamespace(direction=np.asarray(anchor), curvature=1.,
            real_curvature=1., residual_norm=1., force_calls=2,
            converged=False, stop_reason='force_budget', stage='CBD_PreRot',
            stage_complete=False, trace=(), bias_reference=np.asarray(anchor),
            rotation_weight=0.)
    monkeypatch.setattr(module, '_active_rotation_callback', callback)
    result = run_constrained_ssw(atoms(), Harmonic(), steps=1,
        config=config(recovered_rotation=recovered(), rotation_exit_policy='force_or_budget'),
        rng=np.random.default_rng(2))
    stage = result.records[1]['climb'][0]
    assert stage['rotation_budget_released'] is True
    assert stage['rotation_converged'] is False


def test_fixed_atoms_and_direction_subspace_remain_exact():
    a = atoms(); a.set_constraint(FixAtoms(indices=[0]))
    result = run_constrained_ssw(a, Harmonic(), steps=1,
        config=config(recovered_rotation=recovered(), rotation_exit_policy='force_or_budget'),
        rng=np.random.default_rng(3), direction_fixed_indices=[1])
    assert np.array_equal(result.current.atoms.positions[0], a.positions[0])
    assert result.records[1]['rotation_coordinate_indices'].tolist() == [3, 4, 5]
    mode = result.records[1]['climb'][0]['mode']
    assert np.asarray(mode.bias_reference).shape == np.asarray(mode.direction).shape
    np.testing.assert_array_equal(np.asarray(mode.bias_reference)[:3], 0.)


def test_hookean_constraint_is_represented_by_the_bound_objective():
    a = atoms(); a.set_constraint(Hookean(a1=0, a2=1, rt=1.0, k=2.0))
    surface = Harmonic()
    result = run_constrained_ssw(a, surface, steps=0, config=config(),
                                 rng=np.random.default_rng(4))
    assert result.requests > 0
    assert result.initial.certificate['scope'] == 'physical_plus_hookean'


def test_real_recovered_callback_records_force_calls_and_a_nonzero_step():
    a = Atoms('Cu4', positions=[[0, 0, 0], [2.35, 0, .1],
                                [.3, 2.2, 0], [.15, .25, 2.45]])
    cfg = config(fmax=.05, rotation_tol=.02,
                 recovered_rotation=recovered(pre_rotmax=1, rotmax=2,
                                              pre_ftol=10., ftol=10.,
                                              max_force_calls=8),
                 rotation_exit_policy='force_or_budget')
    result = run_constrained_ssw(a, ASESurface(EMT()), steps=1, config=cfg,
                                 rng=np.random.default_rng(7))
    stage = result.records[1]['climb'][0]
    assert stage['recovered_rotation']['force_calls'] >= 2
    assert np.linalg.norm(np.asarray(stage['mode'].direction)) > 0


def test_force_policy_does_not_release_a_budget_exit(monkeypatch):
    from types import SimpleNamespace
    import pamssw.standalone.constrained_reference as module
    def callback(work, anchor, **kwargs):
        return SimpleNamespace(direction=np.asarray(anchor), curvature=1.,
            real_curvature=1., residual_norm=1., force_calls=2,
            converged=False, stop_reason='force_budget', stage='CBD_PreRot',
            stage_complete=False, trace=(), bias_reference=np.asarray(anchor),
            rotation_weight=0.)
    monkeypatch.setattr(module, '_active_rotation_callback', callback)
    result = run_constrained_ssw(atoms(), Harmonic(), steps=1,
        config=config(recovered_rotation=recovered(), rotation_exit_policy='force'),
        rng=np.random.default_rng(8))
    assert result.records[1]['status'] == 'rotation_failed'
    assert result.records[1]['climb'][0]['rotation_budget_released'] is False


def test_budget_release_rejects_an_invalid_direction(monkeypatch):
    from types import SimpleNamespace
    import pamssw.standalone.constrained_reference as module
    def callback(work, anchor, **kwargs):
        return SimpleNamespace(direction=np.zeros_like(anchor), curvature=1.,
            real_curvature=1., residual_norm=1., force_calls=2,
            converged=False, stop_reason='force_budget', stage='CBD_PreRot',
            stage_complete=False, trace=(), bias_reference=np.asarray(anchor),
            rotation_weight=0.)
    monkeypatch.setattr(module, '_active_rotation_callback', callback)
    result = run_constrained_ssw(atoms(), Harmonic(), steps=1,
        config=config(recovered_rotation=recovered(), rotation_exit_policy='force_or_budget'),
        rng=np.random.default_rng(12))
    assert result.records[1]['status'] == 'evaluation_failed'
    assert 'invalid evaluated direction' in result.records[1]['error']


def test_recovered_hookean_endpoint_matches_direct_ase_objective():
    a = Atoms('H2', positions=[[0., 0., 0.], [1.4, 0., 0.]])
    a.set_constraint(Hookean(a1=0, a2=1, rt=1.0, k=2.0))
    clean = a.copy(); constraints = normalize_constraints(a); clean.set_constraint()
    bound = bind_hookean_surface(Harmonic(), constraints.hookean_specs)
    direct = a.copy()
    direct.set_constraint([Hookean(a1=0, a2=1, rt=1.0, k=2.0)])
    direct_energy = sum(c.adjust_potential_energy(direct) for c in direct.constraints)
    direct_forces = np.zeros((2, 3))
    for c in direct.constraints:
        c.adjust_forces(direct, direct_forces)
    seen = []
    def evaluate(flat):
        candidate = clean.copy(); candidate.positions[:] = np.asarray(flat).reshape(2, 3)
        energy, forces = bound.evaluate(candidate)
        expected_energy = .5 * np.sum(candidate.positions ** 2)
        expected_forces = -candidate.positions.copy()
        for constraint in direct.constraints:
            expected_energy += constraint.adjust_potential_energy(candidate)
            constraint.adjust_forces(candidate, expected_forces)
        assert energy == pytest.approx(expected_energy)
        np.testing.assert_allclose(forces, expected_forces)
        seen.append(candidate.copy())
        return energy, -forces
    anchor = np.array([0., 0., 0., 1., 0., 0.])
    from pamssw.standalone.constrained_reference import _active_rotation_callback
    mode = _active_rotation_callback(clean.positions.ravel(), anchor, evaluate=evaluate,
        rotation_bias=1., fd_step=1e-3, max_hvp=3, tol=10.,
        recovered_rotation=recovered(pre_rotmax=1, rotmax=1, pre_ftol=10., ftol=10., max_force_calls=2))
    assert len(seen) == 2 and np.linalg.norm(seen[1].positions - seen[0].positions) > 0
    energy, forces = bound.evaluate(clean)
    assert bound.last_evaluation['hookean_energy'] == pytest.approx(direct_energy)
    np.testing.assert_allclose(bound.last_evaluation['hookean_forces'], direct_forces)
    assert np.linalg.norm(mode.direction) > 0


def test_recovered_two_step_checkpoint_resume_matches_continuous(tmp_path):
    a = Atoms('Cu4', positions=[[0, 0, 0], [2.35, 0, .1],
                                [.3, 2.2, 0], [.15, .25, 2.45]])
    cfg = config(fmax=.05, rotation_tol=.02,
                 recovered_rotation=recovered(pre_rotmax=1, rotmax=2,
                                              pre_ftol=10., ftol=10., max_force_calls=8),
                 rotation_exit_policy='force_or_budget')
    continuous_path = tmp_path / 'continuous.pkl'
    continuous = run_constrained_ssw(a, ASESurface(EMT()), steps=2, config=cfg,
                                     rng=np.random.default_rng(22), checkpoint_path=continuous_path)
    split_path = tmp_path / 'split.pkl'
    first_surface = ASESurface(EMT())
    run_constrained_ssw(a, first_surface, steps=1, config=cfg,
                        rng=np.random.default_rng(22), checkpoint_path=split_path)
    from pamssw.standalone.constrained_reference import load_constrained_checkpoint
    checkpoint = load_constrained_checkpoint(split_path)
    resumed = run_constrained_ssw(a, first_surface, steps=1, config=cfg,
                                  rng=np.random.default_rng(999), checkpoint=checkpoint,
                                  checkpoint_path=split_path)
    assert resumed.requests == continuous.requests
    np.testing.assert_allclose(resumed.current.atoms.positions, continuous.current.atoms.positions)
    assert resumed.records[-1]['status'] == continuous.records[-1]['status']
    resumed_cp = load_constrained_checkpoint(split_path)
    continuous_cp = load_constrained_checkpoint(continuous_path)
    assert resumed_cp.rng_state == continuous_cp.rng_state

    class Zero:
        requests = 0
        def evaluate(self, atoms):
            raise AssertionError('checkpoint mismatch consumed PES')
    changed = config(fmax=.05, rotation_tol=.02,
                     recovered_rotation=recovered(pre_rotmax=1, rotmax=2,
                                                  pre_ftol=10., ftol=10., max_force_calls=7),
                     rotation_exit_policy='force_or_budget')
    with pytest.raises(ValueError, match='config'):
        run_constrained_ssw(a, Zero(), steps=1, config=changed,
                            rng=np.random.default_rng(1), checkpoint=checkpoint)
