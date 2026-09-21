import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT

from pamssw.standalone.paper_reference import SSWConfig
from pamssw.standalone.surface import ASESurface


def config(**kw):
    base = dict(width=.2, rotation_bias=.5, max_gaussians=1,
                temperature_K=0., fmax=.01, relax_steps=3, fd_step=1e-4,
                rotation_hvp=2, rotation_tol=1e-12,
                direction_sampling='global', cluster_frame='translation_only',
                quench_optimizer='safe-lbfgs-total')
    base.update(kw)
    return SSWConfig(**base)


def test_bias_stage_budget_validation_and_default():
    assert config().bias_stage_steps is None
    assert config(bias_stage_steps=2).bias_stage_steps == 2
    with pytest.raises(ValueError): config(bias_stage_steps=0)
    with pytest.raises(ValueError): config(bias_stage_steps=2, quench_optimizer='ase-lbfgs')


def test_real_cu_safe_quench_optin_marks_iteration_budget():
    """A capped biased stage is eligible for true check only when opted in."""
    from pamssw.standalone.atomic_climb import atomic_climb
    atoms = bulk('Cu', 'fcc', a=3.65, cubic=True)
    atoms.set_pbc(True); atoms.calc = EMT()
    class Stateless(ASESurface):
        def evaluate(self, atoms):
            self.calculator = EMT()
            return super().evaluate(atoms)
    r = atomic_climb(atoms, Stateless(EMT()), reference_energy=-1e9,
                     config=config(bias_stage_steps=1, max_gaussians=2, relax_steps=150,
                                   rotation_hvp=41, rotation_tol=.02), rng=np.random.default_rng(7))
    assert r.status == 'gaussian_limit' and len(r.climb) == 2
    assert all(e['status'] == 'stage_budget' for e in r.climb)
    assert all(e['stage_stop_reason'] == 'iteration_budget' and e['termination_reason'] == 'maxiter' for e in r.climb)
    assert all('true_energy' in e and e['max_force'] > r.checkpoint.config.fmax for e in r.climb)
    assert np.isfinite(r.atoms.positions).all()
    from pamssw.standalone.atomic_climb import resume_atomic_climb
    paused = atomic_climb(atoms, Stateless(EMT()), reference_energy=-1e9,
                          config=config(bias_stage_steps=1, max_gaussians=2, relax_steps=150,
                                        rotation_hvp=41, rotation_tol=.02), rng=np.random.default_rng(7),
                          max_completed_gaussians=1)
    resumed = resume_atomic_climb(paused.checkpoint, Stateless(EMT()), paused.checkpoint.config)
    assert resumed.climb == r.climb
    np.testing.assert_array_equal(resumed.atoms.positions, r.atoms.positions)
    strict = atomic_climb(atoms, Stateless(EMT()), reference_energy=-1e9,
                          config=config(max_gaussians=2, relax_steps=1,
                                        rotation_hvp=41, rotation_tol=.02), rng=np.random.default_rng(7))
    assert strict.status == 'biased_quench_failed' and len(strict.climb) == 1
    assert 'true_energy' not in strict.climb[0]



def test_real_cu_full_walker_still_requires_true_quench():
    from pamssw.standalone.paper_reference import run_ssw
    atoms = bulk('Cu', 'fcc', a=3.65, cubic=True)
    c = config(bias_stage_steps=1, max_gaussians=2, relax_steps=150,
               rotation_hvp=41, rotation_tol=.02)
    result = run_ssw(atoms, ASESurface(EMT()), steps=1, config=c,
                     rng=np.random.default_rng(7))
    record = result.records[0]
    assert record.landing is not None and record.landing.converged
    assert record.landing.surface == 'true'
    assert any(e['stage_stop_reason'] == 'iteration_budget' for e in record.climb)
    fresh = record.landing.atoms.copy()
    fresh.calc = EMT()
    assert np.max(np.linalg.norm(fresh.get_forces(), axis=1)) <= c.fmax
    assert fresh.get_potential_energy() == pytest.approx(record.landing.energy, abs=1e-10)


@pytest.mark.parametrize('failure', ['line_search_failed', 'nonfinite_evaluation',
                                    'nan_energy', 'nan_force', 'nan_positions'])
def test_failed_stage_cannot_be_reclassified_as_budget(monkeypatch, failure):
    import importlib
    from dataclasses import replace
    module = importlib.import_module('pamssw.standalone.atomic_climb')
    original = module.quench
    def failed_quench(*args, **kwargs):
        result = original(*args, **kwargs)
        assert result.optimizer_telemetry.termination_reason == 'maxiter'
        if failure in ('line_search_failed', 'nonfinite_evaluation'):
            return replace(result, optimizer_telemetry=replace(
                result.optimizer_telemetry, termination_reason=failure))
        if failure == 'nan_energy': return replace(result, energy=float('nan'))
        if failure == 'nan_force': return replace(result, max_force=float('nan'))
        atoms = result.atoms.copy(); atoms.positions[0, 0] = float('nan')
        return replace(result, atoms=atoms)
    monkeypatch.setattr(module, 'quench', failed_quench)
    result = module.atomic_climb(bulk('Cu', 'fcc', a=3.65, cubic=True),
        ASESurface(EMT()), reference_energy=-1e9,
        config=config(bias_stage_steps=1, max_gaussians=2, relax_steps=150,
                      rotation_hvp=41, rotation_tol=.02), rng=np.random.default_rng(7))
    assert result.status == 'biased_quench_failed'
    assert result.checkpoint.next_index == 0
    assert 'true_energy' not in result.climb[0]
    assert result.climb[0]['stage_stop_reason'] == 'failure'


def test_real_cu_vacancy_combined_vc_budget_stage_and_fresh_certificate():
    from pamssw.standalone.block_ssw import BlockSSWConfig, run_block_ssw
    from pamssw.standalone.vc_geometry import ASEStressSurface
    atoms = bulk('Cu', 'fcc', a=3.65, cubic=True).repeat((2, 1, 1))
    del atoms[0]
    c = BlockSSWConfig(config(bias_stage_steps=1, max_gaussians=1,
        relax_steps=150, rotation_hvp=41, rotation_tol=.02), 3.6,
        cell_cycles=1, cell_step_fraction=.03, partial_atom_steps=1)
    surface = ASEStressSurface(EMT())
    result = run_block_ssw(atoms, surface, steps=2, config=c,
                          rng=np.random.default_rng(7))
    combined = result.records[2]
    assert combined['atomic_scheduled'] and combined['status'] == 'valid_landing'
    assert combined['atomic'].climb[0]['stage_stop_reason'] == 'iteration_budget'
    assert combined['landing'].converged
    fresh = ASEStressSurface(EMT())
    energy, forces, stress = fresh.evaluate(combined['landing'].evaluation.atoms)
    assert np.max(np.linalg.norm(forces, axis=1)) <= c.atomic.fmax
    assert np.max(np.abs(stress + c.pressure*np.eye(3))) <= c.stress_tol
    assert result.requests == surface.requests == sum(r['requests'] for r in result.records)
