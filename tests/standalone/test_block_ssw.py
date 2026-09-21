"""Short real Cu/EMT block lifecycle checks, not search-efficacy evidence."""
import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from pamssw.standalone.block_ssw import (BlockSSWConfig, _cell_displacement,
                                          run_block_ssw)
from pamssw.standalone.cell_relax import cell_quench
from pamssw.standalone.paper_reference import SSWConfig
from pamssw.standalone.vc_geometry import ASEStressSurface


def config(metric='lattice_frobenius'):
    atomic = SSWConfig(width=.2, rotation_bias=.5, max_gaussians=1,
        temperature_K=300., fmax=.01, relax_steps=150, fd_step=1e-4,
        rotation_hvp=41, rotation_tol=.02, direction_sampling='global',
        cluster_frame='translation_only', quench_optimizer='safe-lbfgs-total')
    return BlockSSWConfig(atomic, 3.6, cell_cycles=1,
                          cell_step_fraction=.03, cell_step_metric=metric,
                          partial_atom_steps=1)


def test_deformation_rms_matches_frobenius_step_on_cube_and_isotropic_on_anisotropic_cell():
    direction = np.arange(1., 10.)
    direction /= np.linalg.norm(direction)
    cube = np.diag([4., 4., 4.])
    old = _cell_displacement(cube, direction, .15, 'lattice_frobenius')
    rms = _cell_displacement(cube, direction, .15, 'deformation_rms')
    np.testing.assert_allclose(old, rms)

    anisotropic = np.diag([3., 5., 8.])
    alt = _cell_displacement(anisotropic, direction, .15, 'deformation_rms')
    relative = np.linalg.solve(anisotropic, alt.reshape(3, 3))
    assert np.linalg.norm(relative) / np.sqrt(3.) == pytest.approx(.15)
    assert not np.isclose(np.linalg.norm(np.linalg.solve(anisotropic, old.reshape(3, 3))) / np.sqrt(3.), .15)


def test_block_metric_is_explicit_and_rejects_unknown_policy():
    assert config().cell_step_metric == 'lattice_frobenius'
    with pytest.raises(ValueError, match='cell_step_metric'):
        BlockSSWConfig(config().atomic, 3.6, cell_step_metric='unknown')
    with pytest.raises(ValueError, match='nonzero'):
        _cell_displacement(np.eye(3), np.zeros(9), .15, 'deformation_rms')


class BoundedSurface(ASEStressSurface):
    def __init__(self):
        super().__init__(EMT())
        self.calls = 0
    def evaluate(self, atoms):
        assert self.calls < 600, 'bounded wiring test exceeded request cap'
        self.calls += 1
        return super().evaluate(atoms)


def test_real_cu_cell_and_combined_blocks_partial_continuation_and_certificates():
    # A vacancy in Cu8 permits internal response to cell deformation; perfect
    # fcc Cu4 has symmetry-zero atomic forces under homogeneous strain.
    atoms = bulk('Cu', 'fcc', a=3.65, cubic=True).repeat((2, 1, 1))
    del atoms[0]
    original = atoms.copy()
    surface = BoundedSurface()
    c = config()
    r = run_block_ssw(atoms, surface, steps=2, config=c,
                      rng=np.random.default_rng(7))
    assert r.status == 'completed' and len(r.minima) == 3
    assert r.requests == surface.requests == surface.calls
    assert r.requests == sum(event['requests'] for event in r.records)
    assert r.records[0]['requests'] == r.initial.requests
    first, second = r.records[1:]
    assert not first['atomic_scheduled'] and first['atomic'] is None
    assert second['atomic_scheduled'] and second['atomic'] is not None
    for event in (first, second):
        assert event['status'] == 'valid_landing'
        cycle, = event['cell_cycles']
        # Actual one-iteration unconverged atomic relaxation is retained and
        # followed by a certified full joint quench, rather than rejected.
        assert cycle['partial_status'] == 'maxiter'
        assert cycle['partial_steps'] == 1 and cycle['partial_error'] is None
        assert np.linalg.norm(cycle['cell_after'] - cycle['cell_before']) > 1e-3
        assert cycle['cell_step_metric'] == 'lattice_frobenius'
        assert np.isfinite(cycle['deformation_rms'])
        assert len(cycle['principal_stretches']) == 3
        assert cycle['mode'].force_calls <= c.cell_rotation_requests
        assert event['landing'].converged
        assert event['requests'] == (cycle['requests'] + event['landing'].requests
            + (0 if event['atomic'] is None else event['atomic'].requests))
    verify = ASEStressSurface(EMT())
    for minimum in r.minima:
        e, f, stress = verify.evaluate(minimum.atoms)
        assert np.linalg.norm(f, axis=1).max() <= c.atomic.fmax
        assert np.abs(stress + c.pressure*np.eye(3)).max() <= c.stress_tol
        assert e + c.pressure*minimum.atoms.get_volume() == pytest.approx(minimum.objective, abs=1e-12)
        assert np.linalg.det(minimum.atoms.cell.array) > 0
    assert verify.requests == 3
    assert r.best.objective == min(m.objective for m in r.minima)
    np.testing.assert_array_equal(atoms.positions, original.positions)
    np.testing.assert_array_equal(atoms.cell.array, original.cell.array)
    np.testing.assert_array_equal(atoms.numbers, original.numbers)


def test_real_cu_deformation_rms_metric_short_block():
    atoms = bulk('Cu', 'fcc', a=3.65, cubic=True).repeat((2, 1, 1))
    del atoms[0]
    surface = BoundedSurface()
    result = run_block_ssw(atoms, surface, steps=1,
        config=config('deformation_rms'), rng=np.random.default_rng(7))
    assert result.status == 'completed' and len(result.minima) == 2
    cycle = result.records[1]['cell_cycles'][0]
    assert cycle['cell_step_metric'] == 'deformation_rms'
    assert cycle['deformation_rms'] == pytest.approx(.03)
    assert len(cycle['principal_stretches']) == 3


def test_real_cu_deformation_rms_metric_two_outer_steps_includes_atomic_branch():
    atoms = bulk('Cu', 'fcc', a=3.65, cubic=True).repeat((2, 1, 1))
    del atoms[0]
    surface = BoundedSurface()
    result = run_block_ssw(atoms, surface, steps=2,
        config=config('deformation_rms'), rng=np.random.default_rng(7))
    assert result.status == 'completed' and len(result.minima) == 3
    first, second = result.records[1:]
    assert first['atomic'] is None and not first['atomic_scheduled']
    assert second['atomic_scheduled'] and second['atomic'] is not None
    for event in (first, second):
        cycle = event['cell_cycles'][0]
        assert cycle['cell_step_metric'] == 'deformation_rms'
        assert cycle['deformation_rms'] == pytest.approx(.03)
        assert event['status'] == 'valid_landing'


def test_initial_backend_failure_is_counted_and_not_a_minimum():
    class Broken:
        requests = 0
        def evaluate(self, atoms):
            self.requests += 1
            raise RuntimeError('intentional backend failure')
    s = Broken()
    r = run_block_ssw(bulk('Cu', 'fcc', cubic=True), s, steps=2,
                      config=config(), rng=np.random.default_rng(7))
    assert r.status == 'initial_quench_failed'
    assert r.current is r.best is None and not r.minima
    assert r.requests == s.requests == r.records[0]['requests'] == 1
    assert not r.initial.converged


def test_fresh_certificate_failure_cannot_certify_optimizer_success():
    class CertificateFailure(ASEStressSurface):
        def evaluate(self, atoms):
            if self.requests == 1:
                self.requests += 1
                raise RuntimeError('fresh certificate failed')
            return super().evaluate(atoms)
    s = CertificateFailure(EMT())
    # Loose tolerances isolate lifecycle fault injection, not convergence or
    # scientific qualification: the real first evaluation succeeds immediately.
    q = cell_quench(bulk('Cu', 'fcc', cubic=True), s, strain_length=3.6,
        pressure=0., fmax=1., stress_tol=1., max_step=.2, maxiter=0)
    assert q.optimizer.converged and not q.converged
    assert q.evaluation is None and not q.certificate['certified']
    assert 'fresh certificate failed' in q.certificate['error']
    assert q.requests == s.requests == 2
