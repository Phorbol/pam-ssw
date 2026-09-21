"""Shared-driver state contracts; these stubs do not measure search quality."""
import numpy as np
import pytest
from ase.cluster import Icosahedron

from pamssw.standalone.paper_reference import SSWConfig, run_ssw
from pamssw.standalone.surface import QuenchResult


def settings():
    from pamssw.standalone.recovered_direction import RecoveredDirectionSettings
    return RecoveredDirectionSettings(ratio_local=50, local_probability=.5,
        group_threshold=.5, pre_rotmax=1, rotmax=1, pre_ftol=.01, ftol=.01,
        metric='euclidean', max_force_calls=8)


class Flat:
    requests = 0

    def evaluate(self, atoms):
        self.requests += 1
        return 0., np.zeros_like(atoms.positions)


def config():
    return SSWConfig(width=.1, rotation_bias=1., max_gaussians=2,
        temperature_K=0., fmax=.03, relax_steps=2, fd_step=.001,
        rotation_hvp=2, rotation_tol=.02, direction_sampling='global',
        cluster_frame='direction_only')


def test_recovered_controller_uses_shared_quench_and_preserves_rotation_trace(monkeypatch):
    import pamssw.standalone.paper_reference as driver
    calls = []

    def quench(atoms, surface, **kwargs):
        calls.append(len(kwargs.get('terms', ())))
        return QuenchResult(atoms.copy(), 0., 0., True, 0, 0, 'stub')

    monkeypatch.setattr(driver, 'quench', quench)
    result = run_ssw(Icosahedron('Cu', 2), Flat(), steps=1,
        config=config(), rng=np.random.default_rng(19), recovered_direction=settings())
    assert result.status == 'completed'
    assert calls == [0, 1, 2, 0]
    assert len(result.minima) == 2
    np.testing.assert_allclose(result.records[0].initial_direction,
        result.records[0].climb[0]['recovered_direction']['proposal'])
    assert result.records[0].climb[0]['recovered_direction']['coefficients'][1] == 1.
    updated = result.records[0].climb[1]['recovered_direction']['coefficients']
    assert updated[1] == 0.
    assert np.isclose(updated[9], 1.2*sum(updated[4:7]))
    for event in result.records[0].climb:
        assert event['rotation_solver'] == 'recovered-cbd'
        assert event['recovered_direction']['pair']
        assert event['recovered_rotation']['trace']
        assert event['rotation_converged']


def test_selection_failure_preserves_qualified_landing_without_mc(monkeypatch):
    import pamssw.standalone.paper_reference as driver
    from pamssw.standalone.recovered_direction import RecoveredDirectionController

    def quench(atoms, surface, **kwargs):
        return QuenchResult(atoms.copy(), 0., 0., True, 0, 0, 'stub')

    def fail(self, atoms, rng):
        raise ValueError('selection fixture exhausted')

    monkeypatch.setattr(driver, 'quench', quench)
    monkeypatch.setattr(RecoveredDirectionController, 'observe_landing', fail)
    result = run_ssw(Icosahedron('Cu', 2), Flat(), steps=2, config=config(),
        rng=np.random.default_rng(19), recovered_direction=settings())
    assert result.status == 'direction_selection_failed'
    assert len(result.records) == 1 and len(result.minima) == 2
    assert result.records[0].landing.converged
    assert not result.records[0].accepted
    assert 'selection fixture exhausted' in result.records[0].error


def test_zero_generated_direction_requests_true_quench_without_gaussian(monkeypatch):
    import pamssw.standalone.paper_reference as driver
    from pamssw.standalone.recovered_direction import RecoveredDirectionController
    from pamssw.standalone.native_direction_control import LocalDirectionResult
    calls = []
    original = RecoveredDirectionController.begin_escape

    def zero(self, current, work, rng):
        original(self, current, work, rng)
        return LocalDirectionResult(np.zeros_like(work.positions), True, 'forbidden', 0)

    def quench(atoms, surface, **kwargs):
        calls.append(len(kwargs.get('terms', ())))
        return QuenchResult(atoms.copy(), 0., 0., True, 0, 0, 'stub')

    monkeypatch.setattr(driver, 'quench', quench)
    monkeypatch.setattr(RecoveredDirectionController, 'begin_escape', zero)
    result = run_ssw(Icosahedron('Cu', 2), Flat(), steps=1, config=config(),
        rng=np.random.default_rng(19), recovered_direction=settings())
    assert calls == [0, 0]
    assert result.records[0].status == 'stage_release'
    assert result.records[0].climb[0]['status'] == 'direction_zero_release'
    assert len(result.minima) == 2


def test_generator_domain_failure_is_recorded_not_lost(monkeypatch):
    from pamssw.standalone.recovered_direction import RecoveredDirectionController
    import pamssw.standalone.paper_reference as driver

    def fail(*args):
        raise ValueError('generator domain fixture')

    monkeypatch.setattr(RecoveredDirectionController, 'begin_escape', fail)
    monkeypatch.setattr(driver, 'quench', lambda atoms, surface, **kwargs:
        QuenchResult(atoms.copy(), 0., 0., True, 0, 0, 'stub'))
    result = run_ssw(Icosahedron('Cu', 2), Flat(), steps=2, config=config(),
        rng=np.random.default_rng(19), recovered_direction=settings())
    assert result.status == 'evaluation_failed'
    assert len(result.minima) == len(result.records) == 1
    assert 'generator domain fixture' in result.records[0].error
