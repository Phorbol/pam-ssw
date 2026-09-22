import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes


class Harmonic(Calculator):
    implemented_properties = ['energy', 'forces']
    def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {'energy': float((atoms.positions ** 2).sum() / 2),
                        'forces': -atoms.positions.copy()}


def _config():
    from pamssw.standalone.paper_reference import SSWConfig
    return SSWConfig(width=.2, rotation_bias=2., max_gaussians=1,
                     temperature_K=300., fmax=1e-4, relax_steps=20,
                     fd_step=.001, rotation_hvp=4, rotation_tol=1e-5,
                     direction_sampling='global')


def _surface():
    from pamssw.standalone.surface import ASESurface
    return ASESurface(Harmonic())


def test_starter_hook_is_explicitly_accepted():
    from pamssw.standalone.paper_reference import run_ssw
    result = run_ssw(Atoms('H', positions=[[.1, .2, .3]]), _surface(), steps=0,
                     config=_config(), rng=np.random.default_rng(1),
                     starter_selector=lambda snapshot, selector_rng: None,
                     selector_rng=np.random.default_rng(2))
    assert result.status == 'completed'


def test_snapshot_is_copy_and_mc_rejected_landing_is_visible(monkeypatch):
    import pamssw.standalone.paper_reference as paper
    from pamssw.standalone.native_mc import NativeMCSettings
    seen = []

    def reject(*args, **kwargs):
        decision = paper.native_metropolis(*args, **kwargs)
        return decision

    def selector(snapshot, selector_rng):
        seen.append(snapshot)
        snapshot.observations[0].atoms.positions[:] = 99.
        return None

    # Force the MC branch to retain a landing while preserving its normal shape.
    monkeypatch.setattr(paper, 'native_metropolis',
                        lambda *args, **kwargs: type('D', (), {
                            'state': kwargs.get('state'), 'accepted': False})())
    atoms = Atoms('H', positions=[[.1, .2, .3]])
    result = paper.run_ssw(atoms, _surface(), steps=1, config=_config(),
                           rng=np.random.default_rng(3),
                           mc=NativeMCSettings(.1, 2),
                           starter_selector=selector,
                           selector_rng=np.random.default_rng(4))
    assert seen and len(seen[0].observations) >= 2
    assert result.records[0].starter_selection['chosen_index'] is None
    assert not result.records[0].accepted
    assert not np.all(result.initial.atoms.positions == 99.)


@pytest.mark.parametrize('choice, error', [(True, TypeError), (999, IndexError)])
def test_selector_index_is_strict(choice, error):
    from pamssw.standalone.paper_reference import run_ssw
    with pytest.raises(error):
        run_ssw(Atoms('H', positions=[[.1, .2, .3]]), _surface(), steps=1,
                config=_config(), rng=np.random.default_rng(5),
                starter_selector=lambda snapshot, selector_rng: choice,
                selector_rng=np.random.default_rng(6))


def test_selector_requires_independent_rng_and_rejects_checkpoint(tmp_path):
    from pamssw.standalone.paper_reference import run_ssw
    main_rng = np.random.default_rng(7)
    with pytest.raises(ValueError, match='independent'):
        run_ssw(Atoms('H', positions=[[.1, .2, .3]]), _surface(), steps=0,
                config=_config(), rng=main_rng,
                starter_selector=lambda snapshot, selector_rng: None,
                selector_rng=main_rng)
    shared = np.random.default_rng(11)
    shared_wrapper = np.random.Generator(shared.bit_generator)
    with pytest.raises(ValueError, match='share'):
        run_ssw(Atoms('H', positions=[[.1, .2, .3]]), _surface(), steps=0,
                config=_config(), rng=shared,
                starter_selector=lambda snapshot, selector_rng: None,
                selector_rng=shared_wrapper)
    with pytest.raises(ValueError, match='requires an independent'):
        run_ssw(Atoms('H', positions=[[.1, .2, .3]]), _surface(), steps=0,
                config=_config(), rng=np.random.default_rng(8),
                starter_selector=lambda snapshot, selector_rng: None)
    with pytest.raises(ValueError, match='checkpointing'):
        run_ssw(Atoms('H', positions=[[.1, .2, .3]]), _surface(), steps=0,
                config=_config(), rng=np.random.default_rng(9),
                checkpoint_path=tmp_path / 'cp.pkl',
                starter_selector=lambda snapshot, selector_rng: None,
                selector_rng=np.random.default_rng(10))
def test_transparent_selector_preserves_rng_and_trajectory():
    from pamssw.standalone.paper_reference import run_ssw
    atoms = Atoms('H', positions=[[.1, .2, .3]])
    a_rng, b_rng = np.random.default_rng(22), np.random.default_rng(22)
    a = run_ssw(atoms, _surface(), steps=3, config=_config(), rng=a_rng)
    b = run_ssw(atoms, _surface(), steps=3, config=_config(), rng=b_rng,
                starter_selector=lambda snapshot, policy_rng: None,
                selector_rng=np.random.default_rng(23))
    assert a.status == b.status
    assert a.evaluation_requests == b.evaluation_requests
    assert a_rng.bit_generator.state == b_rng.bit_generator.state
    np.testing.assert_array_equal(a.current.positions, b.current.positions)
    assert len(a.minima) == len(b.minima)
    for x, y in zip(a.minima, b.minima):
        assert x.energy == y.energy
        np.testing.assert_array_equal(x.atoms.positions, y.atoms.positions)
    assert [(r.status, r.accepted) for r in a.records] == [(r.status, r.accepted) for r in b.records]


@pytest.mark.parametrize('restart', [False, True])
@pytest.mark.parametrize('reject', [False, True])
def test_recovered_controller_lifetime_and_restart_without_requench(monkeypatch, restart, reject):
    from dataclasses import replace
    from ase.cluster import Icosahedron
    import pamssw.standalone.paper_reference as driver
    from pamssw.standalone.recovered_direction import (
        RecoveredDirectionController, RecoveredDirectionSettings)
    from pamssw.standalone.surface import QuenchResult
    import pamssw.standalone.recovered_cbd as cbd
    # Isolate controller ownership from CBD's strict acos endpoint arithmetic.
    # Real CBD/MLIP behavior is qualified separately, without this stub.
    monkeypatch.setattr(cbd, 'recovered_cbd_direction',
        lambda atoms, anchor, **kwargs: cbd.RecoveredCBDResult(
            anchor.copy(), 0., 0., 0., 0, 'CBD_PreRot', True, True,
            'converged', 0., (), anchor.copy()))
    initialized, observed, starts, true_quenches = [], [], [], []
    initialize = RecoveredDirectionController.initialize
    observe = RecoveredDirectionController.observe_landing
    begin = RecoveredDirectionController.begin_escape

    def init(self, source, minimum, rng):
        initialized.append(self)
        assert self._group_marker is None
        return initialize(self, source, minimum, rng)

    def observe_spy(self, atoms, rng):
        observed.append(self)
        return observe(self, atoms, rng)

    def begin_spy(self, current, work, rng):
        starts.append(self)
        return begin(self, current, work, rng)

    def quench(atoms, surface, **kwargs):
        if not kwargs.get('terms'):
            true_quenches.append(atoms.copy())
        return QuenchResult(atoms.copy(), 0., 0., True, 0, 0, 'stub')

    class Flat:
        requests = 0
        def evaluate(self, atoms):
            self.requests += 1
            return 0., np.zeros_like(atoms.positions)

    monkeypatch.setattr(RecoveredDirectionController, 'initialize', init)
    monkeypatch.setattr(RecoveredDirectionController, 'observe_landing', observe_spy)
    monkeypatch.setattr(RecoveredDirectionController, 'begin_escape', begin_spy)
    monkeypatch.setattr(driver, 'quench', quench)
    from pamssw.standalone.native_mc import NativeMCSettings
    if reject:
        from types import SimpleNamespace
        monkeypatch.setattr(driver, 'native_metropolis',
            lambda *args, **kwargs: SimpleNamespace(accepted=False, state=kwargs['state']))
    settings = RecoveredDirectionSettings(ratio_local=50, local_probability=.5,
        group_threshold=.5, pre_rotmax=1, rotmax=1, pre_ftol=.01,
        ftol=.01, metric='euclidean', max_force_calls=8)
    result = driver.run_ssw(Icosahedron('Cu', 2), Flat(), steps=3,
        config=replace(_config(), cluster_frame='direction_only'),
        rng=np.random.default_rng(19), recovered_direction=settings, mc=NativeMCSettings(.1, 99999),
        starter_selector=lambda snapshot, policy_rng: 0 if restart else None,
        selector_rng=np.random.default_rng(20))
    assert result.status == 'completed', [(r.status, r.error) for r in result.records]
    assert len(observed) == len(starts) == 3
    expected_restart = restart and not reject
    assert len(initialized) == (4 if expected_restart else 1)
    assert len({id(controller) for controller in starts}) == (3 if expected_restart else 1)
    assert len(true_quenches) == 4  # initial + one true landing per outer attempt
    assert len(result.minima) == 4
    assert [r.starter_selection['restarted'] for r in result.records] == [expected_restart] * 3
    assert [r.starter_selection['mc_current_index'] for r in result.records] == ([0, 0, 0] if reject else [1, 2, 3])
