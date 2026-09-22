import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT
from ase.cluster.icosahedron import Icosahedron
from pamssw.standalone.surface import ASESurface


def _native_config():
    from pamssw.standalone.paper_reference import SSWConfig
    return SSWConfig(width=.2, rotation_bias=.5, max_gaussians=1,
                     temperature_K=300., fmax=.01, relax_steps=150,
                     fd_step=1e-4, rotation_hvp=41, rotation_tol=.02,
                     direction_sampling='global', rotation_solver='ritz',
                     cluster_frame='direction_only',
                     quench_optimizer='safe-lbfgs-total')


class _BoundedEMT(ASESurface):
    def __init__(self):
        super().__init__(EMT())

    def evaluate(self, atoms):
        assert self.requests < 500
        return super().evaluate(atoms)


def _native_ls():
    from pamssw.standalone.ls_native_reference import NativeLSSettings
    return NativeLSSettings({(29, 29): 3.}, {(29, 29): 2.8}, scale=.1)


def _run_selector(choice, rng):
    import pamssw.standalone.paper_reference as paper
    from pamssw.standalone.native_mc import NativeMCSettings

    return paper.run_ssw(
        Icosahedron('Cu', 2), _BoundedEMT(), steps=2,
        config=_native_config(), rng=rng, ls=_native_ls(),
        mc=NativeMCSettings(.1, 2),
        starter_selector=(None if choice == 'none' else
                          (lambda snapshot, selector_rng: snapshot.current_index if choice == "current" else choice)),
        selector_rng=(None if choice == 'none' else np.random.default_rng(8)))


def test_pool_jump_restarts_native_ls_local_step_counter(monkeypatch):
    import pamssw.standalone.paper_reference as paper
    from pamssw.standalone.native_mc import NativeMCSettings

    monkeypatch.setattr(paper, 'native_metropolis',
                        lambda *args, **kwargs: type('D', (), {
                            'state': kwargs['state'], 'accepted': False})())
    result = paper.run_ssw(
        Icosahedron('Cu', 2), _BoundedEMT(), steps=2,
        config=_native_config(), rng=np.random.default_rng(7), ls=_native_ls(),
        mc=NativeMCSettings(.1, 2),
        starter_selector=lambda snapshot, selector_rng: 1,
        selector_rng=np.random.default_rng(8))

    assert result.status == 'completed'
    assert [record.starter_selection['restarted'] for record in result.records] == [True, False]
    assert [record.ls_update['step'] for record in result.records] == [1, 1]
    assert result.records[0].starter_selection['ls_reinitialized']


def test_pool_restart_failure_keeps_previous_ls_state_and_records_failure(monkeypatch):
    import pamssw.standalone.paper_reference as paper
    from pamssw.standalone.native_mc import NativeMCSettings

    monkeypatch.setattr(paper, 'native_metropolis',
                        lambda *args, **kwargs: type('D', (), {
                            'state': kwargs['state'], 'accepted': False})())
    monkeypatch.setattr(paper, '_prepare_pool_restart',
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            ValueError('restart fixture failure')))
    atoms = Icosahedron('Cu', 2)
    result = paper.run_ssw(
        atoms, _BoundedEMT(), steps=1, config=_native_config(),
        rng=np.random.default_rng(7), ls=_native_ls(),
        mc=NativeMCSettings(.1, 2),
        starter_selector=lambda snapshot, selector_rng: 1,
        selector_rng=np.random.default_rng(8))

    assert result.status == 'starter_selection_failed'
    assert result.records[0].ls_update['step'] == 1
    assert result.records[0].starter_selection['restarted'] is False
    assert result.records[0].starter_selection['ls_reinitialized'] is False
    np.testing.assert_array_equal(result.current.positions, result.initial.atoms.positions)


def test_pool_selector_none_and_current_index_preserve_ls_trajectory():
    rngs = [np.random.default_rng(7) for _ in range(3)]
    baseline = _run_selector('none', rngs[0])
    explicit_none = _run_selector(None, rngs[1])
    current_index = _run_selector('current', rngs[2])
    assert rngs[0].bit_generator.state == rngs[1].bit_generator.state == rngs[2].bit_generator.state

    for result in (explicit_none, current_index):
        assert result.status == baseline.status == 'completed'
        assert result.evaluation_requests == baseline.evaluation_requests
        assert [record.ls_update['step'] for record in result.records] == [1, 2]
        np.testing.assert_array_equal(result.current.positions, baseline.current.positions)
    assert all(not record.starter_selection['restarted'] for record in explicit_none.records)
    assert all(not record.starter_selection['restarted'] for record in current_index.records)


def test_paper_ls_restart_rebuilds_response_controller():
    from pamssw.standalone.paper_reference import LSSettings, _initialize_ls_state

    atoms = Atoms('Cu2', positions=[[0., 0., 0.], [2.4, 0., 0.]])
    settings = LSSettings({(29, 29): 3.}, {(29, 29): 2.8}, target_per_atom=.02)
    first_frozen, first_response = _initialize_ls_state(atoms, settings)
    first_response.steps = 4
    second_frozen, second_response = _initialize_ls_state(atoms, settings)

    assert first_frozen is not second_frozen
    assert first_response is not second_response
    assert second_response.steps == 0


def test_pool_restart_prepares_ls_before_direction(monkeypatch):
    import pamssw.standalone.paper_reference as paper

    calls = []
    frozen = object()
    response = object()

    def initialize_ls(atoms, settings):
        calls.append(('ls', atoms.copy()))
        return frozen, response

    class Controller:
        def __init__(self, settings):
            self.settings = settings

        def initialize(self, source, minimum, rng):
            calls.append(('direction', source.copy(), minimum.copy()))

    monkeypatch.setattr(paper, '_initialize_ls_state', initialize_ls)
    import pamssw.standalone.recovered_direction as recovered_direction
    monkeypatch.setattr(recovered_direction, 'RecoveredDirectionController', Controller)
    selected = Atoms('H', positions=[[1., 2., 3.]])
    settings = object()
    direction = object()
    ls_frozen, ls_response, controller = paper._prepare_pool_restart(
        selected, ls=settings, recovered_direction=direction,
        rng=np.random.default_rng(1))

    assert (ls_frozen, ls_response) == (frozen, response)
    assert controller.settings is direction
    assert [item[0] for item in calls] == ['ls', 'direction']
    np.testing.assert_array_equal(calls[0][1].positions, selected.positions)
    np.testing.assert_array_equal(calls[1][1].positions, selected.positions)
    np.testing.assert_array_equal(calls[1][2].positions, selected.positions)


def test_pool_restart_does_not_hide_initializer_failure(monkeypatch):
    import pamssw.standalone.paper_reference as paper

    def fail(atoms, settings):
        raise ValueError('new LS domain is invalid')

    monkeypatch.setattr(paper, '_initialize_ls_state', fail)
    with pytest.raises(ValueError, match='new LS domain'):
        paper._prepare_pool_restart(
            Atoms('H', positions=[[1., 2., 3.]]), ls=object(),
            recovered_direction=object(), rng=np.random.default_rng(2))


def test_paper_ls_pool_jump_uses_fresh_history_on_next_step(monkeypatch):
    """Reuse existing Cu2 EMT LS fixture; coefficients are interface-test inputs."""
    import pamssw.standalone.paper_reference as paper
    from pamssw.standalone.softening import LSResponseState
    from pamssw.standalone.surface import quench
    from pamssw.standalone.native_mc import NativeMCSettings
    from ase.optimize import BFGS

    atoms = quench(Atoms('Cu2', positions=[[0, 0, 0], [2.7, 0, 0]]),
                   _BoundedEMT(), fmax=1e-6, steps=100, optimizer=BFGS).atoms
    config = paper.SSWConfig(width=.1, rotation_bias=2., max_gaussians=1,
        temperature_K=300., fmax=1e-5, relax_steps=100, fd_step=1e-4,
        rotation_hvp=8, rotation_tol=1e-3, direction_sampling='global')
    settings = paper.LSSettings({(29, 29): 1.}, {(29, 29): 3.}, target_per_atom=.001)
    seen = []
    original = LSResponseState.update
    def capture(self, frozen, next_atoms, **kwargs):
        seen.append(self.steps)
        return original(self, frozen, next_atoms, **kwargs)
    monkeypatch.setattr(LSResponseState, 'update', capture)
    monkeypatch.setattr(paper, 'native_metropolis',
        lambda *args, **kwargs: type('Decision', (), {'state': kwargs['state'], 'accepted': False})())
    result = paper.run_ssw(atoms, _BoundedEMT(), steps=2, config=config,
        rng=np.random.default_rng(19), ls=settings, mc=NativeMCSettings(.1, 2),
        starter_selector=lambda snapshot, rng: 1, selector_rng=np.random.default_rng(8))
    assert result.status == 'completed'
    assert seen == [0, 0]
    assert [r.starter_selection['ls_reinitialized'] for r in result.records] == [True, False]
