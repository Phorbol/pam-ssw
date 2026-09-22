import numpy as np
import pytest
from ase import Atoms


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
