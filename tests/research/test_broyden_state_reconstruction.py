import numpy as np

from research.ga_ssw.broyden_state_reconstruction import BroydenState


def test_first_step_uses_fixed_g0_and_stores_preupdate_state():
    g0 = np.array([2.0, 3.0, 4.0])
    state = BroydenState(g0, weight=1000.0, metric="euclidean",
                         history_limit=50, spectral_limit=1e7)
    x = np.array([1.0, 2.0, 3.0])
    force = np.array([0.5, -1.0, 2.0])
    result = state.step(x, force)
    np.testing.assert_allclose(result.x, x + g0 * force)
    assert state.x_previous is not None
    assert state.force_previous is not None
    np.testing.assert_array_equal(state.x_previous, x)
    np.testing.assert_array_equal(state.force_previous, force)
    assert state.history_size == 0


def test_rejects_invalid_configuration_and_input():
    g0 = np.ones(3)
    for kwargs in (
        dict(weight=np.nan, metric="euclidean", history_limit=2, spectral_limit=1e7),
        dict(weight=1000.0, metric="bad", history_limit=2, spectral_limit=1e7),
        dict(weight=1000.0, metric="euclidean", history_limit=0, spectral_limit=1e7),
        dict(weight=1000.0, metric="euclidean", history_limit=2, spectral_limit=np.inf),
    ):
        with np.testing.assert_raises(ValueError):
            BroydenState(g0, **kwargs)
    state = BroydenState(g0, weight=1000.0, metric="euclidean",
                         history_limit=2, spectral_limit=1e7)
    with np.testing.assert_raises(ValueError):
        state.step(np.ones(2), np.ones(3))
    with np.testing.assert_raises(ValueError):
        state.step(np.ones(3), np.array([1.0, np.nan, 1.0]))


def test_one_dimensional_secant_reaches_quadratic_root_and_trims():
    # F=-2x, g0=0.1: after the initial move, the first secant is exact
    # except for regularization. The native rule requires m < history_limit.
    state = BroydenState([0.1], weight=1000., metric='euclidean',
                         history_limit=2, spectral_limit=1e7)
    x = np.array([1.])
    x = state.step(x, -2*x).x
    update = state.step(x, -2*x)
    np.testing.assert_allclose(update.x, [0.64 / 1000001.], atol=2e-16)
    assert state.history_size == 1
    update = state.step(update.x, -2*update.x)
    assert update.dropped == 1
    assert state.history_size == 1


def test_native_null_force_difference_does_not_silently_restart():
    state = BroydenState(np.ones(3), weight=1000., metric='native_block_sum',
                         history_limit=50, spectral_limit=1e7)
    state.step(np.zeros(3), np.zeros(3))
    with np.testing.assert_raises_regex(ValueError, 'normalizer'):
        state.step(np.ones(3), np.array([1., -1., 0.]))
    assert state.history_size == 0
    np.testing.assert_array_equal(state.x_previous, np.zeros(3))


def test_minimal_spectral_failure_restarts_in_same_call():
    # The order-one exact spectrum is ~1/(2*g0), above the native 1e7 limit.
    state = BroydenState([1e-8], weight=1000., metric='euclidean',
                         history_limit=50, spectral_limit=1e7)
    x = state.step(np.array([1.]), np.array([-2.])).x
    force = -2*x
    update = state.step(x, force)
    assert update.restarted
    assert state.history_size == 0
    np.testing.assert_array_equal(update.x, x + 1e-8*force)
    np.testing.assert_array_equal(state.x_previous, x)
