import numpy as np

from research.ga_ssw.broyden_history_reconstruction import reconstruct_history_action


def test_uniform_weight_history_equations_and_residual():
    rng = np.random.default_rng(7)
    ndim, m = 7, 3
    df = rng.normal(size=(ndim, m))
    u = rng.normal(size=(ndim, m))
    z_old = np.zeros((ndim, m))
    x = rng.normal(size=ndim)
    force = rng.normal(size=ndim)
    g0 = rng.normal(size=ndim)
    result = reconstruct_history_action(df, u, z_old, x, force, g0, weight=1000.0,
                                        metric="euclidean")
    # Check the defining linear equations rather than recomputing the solver.
    np.testing.assert_allclose(result.a @ result.beta, np.eye(m), atol=1e-12)
    np.testing.assert_allclose(result.z @ result.a, 1000.0**2*u + z_old,
                               rtol=1e-12, atol=1e-8)
    assert result.residual < 1e-10


def test_native_block_sum_metric_is_explicit_and_distinct():
    df = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    common = dict(u=np.eye(3, 2), z_old=np.zeros((3, 2)), x=np.zeros(3),
                  force=np.ones(3), g0=np.ones(3), weight=2.0)
    euclidean = reconstruct_history_action(df, metric="euclidean", **common)
    block = reconstruct_history_action(df, metric="native_block_sum", **common)
    assert not np.allclose(euclidean.a, block.a)
    assert euclidean.residual < 1e-12
    assert block.residual < 1e-12


def test_rejects_shape_nonfinite_and_unknown_metric():
    good = np.ones((3, 1))
    kwargs = dict(u=good, z_old=good, x=np.ones(3), force=np.ones(3),
                  g0=np.ones(3), weight=1000.0, metric="euclidean")
    for bad in (np.nan, np.inf):
        with np.testing.assert_raises(ValueError):
            reconstruct_history_action(good * bad, **kwargs)
    with np.testing.assert_raises(ValueError):
        reconstruct_history_action(good, **kwargs | {"metric": "mass_weighted"})
    with np.testing.assert_raises(ValueError):
        reconstruct_history_action(good, **kwargs | {"u": np.ones((4, 1))})
