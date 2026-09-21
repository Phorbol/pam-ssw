"""Contract tests for the typed recovered-rotation SSW integration.

These tests exercise lifecycle and checkpoint contracts on a small Cu cluster;
they are not search-quality or material-efficacy tests.
"""

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT
from dataclasses import replace

from pamssw.standalone import (
    ASESurface,
    RecoveredRotationSettings,
    SSWConfig,
    load_ssw_checkpoint,
    run_ssw,
)


def _atoms():
    # Noncollinear and nonperiodic: required by the ClusterFrame contract.
    return Atoms("Cu4", positions=[
        [0.00, 0.00, 0.00],
        [2.35, 0.00, 0.10],
        [0.30, 2.20, 0.00],
        [0.15, 0.25, 2.45],
    ])


def _config(**changes):
    values = dict(
        width=0.08, rotation_bias=1.0, max_gaussians=2,
        temperature_K=300.0, fmax=0.05, relax_steps=80,
        fd_step=1e-3, rotation_hvp=6, rotation_tol=1e-2,
        direction_sampling="global", cluster_frame="direction_only",
        rotation_exit_policy="force_or_budget",
    )
    values.update(changes)
    return SSWConfig(**values)


def _rotation():
    # Loose tolerances keep this a lifecycle test while preserving explicit
    # settings and a finite stopping path on the small EMT cluster.
    return RecoveredRotationSettings(
        pre_rotmax=1, rotmax=2, pre_ftol=10.0, ftol=10.0,
        metric="euclidean", max_force_calls=8,
    )


def _events(result):
    return [event for record in result.records for event in record.climb
            if isinstance(event, dict)]


def test_recovered_rotation_settings_are_typed_and_explicit():
    settings = _rotation()
    assert settings.metric == "euclidean"
    assert settings.max_force_calls == 8
    with pytest.raises((TypeError, ValueError)):
        run_ssw(_atoms(), ASESurface(EMT()), steps=0, config=_config(),
                rng=np.random.default_rng(4), recovered_rotation={})
    with pytest.raises(ValueError, match="pre_rotmax"):
        RecoveredRotationSettings(pre_rotmax=-1, rotmax=2, pre_ftol=1.0, ftol=1.0,
                                  metric="euclidean", max_force_calls=8)
    with pytest.raises(ValueError, match="metric"):
        RecoveredRotationSettings(pre_rotmax=1, rotmax=2, pre_ftol=1.0,
                                  ftol=1.0, metric="bad", max_force_calls=8)
    with pytest.raises(ValueError, match="at least 2"):
        RecoveredRotationSettings(pre_rotmax=1, rotmax=2, pre_ftol=1.0,
                                  ftol=1.0, metric="euclidean", max_force_calls=1)


def test_recovered_rotation_lifecycle_records_anchor_bias_and_stop_reason():
    result = run_ssw(_atoms(), ASESurface(EMT()), steps=1, config=_config(),
                     rng=np.random.default_rng(7), recovered_rotation=_rotation())
    events = _events(result)
    assert len(events) >= 2, "the EMT lifecycle must reach two Gaussian rotation events"
    event = events[0]
    assert "recovered_rotation" in event
    telemetry = event["recovered_rotation"]
    assert telemetry["trace"]
    assert telemetry["stage"] in {"CBD_PreRot", "CBD_UnbiasedRot", "CBD_biasedRot"}
    assert "rotation_stop_reason" in event
    assert np.asarray(event["actual_anchor"]).shape == _atoms().positions.shape
    assert np.isfinite(float(event["actual_rotation_bias"]))
    assert result.evaluation_requests > 0


def test_default_and_pre_rotation_paths_remain_mutually_exclusive():
    # The unchanged default still runs without recovered telemetry.
    result = run_ssw(_atoms(), ASESurface(EMT()), steps=1, config=_config(),
                     rng=np.random.default_rng(8))
    assert result.checkpoint is None
    assert all("recovered_rotation" not in event for event in _events(result))

    with pytest.raises(ValueError, match="pre_rotation_hvp|recovered_rotation"):
        run_ssw(_atoms(), ASESurface(EMT()), steps=0,
                config=_config(pre_rotation_hvp=2), rng=np.random.default_rng(8),
                recovered_rotation=_rotation())


def test_rotation_budget_exit_depends_on_explicit_policy():
    strict = RecoveredRotationSettings(
        pre_rotmax=0, rotmax=0, pre_ftol=1e-12, ftol=1e-12,
        metric="euclidean", max_force_calls=2,
    )
    released = run_ssw(_atoms(), ASESurface(EMT()), steps=1,
                       config=_config(rotation_exit_policy="force_or_budget"),
                       rng=np.random.default_rng(10), recovered_rotation=strict)
    released_events = _events(released)
    assert released_events
    assert released_events[0]["rotation_budget_released"] is True
    assert released_events[0]["rotation_stop_reason"] in {"rotation_limit", "force_budget"}

    failed = run_ssw(_atoms(), ASESurface(EMT()), steps=1,
                     config=_config(rotation_exit_policy="force"),
                     rng=np.random.default_rng(10), recovered_rotation=strict)
    assert failed.records[0].status == "rotation_failed"


def test_checkpoint_restores_recovered_rotation_settings_and_rejects_change_before_pes(tmp_path):
    atoms = _atoms()
    config = _config(max_gaussians=1)
    path = tmp_path / "recovered-rotation.pkl"
    first_surface = ASESurface(EMT())
    first = run_ssw(atoms, first_surface, steps=0, config=config,
                    rng=np.random.default_rng(9), recovered_rotation=_rotation(),
                    checkpoint_path=path)
    checkpoint = load_ssw_checkpoint(path)
    assert checkpoint.recovered_rotation == _rotation()
    assert first.checkpoint is not None

    resumed_surface = ASESurface(EMT())
    resumed = run_ssw(atoms, resumed_surface, steps=1, config=config,
                      rng=np.random.default_rng(99), checkpoint=checkpoint,
                      recovered_rotation=None)
    assert resumed.checkpoint is not None
    assert resumed.checkpoint.schema_version == 3
    assert resumed.checkpoint.recovered_rotation == _rotation()

    changed_surface = ASESurface(EMT())
    with pytest.raises(ValueError, match="recovered_rotation|checkpoint"):
        run_ssw(atoms, changed_surface, steps=0,
                config=config, rng=np.random.default_rng(99),
                recovered_rotation=replace(_rotation(), ftol=9.0),
                checkpoint=checkpoint)
    assert changed_surface.requests == 0


def test_default_and_native_mc_checkpoints_keep_rotation_field_empty_or_typed(tmp_path):
    from pamssw.standalone import NativeMCSettings

    path = tmp_path / "default.pkl"
    run_ssw(_atoms(), ASESurface(EMT()), steps=1, config=_config(),
            rng=np.random.default_rng(11), checkpoint_path=path)
    default = load_ssw_checkpoint(path)
    assert default.schema_version == 1
    assert getattr(default, "recovered_rotation", None) is None

    mc_path = tmp_path / "mc.pkl"
    run_ssw(_atoms(), ASESurface(EMT()), steps=1, config=_config(),
            rng=np.random.default_rng(11), mc=NativeMCSettings(.1, 2),
            checkpoint_path=mc_path)
    mc_checkpoint = load_ssw_checkpoint(mc_path)
    assert mc_checkpoint.schema_version == 2
    assert getattr(mc_checkpoint, "recovered_rotation", None) is None

    cbd_path = tmp_path / "cbd.pkl"
    run_ssw(_atoms(), ASESurface(EMT()), steps=1, config=_config(),
            rng=np.random.default_rng(11), recovered_rotation=_rotation(),
            mc=NativeMCSettings(.1, 2), checkpoint_path=cbd_path)
    cbd_checkpoint = load_ssw_checkpoint(cbd_path)
    assert cbd_checkpoint.schema_version == 3
    assert cbd_checkpoint.recovered_rotation == _rotation()


@pytest.mark.parametrize('native_mc', [False, True])
def test_continuous_two_steps_equal_checkpoint_one_plus_one(tmp_path, native_mc):
    from pamssw.standalone import NativeMCSettings
    mc = NativeMCSettings(.1, 2) if native_mc else None
    atoms, config = _atoms(), _config()
    continuous = run_ssw(atoms, ASESurface(EMT()), steps=2, config=config,
                         rng=np.random.default_rng(19), mc=mc,
                         recovered_rotation=_rotation())
    path = tmp_path / 'split.pkl'
    first = run_ssw(atoms, ASESurface(EMT()), steps=1, config=config,
                    rng=np.random.default_rng(19), mc=mc,
                    recovered_rotation=_rotation(), checkpoint_path=path)
    resumed = run_ssw(atoms, ASESurface(EMT()), steps=1, config=config,
                      rng=np.random.default_rng(777), mc=mc,
                      checkpoint=load_ssw_checkpoint(path))
    assert first.status == continuous.status == resumed.status == 'completed'
    assert len(first.records) == 1
    assert len(resumed.records) == len(continuous.records) == 2
    assert resumed.evaluation_requests == continuous.evaluation_requests
    np.testing.assert_allclose(resumed.current.positions, continuous.current.positions, atol=1e-12, rtol=0)
    for actual, expected in zip(resumed.records, continuous.records):
        assert (actual.index, actual.status, actual.accepted, actual.evaluation_requests) == (
            expected.index, expected.status, expected.accepted, expected.evaluation_requests)
        assert actual.mc_telemetry == expected.mc_telemetry
        np.testing.assert_allclose(actual.initial_direction, expected.initial_direction, atol=1e-12, rtol=0)
        assert len(actual.climb) == len(expected.climb)
        for got, want in zip(actual.climb, expected.climb):
            assert got['rotation_stop_reason'] == want['rotation_stop_reason']
            np.testing.assert_allclose(got['direction'], want['direction'], atol=1e-12, rtol=0)
            assert got['actual_rotation_bias'] == want['actual_rotation_bias']
