"""Recovered rotation plus NativeLS composition contracts on Cu13/EMT."""

import numpy as np
import pytest
from ase.calculators.emt import EMT
from ase.cluster.icosahedron import Icosahedron

from pamssw.standalone import (
    ASESurface,
    NativeLSSettings,
    RecoveredRotationSettings,
    SSWConfig,
    load_ssw_checkpoint,
    run_ssw,
)


def _config():
    return SSWConfig(
        width=0.08, rotation_bias=1.0, max_gaussians=1,
        temperature_K=300.0, fmax=0.05, relax_steps=80,
        fd_step=1e-3, rotation_hvp=6, rotation_tol=1e-2,
        direction_sampling="global", cluster_frame="direction_only",
        rotation_exit_policy="force_or_budget",
    )


def _rotation():
    return RecoveredRotationSettings(
        pre_rotmax=1, rotmax=2, pre_ftol=10.0, ftol=10.0,
        metric="euclidean", max_force_calls=8,
    )


def _ls():
    # Explicit Cu fixture values used by the existing native-LS lifecycle test.
    return NativeLSSettings({(29, 29): 3.0}, {(29, 29): 2.8}, scale=0.1)


def test_recovered_rotation_native_ls_has_biased_stage_bare_landing_and_cost():
    atoms = Icosahedron("Cu", 2)
    surface = ASESurface(EMT())
    result = run_ssw(
        atoms, surface, steps=1, config=_config(), rng=np.random.default_rng(7),
        recovered_rotation=_rotation(), ls=_ls(),
    )

    assert result.status == "completed"
    assert len(result.records) == 1
    record = result.records[0]
    assert record.ls_preparation is not None
    assert record.ls_preparation["soft_quench"].surface == "modified"
    events = [event for event in record.climb if isinstance(event, dict)]
    assert events
    event = events[0]
    assert event["recovered_rotation"]["trace"]
    assert event["actual_rotation_bias"] > 0.0
    assert record.landing is not None and record.landing.surface == "true"
    fresh_energy, fresh_forces = ASESurface(EMT()).evaluate(record.landing.atoms)
    assert fresh_energy == pytest.approx(record.landing.energy, abs=1e-12)
    assert np.linalg.norm(fresh_forces, axis=1).max() <= _config().fmax
    assert record.energy_response is not None
    assert record.ls_update["observed_response_mev_per_atom"] == pytest.approx(
        1000.0 * record.energy_response
    )
    assert result.evaluation_requests == surface.requests
    assert result.evaluation_requests == result.initial.evaluation_requests + sum(
        item.evaluation_requests for item in result.records
    )


def test_recovered_rotation_native_ls_checkpoint_resume_preserves_combination(tmp_path):
    atoms = Icosahedron("Cu", 2)
    config = _config()
    rotation = _rotation()
    ls = _ls()
    continuous = run_ssw(
        atoms, ASESurface(EMT()), steps=2, config=config,
        rng=np.random.default_rng(19), recovered_rotation=rotation, ls=ls,
        checkpoint_path=tmp_path / "continuous.pkl",
    )
    path = tmp_path / "recovered-rotation-native-ls.pkl"
    first = run_ssw(
        atoms, ASESurface(EMT()), steps=1, config=config,
        rng=np.random.default_rng(19), recovered_rotation=rotation, ls=ls,
        checkpoint_path=path,
    )
    checkpoint = load_ssw_checkpoint(path)
    resumed = run_ssw(
        atoms, ASESurface(EMT()), steps=1, config=config,
        rng=np.random.default_rng(777), ls=ls, checkpoint=checkpoint,
    )

    assert first.status == continuous.status == resumed.status == "completed"
    assert checkpoint.recovered_rotation == rotation
    assert checkpoint.ls == ls
    assert resumed.evaluation_requests == continuous.evaluation_requests
    assert resumed.records[1].ls_update == continuous.records[1].ls_update
    assert resumed.checkpoint.response.steps == 2
    assert resumed.checkpoint.response.state.table == continuous.checkpoint.response.state.table
    np.testing.assert_allclose(
        resumed.current.positions, continuous.current.positions, atol=1e-12, rtol=0
    )
