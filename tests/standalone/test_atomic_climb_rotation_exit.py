"""Pure-stub contract tests for the explicit atomic rotation-exit policy."""
from dataclasses import replace
import importlib
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms

from pamssw.standalone.paper_reference import SSWConfig
from pamssw.standalone.surface import QuenchResult


climb = importlib.import_module("pamssw.standalone.atomic_climb")


def _config(**changes):
    values = dict(width=.1, rotation_bias=1., max_gaussians=1,
        temperature_K=0., fmax=.03, relax_steps=2, fd_step=.001,
        rotation_hvp=2, rotation_tol=.02, direction_sampling="global",
        rotation_solver="dimer", cluster_frame="translation_only",
        quench_optimizer="safe-lbfgs-total")
    values.update(changes)
    return SSWConfig(**values)


def _atoms():
    return Atoms("H2", positions=[[0., 0., 0.], [1., 0., 0.]],
                 cell=np.eye(3) * 5., pbc=True)


class FlatSurface:
    def __init__(self):
        self.requests = 0

    def evaluate(self, atoms):
        self.requests += 1
        return 0., np.zeros_like(atoms.positions)


def _solver(monkeypatch, *, stop="budget_exhausted", direction="anchor", converged=False):
    calls = []
    def solve(atoms, anchor, **kwargs):
        calls.append((atoms.copy(), anchor.copy()))
        value = (anchor.copy() if isinstance(direction, str) and direction == "anchor"
                 else np.asarray(direction, dtype=float))
        return SimpleNamespace(direction=value, curvature=-1., residual_norm=.4,
            force_calls=3, hvp_calls=2, converged=converged,
            projected_symmetry_error=0., stop_reason=stop)
    monkeypatch.setattr(climb, "paper_dimer_direction", solve)
    return calls


def _stub_quench(monkeypatch, *, converged=True):
    calls = []
    def quench(atoms, surface, **kwargs):
        calls.append(bool(kwargs.get("terms")))
        return QuenchResult(atoms.copy(), 0., 0., converged, 0, 0, "stub")
    monkeypatch.setattr(climb, "quench", quench)
    return calls


def test_default_force_policy_still_stops_on_budget_exhaustion(monkeypatch):
    solve_calls = _solver(monkeypatch)
    quench_calls = _stub_quench(monkeypatch)
    result = climb.atomic_climb(_atoms(), FlatSurface(), reference_energy=-10.,
        config=_config(), rng=np.random.default_rng(3))
    assert len(solve_calls) == 1
    assert quench_calls == []
    assert result.status == "rotation_failed"
    event = result.climb[0]
    assert event["rotation_converged"] is False
    assert event["rotation_budget_released"] is False
    assert event["rotation_stop_reason"] == "budget_exhausted"


def test_budget_opt_in_releases_only_finite_evaluated_direction(monkeypatch):
    _solver(monkeypatch)
    quench_calls = _stub_quench(monkeypatch)
    result = climb.atomic_climb(_atoms(), FlatSurface(), reference_energy=-10.,
        config=_config(rotation_exit_policy="force_or_budget"),
        rng=np.random.default_rng(3))
    assert quench_calls == [True]
    event = result.climb[0]
    assert event["rotation_stop_reason"] == "budget_exhausted"
    assert event["rotation_converged"] is False
    assert event["rotation_budget_released"] is True
    assert result.status == "gaussian_limit"
    assert len(result.checkpoint.climb) == 1


def test_budget_release_does_not_bypass_biased_quench_convergence_gate(monkeypatch):
    _solver(monkeypatch)
    quench_calls = _stub_quench(monkeypatch, converged=False)
    result = climb.atomic_climb(_atoms(), FlatSurface(), reference_energy=-10.,
        config=_config(rotation_exit_policy="force_or_budget"),
        rng=np.random.default_rng(3))
    assert quench_calls == [True]
    assert result.status == "biased_quench_failed"
    assert result.climb[0]["rotation_converged"] is False
    assert result.climb[0]["rotation_budget_released"] is True
    assert "true_energy" not in result.climb[0]
    assert result.checkpoint.next_index == 0


@pytest.mark.parametrize("reason", ["subspace_exhausted", "unspecified", "rotation_limit"])
def test_only_verified_budget_stop_reason_can_release(monkeypatch, reason):
    _solver(monkeypatch, stop=reason)
    quench_calls = _stub_quench(monkeypatch)
    result = climb.atomic_climb(_atoms(), FlatSurface(), reference_energy=-10.,
        config=_config(rotation_exit_policy="force_or_budget"),
        rng=np.random.default_rng(3))
    assert result.status == "rotation_failed"
    assert quench_calls == []
    assert result.climb[0]["rotation_budget_released"] is False
    assert result.climb[0]["rotation_stop_reason"] == reason


@pytest.mark.parametrize("invalid_direction", ["nan", "infinite", "zero", "wrong_shape"])
def test_nonfinite_or_malformed_budget_direction_is_rejected_before_quench(monkeypatch, invalid_direction):
    direction = {"nan": np.array([[np.nan, 0., 0.], [0., 1., 0.]]),
        "infinite": np.array([[np.inf, 0., 0.], [0., 1., 0.]]),
        "zero": np.zeros((2, 3)), "wrong_shape": np.ones((1, 3))}[invalid_direction]
    _solver(monkeypatch, direction=direction)
    quench_calls = _stub_quench(monkeypatch)
    result = climb.atomic_climb(_atoms(), FlatSurface(), reference_energy=-10.,
        config=_config(rotation_exit_policy="force_or_budget"),
        rng=np.random.default_rng(3))
    assert result.status == "evaluation_failed"
    assert "invalid evaluated direction" in result.error
    assert quench_calls == []
    assert len(result.checkpoint.climb) == 0


def test_resume_keeps_saved_policy_and_does_not_resample_initial_direction(monkeypatch):
    cfg = _config(rotation_exit_policy="force_or_budget", max_gaussians=2)
    sample_calls = []
    original_sample = climb.sample_initial_direction
    def sample(atoms, rng, *, mode):
        sample_calls.append(mode)
        return original_sample(atoms, rng, mode=mode)
    monkeypatch.setattr(climb, "sample_initial_direction", sample)
    _solver(monkeypatch)
    quench_calls = _stub_quench(monkeypatch)
    surface = FlatSurface()
    first = climb.atomic_climb(_atoms(), surface, reference_energy=-10., config=cfg,
        rng=np.random.default_rng(3), max_completed_gaussians=1)
    assert first.status == "checkpoint_boundary"
    assert first.checkpoint.config.rotation_exit_policy == "force_or_budget"
    saved_direction = first.initial_direction.copy()
    before_resume_requests = surface.requests

    resumed = climb.resume_atomic_climb(first.checkpoint, surface)
    assert resumed.status == "gaussian_limit"
    assert sample_calls == ["global"]
    np.testing.assert_array_equal(resumed.initial_direction, saved_direction)
    assert surface.requests > before_resume_requests
    assert quench_calls == [True, True]
    assert all(event["rotation_budget_released"] is True for event in resumed.climb)
    assert all(event["rotation_converged"] is False for event in resumed.climb)


def test_resume_rejects_changed_rotation_exit_policy_before_surface_work(monkeypatch):
    cfg = _config(rotation_exit_policy="force_or_budget")
    _solver(monkeypatch)
    _stub_quench(monkeypatch)
    surface = FlatSurface()
    first = climb.atomic_climb(_atoms(), surface, reference_energy=-10., config=cfg,
        rng=np.random.default_rng(3), max_completed_gaussians=1)
    assert first.checkpoint.config.rotation_exit_policy == "force_or_budget"
    before = surface.requests
    with pytest.raises(ValueError, match="checkpoint config must match"):
        climb.resume_atomic_climb(first.checkpoint, surface,
                                  replace(cfg, rotation_exit_policy="force"))
    assert surface.requests == before
