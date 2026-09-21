"""Minimal lifecycle contract for recovered CBD plus native height policy.

This is an EMT interface test, not a search-quality or material-efficacy test.
"""

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT

from pamssw.standalone import ASESurface, RecoveredRotationSettings, SSWConfig, run_ssw
from pamssw.standalone.native_height_policy import ConservativeNativeHeightPolicy


def _atoms():
    return Atoms("Cu4", positions=[
        [0.00, 0.00, 0.00], [2.35, 0.00, 0.10],
        [0.30, 2.20, 0.00], [0.15, 0.25, 2.45],
    ])


def _config():
    return SSWConfig(width=0.08, rotation_bias=1.0, max_gaussians=2,
                     temperature_K=300.0, fmax=0.05, relax_steps=80,
                     fd_step=1e-3, rotation_hvp=6, rotation_tol=1e-2,
                     direction_sampling="global", cluster_frame="direction_only",
                     rotation_exit_policy="force_or_budget")


def _rotation():
    return RecoveredRotationSettings(pre_rotmax=1, rotmax=2,
                                     pre_ftol=10.0, ftol=10.0,
                                     metric="euclidean", max_force_calls=8)


def test_recovered_rotation_and_native_height_share_curvature_and_frozen_terms():
    policy = ConservativeNativeHeightPolicy(2.0, 0.2, 1, 10.0, 1.05, 1.25)
    result = run_ssw(_atoms(), ASESurface(EMT()), steps=1, config=_config(),
                     rng=np.random.default_rng(7), recovered_rotation=_rotation(),
                     height_policy=policy)
    events = [event for record in result.records for event in record.climb
              if isinstance(event, dict) and "height_preparation" in event]
    assert events, "combined lifecycle did not reach height preparation"
    for event in events:
        telemetry = event["recovered_rotation"]
        preparation = event["height_preparation"]
        assert "real_curvature" in telemetry
        assert preparation.curvature_input == pytest.approx(telemetry["real_curvature"], abs=1e-12)
        assert preparation.curvature_scope == "physical PES; rotation-only bias excluded"
        assert all(term.center.flags.writeable is False and term.direction.flags.writeable is False
                   for term in preparation.terms)

        point = (np.asarray(event["center"], dtype=float) +
                 float(event["width"]) * np.asarray(event["direction"], dtype=float)).ravel()
        term_energy = sum(term.evaluate(point)[0] for term in preparation.terms)
        term_force = sum((term.evaluate(point)[1] for term in preparation.terms), np.zeros_like(point))
        assert preparation.bias_energy == pytest.approx(term_energy, abs=1e-12)
        for coordinate in (0, min(3, point.size - 1)):
            step = 1e-6
            plus = point.copy(); plus[coordinate] += step
            minus = point.copy(); minus[coordinate] -= step
            fd = (sum(term.evaluate(plus)[0] for term in preparation.terms) -
                  sum(term.evaluate(minus)[0] for term in preparation.terms)) / (2.0 * step)
            assert term_force[coordinate] == pytest.approx(-fd, abs=2e-8)
