"""Tests for the private, nonperiodic S1 analytic reference."""
from __future__ import annotations

import numpy as np
from ase.build import molecule

from research.ga_ssw.s1_reference import (
    center_neighbors_value_gradient,
    radial_value_derivative,
)


def test_radial_derivative_matches_finite_difference_inside_guard():
    cutoff, exponent, epsilon = 6.48507, -1, 1.0e-5
    r = 1.37
    h = 1.0e-6
    plus = radial_value_derivative(r + h, cutoff, exponent, epsilon)[0]
    minus = radial_value_derivative(r - h, cutoff, exponent, epsilon)[0]
    derivative = radial_value_derivative(r, cutoff, exponent, epsilon)[1]
    np.testing.assert_allclose((plus - minus) / (2.0 * h), derivative, rtol=2e-6, atol=2e-8)


def test_water_geometry_has_full_center_neighbor_invariants():
    # A molecular geometry checks the algebra, not PES exploration efficacy.
    atoms = molecule("H2O")
    relative = atoms.positions[1:] - atoms.positions[0]
    value, center_gradient, neighbors_gradient = center_neighbors_value_gradient(
        relative, 6.48507, -1, 1.0e-5)
    assert np.isfinite(value)
    full_positions = atoms.positions.copy()
    full_gradient = np.vstack((center_gradient, neighbors_gradient))
    np.testing.assert_allclose(full_gradient.sum(axis=0), 0.0, atol=2e-15)
    torque = np.cross(full_positions, full_gradient).sum(axis=0)
    np.testing.assert_allclose(torque, 0.0, atol=2e-14)


def test_native_cutoff_guard_skips_boundary_pair():
    cutoff, epsilon = 6.48507, 1.0e-5
    value, derivative = radial_value_derivative(cutoff - 0.5 * epsilon,
                                                cutoff, -1, epsilon)
    assert value == 0.0
    assert derivative == 0.0
