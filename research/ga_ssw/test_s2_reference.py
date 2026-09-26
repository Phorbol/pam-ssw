"""Tests for the paper-derived, nonperiodic S2 research primitive."""
from __future__ import annotations

import math

import numpy as np
import pytest

from research.ga_ssw.s2_reference import center_neighbors_value_gradient
from research.ga_ssw.s1_reference import center_neighbors_value_gradient as s1_value_gradient


def _geometry():
    # Explicit noncollinear C/H-like neighbor vectors, in Angstrom.
    return np.array([
        [1.08, 0.12, -0.04],
        [-0.31, 1.24, 0.19],
        [0.27, -0.42, 1.37],
        [-1.16, -0.73, 0.51],
    ])


def _value(vectors, degree=3):
    return center_neighbors_value_gradient(
        vectors, degree, exponent=1, cutoff=4.2, epsilon=1.0e-6)[0]


def test_s2_analytic_gradients_match_finite_differences():
    vectors = _geometry()
    _, center, neighbors = center_neighbors_value_gradient(
        vectors, 3, exponent=1, cutoff=4.2, epsilon=1.0e-6)
    h = 2.0e-6
    numeric = np.empty_like(vectors)
    for j in range(len(vectors)):
        for axis in range(3):
            plus = vectors.copy(); plus[j, axis] += h
            minus = vectors.copy(); minus[j, axis] -= h
            numeric[j, axis] = (_value(plus) - _value(minus)) / (2.0 * h)
    np.testing.assert_allclose(neighbors, numeric, rtol=2e-6, atol=2e-8)
    np.testing.assert_allclose(center, -numeric.sum(axis=0), rtol=2e-6, atol=2e-8)


def test_s2_gradient_is_translation_and_rotation_covariant():
    vectors = _geometry()
    value, center, neighbors = center_neighbors_value_gradient(
        vectors, 2, exponent=1, cutoff=4.2, epsilon=1.0e-6)
    full_gradient = np.vstack((center, neighbors))
    np.testing.assert_allclose(full_gradient.sum(axis=0), 0.0, atol=2e-14)
    torque = np.cross(vectors, neighbors).sum(axis=0)
    np.testing.assert_allclose(torque, 0.0, atol=2e-13)

    angle = 0.73
    rotation = np.array([
        [math.cos(angle), -math.sin(angle), 0.0],
        [math.sin(angle), math.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    rotated_value, rotated_center, rotated_neighbors = center_neighbors_value_gradient(
        vectors @ rotation.T, 2, exponent=1, cutoff=4.2, epsilon=1.0e-6)
    np.testing.assert_allclose(rotated_value, value, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(rotated_center, center @ rotation.T, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(rotated_neighbors, neighbors @ rotation.T,
                               rtol=1e-12, atol=1e-13)


def test_degree_zero_is_s1_with_addition_theorem_normalization():
    vectors = _geometry()
    s2, _, _ = center_neighbors_value_gradient(
        vectors, 0, exponent=1, cutoff=4.2, epsilon=1.0e-6)
    s1, _, _ = s1_value_gradient(vectors, 4.2, 1, 1.0e-6)
    np.testing.assert_allclose(s2, s1 / math.sqrt(4.0 * math.pi), rtol=1e-14, atol=1e-14)


def test_zero_descriptor_is_an_explicit_domain_error():
    outside = np.array([[5.0, 0.0, 0.0], [0.0, 5.1, 0.0]])
    with pytest.raises(ValueError, match="zero S2 descriptor"):
        center_neighbors_value_gradient(
            outside, 2, exponent=1, cutoff=4.2, epsilon=1.0e-6)


def test_zero_norm_and_nonfinite_neighbor_vectors_are_rejected():
    with pytest.raises(ValueError, match="nonzero norms"):
        center_neighbors_value_gradient(
            np.array([[0.0, 0.0, 0.0]]), 2,
            exponent=1, cutoff=4.2, epsilon=1.0e-6)
    with pytest.raises(ValueError, match=r"finite \(M,3\)"):
        center_neighbors_value_gradient(
            np.array([[np.nan, 0.0, 1.0]]), 2,
            exponent=1, cutoff=4.2, epsilon=1.0e-6)


def test_degree_must_be_a_nonnegative_integer():
    vectors = _geometry()
    for degree in (-1, 1.5, True):
        with pytest.raises(ValueError, match="degree"):
            center_neighbors_value_gradient(
                vectors, degree, exponent=1, cutoff=4.2, epsilon=1.0e-6)
