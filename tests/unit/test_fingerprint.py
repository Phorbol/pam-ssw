import numpy as np

from pamssw.fingerprint import (
    contact_count,
    pair_distance_fingerprint,
    rdf_histogram_fingerprint,
)
from pamssw.state import State


def _tetrahedron() -> State:
    return State(
        numbers=np.full(4, 18),
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 0.8660254, 0.0],
                [0.5, 0.2886751, 0.8164966],
            ]
        ),
    )


def test_pair_and_rdf_fingerprints_are_translation_rotation_invariant():
    base = _tetrahedron()
    angle = np.deg2rad(41.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    moved = State(
        numbers=base.numbers.copy(),
        positions=base.positions @ rotation.T + np.array([3.0, -2.0, 0.5]),
    )

    np.testing.assert_allclose(pair_distance_fingerprint(base), pair_distance_fingerprint(moved))
    np.testing.assert_allclose(rdf_histogram_fingerprint(base), rdf_histogram_fingerprint(moved))


def test_rdf_and_contact_descriptors_distinguish_compact_and_elongated_clusters():
    compact = _tetrahedron()
    elongated = State(
        numbers=np.full(4, 18),
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ]
        ),
    )

    assert contact_count(compact) > contact_count(elongated)
    assert np.linalg.norm(rdf_histogram_fingerprint(compact) - rdf_histogram_fingerprint(elongated)) > 0.25
