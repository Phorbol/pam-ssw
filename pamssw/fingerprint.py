from __future__ import annotations

import numpy as np

from .state import State


def pair_distances(state: State) -> np.ndarray:
    distances: list[float] = []
    for atom_i in range(state.n_atoms):
        delta = state.positions[atom_i + 1 :] - state.positions[atom_i]
        if delta.size:
            distances.extend(np.linalg.norm(delta, axis=1).tolist())
    return np.asarray(distances, dtype=float)


def pair_distance_fingerprint(state: State) -> np.ndarray:
    return np.sort(pair_distances(state))


def rdf_histogram_fingerprint(state: State, n_bins: int = 16, r_max: float | None = None) -> np.ndarray:
    distances = pair_distances(state)
    if distances.size == 0:
        return np.zeros(n_bins, dtype=float)
    upper = float(r_max if r_max is not None else max(4.0, distances.max() + 1e-6))
    histogram, _ = np.histogram(distances, bins=n_bins, range=(0.0, upper))
    vector = histogram.astype(float)
    norm = np.linalg.norm(vector)
    if norm > 0.0:
        vector /= norm
    return vector


def contact_count(state: State, cutoff: float = 1.35) -> int:
    distances = pair_distances(state)
    if distances.size == 0:
        return 0
    return int(np.count_nonzero(distances <= cutoff))


def structural_descriptor(state: State) -> np.ndarray:
    distances = pair_distances(state)
    if distances.size == 0:
        stats = np.zeros(4, dtype=float)
    else:
        stats = np.array(
            [
                distances.min(),
                distances.mean(),
                distances.std(),
                contact_count(state) / max(1.0, state.n_atoms * (state.n_atoms - 1) / 2.0),
            ],
            dtype=float,
        )
    return np.concatenate([rdf_histogram_fingerprint(state), stats])


def descriptor_distance(lhs: np.ndarray, rhs: np.ndarray) -> float:
    if lhs.shape != rhs.shape:
        return float("inf")
    return float(np.linalg.norm(lhs - rhs))
