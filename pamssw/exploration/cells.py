from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .actions import PolicySnapshot


@dataclass(frozen=True)
class FPSCellPartition:
    """A full-support, equal-cell-mass partition of starter features."""

    starter_ids: tuple[int, ...]
    center_ids: tuple[int, ...]
    members_by_center: tuple[tuple[int, ...], ...]
    probabilities: tuple[float, ...]

    def policy_snapshot(self, *, version: int, archive_version: int) -> PolicySnapshot:
        return PolicySnapshot(
            version=version,
            archive_version=archive_version,
            policy_name="fps_cell_uniform",
            eligible_starter_ids=self.starter_ids,
            probabilities=self.probabilities,
            support_complete=True,
        )

    def probability_for(self, starter_id: int) -> float:
        return self.probabilities[self.starter_ids.index(starter_id)]

    def cell_probability_for(self, starter_id: int) -> float:
        self._cell_index(starter_id)
        return 1.0 / len(self.center_ids)

    def conditional_probability_for(self, starter_id: int) -> float:
        members = self.members_by_center[self._cell_index(starter_id)]
        return 1.0 / len(members)

    def _cell_index(self, starter_id: int) -> int:
        for index, members in enumerate(self.members_by_center):
            if starter_id in members:
                return index
        raise KeyError(f"unknown starter_id: {starter_id!r}")


def build_fps_cell_partition(
    starter_ids: tuple[int, ...],
    features: np.ndarray,
    *,
    max_cells: int,
) -> FPSCellPartition:
    """Partition starters around deterministic farthest-point centers.

    The smallest starter ID is the first center.  Every nonempty cell receives
    equal probability and its members share that mass uniformly.
    """
    ids = tuple(starter_ids)
    matrix = np.asarray(features, dtype=float)
    if not ids:
        raise ValueError("starter_ids cannot be empty")
    if len(set(ids)) != len(ids):
        raise ValueError("starter_ids must be unique")
    if matrix.ndim != 2 or matrix.shape[0] != len(ids):
        raise ValueError("features must be a two-dimensional row-aligned matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("features must be finite")
    if isinstance(max_cells, bool) or not isinstance(max_cells, int) or max_cells <= 0:
        raise ValueError("max_cells must be a positive integer")

    canonical_order = np.argsort(np.asarray(ids), kind="stable")
    sorted_ids = tuple(ids[index] for index in canonical_order)
    sorted_features = matrix[canonical_order]

    center_indices = [0]
    minimum_squared_distance = _squared_distances(sorted_features, sorted_features[0])
    while len(center_indices) < min(max_cells, len(sorted_ids)):
        minimum_squared_distance[center_indices] = -1.0
        next_index = int(np.argmax(minimum_squared_distance))
        if minimum_squared_distance[next_index] <= 0.0:
            break
        center_indices.append(next_index)
        minimum_squared_distance = np.minimum(
            minimum_squared_distance,
            _squared_distances(sorted_features, sorted_features[next_index]),
        )

    centers = sorted_features[center_indices]
    assignments = np.argmin(
        np.sum((sorted_features[:, None, :] - centers[None, :, :]) ** 2, axis=2),
        axis=1,
    )
    members_by_center = tuple(
        tuple(
            starter_id
            for starter_id, assignment in zip(sorted_ids, assignments, strict=True)
            if assignment == center_index
        )
        for center_index in range(len(center_indices))
    )
    cell_count = len(members_by_center)
    probability_by_id = {
        starter_id: 1.0 / cell_count / len(members)
        for members in members_by_center
        for starter_id in members
    }
    return FPSCellPartition(
        starter_ids=sorted_ids,
        center_ids=tuple(sorted_ids[index] for index in center_indices),
        members_by_center=members_by_center,
        probabilities=tuple(probability_by_id[starter_id] for starter_id in sorted_ids),
    )


def _squared_distances(matrix: np.ndarray, point: np.ndarray) -> np.ndarray:
    delta = matrix - point
    return np.einsum("ij,ij->i", delta, delta)


__all__ = ["FPSCellPartition", "build_fps_cell_partition"]
