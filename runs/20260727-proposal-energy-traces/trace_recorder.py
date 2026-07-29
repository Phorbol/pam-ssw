"""Zero-extra-call tracing for frozen proposal-relaxation evaluations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from hashlib import sha256
from typing import Any

import numpy as np

from pamssw.state import State
from pamssw.walker import ProposalPotential


def position_hash(state_or_positions: State | np.ndarray) -> str:
    """Return a stable SHA-256 digest for Cartesian coordinates in Angstrom."""

    positions = state_or_positions.positions if isinstance(state_or_positions, State) else state_or_positions
    coordinates = np.asarray(positions, dtype=np.dtype("<f8"))
    if coordinates.ndim == 1:
        if coordinates.size % 3:
            raise ValueError("flat positions must contain a multiple of three coordinates")
        coordinates = coordinates.reshape(-1, 3)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("positions must have shape (n_atoms, 3)")
    canonical = np.array(coordinates, dtype=np.dtype("<f8"), order="C", copy=True)
    digest = sha256()
    digest.update(str(canonical.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical.tobytes())
    return digest.hexdigest()


class RecordingProposalPotential(ProposalPotential):
    """Proposal potential that records the components of its existing evaluations."""

    def __init__(self, calculator, biases=None, softening=None) -> None:
        super().__init__(calculator, biases=biases, softening=softening)
        self.records: list[dict[str, float | int | str]] = []

    def evaluate_parts(self, flat_positions: np.ndarray, template: State):
        evaluation = super().evaluate_parts(flat_positions, template)
        positions = np.asarray(flat_positions, dtype=float).reshape(template.n_atoms, 3)
        total_gradient = np.asarray(evaluation.total_gradient, dtype=float).reshape(template.n_atoms, 3)
        active_gradient = total_gradient[template.movable_mask]
        active_max_force = (
            0.0
            if active_gradient.size == 0
            else float(np.linalg.norm(active_gradient, axis=1).max())
        )
        self.records.append(
            {
                "evaluation_index": len(self.records) + 1,
                "positions_sha256": position_hash(positions),
                "true_energy_eV": float(evaluation.true_energy),
                "bias_energy_eV": float(evaluation.bias_energy),
                "softening_energy_eV": float(evaluation.softening_energy),
                "total_energy_eV": float(evaluation.total_energy),
                "active_max_total_force_eV_per_A": active_max_force,
            }
        )
        return evaluation


def mark_accepted_state_evaluations(
    records: Iterable[Mapping[str, Any]],
    accepted_states_or_hashes: Iterable[State | np.ndarray | str] | State | np.ndarray | str,
) -> list[dict[str, Any]]:
    """Copy records and label only coordinate hashes observed by a trajectory callback."""

    if isinstance(accepted_states_or_hashes, (State, np.ndarray, str)):
        accepted_items = (accepted_states_or_hashes,)
    else:
        accepted_items = accepted_states_or_hashes
    accepted_hashes = {
        item if isinstance(item, str) else position_hash(item)
        for item in accepted_items
    }
    marked_records: list[dict[str, Any]] = []
    for record in records:
        marked = deepcopy(dict(record))
        marked["accepted_state"] = marked.get("positions_sha256") in accepted_hashes
        marked_records.append(marked)
    return marked_records
