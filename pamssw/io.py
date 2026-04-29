from __future__ import annotations

from pathlib import Path
from typing import Any

from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import read, write
import numpy as np

from .state import State


def state_from_atoms(
    atoms: Atoms,
    *,
    fixed_mask: np.ndarray | list[bool] | None = None,
    metadata: dict[str, Any] | None = None,
) -> State:
    """Convert an ASE Atoms object into a PAM-SSW State."""
    mask = _fixed_mask_from_atoms(atoms) if fixed_mask is None else np.asarray(fixed_mask, dtype=bool)
    cell = atoms.cell.array.copy() if atoms.cell.rank > 0 else None
    return State(
        numbers=atoms.numbers.copy(),
        positions=atoms.positions.copy(),
        cell=cell,
        pbc=tuple(bool(value) for value in atoms.pbc),
        fixed_mask=mask,
        metadata={} if metadata is None else dict(metadata),
    )


def read_state(
    path: str | Path,
    *,
    index: int | str | None = None,
    format: str | None = None,
    fixed_mask: np.ndarray | list[bool] | None = None,
    metadata: dict[str, Any] | None = None,
    **read_kwargs: Any,
) -> State:
    """Read any ASE-supported structure file as a PAM-SSW State."""
    if index is None:
        atoms = read(path, format=format, **read_kwargs)
    else:
        atoms = read(path, index=index, format=format, **read_kwargs)
    if not isinstance(atoms, Atoms):
        raise ValueError("read_state expects a single structure; pass a single index for multi-frame files")
    return state_from_atoms(atoms, fixed_mask=fixed_mask, metadata=metadata)


def state_to_atoms(state: State) -> Atoms:
    """Convert a PAM-SSW State into an ASE Atoms object."""
    atoms = Atoms(
        numbers=state.numbers,
        positions=state.positions,
        cell=state.cell,
        pbc=state.pbc,
    )
    if np.any(state.fixed_mask):
        atoms.set_constraint(FixAtoms(mask=state.fixed_mask))
    atoms.info.update(state.metadata)
    return atoms


def write_state(path: str | Path, state: State, *, format: str | None = None, **write_kwargs: Any) -> None:
    """Write a PAM-SSW State through ASE."""
    write(path, state_to_atoms(state), format=format, **write_kwargs)


def _fixed_mask_from_atoms(atoms: Atoms) -> np.ndarray:
    mask = np.zeros(len(atoms), dtype=bool)
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            mask[np.asarray(constraint.get_indices(), dtype=int)] = True
    return mask
