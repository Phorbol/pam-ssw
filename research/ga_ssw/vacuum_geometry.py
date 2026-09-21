"""Geometry-only qualification for converting a large orthogonal PBC box.

The helper identifies a cut through the largest empty interval on each box
axis, moves atoms across that cut by integer cell translations, and returns a
centered ``pbc=False`` copy.  It proves only equality of a finite-cutoff
neighbor graph under the stated geometric conditions; it does not qualify a
structure chemically or support nonlocal calculators.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from ase import Atoms


def _machine_margin(cell: np.ndarray, cutoff: float) -> float:
    """Use only floating-point roundoff as comparison margin."""
    scale = max(1.0, float(np.max(np.abs(cell))), abs(float(cutoff)))
    return 8.0 * np.finfo(float).eps * scale


def _failure(reason: str, **extra: Any) -> tuple[None, dict[str, Any]]:
    diagnostics = {"eligible": False, "reason": reason}
    diagnostics.update(extra)
    return None, diagnostics


def _connected_components(atoms: Atoms, cutoff: float) -> int:
    """Count components of the original finite-cutoff MIC neighbor graph."""
    n = len(atoms)
    if n == 0:
        return 0
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    diagonal = np.diag(np.asarray(atoms.cell.array, dtype=float))
    delta = np.asarray(atoms.positions, dtype=float)[:, None, :] - np.asarray(atoms.positions, dtype=float)[None, :, :]
    scaled = delta / diagonal
    scaled -= np.round(scaled)
    distances = np.linalg.norm(scaled * diagonal, axis=2)
    for i in range(n):
        for j in range(i + 1, n):
            if distances[i, j] <= cutoff:
                union(i, j)
    return len({find(i) for i in range(n)})


def inspect_vacuum(atoms: Atoms, cutoff: float) -> tuple[Atoms | None, dict[str, Any]]:
    """Inspect and canonicalize a fully periodic orthorhombic cell.

    Returns ``(canonical, diagnostics)`` when the largest empty interval on
    every axis is strictly larger than ``cutoff``.  The canonical object has
    the same cell but ``pbc=False`` and is centered in the cell.  For any
    unsupported or geometrically ineligible input, returns ``(None, diag)``;
    no exception is needed for ordinary qualification failure.

    ``connected`` refers only to the graph formed by MIC distances at
    ``cutoff``.  It is recorded for downstream validation and is never a
    claim of chemical bonding or physical stability.
    """
    if not isinstance(atoms, Atoms):
        return _failure("atoms_must_be_ase_atoms")
    try:
        cutoff = float(cutoff)
    except (TypeError, ValueError):
        return _failure("cutoff_must_be_finite_nonnegative")
    if not math.isfinite(cutoff) or cutoff < 0.0:
        return _failure("cutoff_must_be_finite_nonnegative")
    cell = np.asarray(atoms.cell.array, dtype=float)
    if cell.shape != (3, 3) or not np.isfinite(cell).all():
        return _failure("cell_must_be_finite_3x3")
    if not bool(np.all(atoms.pbc)):
        return _failure("full_pbc_required")
    if atoms.constraints:
        return _failure("constraints_not_supported")
    if not np.isfinite(np.asarray(atoms.positions, dtype=float)).all():
        return _failure("positions_must_be_finite")
    margin = _machine_margin(cell, cutoff)
    diagonal = np.diag(cell)
    if np.any(diagonal <= margin) or np.any(np.abs(cell - np.diag(diagonal)) > margin):
        return _failure("orthorhombic_cell_required")
    if not math.isfinite(float(np.linalg.det(cell))) or float(np.linalg.det(cell)) <= 0.0:
        return _failure("cell_must_have_positive_volume")
    if len(atoms) == 0:
        return _failure("empty_atoms_not_supported")

    scaled = np.asarray(atoms.get_scaled_positions(wrap=True), dtype=float)
    scaled %= 1.0
    cuts = []
    gaps = []
    canonical_scaled = np.empty_like(scaled)
    for axis in range(3):
        values = np.sort(scaled[:, axis])
        cyclic_gaps = np.diff(np.r_[values, values[0] + 1.0])
        index = int(np.argmax(cyclic_gaps))
        largest = float(cyclic_gaps[index])
        gap_A = largest * float(diagonal[axis])
        gaps.append(gap_A)
        cut = float((values[index] + largest / 2.0) % 1.0)
        cuts.append(cut)
        canonical_scaled[:, axis] = (scaled[:, axis] - cut) % 1.0

    if any(gap <= cutoff + margin for gap in gaps):
        return _failure(
            "largest_gap_not_strictly_above_cutoff",
            largest_gap_A=gaps,
            cut_fraction=cuts,
            cutoff_A=cutoff,
            comparison_margin_A=margin,
        )

    # The largest-gap cut puts all atoms in one interval on each axis.  A
    # common translation then centers that interval without re-wrapping it.
    centered = canonical_scaled + (0.5 - np.mean(canonical_scaled, axis=0))
    canonical = atoms.copy()
    canonical.set_positions(centered @ cell)
    canonical.set_cell(cell)
    canonical.set_pbc(False)
    components = _connected_components(atoms, cutoff)
    diagnostics = {
        "eligible": True,
        "reason": "ok",
        "cutoff_A": cutoff,
        "comparison_margin_A": margin,
        "largest_gap_A": gaps,
        "cut_fraction": cuts,
        "connected_components": components,
        "connected": components == 1,
        "physical_qualified": False,
        "physical_qualification_note": "finite-cutoff MIC graph connectivity only; physical validity requires a separate check",
    }
    return canonical, diagnostics
