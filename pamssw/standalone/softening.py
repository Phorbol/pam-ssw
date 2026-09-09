"""Independent, fixed-cell LS pair potential and paper response controller.

Reference: Guan, Shang and Liu, JCTC (2024), DOI 10.1021/acs.jctc.4c01081,
sections 2.3–2.4, equations 11–15. The published xi=0.2 is dimensionless;
the initial pair strength is 3% of a supplied standard bond energy and the
response learning rate is 1.8. These are paper settings, not universal optima.

This module never evaluates a physical calculator or runs an optimizer. The
caller must pre-relax E+V_LS, measure E (without LS) before and after that
pre-relaxation, retain the frozen potential throughout one SSW walk, and
remove all biases for final quenching and Metropolis decisions.

Bond tables are explicit mappings (atomic_number, atomic_number) -> value.
`bond_energies` contains positive standard bond energies in eV. `bond_lengths`
contains positive *maximum bonding distances* in Angstrom, not equilibrium
lengths: a pair is selected when its initial MIC distance <= the table value.
No element table, covalent-radius rule, tolerance, cutoff or fallback is
invented here. The caller owns the provenance of its bond selection rule.
Only numeric/contract tests have been performed; scientific validity is untested.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping
import math

import numpy as np
from ase import Atoms
from ase.geometry import find_mic

BondTable = Mapping[tuple[int, int], float]


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return value


def _table(table: BondTable) -> dict[tuple[int, int], float]:
    result = {}
    for pair, value in table.items():
        if len(pair) != 2 or any(int(z) != z or z <= 0 for z in pair):
            raise ValueError('bond table keys must be two positive atomic numbers')
        key = tuple(sorted(map(int, pair)))
        value = _positive(value, 'bond table value')
        if key in result and result[key] != value:
            raise ValueError(f'conflicting bond table entries for {key}')
        result[key] = value
    return result


def _geometry(atoms: Atoms):
    if len(atoms) < 2:
        raise ValueError('at least two atoms are required to define bond pairs')
    if not np.isfinite(atoms.positions).all() or not np.isfinite(atoms.cell).all():
        raise ValueError('positions and cell must be finite')
    if any(atoms.pbc) and np.any(np.linalg.norm(atoms.cell.array, axis=1)[atoms.pbc] == 0):
        raise ValueError('periodic cell vectors must be nonzero')
    return (tuple(map(int, atoms.numbers)),
            tuple(tuple(map(float, row)) for row in atoms.cell.array),
            tuple(map(bool, atoms.pbc)))


@dataclass(frozen=True)
class FrozenBondSoftening:
    """Immutable pair list, reference distances and strengths for one SSW step.

    Use `from_atoms` to freeze neighbors at a true minimum. `evaluate` returns
    the additional energy (eV) and Cartesian forces (eV/Angstrom); it does not
    call `atoms.calc`. Coordinates may move across the periodic boundary but
    species/order, PBC and cell must remain unchanged. MIC branch boundaries
    themselves are not smooth; callers must avoid them in derivative tests.
    """
    numbers: tuple[int, ...]
    cell: tuple[tuple[float, float, float], ...]
    pbc: tuple[bool, bool, bool]
    pairs: tuple[tuple[int, int], ...]
    reference_distances: tuple[float, ...]
    strengths: tuple[float, ...]
    xi: float = 0.2

    def __post_init__(self):
        # A frozen dataclass alone does not own mutable ndarray/list inputs.
        # Canonical immutable values also keep ASE's cached bias meaningful.
        raw_numbers = tuple(self.numbers)
        if len(raw_numbers) < 2 or any(isinstance(z, (bool, np.bool_)) or
                                      not isinstance(z, (int, np.integer)) or z <= 0
                                      for z in raw_numbers):
            raise ValueError('numbers must contain at least two positive atomic integers')
        numbers = tuple(map(int, raw_numbers))
        cell = np.asarray(self.cell, dtype=float)
        pbc = tuple(self.pbc)
        if cell.shape != (3, 3) or not np.isfinite(cell).all():
            raise ValueError('cell must be a finite 3 by 3 matrix')
        if len(pbc) != 3 or any(not isinstance(v, (bool, np.bool_)) for v in pbc):
            raise ValueError('periodic boundary flags must be three booleans')
        if any(np.linalg.norm(cell[axis]) == 0 for axis in range(3) if pbc[axis]):
            raise ValueError('periodic cell vectors must be nonzero')
        pairs = tuple(tuple(pair) for pair in self.pairs)
        if not pairs:
            raise ValueError('no bonded pairs: LS response is undefined')
        if any(len(pair) != 2 or any(isinstance(i, (bool, np.bool_)) or
                                    not isinstance(i, (int, np.integer)) or not 0 <= i < len(numbers)
                                    for i in pair) or pair[0] == pair[1] for pair in pairs):
            raise ValueError('pairs must contain distinct valid zero-based atom indices')
        pairs = tuple(tuple(sorted(map(int, pair))) for pair in pairs)
        if len(set(pairs)) != len(pairs):
            raise ValueError('duplicate unordered pairs are not allowed')
        references = tuple(_positive(r, 'reference distance') for r in self.reference_distances)
        strengths = tuple(map(float, self.strengths))
        if len(pairs) != len(references) or len(pairs) != len(strengths):
            raise ValueError('pairs, reference distances and strengths must have equal lengths')
        if any(not math.isfinite(a) or a < 0 for a in strengths):
            raise ValueError('pair strength must be finite and nonnegative')
        for name, value in [('numbers', numbers), ('cell', tuple(tuple(map(float, row)) for row in cell)),
                            ('pbc', tuple(map(bool, pbc))), ('pairs', pairs),
                            ('reference_distances', references), ('strengths', strengths),
                            ('xi', _positive(self.xi, 'xi'))]:
            object.__setattr__(self, name, value)

    @classmethod
    def from_atoms(cls, atoms: Atoms, *, bond_energies: BondTable,
                   bond_lengths: BondTable, initial_fraction: float = 0.03,
                   xi: float = 0.2) -> FrozenBondSoftening:
        """Freeze pairs and set A_pq = initial_fraction * bond_energy_pq.

        Every species pair encountered must exist in both caller-supplied
        tables, including nonbonded pairs; missing data is never treated as
        evidence that a species cannot bond. Actual r0 is taken from atoms.
        """
        initial_fraction = _positive(initial_fraction, 'initial_fraction')
        return cls._build(atoms, bond_energies, bond_lengths, xi,
                          initial_fraction=initial_fraction)

    @classmethod
    def _build(cls, atoms, bond_energies, bond_lengths, xi, *,
               initial_fraction=None, total_strength=None):
        numbers, cell, pbc = _geometry(atoms)
        energies, lengths = _table(bond_energies), _table(bond_lengths)
        pairs, distances, weights = [], [], []
        for i in range(len(atoms)-1):
            for j in range(i+1, len(atoms)):
                key = tuple(sorted((numbers[i], numbers[j])))
                if key not in energies or key not in lengths:
                    raise ValueError(f'missing bond table entry for {key}')
                _, distance = find_mic(atoms.positions[j]-atoms.positions[i], atoms.cell, atoms.pbc)
                distance = float(distance)
                if distance <= 0:
                    raise ValueError('pair distance must be positive')
                if distance <= lengths[key]:
                    pairs.append((i, j))
                    distances.append(distance)
                    weights.append(energies[key])
        if not pairs:
            raise ValueError('no bonded pairs: LS response is undefined')
        if total_strength is None:
            strengths = tuple(initial_fraction * energy for energy in weights)
        else:
            strengths = tuple(total_strength * energy / math.fsum(weights) for energy in weights)
        return cls(numbers, cell, pbc, tuple(pairs), tuple(distances), strengths, xi)

    def _validate_atoms(self, atoms):
        numbers, cell, pbc = _geometry(atoms)
        if numbers != self.numbers:
            raise ValueError('atom identity/order differs from frozen reference')
        if pbc != self.pbc:
            raise ValueError('periodic boundary conditions differ from frozen reference')
        if cell != self.cell:
            raise ValueError('cell differs from fixed-cell frozen reference')

    def evaluate(self, atoms: Atoms) -> tuple[float, np.ndarray]:
        self._validate_atoms(atoms)
        forces = np.zeros((len(atoms), 3))
        energies = []
        for (i, j), r0, strength in zip(self.pairs, self.reference_distances, self.strengths):
            delta, distance = find_mic(atoms.positions[j]-atoms.positions[i], atoms.cell, atoms.pbc)
            distance = float(distance)
            if distance <= 0:
                raise ValueError('pair distance must be positive')
            scale = self.xi*r0
            energy = strength * math.exp(-(distance-r0)/scale)
            outward = (energy/scale) * delta/distance
            forces[i] -= outward
            forces[j] += outward
            energies.append(energy)
        return float(math.fsum(energies)), forces


@dataclass
class LSResponseState:
    """Across-step LS strength controller using *true* pre-relaxation energies.

    `target_per_atom` is the paper's Upsilon in eV/atom, not V_LS/N or kT.
    At each update, current is the potential used for the completed pre-relax,
    and next_atoms is the next true minimum (after the walk/acceptance decision).
    `energy_before` and `energy_after` must refer to the physical PES at the
    original current-step minimum and its soft-only pre-relaxed structure.

    For unordered pairs, total_A_next = total_A_current - N*learning_rate*
    (true_energy_rise/N - target). Redistribute total_A_next over the next
    neighbors proportional to their supplied standard bond energies (eq.15).
    Negative strengths/empty neighbors fail explicitly; no clipping or rescue.
    State changes only after a valid next potential has been constructed.
    """
    target_per_atom: float
    learning_rate: float = 1.8
    steps: int = field(default=0, init=False)
    last_response: float | None = field(default=None, init=False)

    def __post_init__(self):
        self.target_per_atom = _positive(self.target_per_atom, 'target_per_atom')
        self.learning_rate = _positive(self.learning_rate, 'learning_rate')

    def update(self, current: FrozenBondSoftening, next_atoms: Atoms, *,
               energy_before: float, energy_after: float,
               bond_energies: BondTable, bond_lengths: BondTable) -> FrozenBondSoftening:
        current._validate_atoms(next_atoms)
        if not math.isfinite(energy_before) or not math.isfinite(energy_after):
            raise ValueError('true pre-relaxation energies must be finite')
        response = (energy_after-energy_before)/len(current.numbers)
        total = math.fsum(current.strengths) - len(current.numbers)*self.learning_rate*(response-self.target_per_atom)
        if not math.isfinite(total) or total < 0:
            raise ValueError('response update produces a negative/nonfinite strength; no clipping applied')
        result = FrozenBondSoftening._build(next_atoms, bond_energies, bond_lengths,
                                           current.xi, total_strength=total)
        self.steps += 1
        self.last_response = float(response)
        return result
