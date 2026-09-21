"""Independent TYPE0 atomic/alloy crossover and mutation from uploaded sgn.jar.

Sources: ga_cluster_cell/{Cut,CutBasicAbstract,Cross,Compete,Mutate}.java and TYPE0. This is
an experimental geometric proposal, not a relaxation, complete GA controller,
or a demonstrated global optimizer. No calculator or Java subprocess is called.
The original nonuniform rotations, 0.3 Angstrom half-separation and default
10**(number_of_elements+1) cuts per parent slot are compatibility rules, not
universal physical parameters. Parents are copied; infinite native sampling
loops are bounded explicitly. No collision repair or fake fallback is applied. Reinsertion explicitly corrects
the released JAR collision-predicate direction; see type0-mutation-contract.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from ase import Atoms

from .ga_operators import (SamplingExhausted, _competition_cumulative,
                           _rotation, rotate_coordinates)


@dataclass(frozen=True)
class AtomicGamete:
    atoms: Atoms
    parent_index: int
    source_atom_indices: tuple[int, ...]


@dataclass(frozen=True)
class AtomicCut:
    son: AtomicGamete
    daughter: AtomicGamete
    plane_slope: float
    attempts: int


@dataclass(frozen=True)
class AtomicPool:
    sons: tuple[AtomicGamete, ...]
    daughters: tuple[AtomicGamete, ...]
    composition: tuple[int, ...]
    parent_slots: tuple[int, ...]
    cuts_per_parent_slot: int
    cut_attempts: int


@dataclass(frozen=True)
class AtomicChild:
    atoms: Atoms
    parent_indices: tuple[int, ...]
    source_atom_indices: tuple[int, ...]
    son_pool_index: int
    daughter_pool_index: int
    pair_attempts: int


def _positive_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


def _validate_atoms(atoms):
    if len(atoms) < 2 or atoms.pbc.any() or atoms.constraints:
        raise ValueError('atomic crossover needs >=2 unconstrained nonperiodic atoms')
    if not np.isfinite(atoms.positions).all() or np.any(atoms.numbers <= 0):
        raise ValueError('coordinates must be finite and atomic numbers positive')


def cut_atoms(atoms: Atoms, rng: np.random.Generator, *, max_attempts: int,
              parent_index: int = -1) -> AtomicCut:
    """Cut a centered copy into near-equal halves; retain species and lineage.

    Rotate cumulatively across unsuccessful attempts, then classify by
    z+x/pl >= 0. Align the plane by the source row-vector y rotation and
    displace positive/negative halves by +/-0.3 Angstrom along z. The source
    centers its parent in-place; the output here is equivalent up to roundoff
    while never mutating the caller. A singular undefined cut fails explicitly.
    """
    _validate_atoms(atoms)
    max_attempts = _positive_int(max_attempts, 'max_attempts')
    coordinates = atoms.positions - atoms.positions.mean(axis=0)
    for attempt in range(1, max_attempts + 1):
        coordinates = rotate_coordinates(coordinates, rng)
        slope = float(2. * (rng.random() - .5))
        with np.errstate(divide='ignore', invalid='ignore'):
            height = coordinates[:, 2] + coordinates[:, 0] / slope
        positive, negative = height >= 0., height < 0.
        if abs(int(positive.sum()) - int(negative.sum())) < 2:
            if not (positive | negative).all():
                raise SamplingExhausted('singular legacy cut has undefined atom membership')
            break
    else:
        raise SamplingExhausted(f'balanced atomic cut not found in {max_attempts} attempts')
    with np.errstate(divide='ignore'):
        angle = float(np.arctan(np.divide(-1., slope)))
    aligned = coordinates @ _rotation(1, angle)

    def gamete(mask, separation):
        ids = tuple(int(i) for i in np.flatnonzero(mask))
        xyz = aligned[mask].copy()
        xyz[:, 2] += separation
        # Native AtoCoo carries only number and position. Do not silently carry
        # index-bound per-atom metadata from incompatible parents into children.
        fragment = Atoms(numbers=atoms.numbers[mask], positions=xyz)
        return AtomicGamete(fragment, parent_index, ids)

    return AtomicCut(gamete(positive, .3), gamete(negative, -.3), slope, attempt)


def build_atomic_pool(parents: Sequence[Atoms], energies: Sequence[float],
                      rng: np.random.Generator, *, max_cut_attempts: int,
                      cuts_per_parent_slot: int | None = None) -> AtomicPool:
    """Cross.genePool, with explicit sampling budget and parent provenance.

    All parents must share composition, not atom ordering. Compete requires
    >2 parents and nonzero energy span. Both index-column draws are consumed,
    but only column zero seeds cuts, exactly as the native source. An explicit
    cuts_per_parent_slot changes native sampling density and must be recorded.
    Parent energies are supplied metadata; no energy evaluations are performed.
    """
    if len(parents) == 0:
        raise ValueError('nonempty parents required')
    energy = np.asarray(energies, dtype=float)
    if energy.shape != (len(parents),) or not np.isfinite(energy).all():
        raise ValueError('one finite energy per parent required')
    max_cut_attempts = _positive_int(max_cut_attempts, 'max_cut_attempts')
    for atoms in parents:
        _validate_atoms(atoms)
    composition = tuple(sorted(int(z) for z in parents[0].numbers))
    if any(tuple(sorted(int(z) for z in a.numbers)) != composition for a in parents):
        raise ValueError('all parents must have identical composition')
    cuts = 10 ** (len(set(composition)) + 1) if cuts_per_parent_slot is None else cuts_per_parent_slot
    cuts = _positive_int(cuts, 'cuts_per_parent_slot')
    cumulative = _competition_cumulative(energy)
    slots = []
    for _ in parents:
        pair = [int(np.searchsorted(cumulative, rng.random() * cumulative[-1], side='right'))
                for _ in range(2)]
        slots.append(pair[0])
    sons, daughters = [], []
    attempts = 0
    # Native cuts recenter Model in-place. Keep this repeated-centering behavior
    # on private copies so later cuts follow the same floating-point operations.
    private = [a.copy() for a in parents]
    for index in slots:
        for _ in range(cuts):
            cut = cut_atoms(private[index], rng, max_attempts=max_cut_attempts, parent_index=index)
            private[index].positions -= private[index].positions.mean(axis=0)
            sons.append(cut.son)
            daughters.append(cut.daughter)
            attempts += cut.attempts
    return AtomicPool(tuple(sons), tuple(daughters), composition, tuple(slots), cuts, attempts)


def cross_atomic_pool(pool: AtomicPool, rng: np.random.Generator, *,
                      max_pair_attempts: int) -> AtomicChild:
    """Uniformly pair independent gametes until exact composition matches.

    Same-parent pairing is allowed by Cross.joint. Child order is son then
    daughter, without sorting. No overlap filter, docking, quench, artificial
    cell or energy-based acceptance is hidden here: these belong to subsequent
    caller stages. Exhaustion raises SamplingExhausted without fabricated child.
    """
    max_pair_attempts = _positive_int(max_pair_attempts, 'max_pair_attempts')
    if not pool.sons or not pool.daughters:
        raise ValueError('both gamete pools must be nonempty')
    for attempt in range(1, max_pair_attempts + 1):
        i = int(np.floor(rng.random() * len(pool.sons)))
        j = int(np.floor(rng.random() * len(pool.daughters)))
        son, daughter = pool.sons[i], pool.daughters[j]
        numbers = np.concatenate((son.atoms.numbers, daughter.atoms.numbers))
        if tuple(sorted(int(z) for z in numbers)) == pool.composition:
            positions = np.concatenate((son.atoms.positions, daughter.atoms.positions))
            child = Atoms(numbers=numbers, positions=positions)
            lineage = (son.parent_index,) * len(son.atoms) + (daughter.parent_index,) * len(daughter.atoms)
            return AtomicChild(child, lineage, son.source_atom_indices + daughter.source_atom_indices,
                               i, j, attempt)
    raise SamplingExhausted(f'no composition-matching pair in {max_pair_attempts} attempts')


def disturb_atoms(atoms, n, width, rng):
    """Native disturbance: n selections WITH replacement, Cartesian U(-w/2,w/2)."""
    _validate_atoms(atoms)
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 0:
        raise ValueError('n must be a nonnegative integer')
    if not np.isfinite(width) or width < 0:
        raise ValueError('width must be finite and nonnegative')
    result = Atoms(numbers=atoms.numbers, positions=atoms.positions)
    draws = []
    for _ in range(n):
        index = int(rng.random() * len(result))
        result.positions[index] += width * np.array([rng.random() - .5 for _ in range(3)])
        draws.append(index)
    return result, tuple(draws)


def exchange_atoms(atoms, rng):
    """Native alloy mutation: 10*N random index-pair swaps of species only."""
    _validate_atoms(atoms)
    result = Atoms(numbers=atoms.numbers, positions=atoms.positions)
    origins = np.arange(len(result))
    for _ in range(10 * len(result)):
        i, j = int(rng.random() * len(result)), int(rng.random() * len(result))
        result.numbers[[i, j]] = result.numbers[[j, i]]
        origins[[i, j]] = origins[[j, i]]
    return result, tuple(int(i) for i in origins)


def reinsert_undercoordinated_atoms(atoms, count, rng, *, max_insertion_attempts):
    """Reconstruct interMu geometry with explicitly CORRECTED collision acceptance.

    Remove count lowest-CN atoms (strict 3.2 A cutoff, stable ties), then reinsert
    their species in that order. Recenter before each insertion and use original
    bounding-box R and nonuniform spherical draws. Accept only all distances
    >=0.3 A. Uploaded JAR instead loops while this condition is true; that anomaly
    and its origin-padding timeout are deliberately not reproduced here.
    """
    _validate_atoms(atoms)
    count = _positive_int(count, 'count')
    max_insertion_attempts = _positive_int(max_insertion_attempts, 'max_insertion_attempts')
    if count >= len(atoms):
        raise ValueError('reinsertion requires at least one surviving atom')
    from .ga_operators import _passes_bond_limit
    distances = np.linalg.norm(atoms.positions[:, None] - atoms.positions[None, :], axis=2)
    coordination = ((distances < 3.2) & ~np.eye(len(atoms), dtype=bool)).sum(axis=1)
    removed = np.argsort(coordination, kind='stable')[:count]
    survivors = [i for i in range(len(atoms)) if i not in removed]
    result = Atoms(numbers=atoms.numbers[survivors], positions=atoms.positions[survivors])
    order = list(survivors); attempts = []
    for index in removed:
        result.positions -= result.positions.mean(axis=0)
        radius = np.linalg.norm(np.ptp(result.positions, axis=0)) / 2. + 1.5
        for attempt in range(1, max_insertion_attempts + 1):
            r = (.1 + .5 * rng.random()) * radius
            elevation = (rng.random() - .5) * 2. * np.pi
            azimuth = rng.random() * 2. * np.pi
            xyz = r * np.array([np.cos(elevation) * np.cos(azimuth),
                                np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])
            candidate = result + Atoms(numbers=[atoms.numbers[index]], positions=[xyz])
            zs = set(int(z) for z in candidate.numbers)
            if _passes_bond_limit(candidate, {(a, b): .3 for a in zs for b in zs}):
                result = candidate; attempts.append(attempt); order.append(int(index)); break
        else:
            raise SamplingExhausted(f'corrected interMu failed after {max_insertion_attempts} insertion attempts')
    return result, dict(source_atom_indices=tuple(order), removed_indices=tuple(int(i) for i in removed),
                        insertion_attempts=tuple(attempts), collision_acceptance='corrected_all_pairs_ge_0.3A')


def mutate_type0(parents, energies, rng, *, n, max_insertion_attempts):
    """Original pure/doped mutation integer quotas, stable energy-ranked parents.

    n is the native requested count, NOT the actual output count. No parent is
    mutated in-place. The only intentional operator change is documented interMu
    collision acceptance and explicit failure instead of an origin fallback.
    """
    from .ga_operators import GeneticCandidate
    if not parents or not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 0:
        raise ValueError('nonempty parents and nonnegative integer n required')
    max_insertion_attempts = _positive_int(max_insertion_attempts, 'max_insertion_attempts')
    energy = np.asarray(energies, dtype=float)
    if energy.shape != (len(parents),) or not np.isfinite(energy).all():
        raise ValueError('one finite energy per parent required')
    composition = tuple(sorted(parents[0].numbers))
    for a in parents:
        _validate_atoms(a)
        if tuple(sorted(a.numbers)) != composition:raise ValueError('all parents must share composition')
    pure = len(set(composition)) == 1
    if len(composition) <= (10 if pure else 5):
        raise ValueError('full mutation requires N>10 for pure or N>5 for alloy')
    ranked = np.argsort(energy, kind='stable'); candidates = []; size = len(composition)
    def parent(random):return int(ranked[int(rng.random() * len(ranked)) if random else 0])
    def append(atoms, index, operation, details):
        candidates.append(GeneticCandidate(atoms, tuple((i,) for i in range(size)), operation,
                                          (index,) * size, details))
    if not pure:
        for _ in range((n // 4) * 3):
            index = parent(True); atoms, origins = exchange_atoms(parents[index], rng)
            append(atoms, index, 'atomic_exchange', {'species_source_atom_indices': origins})
    counts = (n // 4 + 1, n // 4 + 1, n // 2, n // 2) if pure else (n // 8,) * 4
    for amount, moves, width, random in zip(counts, (size // 10, size // 2, size // 2, size // 10), (.3, .5, .7, .7), (False, False, True, True)):
        for _ in range(amount):
            index = parent(random); atoms, changed = disturb_atoms(parents[index], moves, width, rng)
            append(atoms, index, 'atomic_disturbance', {'moves': moves, 'width_A': width, 'selected_indices': changed})
    for _ in range(n // 4 + 1 if pure else n // 8):
        index = parent(True); atoms, details = reinsert_undercoordinated_atoms(parents[index], 5, rng, max_insertion_attempts=max_insertion_attempts)
        append(atoms, index, 'atomic_reinsertion_corrected', details)
    if pure:
        index = parent(False); atoms, details = reinsert_undercoordinated_atoms(parents[index], 10, rng, max_insertion_attempts=max_insertion_attempts)
        append(atoms, index, 'atomic_reinsertion_corrected', details)
    return candidates


def propose_type0(parents, energies, rng, *, min_ga, bond_limits, max_batches,
                  max_cut_attempts, max_pair_attempts, parent_regions=None,
                  max_insertion_attempts=10000):
    """Bounded TYPE0 batch schedule with explicit caller-side distance filtering.

    Each batch: G//4 crossovers from all parents; mutations using n=G//2 in
    region zero and n=G//8 in each other region. Whole passing batches append
    without truncation. JAR TYPE0 itself has no BLLimit filter; explicit
    bond_limits is an independent input-domain filter, empty means no extra cut.
    """
    from .ga_operators import GeneticCandidate, ProposalResult, _passes_bond_limit
    for name, value in [('min_ga', min_ga), ('max_batches', max_batches), ('max_cut_attempts', max_cut_attempts), ('max_pair_attempts', max_pair_attempts), ('max_insertion_attempts', max_insertion_attempts)]:_positive_int(value, name)
    if not parents:raise ValueError('nonempty parents required')
    energy = np.asarray(energies, dtype=float)
    if energy.shape != (len(parents),) or not np.isfinite(energy).all():raise ValueError('one finite energy per parent required')
    composition = tuple(sorted(parents[0].numbers))
    for a in parents:
        _validate_atoms(a)
        if tuple(sorted(a.numbers)) != composition:raise ValueError('all parents must share composition')
    if len(composition) <= (10 if len(set(composition)) == 1 else 5):raise ValueError('full mutation requires N>10 for pure or N>5 for alloy')
    regions = [list(range(len(parents)))] if parent_regions is None else [list(r) for r in parent_regions]
    if not regions or any(not r for r in regions) or any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer)) for r in regions for i in r) or sorted(i for r in regions for i in r) != list(range(len(parents))):raise ValueError('parent_regions must partition all parent indices')
    for key, value in bond_limits.items():
        if len(key) != 2 or not np.isfinite(value) or value < 0:raise ValueError('bond_limits require element pairs and finite nonnegative cutoffs')
    selected = []; rejected = 0
    working = [a.copy() for a in parents]
    for batch_number in range(1, max_batches + 1):
        try:
            pool = build_atomic_pool(working, energy, rng, max_cut_attempts=max_cut_attempts)
            # Cross's Cut centers selected source Models in-place. Reproduce
            # that effect only on private parents before subsequent mutation.
            for index in pool.parent_slots:
                for _ in range(pool.cuts_per_parent_slot):
                    working[index].positions -= working[index].positions.mean(axis=0)
            batch = []
            for _ in range(min_ga // 4):
                child = cross_atomic_pool(pool, rng, max_pair_attempts=max_pair_attempts)
                batch.append(GeneticCandidate(child.atoms, tuple((i,) for i in range(len(child.atoms))), 'atomic_crossover', child.parent_indices,
                    {'source_atom_indices': child.source_atom_indices, 'pair_attempts': child.pair_attempts}))
            for region_number, indices in enumerate(regions):
                local = mutate_type0([working[i] for i in indices], energy[indices], rng, n=min_ga // (2 if region_number == 0 else 8), max_insertion_attempts=max_insertion_attempts)
                for candidate in local:
                    candidate.details['parent_region'] = region_number
                    batch.append(GeneticCandidate(candidate.atoms, candidate.groups, candidate.operation,
                                tuple(indices[i] for i in candidate.group_parent_indices), candidate.details))
        except SamplingExhausted as error:
            return ProposalResult(tuple(selected), 'budget_exhausted', batch_number, rejected, str(error))
        valid = [c for c in batch if _passes_bond_limit(c.atoms, bond_limits)]
        rejected += len(batch) - len(valid)
        for c in valid:c.details['batch'] = batch_number
        selected.extend(valid)
        if len(selected) >= min_ga:return ProposalResult(tuple(selected), 'target_reached', batch_number, rejected)
        if not valid:return ProposalResult(tuple(selected), 'empty_batch', batch_number, rejected, 'no passing TYPE0 candidates')
    return ProposalResult(tuple(selected), 'budget_exhausted', max_batches, rejected, 'whole-batch count did not reach min_ga')
