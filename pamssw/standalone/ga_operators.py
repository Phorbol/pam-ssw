"""Experimental, independent reconstruction of selected uploaded TYPE3 primitives.

Sources: sgn.jar decompiled other/CooHandle.java:56-88;
ga_cluster_cell/{Cut,CutBasicAbstract}.java; ga_molecular_crystal/CutMC.java;
CrossMC.checkGameteMatching; ga_monomer/MutateMonomer.monomerRotation.
These are compatibility rules, not proposed general molecular GA defaults.

Implemented: original Euler-angle distribution, whole-monomer cuts and docking,
Compete/gamete pools, all three fixed-internal-monomer mutation modes, and TYPE3
proposal batches with explicit bond limits and budgets, including changeType=1
internal atom-level reconstruction from an explicit mutable monomer library.
Initialization and the quick/fine population controller are provided by the
separate paper_ga module; this module does not perform SSW. A cut pair is not itself a finished
crossover child; dock_gametes completes it. No calculator/Java process is used.

Only unconstrained, nonperiodic structures and disjoint, exhaustive, zero-based
monomer partitions are supported. In particular these groups are not rigid-chain
joint definitions (which can overlap). Input coordinates are never mutated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from ase import Atoms


class SamplingExhausted(RuntimeError):
    """Operational sampling budget exhausted; no fallback child was fabricated."""


@dataclass(frozen=True)
class MolecularGamete:
    group_ids: tuple[int, ...]
    fragments: tuple[Atoms, ...]


@dataclass(frozen=True)
class MolecularCut:
    son: MolecularGamete
    daughter: MolecularGamete
    plane_slope: float
    attempts: int


@dataclass(frozen=True)
class MolecularMutation:
    atoms: Atoms
    groups: tuple[tuple[int, ...], ...]
    parent_index: int
    group_index: int


def _partition(atoms: Atoms, groups) -> tuple[tuple[int, ...], ...]:
    groups = tuple(tuple(group) for group in groups)
    flat = [i for group in groups for i in group]
    if (not groups or any(not group for group in groups)
            or any(not isinstance(i, (int, np.integer)) for i in flat)
            or sorted(flat) != list(range(len(atoms)))):
        raise ValueError('groups must form a nonempty zero-based atom partition')
    if atoms.pbc.any() or atoms.constraints:
        raise ValueError('only unconstrained nonperiodic molecular clusters are supported')
    if not np.isfinite(atoms.positions).all():
        raise ValueError('coordinates must be finite')
    return groups


def _rotation(axis: int, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    if axis == 0:
        return np.array([[1., 0., 0.], [0., c, s], [0., -s, c]])
    if axis == 1:
        return np.array([[c, 0., -s], [0., 1., 0.], [s, 0., c]])
    return np.array([[c, s, 0.], [-s, c, 0.], [0., 0., 1.]])


def rotate_coordinates(positions, rng: np.random.Generator) -> np.ndarray:
    """CooHandle.randomRotation: row vectors, x then y then z, about origin.

    Angle = atan(tan(3.14159*(U-.5))). This is not uniform SO(3) sampling.
    Preserving 3.14159 is intentional. Coordinates retain their input length unit.
    """
    result = np.asarray(positions, dtype=float).copy()
    if result.ndim != 2 or result.shape[1] != 3 or not np.isfinite(result).all():
        raise ValueError('positions must be a finite N by 3 array')
    angles = np.arctan(np.tan(3.14159 * (rng.random(3) - .5)))
    for axis, angle in enumerate(angles):
        result = result @ _rotation(axis, angle)
    return result


def cut_monomers(atoms: Atoms, groups, rng: np.random.Generator, *,
                 max_attempts: int) -> MolecularCut:
    """Reproduce CutMC's balanced virtual-center cut and full-group restoration.

    Source repeatedly rotates virtual centers until the two group counts differ
    by <2. Full atoms are restored using ONLY the final cut-plane y rotation,
    as in CutMC.restoreToMonomer; the virtual-center random rotation is not
    applied to them. This surprising source behavior is preserved explicitly.
    Cut's +/-0.3 Angstrom separation is canceled by CutMC before restoration.

    max_attempts is a caller-specified operational limit; the Java loop has no
    bound. Exhaustion raises rather than returning a changed sampling rule.
    """
    groups = _partition(atoms, groups)
    if len(groups) < 2:
        raise ValueError('a molecular crossover cut requires at least two groups')
    if not isinstance(max_attempts, (int, np.integer)) or max_attempts < 1:
        raise ValueError('max_attempts must be a positive integer')
    centered = atoms.positions - atoms.positions.mean(axis=0)
    centers = np.array([centered[list(group)].mean(axis=0) for group in groups])
    centers -= centers.mean(axis=0)  # CutBasicAbstract centers the virtual model.
    for attempt in range(1, max_attempts + 1):
        centers = rotate_coordinates(centers, rng)
        slope = float(2. * (rng.random() - .5))
        # Match Java's IEEE division at slope=0; NaN is in neither count.
        with np.errstate(divide='ignore', invalid='ignore'):
            height = centers[:, 2] + centers[:, 0] / slope
        positive, negative = height >= 0., height < 0.
        if abs(int(positive.sum()) - int(negative.sum())) < 2:
            if not (positive | negative).all():
                raise SamplingExhausted('singular legacy cut produced undefined group membership')
            break
    else:
        raise SamplingExhausted(f'balanced cut not found in {max_attempts} attempts')
    with np.errstate(divide='ignore'):
        angle = float(np.arctan(np.divide(-1., slope)))
    restored = centered @ _rotation(1, angle)

    def gamete(mask):
        ids = tuple(int(i) for i in np.flatnonzero(mask))
        fragments = []
        for gid in ids:
            fragment = atoms[list(groups[gid])]
            fragment.positions = restored[list(groups[gid])]
            fragments.append(fragment)
        return MolecularGamete(ids, tuple(fragments))

    return MolecularCut(gamete(positive), gamete(negative), slope, attempt)


def gametes_match(son: MolecularGamete, daughter: MolecularGamete,
                  n_groups: int) -> bool:
    """CrossMC identity test: concatenated IDs must be exactly 0..n_groups-1.

    Caller must use gametes with the same monomer topology. This does not test
    geometry or dock them; shared monomer identities invalidate a pairing.
    """
    return sorted(son.group_ids + daughter.group_ids) == list(range(n_groups))


def mutate_single_monomer(parents: Sequence[Atoms], energies: Sequence[float],
                          groups, rng: np.random.Generator, *, max_attempts: int) -> MolecularMutation:
    """Rotate one non-singleton group in the lowest-energy parent, about origin.

    Rejection sampling of singleton groups follows MutateMonomer.monomerRotation.
    Unlike the source's infinite loop, an all-singleton partition raises. Output
    atom order is the concatenated group order, matching atomUtilToAtoCoos.
    No energy evaluation is performed: energies are explicit parent metadata.
    """
    energy = np.asarray(energies, dtype=float)
    if not parents or energy.shape != (len(parents),) or not np.isfinite(energy).all():
        raise ValueError('one finite energy per nonempty parent list is required')
    parent_index = int(np.argmin(energy))  # First minimum agrees with stable sort.
    parent = parents[parent_index]
    groups = _partition(parent, groups)
    if all(len(group) == 1 for group in groups):
        raise ValueError('mutation needs a non-singleton monomer')
    if not isinstance(max_attempts, (int, np.integer)) or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    for _ in range(max_attempts):
        group_index = int(np.floor(rng.random() * len(groups)))
        if len(groups[group_index]) > 1:
            break
    else:
        raise SamplingExhausted(f"no non-singleton monomer selected in {max_attempts} attempts")
    positions = parent.positions.copy()
    chosen = list(groups[group_index])
    positions[chosen] = rotate_coordinates(positions[chosen], rng)
    order = [i for group in groups for i in group]
    result = parent[order]
    result.positions = positions[order]
    result.calc = None
    offset = 0
    output_groups = []
    for group in groups:
        output_groups.append(tuple(range(offset, offset + len(group))))
        offset += len(group)
    return MolecularMutation(result, tuple(output_groups), parent_index, group_index)


@dataclass(frozen=True)
class DockingResult:
    atoms: Atoms
    candidate_index: int


@dataclass(frozen=True)
class MolecularChild:
    atoms: Atoms
    groups: tuple[tuple[int, ...], ...]
    candidate_index: int


def _monomer_radius(positions):
    """BasicInfo.getMonR: half bounding-box diagonal plus 1.5 Angstrom."""
    return float(np.linalg.norm(np.ptp(positions, axis=0)) / 2. + 1.5)


def _docking_directions(accuracy):
    # CooHandle.sphericalFibonacciCoordinates(accuracy,10.0): preserve the
    # supplied radius. translate() does NOT normalize these length-10 vectors.
    phi = np.arccos(-1. + (2. * np.arange(1, accuracy + 1) - 1.) / accuracy)
    theta = np.sqrt(accuracy * np.pi) * phi
    return 10. * np.column_stack((np.cos(theta) * np.sin(phi),
                                  np.sin(theta) * np.sin(phi), np.cos(phi)))


def _docking_rotation(direction):
    # Exactly getEulerAngles([0,0,1],direction) followed by rotateM; replacing
    # this with an axis-angle alignment would change the recovered algorithm.
    pitch = np.arctan2(direction[0], 0.)
    yaw = np.arctan2(-direction[1], abs(direction[0]))
    roll = np.arccos(direction[2] / np.linalg.norm(direction))
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cr, sr = np.cos(roll), np.sin(roll)
    return np.array([[cy*cp, sy*cp, -sp],
                     [-sy*cr+cy*sp*sr, cy*cr+sy*sp*sr, cp*sr],
                     [sy*sr+cy*sp*cr, -cy*sr+sy*sp*cr, cp*cr]])


def _fit_binding(subject: Atoms, object_: Atoms, *, min_distance: float = 1.5,
                 accuracy: int = 5, radii=None) -> DockingResult:
    """Independent MC_Base.fitBinding(Model,Model) plus moleculeDocking/maxDock.

    CrossMC.butt supplies 1.5 Angstrom and accuracy=5. Each of accuracy**2
    direction/orientation pairs translates object by -distance*direction,
    decreasing distance by .1 until an inter-fragment pair is closer than
    min_distance or distance<0, then backs up .1. The direction has length 10,
    as in the original, so a .1 distance decrement translates by 1 Angstrom.

    maxDock deliberately excludes candidate zero, retains up to ten smallest
    bounding radii, then minimizes the sum of all inter-fragment distances.
    This is geometric docking; there is no force optimization or energy score.
    """
    for fragment in (subject, object_):
        _partition(fragment, (tuple(range(len(fragment))),))
    if not np.isfinite(min_distance) or min_distance <= 0:
        raise ValueError('min_distance must be positive and finite')
    if not isinstance(accuracy, (int, np.integer)) or accuracy < 2:
        raise ValueError('accuracy must be >=2; legacy maxDock excludes candidate zero')
    directions = _docking_directions(accuracy)
    s = subject.positions
    o = object_.positions
    rotations = [o @ _docking_rotation(direction) for direction in directions]
    starting_distance = (_monomer_radius(s) + _monomer_radius(o)
                         if radii is None else float(sum(radii)))
    candidates = []
    for direction in directions:
        for rotated in rotations:
            distance = starting_distance
            while True:
                distance -= .1
                trial = rotated - distance * direction
                if distance < 0. or np.any(np.linalg.norm(s[:, None] - trial[None], axis=2) < min_distance):
                    break
            trial = rotated - (distance + .1) * direction
            candidates.append(trial)
    ranked = sorted(range(1, len(candidates)),
                    key=lambda i: _monomer_radius(np.vstack((s, candidates[i]))))[:10]
    best = min(ranked, key=lambda i: float(np.linalg.norm(s[:, None] - candidates[i][None], axis=2).sum()))
    atoms = subject + object_
    atoms.positions = np.vstack((s, candidates[best]))
    atoms.calc = None
    return DockingResult(atoms, best)


def _flatten_fragments(fragments):
    atoms = Atoms()
    groups = []
    for fragment in fragments:
        # Atoms.extend does not retain the appended object's pbc/constraints.
        # Validate before concatenation so unsupported physics is never lost.
        _partition(fragment, (tuple(range(len(fragment))),))
        groups.append(tuple(range(len(atoms), len(atoms) + len(fragment))))
        atoms += fragment
    return atoms, tuple(groups)


def dock_gametes(son: MolecularGamete, daughter: MolecularGamete,
                 n_groups: int) -> MolecularChild:
    """Complete CrossMC.butt: dock, restore monomer-ID order, and geneCell.

    This produces an actual Atoms child independently of Java. The outer
    competitor/gamete-pool sampling and TYPE3 batch policy remain separate.
    geneCell adds .5 Angstrom to each bounding-box extent without translating
    coordinates. ASE pbc remains False because this interface is for clusters;
    the legacy PBC record itself is packaging, not a periodic-model choice.
    """
    if not gametes_match(son, daughter, n_groups):
        raise ValueError('gamete identities do not form the original monomer set')
    if (not son.fragments or not daughter.fragments
            or len(son.group_ids) != len(son.fragments)
            or len(daughter.group_ids) != len(daughter.fragments)):
        raise ValueError('gametes require nonempty fragments matching their identities')
    subject, sg = _flatten_fragments(son.fragments)
    object_, dg = _flatten_fragments(daughter.fragments)
    fit = fit_binding(subject, object_, min_distance=1.5, accuracy=5)
    docked_groups = sg + tuple(tuple(i + len(subject) for i in group) for group in dg)
    by_id = dict(zip(son.group_ids + daughter.group_ids, docked_groups))
    fragments = [fit.atoms[list(by_id[i])] for i in range(n_groups)]
    atoms, groups = _flatten_fragments(fragments)
    atoms.set_cell(np.ptp(atoms.positions, axis=0) + .5)
    return MolecularChild(atoms, groups, fit.candidate_index)


def fit_binding(subject: Atoms, object_: Atoms, *, min_distance: float = 1.5,
                accuracy: int = 5) -> DockingResult:
    """MC_Base.fitBinding Model overload; see module notes for docking rules."""
    return _fit_binding(subject, object_, min_distance=min_distance, accuracy=accuracy)


@dataclass(frozen=True)
class GeneticCandidate:
    atoms: Atoms
    groups: tuple[tuple[int, ...], ...]
    operation: str
    group_parent_indices: tuple[int | None, ...]
    details: dict


@dataclass(frozen=True)
class ProposalResult:
    candidates: tuple[GeneticCandidate, ...]
    status: str
    batches: int
    rejected_by_bond_limit: int
    reason: str = ''


def _validate_parents(parents, energies, groups, change_types):
    if not parents:
        raise ValueError('parents must be nonempty')
    groups = _partition(parents[0], groups)
    for parent in parents[1:]:
        _partition(parent, groups)
        if not np.array_equal(parent.numbers, parents[0].numbers):
            raise ValueError('parents must have the same ordered element topology')
    if len(change_types) != len(groups) or any(x not in (0, 1) for x in change_types):
        raise ValueError('one binary changeType per group is required')
    energy = np.asarray(energies, dtype=float)
    if energy.shape != (len(parents),) or not np.isfinite(energy).all():
        raise ValueError('one finite energy per parent is required')
    return groups, energy


def _combine_monomers(fragments, radii, rng):
    """MonomerBase.fitMonomerCombination plus the AtomUtil docking overload."""
    order = list(range(len(fragments)))
    # TemTools.upset is 10*N random swaps, not Collections.shuffle.
    for _ in range(10 * len(order)):
        i, j = (int(np.floor(rng.random() * len(order))) for _ in range(2))
        order[i], order[j] = order[j], order[i]
    combined = fragments[order[0]].copy()
    radius = radii[order[0]]
    for index in order[1:]:
        fit = _fit_binding(combined, fragments[index], min_distance=1.5, accuracy=10,
                           radii=(radius, radii[index]))
        combined = fit.atoms
        # Unlike the Model overload, AtomUtil centers after every docking.
        combined.positions -= combined.positions.mean(axis=0)
        radius = _monomer_radius(combined.positions)
    offset = 0
    by_id = {}
    for index in order:
        count = len(fragments[index])
        by_id[index] = combined[offset:offset + count]
        offset += count
    atoms, groups = _flatten_fragments([by_id[i] for i in range(len(fragments))])
    return atoms, groups, tuple(order)


def mutable_monomer_library(parents, energies, rng, *, count, max_attempts,
                            quota_policy='complete_library',cuts_per_parent_slot=None):
    """MutateMonomer internal Cross+Doping library, with explicit quota repair.

    Atom-level parent energies are inherited WHOLE-parent metadata, exactly
    as native; they are not independently evaluated monomer energies.
    """
    from .atomic_ga import (build_atomic_pool,cross_atomic_pool,disturb_atoms,
                            exchange_atoms,reinsert_undercoordinated_atoms)
    if isinstance(count,bool) or not isinstance(count,(int,np.integer)) or count<0:raise ValueError('nonnegative mutable library count required')
    if isinstance(max_attempts,bool) or not isinstance(max_attempts,(int,np.integer)) or max_attempts<1:raise ValueError('positive internal sampling budget required')
    if quota_policy not in ('complete_library','native_quota'):raise ValueError('unknown mutable quota policy')
    if count==0:return (),dict(requested=0,source_request=0,generated=0,discarded=0)
    def size(n):
        cross=2*n//3;m=n-cross
        return cross+3*(m//4)+5*(m//8)
    request=int(count)
    if quota_policy=='native_quota' and size(request)<count:
        raise ValueError(f'native mutable library underflow: requested {count}, generated {size(request)}')
    # Monotone deterministic cardinality calculation, no RNG/adaptive heuristic.
    while size(request)<count:request+=1
    cross_n=2*request//3;mutation_n=request-cross_n
    if mutation_n//8 and len(parents[0])<=5:
        raise ValueError('native mutable reinsertion needs >5 atoms when its quota is nonzero')
    private=[a.copy() for a in parents]
    pool=build_atomic_pool(private,energies,rng,max_cut_attempts=max_attempts,cuts_per_parent_slot=cuts_per_parent_slot)
    for index in pool.parent_slots:
        for _ in range(pool.cuts_per_parent_slot):private[index].positions-=private[index].positions.mean(axis=0)
    items=[]
    for _ in range(cross_n):
        child=cross_atomic_pool(pool,rng,max_pair_attempts=max_attempts)
        items.append((child.atoms,child.parent_indices,child.source_atom_indices,dict(operation='internal_crossover',pair_attempts=child.pair_attempts)))
    ranked=np.argsort(energies,kind='stable');n=len(private[0])
    def append(a,index,origins,details):items.append((a,(index,)*n,tuple(origins),details))
    def select():return int(ranked[int(rng.random()*len(ranked))])
    # Always ForDoping, even for a pure-element mutable unit, as source requires.
    for _ in range((mutation_n//4)*3):
        index=select();a,origins=exchange_atoms(private[index],rng)
        append(a,index,origins,dict(operation='internal_exchange'))
    for moves,width,random_parent in ((n//10,.3,False),(n//2,.5,False),(n//2,.7,True),(n//10,.7,True)):
        for _ in range(mutation_n//8):
            index=select() if random_parent else int(ranked[0]);a,moved=disturb_atoms(private[index],moves,width,rng)
            append(a,index,range(n),dict(operation='internal_disturbance',moved_indices=moved,width_A=width))
    for _ in range(mutation_n//8):
        index=select();a,info=reinsert_undercoordinated_atoms(private[index],5,rng,max_insertion_attempts=max_attempts)
        append(a,index,info['source_atom_indices'],dict(operation='internal_reinsertion_corrected',**info))
    ledger=dict(requested=count,source_request=request,quota_policy=quota_policy,crossovers=cross_n,
                mutation_request=mutation_n,generated=len(items),discarded=len(items)-count,
                pool_cuts=len(pool.sons),pool_cut_attempts=pool.cut_attempts,
                candidates=tuple(dict(parent_indices=p,source_indices=s,**details) for _,p,s,details in items))
    return tuple(items[:count]),ledger


def _mutable_reconstruction(parents,energy,groups,change_types,n,rng,*,max_attempts,
                            quota_policy,cuts_per_parent_slot):
    libraries={};ledgers={}
    for group_index,change in enumerate(change_types):
        if change:
            fragments=[a[list(groups[group_index])] for a in parents]
            libraries[group_index],ledgers[group_index]=mutable_monomer_library(fragments,energy,rng,count=n,
                max_attempts=max_attempts,quota_policy=quota_policy,cuts_per_parent_slot=cuts_per_parent_slot)
    results=[]
    for index in range(n):
        fragments=[];atom_parents=[];atom_sources=[];group_sets=[]
        for group_index,group in enumerate(groups):
            if group_index in libraries:
                fragment,p,s,_=libraries[group_index][index];fragment=fragment.copy()
                # Stable within-species reordering preserves the controller's
                # ordered-element topology without changing geometry/species.
                available={z:list(np.flatnonzero(fragment.numbers==z)) for z in set(fragment.numbers)}
                order=[available[z].pop(0) for z in parents[0].numbers[list(group)]]
                fragment=fragment[order];p=tuple(p[i] for i in order);s=tuple(s[i] for i in order)
            else:fragment=parents[0][list(group)];p=(0,)*len(group);s=tuple(range(len(group)))
            fragments.append(fragment);atom_parents.extend(p);atom_sources.extend(group[i] for i in s);group_sets.append(tuple(sorted(set(p))))
        radii=[_monomer_radius(f.positions) for f in fragments]
        for fragment in fragments:fragment.positions-=fragment.positions.mean(axis=0)
        atoms,new_groups,order=_combine_monomers(fragments,radii,rng)
        results.append(GeneticCandidate(atoms,new_groups,'monomer_reconstruction',
            tuple(p[0] if len(p)==1 else None for p in group_sets),
            dict(docking_order=order,atom_parent_indices=tuple(atom_parents),source_atom_indices=tuple(atom_sources),
                 group_parent_sets=tuple(group_sets),mutable_library_index=index,
                 mutable_library_ledgers=ledgers,
                 mutable_library_cost_scope='shared once per mutation call and group; repeated references are not additional work',
                 topology_correction='stable within-species reordering to original group element order')))
    return results


def mutate_type3(parents: Sequence[Atoms], energies: Sequence[float], groups,
                 change_types, counts: tuple[int, int, int],
                 rng: np.random.Generator, *, max_selection_attempts: int,
                 mutable_quota_policy: str = 'complete_library',
                 mutable_cuts_per_parent_slot: int | None = None) -> list[GeneticCandidate]:
    """Actual three MutateMonomer modes for fixed or atomically mutable units.

    counts correspond to rotation/recombination, monomer reconstruction, and
    single-monomer rotation. Reconstruction with all changeTypes zero really
    recombines the first parent's original monomers; it is not a no-op.
    Mutable internal units build source Cross+Doping libraries. Small native
    underfilled quotas are repaired by the smallest sufficient source request;
    native_quota instead fails explicitly. Full library cost is retained.
    """
    groups, energy = _validate_parents(parents, energies, groups, change_types)
    if len(counts) != 3 or any(not isinstance(n, (int, np.integer)) or n < 0 for n in counts):
        raise ValueError('counts must contain three nonnegative integers')
    if not isinstance(max_selection_attempts, (int, np.integer)) or max_selection_attempts < 1:
        raise ValueError("max_selection_attempts must be a positive integer")
    candidates = []
    for _ in range(counts[0]):
        parent_index = int(np.floor(rng.random() * len(parents)))
        fragments = [parents[parent_index][list(group)] for group in groups]
        radii = [_monomer_radius(fragment.positions) for fragment in fragments]
        for fragment in fragments:
            fragment.positions -= fragment.positions.mean(axis=0)
            fragment.positions = rotate_coordinates(fragment.positions, rng)
        atoms, new_groups, order = _combine_monomers(fragments, radii, rng)
        candidates.append(GeneticCandidate(atoms, new_groups, 'rotation_recombination',
                                          (parent_index,) * len(groups), {'docking_order': order}))
    if any(change_types):
        candidates.extend(_mutable_reconstruction(parents,energy,groups,change_types,counts[1],rng,
            max_attempts=max_selection_attempts,quota_policy=mutable_quota_policy,
            cuts_per_parent_slot=mutable_cuts_per_parent_slot))
    else:
        for _ in range(counts[1]):
            # Source changeType=0 unconditionally takes monomers from parent[0].
            fragments = [parents[0][list(group)] for group in groups]
            radii = [_monomer_radius(fragment.positions) for fragment in fragments]
            for fragment in fragments:
                fragment.positions -= fragment.positions.mean(axis=0)
            atoms, new_groups, order = _combine_monomers(fragments, radii, rng)
            candidates.append(GeneticCandidate(atoms, new_groups, 'monomer_reconstruction',
                                              (0,) * len(groups), {'docking_order': order}))
    for _ in range(counts[2]):
        result = mutate_single_monomer(parents, energy, groups, rng, max_attempts=max_selection_attempts)
        candidates.append(GeneticCandidate(result.atoms, result.groups, 'single_monomer_rotation',
                                          (result.parent_index,) * len(groups),
                                          {'rotated_group': result.group_index}))
    return candidates


def _competition_cumulative(energy):
    # Compete, with explicit rejection of its NaN-producing degeneracies.
    import math
    span = float(max(energy) - min(energy))
    if len(energy) <= 2 or span <= 0:
        raise ValueError('legacy Compete requires >2 parents with positive energy span')
    temperature = -span * 1000. * 2625. / (math.log(2. / len(energy)) * 8.314)
    total = 0.
    cumulative = []
    for value in energy:
        total += math.pow(2.7183, -(float(value) - min(energy)) * 1000. * 2625. / (8.314 * temperature))
        cumulative.append(total)
    return np.asarray(cumulative)


def _molecular_pool(parents, energy, groups, rng, max_cut_attempts):
    cumulative = _competition_cumulative(energy)
    # getParentModelIndex consumes BOTH entries, although genePool uses only 0.
    indices = []
    for _ in range(len(parents) * 10):
        pair = [int(np.searchsorted(cumulative, rng.random() * cumulative[-1], side='right')) for _ in range(2)]
        indices.append(pair[0])
    sons, daughters = [], []
    for parent_index in indices:
        for _ in range(10):
            # Source CutMC recenters its source Model in-place. Preserve the
            # effect on later mutation within this batch, using private copies.
            parent = parents[parent_index]
            parent.positions -= parent.positions.mean(axis=0)
            cut = cut_monomers(parent, groups, rng, max_attempts=max_cut_attempts)
            sons.append((parent_index, cut.son))
            daughters.append((parent_index, cut.daughter))
    return sons, daughters


def _cross_from_pool(sons, daughters, n_groups, rng, max_pair_attempts):
    for attempt in range(1, max_pair_attempts + 1):
        i = int(np.floor(rng.random() * len(sons)))
        j = int(np.floor(rng.random() * len(daughters)))
        sp, son = sons[i]
        dp, daughter = daughters[j]
        if gametes_match(son, daughter, n_groups):
            break
    else:
        raise SamplingExhausted(f'no complementary gametes after {max_pair_attempts} pairs')
    result = dock_gametes(son, daughter, n_groups)
    by_id = dict.fromkeys(son.group_ids, sp)
    by_id.update(dict.fromkeys(daughter.group_ids, dp))
    return GeneticCandidate(result.atoms, result.groups, 'crossover',
                            tuple(by_id[k] for k in range(n_groups)),
                            {'son_pool_index': i, 'daughter_pool_index': j,
                             'pair_attempts': attempt, 'docking_candidate': result.candidate_index})


def _passes_bond_limit(atoms, limits):
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            z1, z2 = int(atoms.numbers[i]), int(atoms.numbers[j])
            distance = float(np.linalg.norm(atoms.positions[i] - atoms.positions[j]))
            # Java independently tests both orientation keys if present.
            for key in ((z1, z2), (z2, z1)):
                if key in limits and distance < limits[key]:
                    return False
    return True


def propose_type3(parents: Sequence[Atoms], energies: Sequence[float], groups,
                  change_types, rng: np.random.Generator, *, min_ga: int,
                  bond_limits: dict[tuple[int, int], float], max_batches: int,
                  max_cut_attempts: int, max_pair_attempts: int,
                  mutable_quota_policy: str = 'complete_library',
                  mutable_cuts_per_parent_slot: int | None = None) -> ProposalResult:
    """TYPE3 whole-batch proposal loop, independently executable for water.

    Caller flattens selected regions in their original order. No partition,
    calculator, SSW, or population controller is invoked. Parent metadata uses
    indices in this caller-supplied ordering. All retry limits are explicit.

    Each batch builds the original 100*n_parent gamete pool; generates G//4
    crossovers and three mutation modes each (G-G//4)//4; applies exact explicit
    BLLimit pair cutoffs; appends the entire passing batch, without truncation.
    Empty bond_limits deliberately means no pair cutoff (source's empty map).
    An empty batch stops. Budget exhaustion returns a reason and any already
    accepted batches, never padded or silently replaced candidates.
    """
    groups, energy = _validate_parents(parents, energies, groups, change_types)
    if any(not isinstance(n, (int, np.integer)) or n < 1
           for n in (min_ga, max_batches, max_cut_attempts, max_pair_attempts)):
        raise ValueError('min_ga and all execution budgets must be positive integers')
    if min_ga < 4:
        raise ValueError('min_ga must be >=4 for native TYPE3 operator allocation')
    for key, value in bond_limits.items():
        if len(key) != 2 or not np.isfinite(value) or value < 0:
            raise ValueError('bond_limits require element-pair keys and finite nonnegative Angstrom cutoffs')
    working = [parent.copy() for parent in parents]
    selected = []
    rejected = 0
    for batch_number in range(1, max_batches + 1):
        try:
            sons, daughters = _molecular_pool(working, energy, groups, rng, max_cut_attempts)
            batch = [_cross_from_pool(sons, daughters, len(groups), rng, max_pair_attempts)
                     for _ in range(min_ga // 4)]
            n_mutation = (min_ga - min_ga // 4) // 4
            batch.extend(mutate_type3(working, energy, groups, change_types,
                                     (n_mutation,) * 3, rng, max_selection_attempts=max_pair_attempts,
                                     mutable_quota_policy=mutable_quota_policy,
                                     mutable_cuts_per_parent_slot=mutable_cuts_per_parent_slot))
        except SamplingExhausted as error:
            return ProposalResult(tuple(selected), 'budget_exhausted', batch_number, rejected, str(error))
        valid = [candidate for candidate in batch if _passes_bond_limit(candidate.atoms, bond_limits)]
        rejected += len(batch) - len(valid)
        for candidate in valid:
            candidate.details['batch'] = batch_number
        selected.extend(valid)
        if len(selected) >= min_ga:
            return ProposalResult(tuple(selected), 'target_reached', batch_number, rejected)
        if not valid:
            return ProposalResult(tuple(selected), 'empty_batch', batch_number, rejected,
                                  'legacy getGA produced no passing candidates')
    return ProposalResult(tuple(selected), 'budget_exhausted', max_batches, rejected,
                          'whole-batch count did not reach min_ga')
