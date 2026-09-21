"""Independent TYPE1 crystal mutation slice recovered from MutateForCell.java.

The native periodic operator itself perturbs Cartesian coordinates at fixed
parent cell, without wrapping; it is not a fractional-coordinate displacement.
It also exchanges species labels at fixed sites. No cluster cut/docking is
reused. Cell-aware crossover and an explicit periodic filter are provided;
no calculator, archive or GA controller is included. See docs/research/type1-periodic-mutation.md for source limits.
"""
from dataclasses import dataclass
import numpy as np
from .periodic_softening import _identity


@dataclass(frozen=True)
class PeriodicMutation:
    atoms: object
    parent_index: int  # Caller parent order, before source energy sorting.
    operation: str
    details: dict


def mutate_type1(parents, energies, n, rng):
    """Return the source's full pure/doping mutation batch, not exactly n items.

    Models are stably energy-sorted without mutating caller order. Draws use
    floor(U*N), preserving scalar Java draw order given an injected stream;
    NumPy seeded draws do not equal Java Math.random trajectories. Displacement
    ranges .3/.5 Angstrom, batch fractions and 10N swaps are empirical native
    constants, not recommended universal physics. Species composition is fixed;
    parent lattices may differ. No child validity or minima claim is made.

    n=0 gives one sparse mutation for pure systems (the source's n/2+1 loop),
    zero for multicomponent systems. Integer-floor zero displacements are
    deliberately retained. Site masses, if explicitly provided, are rejected:
    exchanging species with custom site masses needs an isotope contract.
    """
    if isinstance(n,(bool,np.bool_)) or not isinstance(n,(int,np.integer)) or n<0:
        raise ValueError('n must be a nonnegative integer')
    parents=tuple(parents);energy=np.asarray(energies,dtype=float)
    if not parents or energy.shape!=(len(parents),) or not np.isfinite(energy).all():
        raise ValueError('nonempty parents and matching finite energies required')
    for a in parents:
        _identity(a)
        if 'masses' in a.arrays:
            raise ValueError('explicit isotope/site masses are outside TYPE1 mutation contract')
    composition=np.sort(parents[0].numbers)
    if any(not np.array_equal(np.sort(a.numbers),composition) for a in parents):
        raise ValueError('all parents must have identical elemental composition')
    order=np.argsort(energy,kind='stable');size=len(parents[0]);children=[]
    pure=len(set(composition))==1

    def select(length):
        u=float(rng.random())
        if not np.isfinite(u) or not 0<=u<1:raise ValueError('RNG draws must lie in [0,1)')
        return int(u*length)

    def child(sorted_index, operation, moves=0, extent=0.):
        index=int(order[sorted_index]);a=parents[index].copy();a.calc=None
        # Source clone copies energy metadata; an unevaluated Python offspring
        # must not inherit a misleading energy/force cache or custom atom arrays.
        from ase import Atoms
        a=Atoms(numbers=a.numbers.copy(),positions=a.positions.copy(),cell=a.cell.copy(),pbc=a.pbc.copy())
        changed=[]
        if operation=='exchange':
            for _ in range(10*size):
                i,j=select(size),select(size)
                a.numbers[i],a.numbers[j]=int(a.numbers[j]),int(a.numbers[i])
                changed.append((i,j))
        else:
            for _ in range(moves):
                i=select(size)
                a.positions[i]+=extent*(np.array([float(rng.random()) for _ in range(3)])-.5)
                changed.append(i)
        if not np.isfinite(a.positions).all():raise ValueError('nonfinite mutated positions')
        details=dict(source='ga_cluster_cell.MutateForCell',sorted_parent_index=sorted_index,
                     displacement_draws=moves,cartesian_range_A=extent,
                     exchange_draws=10*size if operation=='exchange' else 0,
                     selected_sites=changed,coordinate_semantics='unwrapped Cartesian, parent cell retained')
        children.append(PeriodicMutation(a,index,operation,details))

    if pure:
        batches=[(n//4,'disturb_best_sparse',size//10,.3,False),
                 (n//4,'disturb_best_half',size//2,.3,False),
                 (n//2,'disturb_random_half',size//2,.5,True),
                 (n//2+1,'disturb_random_sparse',size//10,.5,True)]
    else:
        for _ in range(n//2):child(select(len(parents)),'exchange')
        batches=[(n//8,'disturb_best_sparse',size//5,.3,False),
                 (n//8,'disturb_best_half',size//2,.3,False),
                 (n//4,'disturb_random_sparse',size//5,.5,True),
                 (n//4,'disturb_random_half',size//2,.5,True)]
    for count,operation,moves,extent,random_parent in batches:
        for _ in range(count):child(select(len(parents)) if random_parent else 0,operation,moves,extent)
    return children


@dataclass(frozen=True)
class PeriodicGamete:
    atoms: object
    parent_index: int
    source_atom_indices: tuple
    cell: np.ndarray


@dataclass(frozen=True)
class PeriodicPool:
    sons: tuple
    daughters: tuple
    composition: tuple
    parent_slots: tuple
    slots_per_parent: int
    cuts_per_slot: int


@dataclass(frozen=True)
class PeriodicCandidate:
    atoms: object
    atom_parent_indices: tuple
    source_atom_indices: tuple
    operation: str
    details: dict


@dataclass(frozen=True)
class PeriodicProposal:
    candidates: tuple
    status: str
    batches: tuple


def _periodic_parents(parents, energies):
    parents=tuple(parents);e=np.asarray(energies,dtype=float)
    if not parents or e.shape!=(len(parents),) or not np.isfinite(e).all():
        raise ValueError('nonempty parents and finite energies required')
    for a in parents:_identity(a)
    composition=tuple(sorted(map(int,parents[0].numbers)))
    if any(tuple(sorted(map(int,a.numbers)))!=composition for a in parents):
        raise ValueError('identical parent compositions required')
    return parents,e,composition


def _canonical_parent(atoms):
    """Physical rigid frame change to the native cellpar-derived row lattice.

    Java PBC stores lengths/angles, so its getCM lacks an arbitrary ASE cell
    orientation. Rotate atoms and cell together using fixed fractional coords.
    This explicit input-frame choice preserves physical periodic geometry.
    """
    from ase import Atoms
    from ase.cell import Cell
    cell=Cell.fromcellpar(atoms.cell.cellpar()).array
    return Atoms(numbers=atoms.numbers,positions=atoms.get_scaled_positions(wrap=False)@cell,
                 cell=cell,pbc=True)


def build_periodic_pool(parents, energies, rng, *, max_cut_attempts,
                        slots_per_parent=100, cuts_per_slot=10):
    """CrossForCell.genePool: 100N parent slots, both columns, ten cuts/slot.

    Reuses cut_atoms because the native periodic class literally invokes the
    same Cut.java used by atomic clusters. Only this source-shared cut primitive
    is reused; lattice transfer, output cell and collision geometry are periodic.
    Operational density overrides are recorded, not claimed native parity.
    """
    from ase import Atoms
    from .atomic_ga import cut_atoms,_positive_int
    from .ga_operators import _competition_cumulative
    parents,e,composition=_periodic_parents(parents,energies)
    for value,name in [(max_cut_attempts,'max_cut_attempts'),(slots_per_parent,'slots_per_parent'),(cuts_per_slot,'cuts_per_slot')]:
        _positive_int(value,name)
    cumulative=_competition_cumulative(e)
    slots=[]
    for _ in range(len(parents)*slots_per_parent):
        columns=[int(np.searchsorted(cumulative,rng.random()*cumulative[-1],side='right')) for _ in range(2)]
        slots.append(columns[0])
    private=[_canonical_parent(a) for a in parents]
    sons=[];daughters=[]
    for index in slots:
        for _ in range(cuts_per_slot):
            p=private[index]
            # Native Cut receives Cartesian atoms and leaves PBC untouched.
            fragment=Atoms(numbers=p.numbers,positions=p.positions)
            cut=cut_atoms(fragment,rng,max_attempts=max_cut_attempts,parent_index=index)
            p.positions-=p.positions.mean(axis=0)  # native in-place recenter, private only
            sons.append(PeriodicGamete(cut.son.atoms,index,cut.son.source_atom_indices,p.cell.array.copy()))
            daughters.append(PeriodicGamete(cut.daughter.atoms,index,cut.daughter.source_atom_indices,p.cell.array.copy()))
    return PeriodicPool(tuple(sons),tuple(daughters),composition,tuple(slots),slots_per_parent,cuts_per_slot)


def cross_periodic_pool(pool,rng,*,max_pair_attempts):
    """Cell-aware matching/transfer and Java-sign fractional remainder.

    Intentional correction: daughter branch uses daughter[j]'s cell for BOTH
    transfer and output. Native ova[i] metadata mismatch is not reproduced.
    """
    from ase import Atoms
    from .atomic_ga import _positive_int
    from .ga_operators import SamplingExhausted
    _positive_int(max_pair_attempts,'max_pair_attempts')
    if not pool.sons or not pool.daughters:raise ValueError('nonempty periodic pools required')
    for attempt in range(1,max_pair_attempts+1):
        i=int(rng.random()*len(pool.sons));j=int(rng.random()*len(pool.daughters))
        son,daughter=pool.sons[i],pool.daughters[j]
        numbers=np.concatenate([son.atoms.numbers,daughter.atoms.numbers])
        if tuple(sorted(map(int,numbers)))!=pool.composition:continue
        son_cell=bool(rng.random()>=.5)
        target,foreign=(son,daughter) if son_cell else (daughter,son)
        cell=np.asarray(target.cell,dtype=float)
        transferred=np.linalg.solve(np.asarray(foreign.cell).T,foreign.atoms.positions.T).T@cell
        xyz=np.concatenate([target.atoms.positions,transferred])
        fractional=np.linalg.solve(cell.T,xyz.T).T
        # Java % is signed remainder, not positive wrap to [0,1).
        xyz=np.fmod(fractional,1.)@cell
        atoms=Atoms(numbers=np.concatenate([target.atoms.numbers,foreign.atoms.numbers]),positions=xyz,cell=cell,pbc=True)
        _identity(atoms)
        return PeriodicCandidate(atoms,(target.parent_index,)*len(target.atoms)+(foreign.parent_index,)*len(foreign.atoms),
            target.source_atom_indices+foreign.source_atom_indices,'crossover',
            dict(son_index=i,daughter_index=j,pair_attempts=attempt,cell_parent_index=target.parent_index,
                 selected_cell=cell.tolist(),native_ova_index_defect_corrected=not son_cell,
                 coordinate_frame='canonical native cellpar frame',fractional_remainder='signed Java percent'))
    raise SamplingExhausted(f'no composition-compatible periodic pair in {max_pair_attempts} attempts')


def periodic_collision_free(atoms,bond_limits):
    """Independent corrected all-image periodic lower-distance filter.

    Supplied species pair cutoffs in Angstrom, strict d<cutoff rejects. Empty
    table disables filtering. Includes nonzero self images and skew lattice
    neighbors beyond the native finite 2x2x2 block; no bond order/validity claim.
    """
    from ase.neighborlist import neighbor_list
    _identity(atoms);limits={}
    for pair,cutoff in bond_limits.items():
        if len(pair)!=2 or any(isinstance(z,(bool,np.bool_)) or int(z)!=z or z<=0 for z in pair):
            raise ValueError('cutoff keys require two positive atomic numbers')
        if not np.isfinite(cutoff) or cutoff<0:raise ValueError('finite nonnegative bond cutoff required')
        key=tuple(sorted(map(int,pair)))
        limits[key]=max(limits.get(key,0.),float(cutoff))
    radius=max(limits.values(),default=0.)
    if radius==0:return True
    i,j,d=neighbor_list('ijd',atoms,np.nextafter(radius,np.inf),self_interaction=False)
    return all(distance>=limits.get(tuple(sorted((int(atoms.numbers[a]),int(atoms.numbers[b])))),0.)
               for a,b,distance in zip(i,j,d))


def propose_type1(parents,energies,regions,rng,*,min_ga,bond_limits,max_batches,
                  max_cut_attempts,max_pair_attempts,slots_per_parent=100,cuts_per_slot=10):
    """Complete bounded TYPE1 proposal batches; no relaxation/archive/controller.

    regions lists caller parent indices, first is native best region. All regions
    must partition parents, with >=2 regions. Includes rejected-filter counts and
    finite sampling failure records; no fabricated fallback or truncation of a
    passing batch. Parent selection above100 retains native last-index exclusion.
    """
    from .atomic_ga import _positive_int
    from .ga_operators import SamplingExhausted
    parents,e,_=_periodic_parents(parents,energies)
    for value,name in [(min_ga,'min_ga'),(max_batches,'max_batches')]:_positive_int(value,name)
    regions=tuple(tuple(r) for r in regions);flat=[i for r in regions for i in r]
    if len(regions)<2 or any(not r for r in regions) or sorted(flat)!=list(range(len(parents))):
        raise ValueError('at least two nonempty regions partitioning caller parents required')
    all_candidates=[];records=[]
    for batch in range(max_batches):
        selected=flat.copy()
        if len(selected)>100:
            ordered=sorted(selected,key=lambda i:e[i])
            selected=ordered[:100//3]+[ordered[int(rng.random()*(len(ordered)-1))] for _ in range(100-100//3)]
        try:
            pool=build_periodic_pool([parents[i] for i in selected],e[selected],rng,
                max_cut_attempts=max_cut_attempts,slots_per_parent=slots_per_parent,cuts_per_slot=cuts_per_slot)
            children=[]
            for _ in range(min_ga):
                c=cross_periodic_pool(pool,rng,max_pair_attempts=max_pair_attempts)
                details=dict(c.details);details['cell_parent_index']=selected[details['cell_parent_index']]
                children.append(PeriodicCandidate(c.atoms,tuple(selected[i] for i in c.atom_parent_indices),
                    c.source_atom_indices,c.operation,details))
            for group in [regions[0],tuple(i for r in regions[1:] for i in r)]:
                for c in mutate_type1([parents[i] for i in group],e[list(group)],min_ga//2,rng):
                    parent=group[c.parent_index]
                    source_ids=list(range(len(c.atoms)))
                    if c.operation=='exchange':
                        for i,j in c.details['selected_sites']:
                            source_ids[i],source_ids[j]=source_ids[j],source_ids[i]
                    children.append(PeriodicCandidate(c.atoms,(parent,)*len(c.atoms),tuple(source_ids),
                        c.operation,dict(c.details,cell_parent_index=parent)))
            passing=[]
            for c in children:
                if periodic_collision_free(c.atoms,bond_limits):
                    c.details['batch']=batch;passing.append(c)
            records.append(dict(index=batch,generated=len(children),passed=len(passing),
                rejected_collision=len(children)-len(passing),slots_per_parent=slots_per_parent,cuts_per_slot=cuts_per_slot))
            all_candidates.extend(passing)
            if not passing:return PeriodicProposal(tuple(all_candidates),'empty_batch',tuple(records))
            if len(all_candidates)>=min_ga:return PeriodicProposal(tuple(all_candidates),'target_reached',tuple(records))
        except SamplingExhausted as exc:
            records.append(dict(index=batch,status='sampling_exhausted',error=str(exc)))
            return PeriodicProposal(tuple(all_candidates),'sampling_exhausted',tuple(records))
    return PeriodicProposal(tuple(all_candidates),'budget_exhausted',tuple(records))
