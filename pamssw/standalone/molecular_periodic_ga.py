"""TYPE2 molecular-crystal proposal mechanics from CrossMC and ReComMC.

Unlike TYPE1 cell-transfer crossover, the native molecular operation docks
whole molecules and generates a NEW orthogonal cell from extents+0.5 A.
TYPE2 is therefore not TYPE3 with inherited PBC: its split is half CrossMC,
half accuracy-3 molecular reconstruction, followed by a periodic image filter.
No calculator, molecular topology inference or RC overlapping-body alias.
"""
from dataclasses import dataclass
import numpy as np
from ase import Atoms
from .periodic_softening import _identity
from .periodic_ga import periodic_collision_free
from .ga_operators import (GeneticCandidate,SamplingExhausted,
    _partition,_molecular_pool,_cross_from_pool,_fit_binding,_monomer_radius,
    _flatten_fragments,rotate_coordinates)


@dataclass(frozen=True)
class MolecularPeriodicProposal:
    candidates: tuple
    status: str
    batches: tuple


def _count(value,name,minimum=0):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or value<minimum:
        raise ValueError(f'{name} must be integer >= {minimum}')


def _molecular_lift(atoms,groups,image_shifts=None):
    """Explicit complete-molecule Cartesian lift; no inferred MIC bond graph.

    Without shifts, caller asserts each group is already an intact unwrapped
    molecule. With shifts, r'=r+S@cell is applied before any molecular operation.
    Groups must partition atoms contiguously in original molecule-ID order.
    """
    _identity(atoms)
    positions=atoms.positions.copy()
    if image_shifts is not None:
        shifts=np.asarray(image_shifts)
        if shifts.shape!=positions.shape or not np.isfinite(shifts).all() or not np.equal(shifts,np.rint(shifts)).all():
            raise ValueError('explicit image shifts require finite integer N x 3 entries')
        positions+=shifts@atoms.cell.array
    lifted=Atoms(numbers=atoms.numbers,positions=positions)
    groups=_partition(lifted,groups)
    if len(groups)<2 or [i for g in groups for i in g]!=list(range(len(atoms))):
        raise ValueError('at least two contiguous disjoint complete molecules required; overlapping RC bodies unsupported')
    return lifted,groups


def _periodic(atoms):
    result=atoms.copy();result.calc=None
    result.set_cell(np.ptp(result.positions,axis=0)+.5)
    result.pbc=True
    return result


def _reconstruct(lifted,groups,n,rng):
    fragments=[lifted[list(g)] for g in groups]
    radii=[_monomer_radius(f.positions) for f in fragments]
    for fragment in fragments:fragment.positions-=fragment.positions.mean(axis=0)
    results=[]
    for _ in range(n):
        # Deterministic serial sibling: source rotations accumulate over n.
        # Parallel MCP mutates a shared list and is not a reproducible contract.
        for fragment in fragments:fragment.positions=rotate_coordinates(fragment.positions,rng)
        order=list(range(len(fragments)))
        for _ in range(10*len(order)):
            i,j=(int(rng.random()*len(order)) for _ in range(2));order[i],order[j]=order[j],order[i]
        combined=fragments[order[0]].copy();radius=radii[order[0]]
        for index in order[1:]:
            fit=_fit_binding(combined,fragments[index],min_distance=1.5,accuracy=3,radii=(radius,radii[index]))
            combined=fit.atoms;combined.positions-=combined.positions.mean(axis=0)
            radius=_monomer_radius(combined.positions)
        offset=0;by_id={}
        for index in order:
            size=len(fragments[index]);by_id[index]=combined[offset:offset+size];offset+=size
        atoms,new_groups=_flatten_fragments([by_id[k] for k in range(len(groups))])
        results.append(GeneticCandidate(_periodic(atoms),new_groups,'periodic_reconstruction',(0,)*len(groups),
            dict(docking_order=tuple(order),docking_accuracy=3,min_docking_distance_A=1.5,
                 cell_rule='new orthogonal Cartesian extents + 0.5 A',
                 parallel_shared_rotation_corrected='serial ReComMC sibling with cumulative rotations')))
    return results


def reconstruct_type2(atoms,groups,n,rng,*,image_shifts=None):
    """ReComMC serial geometry; n whole-molecule reconstructed periodic cells."""
    _count(n,'n');lifted,groups=_molecular_lift(atoms,groups,image_shifts)
    return _reconstruct(lifted,groups,n,rng)


def propose_type2(parents,energies,groups,rng,*,min_ga,bond_limits,max_batches,
                  max_cut_attempts,max_pair_attempts,image_shifts=None):
    """Bounded TYPE2 batch: floor(G/2) CrossMC + remainder reconstruction.

    Parent0 provides reconstruction molecules, as in TYPE2.getGA; it is not
    silently replaced by the lowest-energy parent. All parents must share
    ordered molecular species topology. Per-parent image_shifts may be supplied.
    Periodic filtering uses corrected full-image geometry, not native2x2x2.
    Return MolecularPeriodicProposal and GeneticCandidate with molecule-level lineage.
    """
    for value,name in [(min_ga,'min_ga'),(max_batches,'max_batches'),(max_cut_attempts,'max_cut_attempts'),(max_pair_attempts,'max_pair_attempts')]:_count(value,name,1)
    parents=tuple(parents);energy=np.asarray(energies,dtype=float)
    if not parents or energy.shape!=(len(parents),) or not np.isfinite(energy).all():raise ValueError('parents and finite matching energies required')
    if image_shifts is None:image_shifts=[None]*len(parents)
    if len(image_shifts)!=len(parents):raise ValueError('one image-shift array per parent required')
    lifted=[]
    for a,shifts in zip(parents,image_shifts):
        p,groups=_molecular_lift(a,groups,shifts);lifted.append(p)
        if not np.array_equal(p.numbers,parents[0].numbers):raise ValueError('ordered molecular species topology must match')
    candidates=[];batches=[]
    for batch in range(max_batches):
        private=[p.copy() for p in lifted]
        try:
            # Actual native CrossMC also used by TYPE3; exact shared primitive.
            sons,daughters=_molecular_pool(private,energy,groups,rng,max_cut_attempts)
            children=[]
            for _ in range(min_ga//2):
                c=_cross_from_pool(sons,daughters,len(groups),rng,max_pair_attempts)
                children.append(GeneticCandidate(_periodic(c.atoms),c.groups,c.operation,c.group_parent_indices,
                    dict(c.details,cell_rule='new orthogonal Cartesian extents + 0.5 A')))
            children.extend(_reconstruct(private[0],groups,min_ga-min_ga//2,rng))
            passed=[]
            for c in children:
                if periodic_collision_free(c.atoms,bond_limits):
                    c.details['batch']=batch;passed.append(c)
            batches.append(dict(index=batch,generated=len(children),passed=len(passed),rejected_collision=len(children)-len(passed)))
            candidates.extend(passed)
            if not passed:return MolecularPeriodicProposal(tuple(candidates),'empty_batch',tuple(batches))
            if len(candidates)>=min_ga:return MolecularPeriodicProposal(tuple(candidates),'target_reached',tuple(batches))
        except SamplingExhausted as error:
            batches.append(dict(index=batch,error=str(error)))
            return MolecularPeriodicProposal(tuple(candidates),'sampling_exhausted',tuple(batches))
    return MolecularPeriodicProposal(tuple(candidates),'budget_exhausted',tuple(batches))
