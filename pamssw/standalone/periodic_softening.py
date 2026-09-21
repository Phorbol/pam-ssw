"""Smooth frozen periodic-image LS bonds on a fixed unwrapped coordinate lift.

Each (i,j,S) and its inverse (j,i,-S) count once per cell. All images within
explicit reference cutoffs are retained, including nonzero self images. This
is an independently derived periodic extension, not native MIC trajectory parity.
"""
from dataclasses import dataclass
import math
import numpy as np
from ase.neighborlist import neighbor_list
from .softening import _table, _positive, _energy_filter, _energy_factor, EnergyFilter


def _identity(atoms):
    if len(atoms)<1 or not atoms.pbc.any() or atoms.constraints:
        raise ValueError('periodic LS requires nonempty unconstrained structure with PBC')
    if len(atoms.pbc) != 3 or any(not isinstance(v, (bool, np.bool_)) for v in atoms.pbc):
        raise ValueError('PBC must be a length-3 boolean mask')
    if not np.isfinite(atoms.positions).all() or not np.isfinite(atoms.cell).all() or np.linalg.det(atoms.cell.array)<=0:
        raise ValueError('finite positions and positive cell determinant required')
    return (tuple(map(int,atoms.numbers)),tuple(map(tuple,atoms.cell.array)),tuple(map(bool,atoms.pbc)))


@dataclass(frozen=True)
class FrozenPeriodicBondSoftening:
    numbers: tuple
    cell: tuple
    pbc: tuple
    pairs: tuple
    image_shifts: tuple
    reference_distances: tuple
    strengths: tuple
    xi: float=.2
    energy_filter: tuple = ()

    def __post_init__(self):
        n=len(self.numbers);cell=np.asarray(self.cell,dtype=float);pbc=tuple(self.pbc)
        if not n or cell.shape!=(3,3) or not np.isfinite(cell).all() or np.linalg.det(cell)<=0:
            raise ValueError('valid positive-volume periodic identity required')
        if len(pbc)!=3 or any(not isinstance(v,(bool,np.bool_)) for v in pbc) or not any(pbc):
            raise ValueError('periodic PBC must be a length-3 boolean mask with at least one axis')
        if not len(self.pairs) or not (len(self.pairs)==len(self.image_shifts)==len(self.reference_distances)==len(self.strengths)):
            raise ValueError('nonempty matching periodic bond arrays required')
        seen=set()
        for (i,j),shift in zip(self.pairs,self.image_shifts):
            if any(isinstance(k,(bool,np.bool_)) or not isinstance(k,(int,np.integer)) or not 0<=k<n for k in (i,j)):
                raise ValueError('invalid atom index')
            if len(shift)!=3 or any(isinstance(k,(bool,np.bool_)) or not isinstance(k,(int,np.integer)) for k in shift):
                raise ValueError('integer image shift required')
            if any(not pbc[axis] and shift[axis] != 0 for axis in range(3)):
                raise ValueError('nonperiodic axes require zero image shift')
            if i==j and not any(shift):raise ValueError('zero-image self bond')
            key=min((i,j,*shift),(j,i,*(-x for x in shift)))
            if key in seen:raise ValueError('duplicate undirected periodic bond')
            seen.add(key)
        for r in self.reference_distances:_positive(r,'reference distance')
        if any(not np.isfinite(a) or a<0 for a in self.strengths):raise ValueError('nonnegative finite strengths required')
        _positive(self.xi,'xi')
        energy_filter = _energy_filter(self.energy_filter)
        for name in ('numbers','pbc','reference_distances','strengths'):
            object.__setattr__(self,name,tuple(getattr(self,name)))
        object.__setattr__(self,'energy_filter',energy_filter)
        for name in ('cell','pairs','image_shifts'):
            object.__setattr__(self,name,tuple(tuple(v) for v in getattr(self,name)))

    @classmethod
    def from_atoms(cls,atoms,*,bond_energies,bond_lengths,initial_fraction=.03,xi=.2,energy_filter=None):
        return cls._build(atoms,bond_energies,bond_lengths,xi,initial_fraction=_positive(initial_fraction,'initial_fraction'),energy_filter=energy_filter)

    @classmethod
    def _build(cls,atoms,bond_energies,bond_lengths,xi,*,initial_fraction=None,total_strength=None,energy_filter=None):
        numbers,cell,pbc=_identity(atoms);energies,lengths=_table(bond_energies),_table(bond_lengths)
        energy_filter = _energy_filter(energy_filter)
        elements=sorted(set(numbers))
        for a in elements:
            for b in elements:
                key=tuple(sorted((a,b)))
                if key not in energies or key not in lengths:raise ValueError(f'missing bond table entry for {key}')
        # nextafter includes the explicitly defined <= cutoff boundary.
        i,j,shifts,distances=neighbor_list('ijSd',atoms,np.nextafter(max(lengths.values()),np.inf),self_interaction=False)
        bonds={}
        for a,b,shift,r in zip(i,j,shifts,distances):
            key=tuple(sorted((numbers[a],numbers[b])))
            if r<=0:raise ValueError('zero bond distance')
            if r>lengths[key]:continue
            shift=tuple(map(int,shift));canonical=min((int(a),int(b),*shift),(int(b),int(a),*(-x for x in shift)))
            bonds[canonical]=(float(r),energies[key])
        if not bonds:raise ValueError('no bonded pairs: LS response is undefined')
        ordered=sorted(bonds)
        weights=[bonds[k][1] * _energy_factor(
            energy_filter, tuple(sorted((numbers[k[0]], numbers[k[1]])))) for k in ordered]
        if not math.isfinite(math.fsum(weights)) or math.fsum(weights) <= 0:
            raise ValueError('all eligible LS energy weights are zero')
        strengths=(tuple(initial_fraction*w for w in weights) if total_strength is None else
                   tuple(total_strength*w/math.fsum(weights) for w in weights))
        return cls(numbers,cell,pbc,tuple(k[:2] for k in ordered),tuple(k[2:] for k in ordered),
                   tuple(bonds[k][0] for k in ordered),strengths,xi,energy_filter)

    def _validate_atoms(self,atoms):
        if _identity(atoms)!=(self.numbers,self.cell,self.pbc):
            raise ValueError('atom identity, PBC and fixed cell must match frozen periodic bonds')

    def _vectors(self,atoms):
        self._validate_atoms(atoms)
        pairs=np.asarray(self.pairs,dtype=int)
        return atoms.positions[pairs[:,1]]-atoms.positions[pairs[:,0]]+np.asarray(self.image_shifts)@atoms.cell.array

    def pair_distances(self,atoms):
        return np.linalg.norm(self._vectors(atoms),axis=1)

    def evaluate(self,atoms):
        delta=self._vectors(atoms);r=np.linalg.norm(delta,axis=1)
        if np.any(r<=0):raise ValueError('positive periodic bond distance required')
        scale=self.xi*np.asarray(self.reference_distances)
        energy=np.asarray(self.strengths)*np.exp(-(r-np.asarray(self.reference_distances))/scale)
        outward=(energy/(scale*r))[:,None]*delta
        if not np.isfinite(energy).all() or not np.isfinite(outward).all():raise FloatingPointError('nonfinite periodic LS potential')
        forces=np.zeros((len(atoms),3));pairs=np.asarray(self.pairs)
        active=pairs[:,0]!=pairs[:,1]  # self-image bond has identically zero atomic derivative
        np.add.at(forces,pairs[active,0],-outward[active]);np.add.at(forces,pairs[active,1],outward[active])
        return math.fsum(energy),forces
