"""Lift explicitly bonded molecules from periodic images; no bond inference."""
from dataclasses import dataclass
from collections import deque
from itertools import product
import numpy as np
from ase.geometry import minkowski_reduce,wrap_positions


@dataclass(frozen=True)
class UnwrappedRigidMolecules:
    atoms: object
    images: np.ndarray
    components: tuple
    provenance: dict


def unwrap_rigid_molecules(atoms,bonds):
    """BFS image lift using exact reduced-cell MIC for explicit zero-based bonds.

    Each bond-graph component keeps its lowest-index root at its original image.
    Degenerate shortest bond images and nonzero cycle winding are rejected.
    Equality tolerance is solely a floating-point cell/coordinate error scale,
    not a chemical radius or inferred connectivity threshold. PBC/cell/species
    are preserved; output positions = input positions + images @ input cell.
    """
    if not len(atoms) or not np.isfinite(atoms.positions).all() or not np.isfinite(atoms.cell.array).all():raise ValueError('finite nonempty atoms required')
    cell=atoms.cell.array;pbc=np.asarray(atoms.pbc,dtype=bool)
    if np.any(pbc) and abs(np.linalg.det(cell))<=np.finfo(float).tiny:raise ValueError('full rank cell required for periodic image lift')
    scale=max(1.,np.linalg.norm(cell),np.linalg.norm(atoms.positions,axis=1).max())
    tolerance=128*np.finfo(float).eps*scale*(max(1.,np.linalg.cond(cell)) if np.any(pbc) else 1.)
    reduced,_=minkowski_reduce(cell,pbc=pbc) if np.any(pbc) else (cell,None)
    shifts=np.array(list(product(*[(-1,0,1) if flag else (0,) for flag in pbc])))
    neighbors=[[] for _ in atoms];seen=set()
    for raw in bonds:
        pair=tuple(raw)
        if len(pair)!=2 or any(isinstance(i,(bool,np.bool_)) or not isinstance(i,(int,np.integer)) or not 0<=i<len(atoms) for i in pair) or pair[0]==pair[1]:raise ValueError('two distinct valid atom indices required')
        i,j=sorted(pair)
        if (i,j) in seen:raise ValueError('duplicate bond')
        seen.add((i,j));delta=atoms.positions[j]-atoms.positions[i]
        if np.any(pbc):
            wrapped=wrap_positions(delta[None,:],reduced,pbc=pbc,eps=0.)[0]
            vectors=wrapped+shifts@reduced;lengths=np.linalg.norm(vectors,axis=1)
            order=np.argsort(lengths);mic=vectors[order[0]]
            if len(order)>1 and lengths[order[1]]-lengths[order[0]]<=tolerance:
                raise ValueError(f'ambiguous shortest periodic image for bond {(i,j)}')
            image_float=np.linalg.solve(cell.T,mic-delta)
            image=np.rint(image_float).astype(np.int64)
            if np.linalg.norm(delta+image@cell-mic)>tolerance or np.any(image[~pbc]):raise ValueError('unresolved integer bond image')
        else:mic=delta;image=np.zeros(3,dtype=np.int64)
        if np.linalg.norm(mic)<=tolerance:raise ValueError('zero-length bonded axis')
        neighbors[i].append((j,image));neighbors[j].append((i,-image))
    images=np.zeros((len(atoms),3),dtype=np.int64);visited=set();components=[]
    for root in range(len(atoms)):
        if root in visited:continue
        queue=deque([root]);visited.add(root);component=[]
        while queue:
            i=queue.popleft();component.append(i)
            for j,image in neighbors[i]:
                expected=images[i]+image
                if j in visited:
                    if not np.array_equal(images[j],expected):raise ValueError('periodically winding bond cycle cannot be a finite rigid molecule')
                else:images[j]=expected;visited.add(j);queue.append(j)
        components.append(tuple(sorted(component)))
    out=atoms.copy();out.positions=atoms.positions+images@cell
    images.setflags(write=False)
    return UnwrappedRigidMolecules(out,images,tuple(components),dict(
        bonds=tuple(sorted(seen)),method='explicit bond BFS, Minkowski-reduced closest images',
        root_convention='lowest atom index unchanged per bond component',numerical_tolerance_A=tolerance,
        limitations='shortest bonded image must be unique; winding polymers rejected; no chemical bond inference'))
