"""Corrected image-aware three-shell NNA geometry for periodic GA routing.

Source structure: NeighbourPeriodic.getSecondNeighbor/getThirdNeighbor,
getConfigureInfo/getBaseDI; similarity projection: Classify.sims. Full ASE
periodic neighbor enumeration replaces finite native expansion, and distances
are center-specific rather than mutable shared neighbor objects. This is NOT
native slow/quick descriptor parity, structural identity or PES connectivity.
All bond reference lengths, neighbor multiplier and similarity weights must be
supplied. Native numerical quirks/changes are documented in periodic-descriptor.md.
"""
import math
import numpy as np
from ase.neighborlist import neighbor_list
from .periodic_softening import _identity
from .softening import _table,_positive
from .legacy_descriptor import descriptor_similarity


_KEYS=('n1','n2','n3','d1','d2','d3')


def periodic_descriptor(atoms,bond_lengths,neighbor_range):
    """Return N-row six-component descriptor, explicitly preserving image IDs.

    Nodes are (atom index, integer cell shift); shell k is graph distance k
    under the first-shell cutoff r < neighbor_range*b_species. Nonzero self
    images count normally; exact overlapping sites are rejected, not silently
    removed via the native quick branch's 0.1-A exclusion. Counts are per
    neighbor species; d1 is RMS(r-b/2) per species, d2/d3 pooled over each shell.
    Stable physical row ordering uses central species, all shell counts and
    distances; native sgn sorting omits the central species. Fixed-composition routing only:
    repetition preserves local row values but changes the row count.
    """
    numbers,_,_=_identity(atoms);z=np.asarray(numbers);n=len(z)
    table=_table(bond_lengths);multiplier=_positive(neighbor_range,'neighbor_range')
    elements=sorted(set(numbers))
    for a in elements:
        for b in elements:
            if tuple(sorted((a,b))) not in table:raise ValueError('complete explicit bond-length table required')
    radius=multiplier*max(table[tuple(sorted((a,b)))] for a in elements for b in elements)
    i,j,shifts,distances=neighbor_list('ijSd',atoms,np.nextafter(radius,np.inf),self_interaction=False)
    first=[set() for _ in range(n)]
    for a,b,s,r in zip(i,j,shifts,distances):
        if r<=0:raise ValueError('overlapping periodic sites are not valid routing inputs')
        if r<multiplier*table[tuple(sorted((int(z[a]),int(z[b]))))]:
            first[a].add((int(b),*map(int,s)))

    def expand(nodes):
        result=set()
        for atom,x,y,zshift in nodes:
            for target,u,v,w in first[atom]:result.add((target,x+u,y+v,zshift+w))
        return result

    second=[expand(first[k])-first[k]-{(k,0,0,0)} for k in range(n)]
    third=[expand(second[k])-first[k]-second[k]-{(k,0,0,0)} for k in range(n)]
    shells=[first,second,third]
    counts=[[[sum(z[node[0]]==element for node in shell[k]) for element in elements] for k in range(n)] for shell in shells]

    def rms(center,nodes):
        terms=[]
        for atom,x,y,zshift in sorted(nodes):
            r=float(np.linalg.norm(atoms.positions[atom]-atoms.positions[center]+np.array([x,y,zshift])@atoms.cell.array))
            b=table[tuple(sorted((int(z[center]),int(z[atom]))))]
            terms.append((r-.5*b)**2)
        return math.sqrt(math.fsum(terms)/len(terms)) if terms else 0.

    d1=[[rms(k,[node for node in first[k] if z[node[0]]==element]) for element in elements] for k in range(n)]
    d2=[rms(k,second[k]) for k in range(n)];d3=[rms(k,third[k]) for k in range(n)]
    order=sorted(range(n),key=lambda k:(int(z[k]),*counts[0][k],*counts[1][k],*counts[2][k],*d1[k],d2[k],d3[k]))
    arrays=dict(n1=counts[0],n2=counts[1],n3=counts[2],d1=d1,d2=d2,d3=d3)
    result={key:np.asarray(value)[order].tolist() for key,value in arrays.items()}
    result.update(composition=sorted(numbers),elements=elements,
        contract='corrected_periodic_image_shells_v1',neighbor_range=multiplier,
        bond_lengths={f'{a}-{b}':v for (a,b),v in sorted(table.items())})
    return result


def periodic_projection(descriptor,bases,weights):
    """Exactly three explicit fixed basis descriptors → three similarities.

    This is the Classify.sims projection form with corrected periodic geometry.
    Three coordinates suit existing population.partition, which only consumes
    its first three entries. Caller fixes these bases for the entire campaign;
    no random virtual basis or changing archive-dependent embedding is hidden.
    Require same composition and same geometric definition across all bases.
    """
    if len(bases)!=3:raise ValueError('exactly three explicit fixed periodic bases required')
    for basis in bases:
        for field in ['contract','composition','elements','neighbor_range','bond_lengths']:
            if descriptor.get(field)!=basis.get(field):raise ValueError('basis composition/descriptor contract mismatch')
        for key in _KEYS:
            x=np.asarray(descriptor[key],dtype=float);y=np.asarray(basis[key],dtype=float)
            if x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(x<0) or np.any(y<0):
                raise ValueError('finite nonnegative matching descriptor arrays required')
    return [descriptor_similarity(descriptor,basis,weights) for basis in bases]
