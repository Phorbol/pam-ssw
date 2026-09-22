"""Experimental behavioral reconstruction of the supplied GA-SSW Java release.

This module intentionally preserves observed legacy semantics. It is not a
canonical DCCD implementation, a production archive, or a full GA-SSW walker.
Geometry support is currently the nonperiodic NNA branch only. Callers supply
the original element-pair bond-length table; no physical defaults are invented.
"""
from __future__ import annotations
import copy
import math
from typing import Mapping, Sequence
import numpy as np


def _full_fingerprint_order(descriptor):
    """Return an untouched descriptor copy ordered by every row component."""
    fields = ('n1', 'n2', 'n3', 'd1', 'd2', 'd3')
    if not isinstance(descriptor, Mapping) or any(field not in descriptor for field in fields):
        raise TypeError('full_fingerprint ordering requires a complete descriptor mapping')
    length = len(descriptor['n1'])
    if any(len(descriptor[field]) != length for field in fields):
        raise ValueError('descriptor fields have inconsistent row counts')

    def fingerprint(index):
        key = []
        for field in fields:
            value = descriptor[field][index]
            if field in ('n1', 'n2', 'n3', 'd1'):
                key.extend(value)
            else:
                key.append(value)
        return tuple(key)

    order = sorted(range(length), key=fingerprint)
    return {field: [copy.deepcopy(descriptor[field][index]) for index in order]
            for field in fields}


def cluster_descriptor(numbers, positions, bond_lengths: Mapping, neighbor_range: float):
    """Reconstruct nna.Neighbour's nonperiodic three-shell descriptor.

    First-shell distance is RMS(r - b/2), higher shells RMS(r - 3b).
    Atom rows are stably sorted by counts only, preserving the original tie
    behavior (and therefore its possible atom-order dependence).
    """
    z=np.asarray(numbers,dtype=int); xyz=np.asarray(positions,dtype=float)
    if xyz.shape!=(len(z),3) or not len(z) or not np.isfinite(xyz).all():
        raise ValueError('expected nonempty finite Nx3 coordinates')
    if not math.isfinite(neighbor_range) or neighbor_range<=0:
        raise ValueError('neighbor_range must be positive and finite')
    elements=sorted(set(z.tolist()))
    def bond(i,j):
        key=tuple(sorted((int(z[i]),int(z[j]))))
        value=float(bond_lengths[key])
        if not math.isfinite(value) or value<=0: raise ValueError('invalid bond length')
        return value
    n=len(z)
    distance=np.linalg.norm(xyz[:,None,:]-xyz[None,:,:],axis=2)
    first=[set() for _ in z]
    for i in range(n):
        for j in range(i+1,n):
            if distance[i,j]<=bond(i,j)*neighbor_range:
                first[i].add(j);first[j].add(i)
    second=[set().union(*(first[j] for j in first[i]))-first[i]-{i} for i in range(n)]
    third=[set().union(*(first[j] for j in second[i]))-first[i]-second[i]-{i} for i in range(n)]
    counts=[[[sum(z[j]==element for j in neighbors[i]) for element in elements] for i in range(n)] for neighbors in (first,second,third)]
    d1=[]
    for i in range(n):
        row=[]
        for element in elements:
            js=sorted(j for j in first[i] if z[j]==element)
            row.append(math.sqrt(sum((distance[i,j]-.5*bond(i,j))**2 for j in js)/len(js)) if js else 0.)
        d1.append(row)
    high=[]
    for neighbors in (second,third):
        high.append([math.sqrt(sum((distance[i,j]-3.*bond(i,j))**2 for j in sorted(neighbors[i]))/len(neighbors[i])) if neighbors[i] else 0. for i in range(n)])
    order=sorted(range(n),key=lambda i: counts[0][i]+counts[1][i]+counts[2][i])
    arrays=dict(n1=counts[0],n2=counts[1],n3=counts[2],d1=d1,d2=high[0],d3=high[1])
    return {key:np.asarray(value)[order].tolist() for key,value in arrays.items()}


def _vector_similarity(a,b):
    if len(a)!=len(b): raise ValueError('descriptor lengths differ')
    result=0.
    weight=1./len(a) if len(a) else 1.
    for x,y in zip(a,b):
        result+=(1. if x==0 and y==0 else 1.-abs(x-y)/max(x,y))*weight
    return result


def descriptor_similarity(a,b,weights: Sequence[float]):
    """Original NNA six-component weighted similarity on extracted descriptors."""
    if len(weights)!=6 or not all(math.isfinite(x) and x>=0 for x in weights) or sum(weights)<=0:
        raise ValueError('six nonnegative finite weights with positive sum required')
    values=[]
    for key in ('n1','n2','n3','d1'):
        x,y=a[key],b[key]
        if len(x)!=len(y): raise ValueError('descriptor atom counts differ')
        values.append(sum(_vector_similarity(row,other)/len(x) for row,other in zip(x,y)))
    values.extend(_vector_similarity(a[key],b[key]) for key in ('d2','d3'))
    return sum(value*(weight/sum(weights)) for value,weight in zip(values,weights))


def same_projection(a,b,tolerance):
    """Componentwise original projection threshold, not a structural equivalence proof."""
    if len(a['sims'])!=len(b['sims']): raise ValueError('projection dimensions differ')
    return all(abs(x-y)<=tolerance for x,y in zip(a['sims'],b['sims']))


def remove_duplicates(rows,tolerance):
    result=[]
    for current in rows:
        for i,existing in enumerate(result):
            if same_projection(current,existing,tolerance):
                if current['energy']<existing['energy']:result[i]=current
                break
        else: result.append(current)
    return result


def merge_archive(old,new,tolerance):
    """Legacy merge: incoming rows must already be deduplicated by the caller.

    Newly collected rows are compared only to the old archive in this function,
    exactly as in Classify.mergeModelSimsBySimilarity, not to one another.
    """
    result=list(old);added=[]
    for current in new:
        for i,existing in enumerate(result):
            if same_projection(current,existing,tolerance):
                if current['energy']<existing['energy']:result[i]=current
                break
        else: added.append(current)
    return result+added


def energy_window(rows,window):
    if not rows:return []
    minimum=min(r['energy'] for r in rows)
    return [r for r in rows if r['energy']-minimum<=window]


def compete_cumulative(energies):
    """Original Compete weights, defined here only for n>2 and positive span.

    Degenerate populations fail explicitly instead of propagating legacy NaNs.
    Constants and 2.7183 base intentionally match the binary, not new defaults.
    """
    n=len(energies)
    if n<=2 or not all(math.isfinite(e) for e in energies):
        raise ValueError('requires more than two finite energies')
    low=min(energies);span=max(energies)-low
    if span<=0:raise ValueError('positive energy span required')
    temperature=-span*1000.*2625./(math.log(2./n)*8.314)
    total=0.;out=[]
    for e in energies:
        total+=math.pow(2.7183,-(e-low)*1000.*2625./(8.314*temperature));out.append(total)
    return out
