"""Independent finite-domain port of uploaded Java population selection.

Source: SSWGaSupport.selectParentsByKMeans, lines 529-665 in the archived
CFR output. Row indices preserve object identity; rows contain energy and
sims. NumPy RNG is injectable but is NOT Java's random stream. These
legacy projection regions are not certified PES funnels or kinetic states.
"""
from dataclasses import dataclass
import numpy as np


def partition(rows, k, rng, *, max_draws=100000):
    """Return selected row indices per region, following legacy finite rules.

100 Lloyd iterations, first 3 projections, 20-member cap and 10000 energy
scale are recovered constants, not proposed universal defaults. max_draws
is a resource guard on legacy rejection sampling, never a fallback
selection: exceeding it raises rather than changing the distribution.
"""
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError('positive integer k required')
    if max_draws < 1:
        raise ValueError('positive rejection sampling budget required')
    n = len(rows)
    if not n:
        return []
    energies = np.asarray([r['energy'] for r in rows], dtype=float)
    if not np.isfinite(energies).all():
        raise ValueError('finite energies required')
    if n <= k:
        return [[i] for i in range(n)]
    valid = [i for i, row in enumerate(rows)
             if row.get('sims') is not None and len(row['sims']) >= 3]
    if not valid:
        return []
    features = np.asarray([rows[i]['sims'][:3] for i in valid], dtype=float)
    if not np.isfinite(features).all():
        raise ValueError('finite projections required')
    m = len(valid)
    k = min(k, m)
    chosen = []
    for _ in range(max_draws):
        index = int(rng.integers(m))
        if index not in chosen:
            chosen.append(index)
        if len(chosen) == k:
            break
    else:
        raise RuntimeError('initial-center sampling budget exhausted')
    centers = features[chosen].copy()
    labels = np.zeros(m, dtype=int)  # Original starts with all labels zero.
    for _ in range(100):
        new_labels = np.argmin(((features[:, None] - centers) ** 2).sum(axis=2), axis=1)
        changed = not np.array_equal(labels, new_labels)
        labels = new_labels
        for j in range(k):
            members = features[labels == j]
            centers[j] = members.mean(axis=0) if len(members) else features[int(rng.integers(m))]
        if not changed:
            break
    result = []
    for j in sorted(set(labels.tolist())):
        group = sorted((valid[i] for i in range(m) if labels[i] == j), key=lambda i: energies[i])
        if len(group) > 20:
            selected = [group[0]]
            weights = np.exp(-(energies[group] - energies[group[0]]) / 10000.)
            cumulative = np.cumsum(weights / weights.sum())
            for _ in range(max_draws):
                pick = int(np.searchsorted(cumulative, rng.random(), side='left'))
                if pick == len(group):
                    pick = int(rng.integers(len(group)))
                if group[pick] not in selected:
                    selected.append(group[pick])
                if len(selected) == 20:
                    break
            else:
                raise RuntimeError('region-member sampling budget exhausted')
            group = sorted(selected, key=lambda i: energies[i])
        result.append(group)
    return sorted(result, key=lambda group: energies[group[0]])


@dataclass(frozen=True)
class RankedRegion:
    indices: tuple
    score: float


def rank_regions(rows, regions):
    """Original 0.5 Emin + 0.3 Emean - 0.2 population_variance score.

Use the original energy convention (eV) for reproduction. The mixed
energy/energy-squared formula is empirical and not unit invariant.
"""
    result = []
    for indices in regions:
        if not len(indices):
            raise ValueError('empty region')
        if (any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
                or not 0 <= i < len(rows) for i in indices)
                or len(set(indices)) != len(indices)):
            raise ValueError('region indices must be distinct valid row integers')
        energies = np.asarray([rows[i]['energy'] for i in indices], dtype=float)
        if not np.isfinite(energies).all():
            raise ValueError('finite energies required')
        score = .5 * energies.min() + .3 * energies.mean() - .2 * energies.var()
        result.append(RankedRegion(tuple(indices), float(score)))
    return sorted(result, key=lambda region: region.score)
