"""Research-only reconstruction of the finite isolated-coordinate vapor routine.

No native execution or calculator dependency. Mode 0 reports the nearest
separation from the medoid's connected component, not the global component-pair
minimum. Mode 1 sequentially translates remaining components and returns the
LAST pretranslation separation. Native floating-point medoid ties remain a
parity boundary until checked against the vectorized original reduction.
"""
import numpy as np


def vapor_reference(positions, criterion, *, repair=False):
    x = np.array(positions, dtype=float, copy=True)
    if x.ndim != 2 or x.shape[1] != 3 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('at least two finite isolated Cartesian positions required')
    if not np.isfinite(criterion) or criterion <= 0:
        raise ValueError('positive finite distance criterion required')
    distances = np.linalg.norm(x[:, None] - x[None, :], axis=2)
    # Original code accumulates alternating SIMD lanes before reducing the pair.
    scores = distances[:, ::2].sum(axis=1) + distances[:, 1::2].sum(axis=1)
    medoid = int(np.argmin(scores))
    order = list(range(len(x)))
    order[0], order[medoid] = order[medoid], order[0]

    def expand(start, end):
        cursor = start
        while cursor < end:
            atom = order[cursor]
            for j in range(end, len(x)):
                if np.linalg.norm(x[atom] - x[order[j]]) < criterion:
                    order[end], order[j] = order[j], order[end]
                    end += 1
            cursor += 1
        return end

    end = expand(0, 1)
    initial_component = order[:end].copy()
    moves = []
    result = 0.0
    while end < len(x):
        best = float('inf')
        chosen = None
        displacement = None
        for j in range(end, len(x)):
            for i in range(end):
                vector = x[order[i]] - x[order[j]]
                distance = float(np.linalg.norm(vector))
                if distance < best:
                    best, chosen, displacement = distance, j, vector
        result = best
        if not repair:
            break
        order[end], order[chosen] = order[chosen], order[end]
        new_end = expand(end, end + 1)
        translation = displacement * ((best - 0.7 * criterion) / best)
        members = order[end:new_end].copy()
        x[members] += translation
        moves.append(dict(members=members, separation=best, translation=translation.tolist()))
        end = new_end
    return dict(positions=x, scalar=result, medoid=medoid,
                initial_component=initial_component, moves=moves)
