# Block atomic-climb cost and stopping audit

This zero-PES audit reads the saved Fe7C3 block results. It does not infer
basin identity from energy and does not add a stopping heuristic or extend the
budget.

The reference is the outer starting enthalpy
`E_ref = E_outer - pV_outer` passed to `atomic_climb` at
`pamssw/standalone/block_ssw.py:130-132`. Here `p=0`, so both seeds have
`E_ref = -682.1259013997811 eV`. For each completed Gaussian,
`delta_true = true_energy - E_ref`; the stored bias height is
`weight = (forward_force - force_parallel) * width * exp(0.5)` from
`pamssw/standalone/atomic_climb.py:117-123`.

## Completed prefixes

| seed | completed Gaussian prefix | `delta_true` (eV), in order | atomic status | checkpoint boundary |
|---|---:|---|---|---|
| 7 | 9 | 20.5217, 23.3423, 23.2680, 25.8524, 23.7085, 25.6758, 21.6817, 25.8722, 25.2984 | `evaluation_failed` at budget | `next_index=9`, pending index 9, stage `biased_quench` |
| 101 | 8 | 514.6591, 512.7575, 516.1563, 508.1909, 511.1544, 514.0192, 517.4690, 513.8081 | `evaluation_failed` at budget | `next_index=8`, pending index 8, stage `biased_quench` |

Every completed true energy is above the outer reference. Neither run stopped
because it found `true_energy < E_ref`; both stopped while attempting the next
biased quench after the request cap. Therefore the records do not support the
hypothesis that a low-than-original-energy stop caused the climb to terminate,
and they also did not complete the configured Gaussian count.

The prefixes are not monotone. Seed 7 rises from +20.52 to +23.34, briefly
dips to +23.27, then alternates further. Seed 101 dips from +514.66 to
+512.76, then alternates. This establishes only the observed energy sequence;
it says nothing about basin crossing or structural identity.

## Per-Gaussian costs and directions

The machine-readable companion records `bias_weight`, direction norm, angle
to the previous direction, rotation force requests, and biased-quench requests
for every completed event. Rotation costs are 10 requests per event for seed 7
(12 at event 8), and 42, 38, 40, 40, 42, 40, 40, 44 for seed 101. Biased
quench costs are respectively `253, 74, 147, 104, 267, 224, 165, 85, 75`
and `142, 149, 134, 167, 143, 104, 168, 145`. Direction angles are retained
to expose changes in the sampled mode; they are not treated as a basin metric.

The outer index-0 cell-only preparation had 5 completed cell cycles and cost
222 EFS (seed 7) or 221 EFS (seed 101), yielding rejected high-energy landings.
Outer index 1 then ran 5 completed cell cycles and entered atomic climbing;
the atomic portion consumed 1620 EFS (seed 7) or 1613 EFS (seed 101) before
budget exhaustion. The pending displaced geometry is present, but no partial
optimizer final q is promoted.

The detailed records are in
`research/ga_ssw/fe7c3-block-baseline/atomic-cost-summary.json`. These results
justify preserving the budget failure and reporting the completed prefix; they
do not by themselves justify a policy change or a scientific performance
claim.
