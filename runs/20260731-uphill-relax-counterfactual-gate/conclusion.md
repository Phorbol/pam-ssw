# G-UP0 C60 uphill proposal-relax counterfactual conclusion

## Decision

The current biased-PES proposal relaxation is **not causally redundant**.  It
must remain in the reference SSW kernel until a shorter relaxation is shown to
retain the same basin-level effect.

This is narrower than saying the current 80-step relaxation is optimal.  The
gate isolates only the difference between no biased relaxation and the current
relaxed checkpoint.  It neither promotes the current length nor supports a new
adaptive controller.

## Why this is a clean comparison

For every recorded macro step, the two arms share the same starter state,
selected direction, HVP result, execution scale and accumulated Gaussian
biases.  The explicit arm is trajectory frame zero,

\[
x_k^{\mathrm{explicit}}=x_k+\sigma_k u_k,
\]

which every existing Relaxer backend records before its first optimizer step.
The relaxed arm is the corresponding production checkpoint after minimizing
the biased PES.  Both states use the same true-PES quench and basin
classification.  Replaying the explicit arm performs no direction HVP and no
biased relaxation.

The immutable source is the 24-path first-passage corpus with SHA256
`9ba5fb4fce7b44dea7cfa4bc0115043e3877adc08c345eaae9aa5fc0117fbdd2`.
The compact full G-UP0 evidence has SHA256
`db0dbb47c223149611ae3b5a5eea8b6be90339a725c40e44aa219d58cecc7dfe`.

## Basin-level result

The 34 C60 pairs give:

| Paired result | Count |
|---|---:|
| Both arms return to the starter | 19 |
| Both escape to the same minimum | 4 |
| Both escape, but to different minima | 3 |
| Only the bias-relaxed arm escapes | **5** |
| Numerically or structurally unlearnable | 3 |

The preregistered repeated-context rule is met twice:

1. plateau starter, D0 exact-anchor, horizon 1: relaxed-only escape for seeds
   42 and 43; seed 44 returns in both arms;
2. plateau starter, K4 discrete, horizon 2: relaxed-only escape for seeds 43
   and 44; seed 42 reaches escaped minima in both arms but their identity is
   ambiguous under matcher-versus-descriptor comparison.

These are not energy-ranking effects.  Before true quench, the explicit and
relaxed states differ only by the motion produced while descending the
accumulated biased PES.  In the repeated contexts, that motion moves the state
across the original basin's attraction boundary; the explicit displacement
alone is pulled back to the starter.

At the intermediate starter, all complete h1/h2 contexts return in both arms.
At h4, D0 produces one same escaped landing and one different escaped landing,
whereas all three K4 pairs still return.  The role of proposal relaxation is
therefore state- and horizon-dependent, not a universal monotonic energy
improvement.

## Cost

The explicit replay consumed 2,886/5,000 new force evaluations:

- 34 explicit-state true-PES energy checks;
- 2,818 true-quench evaluations;
- 34 post-relax validation evaluations;
- zero direction HVP;
- zero biased-PES proposal relaxation;
- zero unattributed evaluations.

Per explicit branch the FE distribution is 22 minimum, 38 median, 84.9 mean
and 526 maximum.  Three long true quenches (486, 506 and 526 FE) create most of
the mean--median gap.  The corresponding 34 source relaxed checkpoints used
3,548 quench/validation FE, with median 44 and mean 104.4 FE.  Thus the current
proposal relaxation is not justified as a general true-quench preconditioner:
the relaxed endpoints did not reduce aggregate downstream quench cost in this
cohort.

The source C60 action corpus used 3,626 generation FE, of which 3,130 were
biased proposal relaxation, 406 direction-oracle evaluations and 90 true-PES
checks.  This generation total includes source steps after h4 for paths that
continued, so it cannot be divided into an exact per-pair online cost.  It is
reported as provenance, not added to the 2,886 new-FE gate budget.

GPU kernel wall time was 57.4 s for the explicit gate.  The sum of pair quench
times was 56.0 s.  The reused source corpus recorded 73.2 s of C60 generation
and 68.0 s for the selected relaxed checkpoint quenches.

## Physical interpretation

The Gaussian center is the pre-displacement structure, while the explicit
move starts near one Gaussian width along the chosen direction.  The following
biased minimization does not simply extend that direction.  Orthogonal true-PES
forces relax hard local distortions while the accumulated Gaussian gradients
deflect the path away from previously visited centers.  The earlier trajectory
audit found a micro-path-length/net-displacement ratio near 2.8 for C60; the
present basin labels show that this curved motion sometimes has a discrete
topological consequence.

The three `DIFFERENT_ESCAPED_LANDINGS` pairs reinforce the same point: even
when the explicit state has already crossed a basin boundary, biased
relaxation can redirect it into another attraction basin.  Conversely, two
relaxed checkpoints are geometrically invalid while their explicit states
quench to certified escaped minima.  The propagator is therefore active but
not uniformly beneficial.

## Research consequence

The following claims are now closed:

- removing proposal relaxation and retaining only the current explicit SSW
  displacement;
- treating proposal relaxation as a purely numerical backend with no effect
  on action support;
- using this gate to claim that 80 optimizer steps, adaptive Gaussian weights,
  or the current trust controller are optimal.

The next admissible propagator experiment is a **within-relaxation
first-passage gate**, restricted to the five relaxed-only-escape pairs and the
three different-landing pairs.  It should quench preregistered existing
optimizer frames and determine the earliest accepted relaxation step at which
the basin label changes.  That gate asks whether most of the 3,130 proposal
relax FE can be removed while preserving the demonstrated path-bending effect.
It should precede Gaussian-shape changes, OPES-like deposition, CCQN, or a new
adaptive trust mechanism.

No selector, direction family, posterior model, production default or
system-specific parameter is changed by G-UP0.

## Claim ceiling

This is a 34-pair local C60 counterfactual branching from states generated by
the original relaxed SSW path.  It proves a repeatable basin-level contribution
of nonzero biased relaxation in two tested contexts.  It does not construct an
online explicit-only trajectory, determine the shortest sufficient relax,
or establish transfer to PdO or CuO.
