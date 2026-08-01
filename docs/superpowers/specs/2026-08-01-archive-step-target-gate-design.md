# Archive-Scaled Uphill Target Gate Design

## Problem

The completed U0--U4 sequence already isolated the within-walk mechanisms:

- local sigma and normalized-curvature feedback;
- Gaussian versus center-matched quadratic shape;
- cumulative versus newest-only Gaussian history;
- proposal-relaxation capacity and local-softening scope;
- eight versus fourteen serial microsteps;
- biased-relaxation basin first passage.

Repeating those comparisons would be marginal tuning. One distinct adaptive
block remains un-attributed. Before every macro action, `SurfaceWalker.run()`
calls `StepTargetController.target(archive)`. Once the archive has two minima,
the target is no longer the configured `target_uphill_energy=0.8 eV`; it is

\[
E_{\rm target}
=\operatorname{clip}\left[
0.2\,S(\{E_i-E_{\min}\}),\;0.05E_0,\;5E_0
\right],
\]

where `S` is a median/MAD-based archive energy scale and `E0=0.8 eV`. This
couples starter selection and every previously discovered landing energy back
into direction scoring, explicit displacement, Gaussian width and the
within-walk trust controller.

This rule is physically plausible as a landscape-scale normalization, but its
`0.2`, statistic and bounds are heuristic. It was introduced inside a larger
historical bundle and has not received a clean cross-system ablation.

## Existing evidence and non-overlap

The fresh S-CR1 C60/PdO/CuO cohort used `trial_progress_patience=0`. All twelve
arms ended with `adaptive_step_multiplier=1` and
`adaptive_progress_boost=1`. Therefore the macro target controller's only
active history dependence in this cohort is the archive energy-scale formula;
escape-rate and stagnation feedback are inactive.

The accepted-minimum logs are sufficient to reconstruct the exact target seen
at every completed trial without a PES call. A preliminary read-only
calculation shows a large system separation:

- C60 mean targets are about 0.97--1.19 eV;
- PdO mean targets are about 0.34--0.59 eV;
- CuO mean targets are about 0.19--0.27 eV.

This is not yet evidence that fixed is better. It is evidence that the
untested block is active enough to merit one causal gate.

## Alternatives considered

### A. Repeat local sigma/weight feedback screens

Rejected. U0--U2 already showed context reversals and rejected every global
feedback reset. New floors, interpolation factors or switches would be
post-hoc local heuristics.

### B. Replace Gaussian propagation with OPES or constrained quadratic steps

Rejected at this stage. U3 established that the finite Gaussian tail is
physically active and that an unbounded quadratic runs away. A constrained
quadratic would be a new propagator with its own constraint semantics, not an
ablation of the current one. OPES would also require a CV, kernel bandwidth
and deposition law.

### C. Audit and ablate only the macro target law

Selected. It removes one history-dependent heuristic while retaining the
configured physical reference `0.8 eV`. No new continuous parameter is fitted.

## U-T0: zero-FE target reconstruction

### Inputs

Use the twelve S-CR1 seed-45 accepted-minimum logs and the authoritative raw
evidence under:

```text
runs/20260801-paired-continuation-restart-gate/output/
```

Record SHA-256 for the raw evidence and all twelve accepted logs.

### Reconstruction

For every completed trial:

1. start from the shared bootstrap energy;
2. calculate the target from minima available before that trial;
3. add the accepted new minimum, if one exists;
4. after the last completed trial, calculate the next-attempt target and
   require exact agreement with `stats.adaptive_step_target`.

Report by system and selector:

- mean, median, minimum and maximum target;
- fraction below/equal/above 0.8 eV;
- mean target/reference ratio;
- final multiplier and progress boost;
- archive size and energy range.

U-T0 is descriptive. It admits U-T1 if the reconstruction closes and at least
two systems spend more than 75% of completed actions away from 0.8 eV. It
cannot promote a production change.

## U-T1: fixed-reference full-search gate

### Arms

1. `archive_scaled`: the exact current `StepTargetController.target(archive)`;
2. `fixed_reference`: always return the already configured
   `target_uphill_energy=0.8 eV`.

The fixed arm changes only the macro target source. It retains:

- curvature-dependent direction scoring and explicit step construction;
- the within-walk `sigma_scale` and `weight_scale` trust updates;
- cumulative finite-tail Gaussian history;
- Safe-LBFGS proposal relaxation;
- strict true-PES quench and matcher;
- all geometry and energy-sanity guards.

Implement the fixed controller inside the experimental runner. Do not add a
new `SSWConfig` field or change `pamssw/` defaults unless the repeated gate
later passes.

### Matrix

- systems: C60, fixed-bottom PdO and packaged CuO;
- seed: 46, fresh for this mechanism gate;
- starter policy: `metropolis_chain` for both arms;
- total budget: exactly 20,000 FE per arm, bootstrap included;
- one exact shared bootstrap minimum per system;
- six full searches, 120,000 new FE maximum;
- CuO retains the previously admitted oracle-only local-softening scope in
  both arms.

Metropolis is used because its state transition has a direct physical chain
semantics and it avoids making the current growing-arm UCB-like score a second
experimental variable. The archive remains complete; no top-k/FPS deletion is
introduced.

### Randomness

Both arms begin with the same physical RNG seed and the same separate starter
selection stream. Once different targets produce different states, later
trajectories may diverge; that is part of the algorithm-level treatment
effect. The result is a shared-bootstrap fixed-budget comparison, not a claim
of framewise deterministic replay.

### Metrics

Keep a vector, not a scalar reward:

- best-energy gain AUC over the full 20,000-FE axis;
- terminal best energy and FE at first attainment;
- action count, archive size and duplicate rate;
- purpose-resolved FE and wall time;
- target distribution actually used;
- proposal-relax and true-quench termination/certificate counts;
- geometry-invalid, fragment and energy-sanity rejection counts.

### Validity

Every case must satisfy:

- exact 20,000-FE campaign accounting;
- `unattributed=0`;
- identical bootstrap state, energy and charged cost within a system;
- complete two-arm matrix;
- finite energy trace and reproducible evidence hash.

### Preregistered decision

Advance to repeat seeds 47--48 only if `fixed_reference` has strictly larger
gain AUC than `archive_scaled` in at least two of the three systems. Endpoint
energy, coverage, failures and component costs remain separately reported and
may explain the mechanism, but they do not replace the primary rule after the
result is seen.

If the first gate fails:

- retain archive scaling as the current baseline;
- do not tune the 0.2 factor, statistic, clipping bounds or a system-specific
  fixed target;
- close the macro-target branch and reconsider a physically distinct action
  family rather than another controller.

If it passes, seeds 47--48 compare only these two arms. Production admission
requires a positive fixed-minus-scaled AUC median and at least six positive
system-seed blocks out of nine. No posterior or target interpolation is added.

## Claim ceiling

U-T0 can establish how strongly archive energy history changes the configured
target in the recorded cohort. U-T1 can determine whether removing that one
history dependence improves finite-budget global-minimum search under the
frozen kernel. Neither gate proves canonical unbiasedness, universal optimal
target energy, asymptotic performance, or the superiority of fixed Gaussian
SSW over other propagator families.
