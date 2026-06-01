# Soft-Mode Eigen Experiment

Date: 2026-06-02

## Objective

Compare constrained reference dimer, HVP/Lanczos soft modes, and the existing scored direction pool at two levels:

1. Eigenmode fidelity: how soft/stable the direction is and how much compute is needed to obtain it.
2. Global-optimization usefulness: whether the direction is likely to help SSW escape basins, not merely minimize the Rayleigh quotient.

The experiment is diagnostic. It must not redefine production success as "lowest curvature wins".

## Systems

- `c60`
- `cuo`
- `pdo`

Use the same input/model definitions as `runs/20260602-production-500t-c60-cuo-pdo/run_matrix.py`.

## Methods

- `scored_pool`: existing candidate pool and scored selected direction.
- `pool_candidates`: all generated random/bond/momentum-like candidates before final selection.
- `reference_dimer_a{500,50,10,5,1,0}`: constrained dimer with different bias strength.
- `lanczos_m{8,16,32}`: HVP-only Krylov Ritz soft modes.
- `penalized_lanczos_alpha{1,5,20}`: Lanczos on `H_eff = H - alpha n0 n0^T`; direction is constrained by the proposal prior, while reported curvature is recomputed with the true Hessian.
- `constrained_ritz_eta{0.3,0.6}`: Rayleigh-Ritz candidates from the proposal Krylov subspace, selecting the softest vector that still satisfies an alignment floor to the proposal direction.

## Metrics

### Eigenmode fidelity / cost

- Rayleigh quotient `N^T H N`
- HVP/force evaluation count
- overlap with Lanczos m=32 lowest Ritz vector
- dimer dot to initial mode
- dimer rotations and convergence

### Global-optimization proxy

- small displacement energy response for step sizes `0.05, 0.10, 0.20 A`
- force projection along the direction
- local recovery tendency after a very short true-PES micro-relax
- overlap with scored_pool selected direction and candidate families

## Initial Scope

Run a lightweight single-structure diagnostic first:

- systems: C60/CuO/PdO initial states
- seed: 42
- scored_pool candidates: one generation per system
- dimer samples: 6 bias strengths x 8 initial modes per system
- Lanczos dimensions: 8, 16, 32
- constrained Lanczos/Ritz: first 6 scored-pool proposal seeds per system

This scope is designed to avoid interfering with the running 500t production benchmark while still exposing whether dimer/Lanczos are directionally meaningful.

## Follow-Up Criteria

Only promote a method into full SSW ablation if it passes both gates:

- eigen gate: stable soft mode at acceptable HVP/force cost
- global proxy gate: direction is not just soft, but has better escape proxy than scored_pool

Potential follow-up component ablations after this run:

- stress-selected bond atom pairs instead of random atom pairs
- fixed Gaussian width/height from `/tmp/ssw-reference`
- hybrid scored_pool + Lanczos candidate
- adaptive constrained dimer bias strength based on local curvature scale
