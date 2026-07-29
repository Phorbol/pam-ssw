# C60 Krylov curvature--overlap frontier audit

## Question and claim boundary

This experiment asks whether the already-paid 12-dimensional Krylov space
contains an alternative Ritz eigenvector that retains substantially more of
the random-plus-bond anchor without paying the stiffness of the raw anchor.

The selected direction remains the lowest Ritz vector.  The complete spectrum
is diagnostic and costs no additional HVP or force evaluation.  No unexecuted
Ritz point receives a terminal success label, and no selector, posterior,
overlap weight, curvature threshold, or production default is added.

The cohort contains the two locked accepted C60 starters, seeds 42/43/44, and
two direction constructions:

- detached two-column block Krylov, depth 6;
- exact-anchor-seeded one-column Krylov, depth 12.

Each construction spends 12 HVPs, or 24 force evaluations, per direction
selection.

## Evidence-driven execution amendment

The preregistered design attempted to save terminal-quench cost by reproducing
the old escape structure exactly and inheriting its validated terminal
outcome.  The equality premise failed before any inheritance was allowed.

The same first case was executed twice with the new code and compared with
the prior run.  Even the two new executions differed by:

- up to 0.0106 in selected curvature;
- up to 0.0131 in true curvature;
- 0.00225 A in the largest escape-coordinate component;
- different escape SHA-256 hashes.

The present MACE float32 CUDA path is therefore not bitwise reproducible.
Replacing exact equality with an uncalibrated geometric tolerance would have
introduced the kind of hidden heuristic this project is trying to remove.
The experiment was amended and preregistered again to run fresh strict
terminal quenches for all 12 cases.  The two failed equality probes are
preserved outside the repository under
`/tmp/krylov-frontier-failed-exact-hash-cbc5970-run{1,2}` and are not part of
the scientific cohort.

## Execution integrity

- Execution commit:
  `5d6dd33c64a0811a8eb809af1811876ff62b2cb5`.
- Completed cases: 12/12.
- Fresh strict terminal-quench certificates: 12/12.
- Quench fallbacks: 0/12.
- Independently revalidated structure hashes: 36/36.
- Recomputed evidence equals the stored structured evidence.
- Bootstrap, starter quench, and unattributed force evaluations: 0.
- Process exit code: 0.

## Fresh terminal outcomes

| Starter | Seed | Direction arm | Landing minus starter (eV) | New basin | Force evals | Direction | Proposal relax | True quench |
|---|---:|---|---:|:---:|---:|---:|---:|---:|
| intermediate | 42 | detached Ritz | +0.000153 | no | 348 | 144 | 162 | 33 |
| intermediate | 42 | anchor Lanczos | +0.000183 | no | 462 | 192 | 241 | 17 |
| intermediate | 43 | detached Ritz | +5.673706 | yes | 529 | 192 | 294 | 31 |
| intermediate | 43 | anchor Lanczos | +0.000061 | no | 445 | 192 | 230 | 11 |
| intermediate | 44 | detached Ritz | +0.000092 | no | 336 | 144 | 164 | 18 |
| intermediate | 44 | anchor Lanczos | +0.000061 | no | 433 | 192 | 212 | 17 |
| plateau | 42 | detached Ritz | +0.000061 | no | 418 | 120 | 221 | 69 |
| plateau | 42 | anchor Lanczos | +0.000244 | no | 419 | 120 | 266 | 24 |
| plateau | 43 | detached Ritz | -7.078705 | yes | 399 | 192 | 166 | 29 |
| plateau | 43 | anchor Lanczos | -7.078796 | yes | 531 | 192 | 298 | 29 |
| plateau | 44 | detached Ritz | +0.358551 | yes | 450 | 120 | 275 | 47 |
| plateau | 44 | anchor Lanczos | +0.000031 | no | 475 | 192 | 238 | 33 |

Meaningful means a certified new basin at least 0.001 eV below its starter.

| Direction arm | New basins | Meaningful lower basins | Median landing delta (eV) | Total force evals |
|---|---:|---:|---:|---:|
| detached Ritz | 3/6 | 1/6 | +0.000122 | 2,480 |
| anchor Lanczos | 1/6 | 1/6 | +0.000061 | 2,765 |

Both meaningful outcomes are the paired plateau seed 43 transition to
approximately -7.079 eV.  Every arm and seed again fails to produce a
meaningful lower basin from the intermediate starter.

The prior cohort had two meaningful detached-Ritz outcomes rather than one:
plateau seed 42 previously landed 0.677 eV lower but now returned to the same
basin.  This is direct evidence that three-seed terminal counts on the
float32-CUDA path are descriptive and numerically fragile, not a stable arm
ranking.

## Complete Ritz-spectrum geometry

The audit records 996 zero-extra-HVP Ritz points:

- detached Ritz: 456 points over 38 selections;
- anchor Lanczos: 540 points over 45 selections.

For each selection, the curvature--overlap frontier contains every
successively higher-overlap point when the spectrum is traversed from low to
high curvature.

| Direction arm | Nontrivial frontiers | Median frontier size | Executed overlap | Maximum Ritz overlap | First overlap gain | First total / true curvature cost | Maximum-overlap total / true cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| detached Ritz | 34/38 | 3 | 0.0736 | 0.1570 | +0.0382 | +8.58 / +7.88 | +38.32 / +38.24 |
| anchor Lanczos | 45/45 | 3 | 0.2274 | 0.4193 | +0.1540 | +4.08 / +3.92 | +11.00 / +11.17 |

Additional exact observations:

- The maximum-overlap point is never the executed lowest mode for
  anchor-Lanczos and is executed only 4/38 times for detached Ritz.
- Anchor-Lanczos maximum overlaps range only from 0.381 to 0.482; no individual
  Ritz eigenvector retains even half of the exact anchor.
- Every anchor-Lanczos selection pays positive true-PES curvature for any
  higher-overlap Ritz point.
- Detached Ritz has only 2/38 selections where some higher-overlap point does
  not increase true curvature, and its median minimum true-curvature increase
  is 13.85.

The anchor is itself inside the anchor-seeded Krylov space, but it is
distributed over several Ritz eigenvectors.  Selecting a different single
eigenmode cannot recover the original event intent.  The first alternative
mode offers a real but limited overlap increase, while the maximum-overlap
mode remains substantially stiffer.

The meaningful plateau seed 43 cases do not have a distinctive frontier
signature.  For example, the successful detached case has a median
maximum-overlap curvature cost of 94.26, while the successful anchor-Lanczos
case has 9.99; unsuccessful cases occur on both sides of those values.
Frontier geometry alone is therefore not a terminal-outcome predictor in this
cohort.

## Cost accounting

| Purpose | Force evals | Share |
|---|---:|---:|
| biased proposal relaxation | 2,767 | 52.75% |
| direction oracle | 1,992 | 37.98% |
| landing true quench | 358 | 6.83% |
| escape true-PES checks | 116 | 2.21% |
| post-relax validation | 12 | 0.23% |
| bootstrap, starter quench, unattributed | 0 | 0.00% |
| total | 5,245 | 100.00% |

Measured sequential section time was 86.86 s for direction plus proposal
generation and 5.87 s for terminal quenching, 92.74 s in total.

The spectrum diagnostics add matrix operations and JSON volume but exactly
zero force evaluations.  Proposal relaxation remains the largest cost;
direction HVPs become 38% of this refined-arm-only cohort.

## Decision

1. Keep the full-spectrum diagnostics: they are parameter-free, independently
   useful, and add no PES evaluations.
2. Do not switch from the lowest Ritz vector to the maximum-overlap Ritz
   vector.  The latter remains only moderately aligned and pays a large,
   consistently positive true-curvature cost.
3. Do not add a UCB/TS selector yet.  There is still no action-labelled outcome
   for the unexecuted spectrum points, and all intermediate cases remain
   unproductive.
4. Do not deepen Lanczos further.  The issue is not missing eigenpairs; it is
   that the physical anchor is spread over several eigenmodes.
5. If direction development continues, the next clean mechanism is a
   constrained linear combination or dimer/CBD-like rotation inside the
   already-paid subspace, not selection of a single higher Ritz eigenmode.
   Such a mechanism must first specify how closeness to the anchor is fixed
   without an arbitrary weight or overlap threshold.

The experiment resolves the immediate question negatively: the existing
Krylov spectrum does contain a curvature--overlap frontier, but not a
single-eigenmode direction that is simultaneously soft and strongly faithful
to the physical anchor.
