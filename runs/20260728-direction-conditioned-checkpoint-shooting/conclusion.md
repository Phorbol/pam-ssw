# Direction-conditioned checkpoint shooting: conclusion

## Question and claim boundary

This audit asks one narrow question:

> When a fixed C60 starter/direction arm fails at the terminal escape
> configuration, did an earlier accepted uphill checkpoint already quench to a
> useful lower-energy basin?

It is a descriptive diagnostic, not a stopping-rule benchmark.  Starter
selection, posterior/UCB/TS feedback, archive feedback, and production defaults
were held out.  The two arms use the same 12-HVP (24-force-evaluation)
direction-selection budget at every accepted macro step:

- `balanced_refinement`: two block-Krylov blocks of depth three;
- `deep_refinement`: one block-Krylov block of depth six.

The locked `intermediate_accepted` and `plateau_accepted` C60 states were used
directly, so bootstrap cost is exactly zero.  Each accepted macro checkpoint
was independently quenched with the strict true-PES protocol
(`fmax=0.01 eV/A`, ASE-LBFGS up to 400 steps, preregistered FIRE fallback).
A checkpoint is productive only when the quench is certified, reaches a new
basin, and lowers the starter energy by more than 1 meV.

## Execution and evidence integrity

- Exact cohort: 2 states x 3 seeds x 2 arms = 12 completed trajectories.
- Accepted/reachable checkpoints: 68.
- Discarded rejected macro attempts: 2; they remain charged to generation but
  are not shot as reachable states.
- Strict force-convergence certificates: 68/68.
- FIRE fallbacks: 1/68.  It occurred for
  `plateau_accepted/seed44/deep_refinement/step4`; the fallback converged and is
  not hidden from the cost.
- The main GPU computation wrote `raw.json` and `evidence.json` atomically and
  printed the final aggregate.  The Python/Torch process then hung during
  interpreter cleanup and was interrupted, so the wrapper exit code is 130,
  not a clean zero.  A separate exit-zero validation reloaded both JSON files,
  rebuilt the evidence from the 12 raw cases, closed every purpose ledger, and
  verified the hashes of all 68 effective checkpoints, landings, and raw
  optimizer checkpoints.

This establishes a complete, internally closed scientific cohort despite the
post-write interpreter-cleanup issue.

## Primary result

| Starter | Arm | Productive trajectories | Productive checkpoints | Classification |
|---|---:|---:|---:|---|
| intermediate | balanced | 0/3 | 0/15 | 3 no-productive |
| intermediate | deep | 0/3 | 0/18 | 3 no-productive |
| plateau | balanced | 2/3 | 9/14 | 2 earlier-and-final, 1 no-productive |
| plateau | deep | 3/3 | 15/21 | 3 earlier-and-final |

Across all 12 trajectories:

- `productive_earlier_and_final`: 5;
- `productive_final`: 0;
- `overshoot` (productive earlier, unproductive final): 0;
- `no_productive_checkpoint`: 7.

Therefore this cohort does **not** support categorical terminal overshoot as
the main failure mechanism.  Every trajectory whose earlier checkpoint was
productive remained productive at its terminal checkpoint.  Every failed
trajectory lacked a productive checkpoint throughout its accepted uphill
path.

The earliest productive macro steps were:

- plateau balanced: steps 2 and 3 for seeds 43 and 44;
- plateau deep: steps 3, 2, and 4 for seeds 42, 43, and 44.

Continuing after the first basin-crossing checkpoint did not discover a better
terminal basin in this cohort.  Four terminal landing energies were identical
to the earliest productive landing within about 0.1 meV.  For plateau/deep
seed 42, continuation retained the lower basin but worsened the landing by
16.7 meV.  This is a possible efficiency signal, not evidence for a deployable
stopping rule, because detecting it here required an otherwise unavailable
strict checkpoint quench.

## Energy outcomes

Terminal landing-energy changes relative to each starter (eV):

| Starter | Arm | seed 42 | seed 43 | seed 44 |
|---|---:|---:|---:|---:|
| intermediate | balanced | +0.0002 | +9.6366 | +2.0699 |
| intermediate | deep | +0.0002 | +0.0002 | +0.0002 |
| plateau | balanced | +0.0001 | -0.7622 | -8.6284 |
| plateau | deep | -6.3579 | -8.8252 | -7.0788 |

At the intermediate starter, deep refinement repeatedly returned to the same
basin; balanced refinement sometimes escaped, but only to higher-energy
basins.  This rules out the narrow explanation that the terminal quench merely
missed an earlier lower basin.  The accepted direction-conditioned paths did
not contain one.

## Force-evaluation and time accounting

Generation cost (the cost a proposal path actually paid):

| Purpose | Force evaluations |
|---|---:|
| direction oracle | 1,680 |
| biased proposal relaxation | 2,668 |
| true-PES escape checks | 92 |
| **total generation** | **4,440** |

Thus direction construction consumed 37.8% and biased proposal relaxation
60.1% of generation force evaluations.  Bootstrap, landing quench, and
unattributed generation counts are zero.

Checkpoint shooting is diagnostic overhead, not production proposal cost:

| Purpose | Force evaluations |
|---|---:|
| checkpoint true-PES evaluation | 68 |
| strict landing quench | 2,589 |
| post-relax validation | 68 |
| **total diagnostic shooting** | **2,725** |

Summed measured GPU sections were 120.6 s for generation and 45.5 s for
shooting.  These exclude model loading, WSL filesystem stalls, and the final
interpreter-cleanup hang, so they are kernel-section timings rather than
end-to-end wall clock.

For a production-like accounting that retains only the terminal quench, median
generation-plus-terminal-quench costs were:

| Starter | balanced | deep |
|---|---:|---:|
| intermediate | 373 FE | 298 FE |
| plateau | 415 FE | 495 FE |

Deep refinement is therefore not a universal cost win: it is cheaper but
unproductive at the intermediate starter, and more expensive but more reliable
at the plateau starter.

## Direction mechanism

All selected modes had positive true curvature.  Deep refinement produced
substantially softer and more delocalized directions:

| Arm | Median selected curvature | Median true curvature | Median participation ratio |
|---|---:|---:|---:|
| balanced | 8.09 | 7.33 | 4.56 |
| deep | 3.16 | 3.53 | 23.97 |

However, within the deep arm the nonproductive trajectories were *softer*
than the productive trajectories (median selected curvature 2.43 versus
3.64), while their participation ratios were similar (23.67 versus 24.94).
Balanced productive and nonproductive curvature statistics were also nearly
indistinguishable.

Consequently:

1. more Lanczos depth successfully optimizes the curvature proxy;
2. lower curvature and greater delocalization are not sufficient for a useful
   basin transition;
3. the dominant unresolved variable is the physical identity/continuity of
   the direction, conditional on the starter, rather than insufficient
   minimization of the Rayleigh quotient;
4. a global preference for deep refinement is not justified by six
   starter-seed observations, even though it is clearly better on the locked
   plateau state.

## Verified anchor/softening disconnect

A post-run read-only audit identified a more elementary mechanism than Ritz
branch tracking:

1. `generate_krylov_intents` is called before the macro loop and consumes
   independently sampled random axes plus independently selected pair axes.
2. The random-plus-bond `anchor_direction` is generated later and is not a
   column of those Krylov intent blocks.
3. Local softening is built from that separate anchor, whereas block-Krylov
   selects the lowest-curvature Ritz vector without using anchor overlap in
   its variational selection.
4. `choice_aligned_softening_enabled` was false in the frozen experiment, so
   the softening coordinate was not rebuilt around the selected Ritz vector.
5. The original Krylov intent blocks are reused at subsequent macro steps even
   as the structure changes.

The observed directions are correspondingly almost orthogonal to the physical
anchor:

| Arm | Median absolute anchor cosine | Fraction below 0.1 |
|---|---:|---:|
| balanced | 0.050 | 74.2% |
| deep | 0.067 | 69.2% |

Thus the current block-Krylov experiment is not yet a faithful test of the
original CBD/SSW principle of softening a mode while retaining the
random-plus-bond initial intent.  It commonly softens one coordinate and walks
along another nearly orthogonal coordinate.  This disconnect is a simpler
candidate cause than inadequate depth, a terminal stopping error, or a
missing statistical selector.

## Decision

Do not prioritize a new uphiller termination heuristic, and do not add a
selector or promote TS/UCB from this audit.  The next clean ablation should
hold starter, uphiller, 12-HVP budget ceiling, quench, and softening definition
fixed, and first isolate whether the direction solve uses the exact physical
anchor:

1. current detached lowest-Ritz block (control);
2. unrefined exact random-plus-bond anchor (one central HVP, or two force
   evaluations, to preserve the existing curvature-conditioned uphiller);
3. a single-anchor Lanczos space seeded by that exact anchor, with the same
   total 12-HVP budget.

This removes the independent-intent mismatch without adding an overlap
threshold, penalty weight, posterior selector, or choice-aligned-softening
threshold.  If anchor-seeded lowest Ritz still loses the physical intent, a
second, separately preregistered ablation can compare lowest-Ritz selection
with parameter-free eigenbranch continuation inside the *same* anchored
subspace.  Only after one of these direction transformations has a measured
terminal-outcome advantage should its state-conditioned outcomes feed a
posterior arm selector.
