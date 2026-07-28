# C60 fixed-starter escape ablation conclusion

## Decision

Do not promote any block-Krylov arm and do not add a posterior selector yet.

The experiment supports a narrower conclusion: direction allocation is strongly
state dependent, while the terminal outcome is more tightly associated with
where the biased uphill relaxation ends on the true PES than with the measured
softness of the selected direction. The next experiment should therefore
separate direction quality from uphill propagation by strictly quenching every
stored checkpoint on the same direction-conditioned uphill trajectory.

No `pamssw/` production default was changed by this experiment.

## Frozen protocol

- System: C60 with MACE-OMAT-0-small on CUDA.
- Fixed starters: `bootstrap_quenched`, `intermediate_accepted`, and
  `plateau_accepted`.
- Seeds: 42, 43, and 44.
- Direction arms:
  - `discrete`;
  - `variational_breadth`: 6 blocks x depth 1;
  - `balanced_refinement`: 2 blocks x depth 3;
  - `deep_refinement`: 1 block x depth 6.
- One proposal and one landing quench per case, 36 cases in total.
- Every completed direction selection consumed exactly 12 central-difference
  HVPs, or 24 force evaluations.
- Proposal relaxation: existing Gaussian-bias walk with
  `safe-lbfgs-total`, `fmax=0.05 eV/A`, at most 80 optimizer steps.
- Original landing quench: SciPy L-BFGS-B, `fmax=0.01 eV/A`, at most 400
  iterations.
- Strict replay: the same stored escape structures, ASE-LBFGS primary and
  ASE-FIRE certificate fallback, `fmax=0.01 eV/A`, at most 400 iterations.
- A physically meaningful energy improvement is defined as more than 1 meV,
  matching the archive energy tolerance. Continuous energy differences remain
  available in the evidence.

The starter selector, UCB-like logic, posterior update, and multi-step archive
feedback were absent from this fixed-starter audit.

## Provenance and certification

- Original escape cohort: 36/36 completed at commit
  `01d31d956bf4c8a8573633648cf41008a8eec5a2`.
- Only 4/36 original SciPy landing quenches had a force-convergence
  certificate: 0 discrete, 0 breadth, 2 balanced, and 2 deep.
- Final strict replay: 36/36 certified with ASE-LBFGS at commit
  `b78a3a4fe5a5f7b407b5c1a70820ec51287d740d`.
- ASE-FIRE fallback use: 0/36.
- The source evidence hash and all 36 stored escape hashes close exactly.
- Exact starter references: 24/36. The two stored accepted states are exact.
  The 12 bootstrap cases predate exact starter snapshots and required legacy
  reconstruction. Their reconstructed starter energy drifted by about
  3e-5 eV between loads. Landing energies use the fixed stored escapes and
  landing deltas retain the original starter energies.
- The runner now writes a lossless JSON starter snapshot for future cohorts and
  rejects a stored snapshot whose structural hash differs from the source
  state.

Three strict GPU replays gave identical aggregate new-basin and meaningful
improvement counts. Across 35/36 cases, replay-to-replay landing-energy
differences stayed below 1 meV. One uphill, nonproductive case
(`plateau_accepted`, seed 43, discrete) followed two certified minimization
paths whose landing energies differed by 0.261 eV. This path sensitivity did
not change any arm-level success count, but it prevents treating a single
certified landing as bitwise deterministic.

## Strict outcome and cost

`Meaningful` means a new basin whose landing energy is at least 1 meV below its
starter. `Candidate FE` replaces the failed original SciPy landing-quench cost
with the strict ASE-LBFGS cost; it does not add both quenches.

| arm | new basin | meaningful | median landing delta (eV) | median new-basin delta (eV) | mean direction FE | mean proposal FE | median strict-quench FE | median candidate FE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| discrete | 8/9 | 4/9 | 0.000061 | -0.431229 | 170.67 | 311.22 | 40 | 575 |
| breadth 6x1 | 6/9 | 3/9 | 0.000092 | -0.001221 | 80.00 | 175.44 | 55 | 277 |
| balanced 2x3 | 7/9 | 5/9 | -0.762115 | -0.985901 | 144.00 | 302.89 | 39 | 500 |
| deep 1x6 | 7/9 | 4/9 | 0.000000 | -2.305695 | 146.67 | 255.78 | 26 | 467 |

Observed median per-case wall times were 10.708, 5.348, 9.074, and 8.544
seconds for the original discrete, breadth, balanced, and deep runs. The
standalone strict ASE-LBFGS replay medians were 0.694, 0.984, 0.671, and 0.450
seconds, respectively. These wall times exclude shared model loading and
bootstrap reconstruction and should not be added as a counterfactual runtime.

## State-conditioned result

Each cell reports `meaningful new basins / 3` followed by the median landing
energy difference in eV.

| state | discrete | breadth 6x1 | balanced 2x3 | deep 1x6 |
|---|---:|---:|---:|---:|
| bootstrap | 3/3, -2.559326 | 2/3, -2.202698 | 3/3, -2.603027 | 1/3, +1.176758 |
| intermediate | 0/3, +4.740784 | 0/3, +0.000092 | 0/3, +2.069763 | 0/3, +0.000122 |
| plateau | 1/3, +0.792786 | 1/3, +0.464630 | 2/3, -0.762115 | 3/3, -7.078857 |

The clean contributions and bottlenecks are:

1. Balanced 2x3 is the best general compromise in this small cohort: it has
   the most meaningful outcomes and works on both bootstrap and plateau
   starters. This is descriptive evidence, not enough for default promotion.
2. Deep 1x6 is not generally superior. It is excellent on the plateau state
   and poor on bootstrap and intermediate states. This is evidence for
   conditional direction allocation, not for a fixed global deep setting.
3. Breadth 6x1 is cheaper primarily because it performs fewer direction
   selections and less proposal relaxation. Its intermediate cases all return
   to the starter basin, so the lower cost is not a productivity win.
4. Discrete has the highest new-basin coverage but is expensive and only half
   of its new basins are meaningfully downhill. Candidate diversity alone is
   insufficient.
5. No arm produces a meaningful improvement from the intermediate state.
   A selector cannot recover an action that the present direction/uphiller
   combination does not produce.

## Mechanism audit

Across the 36 cases, Spearman correlation with terminal energy improvement was
approximately:

- true-PES escape energy rise: -0.86;
- first selected curvature: -0.11;
- minimum selected curvature: -0.13;
- mean selected curvature: -0.10;
- last selected curvature: -0.33;
- number of direction selections: +0.09.

The escape-energy association persists within bootstrap (-0.88) and plateau
(-0.92) states and remains -0.48 after subtracting each state-arm three-seed
mean. Curvature correlations change sign between arms. This small descriptive
sample cannot establish causality, but it falsifies the assumption that
selecting the softest available Ritz direction is, by itself, a reliable
terminal-outcome objective.

The evidence instead points to a direction-propagation interaction: productive
cases tend to terminate the biased relaxation at lower true-PES escape
energies, while a locally softer direction can still be propagated into an
unproductive endpoint. This is why a TS/UCB-like direction selector is
premature: it would learn credit for a coupled direction-plus-propagator
outcome.

## Next minimal experiment

Run a direction-conditioned checkpoint shooting audit before changing any
selector:

1. Freeze starter, random seed, direction arm, bias form, and optimizer.
2. Store every post-bias-relax checkpoint along the same uphill walk.
3. Strictly quench every checkpoint independently with ASE-LBFGS. These
   quenches are embarrassingly parallel.
4. Compare checkpoint index, true-PES energy, selected curvature, displacement,
   quench cost, basin identity, and meaningful terminal improvement.
5. Use only the intermediate and plateau starters initially, with balanced 2x3
   and deep 1x6, because they provide the clean failure/success contrast.

If an early checkpoint is productive and the final checkpoint is not, the
primary bottleneck is uphill termination or overshoot. If no checkpoint on a
trajectory is productive, the selected direction is the bottleneck. Only after
this separation should a posterior model be introduced, and its arm should be
the full direction-propagation action rather than a direction label alone.
