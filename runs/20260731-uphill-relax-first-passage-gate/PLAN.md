# G-UP1: accepted-step first passage through biased relaxation

## Question

G-UP0 proved that nonzero biased-PES relaxation changes basin support.  G-UP1
asks when that causal effect first becomes stable and whether one shorter,
absolute Safe-LBFGS accepted-step cutoff transfers without refitting.

## Discovery

Every recorded frame is quenched for the four trajectories forming the two
repeated G-UP0 contexts:

- plateau/D0/h1, seeds 42 and 43;
- plateau/K4/h2, seeds 43 and 44.

These contain 238 frames.  Frame zero and each final frame reuse prior hashed
landings, leaving about 230 new true quenches.  No temporal subsampling or
monotonic basin assumption is allowed.

For each trajectory, the stable final-basin step is the first frame in the
contiguous terminal suffix whose every frame is a certified escape matching
the final relaxed landing.  The proposed fixed cutoff is the maximum of the
four stable final-basin steps.  It is an integer accepted-step count, not an
energy threshold or fitted weight.

## Holdout

The cutoff is evaluated once on four untouched informative trajectories:

- intermediate/D0/h4 seed 42;
- plateau/K4/h1 seed 42;
- plateau/K4/h4 seeds 42 and 44.

It is not refit.  A fixed-cutoff full-action gate opens only if all four have
strictly positive step headroom and all four cutoff landings reproduce their
final certified basin.  Otherwise the present relaxation length remains the
reference and no adaptive rule is added.

## Budgets and exclusions

- additional FE cap: 32,000;
- gate kernel wall cap: 900 seconds;
- replay direction HVP: zero;
- replay biased relaxation: zero;
- unattributed FE: zero;
- no change to `pamssw/`, selector, direction, Gaussian, optimizer, matcher,
  quench, or production defaults.

