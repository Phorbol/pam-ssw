# Two-arm short-uphill-rollout gate

## Question

Can a short, directly observed biased-PES rollout choose between the existing
intent-preserving `static_score` arm and the existing soft
`true_curvature` arm better than committing to either ranker globally?

This is a selector-mechanism gate, not a new optimizer or posterior model.
It adds no direction source, acquisition weight, or production default.

## Frozen factors

- C60 and PdO;
- locked accepted states from trials 100 and 180;
- seeds 42, 43, and 44;
- native K4 candidate generator;
- local softening;
- adaptive serial Gaussian uphill policy;
- safe-LBFGS proposal relaxation;
- the system-specific certified true-quench protocols;
- H8 terminal labels from both completed prospective ranker repeats.

The short probes use the same starting state, seed, ranker, and first selected
direction as their corresponding H8 terminal label. A mismatch invalidates the
case.

## Preregistered decision rule

Both arms are run for H1 and H2 without a true quench. H2 is the only primary
horizon. The primary rule selects the arm with the larger observed true-PES
energy rise after H2. This is a direct barrier-progress signal and uses no
fitted coefficient. H1 is diagnostic only and cannot be promoted post hoc.

For each system independently, the H2 rule must:

1. predict the lower-energy H8+quench arm in at least 4/6 locked groups;
2. make a repeat-stable choice in at least 5/6 groups;
3. have zero median terminal regret;
4. have non-positive mean and median terminal-energy differences relative to
   always using `static_score`.

Only if both C60 and PdO pass the same rule is implementation of an online,
state-reusing, parallel racing policy allowed. Passing this gate would not by
itself promote racing to production: an end-to-end fixed-total-FE experiment
would still be required.

## Cost accounting

Every probe reports purpose-resolved force evaluations and wall time. The
analysis reports:

- actual serial probe cost;
- ideal two-worker probe latency;
- estimated online FE cost if the chosen arm reuses its short trajectory and
  the step-zero K4 HVP pool is shared.

The estimate is diagnostic only; no cost is hidden from the campaign ledger.

## Result

The two repeats completed 96/96 geometry-valid probes with exact first-
direction identity relative to the frozen H8 labels. They used 8,400 force
evaluations and 189.2 seconds of serial wall time.

The preregistered H2 larger-rise rule failed decisively:

- C60: 0/6 terminal winners, median regret 1.269 eV, mean terminal difference
  versus `static_score` +1.037 eV;
- PdO: 2/6 terminal winners, median regret 0.411 eV, mean terminal difference
  versus `static_score` +0.065 eV.

Even with chosen-trajectory reuse and one shared step-zero K4 HVP pool, the
estimated mean overhead was 105 FE per C60 trial and 150 FE per PdO trial.
Therefore the current two-full-arm short racing design is both non-predictive
under its preregistered rule and too expensive to justify an online
implementation.

As a post-hoc diagnostic only, the lower-rise arm was the terminal winner in
6/6 C60 and 4/6 PdO groups. This is consistent with the earlier observation
that stronger short-horizon uphill motion can encode overshoot, strain, or an
already-descending path rather than useful barrier progress. It does **not**
promote the inverse rule: that hypothesis was observed after unblinding and
requires independent preregistered validation before any implementation.
