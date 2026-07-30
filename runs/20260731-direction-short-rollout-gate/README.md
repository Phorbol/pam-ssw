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
