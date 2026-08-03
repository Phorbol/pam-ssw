# G-E1 Online First-Descent Paired Gate Design

## Purpose

G-E0 established one narrow offline fact: when a completed outer SSW micro
step already has true-PES energy below the macro starter by more than the
existing `dedup_energy_tol`, its true quench was a certified lower basin in
all four triggering frozen paths.  It did not establish online force-call
savings because stopping changes both the remaining biased propagation and
the conditioning of the final true quench.

G-E1 asks only:

> On fresh actions, would immediately cashing this sufficient lower-basin
> certificate reduce complete action cost after the true-quench cost is paid?

It does not ask whether first descent is an optimal stopping rule, whether it
finds the deepest basin reachable by the action, or whether it should become a
production default.

## Alternatives considered

### A. Single-path shadow fork through an internal observer — selected

Add one protected walker method that returns no stop reason by default.  The
normal walk calls it after the existing finite true-PES endpoint evaluation.
The G-E1 runner supplies a research-only observer that latches the first
descent but returns no stop reason, so the same fresh path continues to its
natural terminal. G-E1 true-quenches both the exact crossing state and the
terminal state; the early arm's generation ledger is the cumulative purpose
snapshot at the crossing.

This keeps the production path and public configuration unchanged, avoids
duplicating the uphill loop, and makes the early arm an exact prefix of the
reference rather than a numerically approximate GPU replay.

### B. Copy the uphill loop into the research runner — rejected

This would avoid touching `pamssw`, but a copied loop could drift in direction
selection, cumulative Gaussian handling, trust updates or clipping.  A paired
prefix comparison would then test two implementations rather than one stopping
event.

### C. Add a public `walk_early_exit_enabled` configuration — deferred

This would be appropriate only after online evidence.  Adding a user-facing
boolean before G-E1 would expand the algorithm parameter surface and imply a
production maturity that G-E0 did not establish.

### D. Execute two nominally identical GPU arms — rejected by smoke

The initial smoke used separate reference and stop walkers with the same
starter, seed, direction arm and configuration. Before either arm made a
different stopping decision, repeated float32 GPU MACE evaluations already
differed: starter energy by about `3.05e-5 eV`, first endpoint energy by about
`2.14e-4 eV`, and true curvature at the fourth decimal place. Selected
direction SHA, sigma and bias weight were identical, but optimizer endpoint
SHA was not. Relaxing prefix equality after observing this drift would mix
calculator nondeterminism into the stopping effect, so that smoke is excluded
from scientific evidence.

## Algorithm seam

`SurfaceWalker._walk_candidate_from_seed` already computes
`true_energy_after` after every accepted outer micro step.  Immediately after
the accepted state and trust update are committed, and only when no existing
geometry/clipping termination has priority, it calls:

```python
reason = self._walk_early_stop_reason(
    step_index=step_index,
    walk_reference=walk_reference,
    current=current,
    true_energy=true_energy_after,
)
```

The base implementation returns `None`.  No extra force evaluation and no
new configuration field are introduced.

The research observer records the cumulative purpose ledger and latches the
first row exactly when

```text
true_energy < starter_energy - dedup_energy_tol
```

There is no patience count, trend fit, relative percentage, uncertainty term
or second threshold.

## Fresh paired cohort

The gate uses the same hashed locked starters and the same D0/K4 action
definitions as G-E0, but fresh action seeds 45, 46 and 47:

- systems: C60 and fixed-bottom PdO;
- starter classes: `intermediate_accepted`, `plateau_accepted`;
- direction arms: `D0_exact_anchor`, `K4_discrete`;
- paired seeds: 45--47;
- total pairs: 24.

For each fresh action pair:

1. run one unchanged reference action to natural termination while the
   observer records the first descent and cumulative purpose ledger;
2. if a crossing occurs, true-quench its exact persisted checkpoint and the
   natural terminal; otherwise true-quench only the terminal and reuse it for
   the counterfactual early arm;
3. reconstruct early complete-action cost as crossing-prefix generation cost
   plus crossing true-quench cost;
4. compare the complete cost vector and landing outcomes.

No source action from G-E0 is replayed.  Seeds 42--44 are excluded.

## Pair integrity

Before interpreting a pair, require:

- the crossing row, direction row and persisted checkpoint have the same
  one-based step;
- the observer's cumulative ledger is componentwise no larger than the
  terminal generation ledger;
- the early prefix is constructed only from the same reference trace and
  checkpoint SHA, never from a second GPU execution;
- exact purpose-ledger closure and `unattributed = 0`.

A shadow-prefix mismatch invalidates the pair; it is not counted as an
algorithmic failure or repaired with tolerances after observing the result.

## Outcome vector

For every pair, retain separately:

- trigger step and reference terminal step;
- direction, biased-relax, true-energy-check and true-quench FE for each arm;
- total complete-action FE difference;
- wall time;
- landing certificate and geometry validity;
- landing energy relative to the starter;
- early-versus-reference matcher and descriptor relation;
- one of `AVOIDED_OVERSHOOT`, `FORGONE_DEEPER_TERMINAL`,
  `ENERGY_EQUIVALENT`, or `UNLEARNABLE`.

No weighted scalar reward is constructed.

## Preregistered decision

G-E1 admits a later equal-total-budget search gate, G-E2, only if:

1. all 24 shadow prefixes and ledgers close;
2. at least two fresh pairs trigger;
3. every triggered early landing is certified and below its starter by the
   existing energy tolerance;
4. summed complete-action FE over triggered early arms is strictly lower than
   the corresponding reference sum.

Energy trade-offs and basin relations remain veto-free descriptive outputs at
G-E1 because G-E2 exists precisely to test whether cheaper early lower basins
outperform rarer deeper action terminals when saved FE is reinvested.  G-E1
cannot change a production default regardless of outcome.

## Budgets and exclusions

- maximum 25,000 newly executed force evaluations;
- maximum 300 seconds GPU kernel wall time;
- no selector, archive, local-softening, direction-family, Gaussian, trust,
  proposal optimizer, true-quench optimizer or structure-matcher change;
- no public configuration parameter;
- no ThreadPool/batch-relax engineering;
- no Bayesian or learned stopping model.

The only production-file change is the inert protected hook plus focused unit
coverage proving that the base hook preserves the original step cap and an
override can observe or stop immediately after the already-paid true-energy
evaluation.

## Claim ceiling

G-E1 can establish fresh exact-prefix counterfactual complete-action cost and
landing trade-offs for 24 C60/PdO D0/K4 actions. It does not independently
execute the stopped continuation and cannot establish long-search improvement,
CuO transfer, statistical significance, canonical sampling correctness or a
production stopping policy.
