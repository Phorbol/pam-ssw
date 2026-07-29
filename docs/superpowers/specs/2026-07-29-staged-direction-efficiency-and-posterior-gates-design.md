# Staged Direction-Efficiency Ablation and Posterior-Gating Design

## Status

This specification defines the next experimental milestone for `pam-ssw`.
It freezes the existing physical search path and tests, one axis at a time,
whether three expensive pieces of the current direction/proposal loop are
actually buying terminal PES-search outcomes:

1. within-walk momentum;
2. the number of direction candidates;
3. the number and relaxation depth of Gaussian-bias microsteps.

The milestone is deliberately non-learning.  It does not add a Bayesian
model, Thompson sampling, a new UCB-like selector, a MACE regressor, direction
crossover, or another direction family.  Those mechanisms are admissible
only after the present data contract can support selection-bias-aware
comparison.

No production default may be changed from this C60 experiment alone.

## Decision

Run a preregistered sequence of fixed-starter, one-proposal C60 ablations:

```text
Stage M: momentum on/off
    |
    v
Stage K: 4/8/12 direction candidates
    |
    v
Stage B: 5/8 Gaussian-bias microsteps
    |
    v
Stage L: 40/80 proposal-relax iterations, only if the measured bottleneck remains
```

Each stage inherits exactly one setting from the preceding stage.  The search
does not tune intermediate values, reopen a completed axis, or combine axes in
a full factorial.  The maximum non-learning budget is 108 proposal
executions; Stage L is skipped unless its preregistered entry condition is met.

The scientific priority is terminal efficiency under exact force-evaluation
accounting, not mathematical elegance of the direction selector.

## Scientific questions

### Momentum

The current within-walk momentum direction is the normalized displacement
produced by the preceding biased relaxation.  To first order,

\[
\Delta x_k \simeq -H_k^{-1}\Delta g_{\mathrm{bias}},
\]

so it is better interpreted as a secant/tangent continuation estimate along
the current biased escape tube than as heavy-ball momentum.

It is frequently selected in existing C60 traces, but selection frequency and
successful terminal descent are correlated observations, not causal evidence.
Stage M asks:

> At fixed candidate count and downstream policy, does retaining the previous
> biased-relaxation displacement discover a repeat-stable set of lower basins
> that fresh native candidates do not?

### Candidate count

The native discrete selector constructs at most `oracle_candidates`
candidates in the order:

1. valid within-walk momentum;
2. explicit bond candidates;
3. dynamic bond candidates;
4. random candidates to fill the remaining capacity.

Every centrally differenced curvature evaluation costs two force evaluations.
At the current value \(K_{\mathrm{dir}}=12\), direction ranking therefore
costs 24 force evaluations per selection before proposal relaxation.
Stage K asks:

> Is the random breadth beyond the first physically distinct candidates
> producing terminal information, or only increasing the oracle bill?

### Bias-walk depth

`max_steps_per_walk` bounds the number of Gaussian-bias additions and
proposal relaxations inside one SSW proposal.  Prior checkpoint shooting found
the earliest productive C60 checkpoints at zero-based indices 2--4, with no
observed case in which an early productive checkpoint became unproductive at
the terminal checkpoint.  Stage B therefore compares five microsteps with the
current eight, without inventing a new adaptive stopping score.

### Proposal-relax depth

`proposal_relax_steps` bounds safe-LBFGS iterations for each biased
relaxation.  A prior C60 trace contained 70 relaxations with median 34
iterations, 90th percentile 80, and 18 of 70 calls reaching the current cap
of 80.  Stage L tests a cap of 40 only when proposal relaxation remains a
measured bottleneck after Stages M--B.  It does not change `fmax`; the
previous loose-`fmax` experiment is a different physical intervention.

## Alternatives rejected

### Full factorial over momentum, \(K_{\mathrm{dir}}\), \(B_{\mathrm{bias}}\),
and \(L_{\mathrm{relax}}\)

A full factorial would consume substantially more terminal quenches while
making interactions difficult to distinguish from sparse seed effects.
The present goal is to remove non-contributing work, not fit a response
surface over implementation knobs.

### Immediate Bayesian, UCB-like, or Thompson direction selection

The current selector executes only its winning candidate to a terminal
landing basin.  Unselected candidates have curvature proxies but no terminal
counterfactual labels.  Treating them as failures would create selection
bias; fitting a posterior only to winners would confound action quality with
the existing selection policy.

The Bayesian form of a selector is not itself evidence of a better PES
algorithm.  A learning phase is postponed until the data and validation gates
in this specification are met.

### Immediate MACE-feature classifier or regressor

MACE representations may eventually encode useful local environment context,
but a high-capacity representation cannot repair missing terminal labels.
Introducing it now would add model, pooling, regularization, and calibration
choices before a learnable residual signal has been established.

### Direction crossover

Historical direction vectors cannot be crossed safely without resolving
sign, atom correspondence, state-to-state transport, and locality.  If this
mechanism is later tested, it must be restricted to compatible local
environments and validated by a fresh HVP/Ritz calculation.  It is not part
of this milestone.

### Production-length 200-macro-step campaigns

These experiments isolate one proposal from locked starters.  A long adaptive
campaign would mix direction quality with starter selection, archive history,
and changing state occupancy, defeating the causal question.

## Frozen protocol

Every non-learning stage uses:

- the exact accepted C60 `intermediate` and `plateau` starter structures;
- paired seeds 42, 43, and 44;
- two exact repeats for every state/seed/arm condition;
- one SSW proposal followed by one strict true-PES quench;
- the current safe-LBFGS proposal optimizer;
- the same direction scorer, uphiller, local-softening equations, trust
  control, true-quench optimizer, calculator, and structure matcher;
- the same model file and precision;
- strict purpose-resolved force-evaluation accounting;
- fresh terminal convergence and basin certificates.

The experiment excludes:

- starter selection and archive feedback;
- the existing node UCB-like selector;
- Thompson sampling or online posterior updates;
- archive escape momentum;
- reaction-network or edge rewards;
- proposal-pool winner selection;
- changes to true-quench tolerance or maximum iterations.

GPU wall time is recorded but remains descriptive because device contention
and asynchronous execution make it a less stable causal metric than exact
force-evaluation counts.

## Parameter vocabulary

Three costs must not be conflated:

| Symbol | Existing configuration | Meaning | Current value |
|---|---|---|---:|
| \(K_{\mathrm{dir}}\) | `oracle_candidates` | candidates curvature-ranked at each direction selection | 12 |
| \(B_{\mathrm{bias}}\) | `max_steps_per_walk` | maximum Gaussian-bias microsteps in one proposal | 8 |
| \(L_{\mathrm{relax}}\) | `proposal_relax_steps` | maximum safe-LBFGS iterations per biased relaxation | 80 |

With central HVPs, one direction selection costs
\(2K_{\mathrm{dir}}\) force evaluations.  The actual proposal-relax and
true-quench cost remains outcome-dependent and must be measured, not inferred
from iteration caps.

## Common outcome definitions

For starter energy \(E_s\) and certified landing energy \(E_l\), a proposal is
`meaningful` only when:

1. the terminal true-PES quench has a valid strict force-convergence
   certificate;
2. the landing structure is classified as a basin distinct from the starter;
3. \(E_l \le E_s - 0.001\,\mathrm{eV}\).

The 0.001 eV margin is inherited unchanged from the existing fixed-starter
direction experiments and matches the archive energy pre-screen tolerance.
It is not tuned in this milestone.  Every report also retains the continuous
\(E_l-E_s\), so the binary operational margin cannot hide near-threshold
behavior.

For a fixed state, seed, and arm, a condition is
`repeat_stable_meaningful` only when both exact repeats are meaningful.

The primary comparison object is the set

\[
\mathcal S_a =
\{(\text{state},\text{seed}) :
\text{arm } a \text{ is repeat-stable meaningful}\}.
\]

This set-valued definition avoids converting a six-condition experiment into
an unstable scalar reward.  It also prevents one very deep but irreproducible
landing from hiding lost coverage in another condition.

Secondary recorded quantities are:

- landing energy difference from the starter;
- total and purpose-resolved force evaluations;
- direction selections and candidate-source composition;
- selected source at every bias microstep;
- proposal-relax iteration counts and cap hits;
- invalid, damaged, fragmented, or non-converged outcomes;
- strict certificate fields;
- wall time and peak device memory, when available.

`new basin` is local to an isolated state/seed execution.  It does not assert
global uniqueness across independent cases.

## Stage M: causal momentum ablation

### Arms

| Arm | Momentum | \(K_{\mathrm{dir}}\) | \(B_{\mathrm{bias}}\) | \(L_{\mathrm{relax}}\) |
|---|---:|---:|---:|---:|
| M-on | enabled | 12 | 8 | 80 |
| M-off | disabled | 12 | 8 | 80 |

Disabling momentum frees its candidate slot for the existing random fill.
Both arms therefore rank exactly 12 candidates and pay 24 direction force
evaluations per selection.  No other source probability or ordering changes.

### Budget

\[
2\ \text{starters}
\times 3\ \text{seeds}
\times 2\ \text{arms}
\times 2\ \text{repeats}
=24\ \text{proposals}.
\]

### Decision gate

- Confirm a positive momentum contribution only if
  \(\mathcal S_{\mathrm{M-on}}\) strictly contains
  \(\mathcal S_{\mathrm{M-off}}\), and neither M-on repeat has a lower
  meaningful-outcome count than its paired M-off repeat.
- Treat momentum as unproven but retain the current setting for the next stage
  if the sets are equal or incomparable.
- Treat momentum removal as a candidate change only if
  \(\mathcal S_{\mathrm{M-off}}\) strictly contains
  \(\mathcal S_{\mathrm{M-on}}\), and neither M-off repeat is worse.
  Even then, a PdO or second-system validation is required before changing
  the production default.

No momentum scale, mixing coefficient, or persistence length is tuned.

## Stage K: candidate-count ablation

### Arms

Use the momentum setting retained by Stage M and compare
\(K_{\mathrm{dir}}\in\{4,8,12\}\), with
\(B_{\mathrm{bias}}=8\) and \(L_{\mathrm{relax}}=80\).

Under the current native priority, a later microstep with valid momentum
typically contains:

- \(K=4\): momentum, two bond candidates, and one random candidate;
- \(K=8\): the same physical candidates plus five random candidates;
- \(K=12\): the same physical candidates plus nine random candidates.

At the first microstep, where within-walk momentum does not yet exist, the
vacated slot is filled by a random candidate.  Exact realized composition
must be logged rather than assumed.

The direction bill per selection is:

| \(K_{\mathrm{dir}}\) | HVPs | force evaluations |
|---:|---:|---:|
| 4 | 4 | 8 |
| 8 | 8 | 16 |
| 12 | 12 | 24 |

### Budget

\[
2\times3\times3\times2=36\ \text{proposals}.
\]

### Decision gate

Choose the smallest tested \(K\) whose repeat-stable meaningful set is a
superset of the \(K=12\) set, whose strict certificates all pass, and whose
total force-evaluation cost is lower in each exact repeat.

If neither \(K=4\) nor \(K=8\) satisfies all conditions, retain \(K=12\).
Do not test \(K=5,6,7,9,10,\) or \(11\) in this milestone.

## Stage B: bias-microstep ablation

### Arms

Use the retained momentum and \(K_{\mathrm{dir}}\) settings and compare
\(B_{\mathrm{bias}}\in\{5,8\}\), with
\(L_{\mathrm{relax}}=80\).

Five steps include the latest zero-based checkpoint index 4 that was earliest
productive in the prior checkpoint evidence.  This comparison changes only
the maximum number of existing bias additions; it does not add an energy,
curvature, or trajectory-shape stopping heuristic.

### Budget

\[
2\times3\times2\times2=24\ \text{proposals}.
\]

### Decision gate

Choose \(B=5\) only if its repeat-stable meaningful set is a superset of the
\(B=8\) set, all strict certificates pass, and its total force-evaluation
cost is lower in each exact repeat.  Otherwise retain \(B=8\).

## Stage L: conditional proposal-relax cap ablation

### Entry condition

Run Stage L only if both conditions hold in the retained Stage B arm:

1. biased proposal relaxation still consumes more than 50% of total force
   evaluations;
2. at least 20% of biased-relax calls reach the current iteration cap of 80.

The first condition establishes that this axis remains the dominant measured
cost.  The second distinguishes an active cap from a value most relaxations
never approach.  The 20% entry threshold is below the previously observed
18/70 cap-hit fraction (25.7%) and is used only to decide whether the
experiment is worth its fixed budget; it is not an optimizer parameter.

### Arms

Use all retained Stage M--B settings and compare
\(L_{\mathrm{relax}}\in\{40,80\}\).  Proposal `fmax`, true-quench settings,
and all optimizer safeguards remain unchanged.

### Budget

\[
2\times3\times2\times2=24\ \text{proposals maximum}.
\]

### Decision gate

Choose \(L=40\) only if its repeat-stable meaningful set is a superset of the
\(L=80\) set, all strict certificates pass, and both proposal-relax and total
force-evaluation costs are lower in each exact repeat.  Otherwise retain 80.

If the entry condition fails, record Stage L as `not_entered` with the
measured fractions; do not run it to complete a visual experiment matrix.

## Execution and evidence contract

Each stage writes a self-contained package:

```text
runs/<timestamp>-direction-efficiency-<stage>/
  run_stage.py
  effective_configs/
  cases/
  evidence.json
  conclusion.md
```

Every case must record:

- execution commit and dirty-state status;
- complete effective configuration;
- calculator/model path and SHA-256;
- starter path and SHA-256;
- state, seed, arm, and repeat identifiers;
- source composition and selected source for every direction selection;
- exact purpose-resolved evaluation counts;
- proposal trace and termination reason;
- starter, escape, and landing structure hashes;
- strict true-quench convergence certificate;
- basin classification and meaningful-outcome fields;
- wall time and available hardware metadata.

The stage runner fails closed when:

- any calculator call is unattributed;
- configured and observed direction-HVP counts disagree;
- the force-evaluation purpose sum does not equal total evaluations;
- a case is missing, duplicated, or silently skipped;
- a terminal label lacks a strict convergence certificate;
- the two nominal repeats do not share the same locked inputs and effective
  configuration apart from the repeat identifier.

Exact repeats intentionally reuse the same scientific seed.  Their purpose is
to expose nondeterminism in the float32 GPU execution path, not to add new
statistical seeds.

Within each state/seed block, paired arms are executed in a frozen alternating
order, and the second repeat reverses that arm order.  This prevents a fixed
arm from always occupying the earlier thermal/load position without adding a
random scheduling parameter.  Force-evaluation conclusions remain primary.

## Fixed budget and stop rules

The maximum non-learning campaign is:

| Stage | Maximum proposals |
|---|---:|
| M | 24 |
| K | 36 |
| B | 24 |
| L | 24 |
| **Total** | **108** |

There is no automatic extension.  For each axis:

- stop after the preregistered arms and repeats;
- do not inspect an intermediate result and add a nearby parameter;
- do not reinterpret an invalid or unconverged terminal result as a success;
- do not promote a setting based on wall time alone;
- document a null or mixed result as such.

If the complete 108-proposal ceiling produces no repeat-stable evidence that
any tested action component improves terminal efficiency, stop optimizing
these local knobs.  The next scientific question should be a genuinely new
event/direction family, not a classifier trained to rank uniformly
unproductive actions.

## Gate to a posterior direction model

Passing Stages M--L does not automatically authorize an online posterior.
A separate feasibility phase may start only when the accumulated event store
contains:

1. at least two direction families with at least five certified meaningful
   terminal outcomes each;
2. positives spanning at least two locked starter classes, with a later
   second physical system preferred before production use;
3. complete action, context, vector-valued outcome, cost, propensity, and
   terminal-certificate records;
4. a preregistered small context model reduces held-out score-only prediction
   error with the same sign in both exact repeats, demonstrating information
   not already represented by the current heuristic score.

The non-learning stages identify outcomes for executed policies and parameter
arms.  They do not provide counterfactual labels for every ranked candidate.
For this gate, a `direction family` means a predeclared action branch whose
direction-source or portfolio rule remains fixed through the downstream
proposal.  A terminal result from a proposal that mixes changing source types
across microsteps cannot be retrospectively credited to each source family.

### Separate data-collection phase

If the gate passes, preregister at most 60 executed action branches.  Every
candidate branch included in training must be physically executed under the
same frozen downstream continuation and terminal-quench policy.  Unselected
or unexecuted candidates remain unlabeled; they must never be encoded as
negative outcomes.

No online adaptation occurs during this collection phase.

### First model comparison

Compare only:

1. an empirical or Beta--Binomial posterior over coarse action families;
2. a small regularized Bayesian logistic/ridge model over declared
   context/action features;
3. the current heuristic scorer.

Targets remain separate:

- probability of a certified meaningful landing;
- terminal landing energy change;
- total and purpose-resolved force-evaluation cost;
- invalid/damage probability.

The event store retains this vector and must not collapse it into one fixed
scalar reward.

Potential features may include:

- displacement-weighted pooled MACE node embeddings;
- active-region and locality summaries;
- \(u^\mathsf{T}g\), measured curvature, and participation ratio;
- overlap with anchor, momentum, and bond directions;
- starter energy, force, and compatible prior-action history.

MACE features are optional context, not a replacement for the physical
direction invariants.

### Promotion rule

Use leave-one-starter-class-out validation, and later
leave-one-system-out validation when a second system exists.  A model may
proceed to a fixed-budget shadow campaign only if it improves both:

- held-out probability calibration/log loss;
- held-out ranking of certified terminal efficiency;

over both the empirical action prior and the current heuristic scorer in each
held-out starter class.  The shadow campaign must then reproduce or improve
terminal efficiency under an identical force-evaluation budget.

Thompson sampling, UCB-like selection, direct regression, or a MACE feature
model receives no preference merely because its formalism is more Bayesian or
expressive.

## Claim ceiling

This experiment can establish only whether the tested current components can
be removed or reduced without losing repeat-stable certified C60 terminal
outcomes under the locked one-proposal protocol.

It cannot establish:

- thermodynamic or stationary-distribution unbiasedness;
- a universally optimal direction portfolio;
- superior long-campaign minimum discovery;
- transfer to PdO, other clusters, surfaces, or crystals;
- superiority of Bayesian selection, Thompson sampling, or MACE features;
- suitability of any retained setting as a production default.

Here `unbiased` means that every executed action receives its actual terminal
outcome and cost, and that no unexecuted candidate is assigned a fabricated
label.  It does not mean uniform sampling of PES configurations.

## Completion condition

The milestone is complete when:

1. every entered stage reaches its fixed budget with closed accounting;
2. every stage produces its preregistered set comparison and decision;
3. skipped Stage L records its failed entry condition, if applicable;
4. no production default is changed without an external-system validation;
5. the final report separates verified code behavior, empirical C60 evidence,
   physical interpretation, and untested transfer claims.
