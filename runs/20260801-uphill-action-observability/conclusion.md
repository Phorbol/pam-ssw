# U-O1: cumulative-Gaussian uphill action observability

## Decision

The common bottleneck is **not insufficient true-PES elevation** and is **not
failure to leave the starter basin**. In the frozen production profiles, the
current walk usually exceeds the archive-scaled energy proxy in its first
completed micro-step, then spends most of its budget in further biased-PES
proposal relaxation. Those additional steps sometimes matter to basin
identity, but their present fixed H8 cost has not been shown to be optimal.

The production default remains unchanged. The next admissible mechanism test
is a fixed `H4` versus fixed `H8` end-to-end, equal-total-FE comparison. No
adaptive stop rule, posterior, target scan or new propagator is admitted from
this observation alone.

## What was implemented

`SearchResult.action_history` now exposes immutable records linking:

1. starter minimum and requested archive scale;
2. already-computed true-PES energy before and after every completed uphill
   micro-step;
3. direction, biased-relax and true-PES-check force evaluations;
4. exact whole-walk totals, including an interrupted final micro-step;
5. escape endpoint, landing true-quench cost and convergence certificate;
6. new, duplicate, rejected, globally improving or budget-censored outcome.

No coordinate or direction arrays are copied into this history. No calculator
call, random draw, search branch or user-facing parameter was added.

## Execution boundary

Two nominally identical seed-49 cohorts were executed on C60, fixed-bottom PdO
and CuO at 20,000 total force evaluations per system, bootstrap included.

| Execution | Commit | Attributed FE | Wall time | Purpose |
|---|---|---:|---:|---|
| raw | `db28232` | 60,000 | 1,275.3 s | first action history |
| confirmatory | `abe51b6` | 59,989 | 1,273.6 s | exact whole-walk tail cost |

Both have `unattributed=0`. The confirmatory PdO arm left 11 FE, less than one
submitted HVP batch. The trajectories diverged from tiny float32 GPU numerical
differences, so they are two same-configuration mechanism observations, not a
bitwise paired replicate.

## Physical result

### 1. The uphill walk is already strong

Across the six system-execution blocks, 94.1--98.6% of measurable actions
reached the recorded energy scale. The median first-delivery step was one in
every block. Median observed maximum height divided by the recorded scale was
well above one.

This rejects the current hypothesis that the shared cross-system failure is
simply “adaptive uphill is not forceful enough.” Stronger generic bias is not
the next justified change.

### 2. The recorded eV target is not an executed macro-height controller

The reason is visible in the equations. Production uses
`step_length_mode=per_atom_rms`, for which the execution scale is

\[
\sigma = \frac{s_{\rm RMS}}{\operatorname{RMS}(u)}\,\eta,
\]

bounded by `max_step_rms`. The archive-scaled eV value does not appear in this
execution formula. It enters the candidate direction scoring scale and the
trust-controller error floor. The Gaussian weight is then computed from the
selected curvature and executed width.

Therefore `target_delivery_ratio` is a useful diagnostic of scale mismatch,
not proof that a physical barrier was crossed at that height. It also changes
the interpretation of U-T1/U-T2: fixed versus archive-scaled target compared a
direction-scoring/trust scale, not two enforced barrier heights.

### 3. Proposal relaxation is the shared cost bottleneck

Biased proposal relaxation consumed 61.5--75.1% of total FE in every system and
both executions. Direction-oracle work was about 10.7--14.0%; landing quench
about 9.7--25.9%.

After the first proxy delivery, completed later micro-steps alone consumed:

| System | raw FE | confirmatory FE |
|---|---:|---:|
| C60 | 11,543 | 11,070 |
| PdO | 3,399 | 4,533 |
| CuO | 14,178 | 12,491 |

These are not automatically removable costs. Stopping earlier changes the
escape configuration, true-quench conditioning and landing basin. Prior
counterfactual evidence already showed that biased relaxation can causally
move a state across a basin boundary, while H8 versus H14 showed non-monotonic
continuation value.

### 4. Landing is usually novel, but rarely globally improving

New-basin rates were 67.3--84.8% for C60, 85.7--89.9% for PdO and 100% for CuO.
Global-improvement rates were only 28.8--39.1%, 6.5--8.9% and 16.7--20.6%,
respectively.

The physical picture is therefore:

\[
\text{starter basin}
\xrightarrow[\text{usually strong}]{\text{serial biased relaxation}}
\text{new basin often}
\xrightarrow{\text{rare}}
\text{new global best}.
\]

The scarce object is not basin escape itself. It is a cost-effective route to
a basin with lower energy or useful continuation value.

## Why the tempting early stops are not admitted

- “Stop as soon as observed height exceeds the target” is invalid because the
  target is not an enforced physical escape height under `per_atom_rms`.
- “Stop when true energy first falls below the starter” was already tested in
  G-E1. It can make the immediate true quench more expensive and can miss a
  later, deeper basin.
- H14 had mixed basin outcomes and extra cost; longer is not monotonically
  better either.

The clean next test is therefore fixed H4 against fixed H8. It changes one
integer mechanism only, uses terminal quench once per action, and compares
best-energy AUC, new minima per FE, global improvements per FE, invalidity and
purpose-resolved cost under the same total budget.

## Consequence for Bayesian/posterior exploration

This stage creates the data seam but does not yet justify training. Horizon and
direction should first become a small set of stable, globally supported action
arms. Only after H4/H8 and direction-family outcomes are reproducible should a
contextual posterior allocate these arms. The posterior should predict the
vector outcome—new basin, global improvement, cost, damage—not a single mixed
reward. It should not create one growing arm per archive node.

The final budget-aborted proposal remains right-censored rather than a completed
`ActionRecord`; its cost is exact in the global purpose ledger. Censored-action
logging must be added before these histories are called an unbiased training
dataset.
