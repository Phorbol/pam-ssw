# Direction oracle P0: candidate-budget closure

## Question

`SSWConfig.oracle_candidates = K` is the maximum number of native directions
returned by one `CandidateDirectionGenerator.generate` call.  `SoftModeOracle`
then evaluates one central HVP for each returned direction.  `K` is not a
random-direction quota to which momentum or bond directions may be appended.

## Red evidence

Before the change, the focused test command

```bash
python -m pytest -q tests/unit/test_direction_candidate_budget.py
```

failed in the expected ways:

* with `K=4`, a valid momentum, one explicit bond, one dynamic bond, and random
  fill produced six candidates;
* with `K=1`, the same priority stack produced three candidates;
* an all-zero previous direction was emitted as a momentum candidate and then a
  random candidate was appended.

## Minimal change

The generator now has one local bounded append path.  Its existing priority is
unchanged:

```text
valid momentum -> explicit bond -> dynamic bond -> random fill
```

Each accepted candidate consumes one of the `K` slots.  Dynamic bond sampling
is limited to the slots remaining after momentum and explicit bonds.  Random
directions fill any remaining slots.  A previous momentum is accepted only if
it has the full Cartesian shape, finite values, and nonzero norm; an invalid
momentum therefore leaves its slot for the next source.

No score, HVP formula, optimizer, walk step, direction-type priority, random
distribution, or telemetry field was changed.  The logged dynamic-bond request
continues to represent the configured request; the generated count now reports
the part that fit in the native HVP budget.

`K=0` is not a valid public `SSWConfig`, but direct generator use returns no
native candidates, so the hard bound is still closed at that edge.

## Green evidence

```text
python -m pytest -q tests/unit/test_direction_candidate_budget.py
5 passed

python -m pytest -q tests/unit/test_direction_candidate_budget.py tests/unit/test_walker_policy.py
225 passed

python -m pytest -q tests/integration/test_ssw_attempt_worker_integration.py
17 passed
```

The complete 1,325-test suite was also run in seven disjoint file batches
(the execution environment limits a single test command's wall time); all
seven batches passed.  The final focused closure after refactoring was:

```text
python -m pytest -q tests/unit/test_direction_candidate_budget.py \
  tests/unit/test_walker_policy.py tests/integration/test_ssw.py \
  tests/integration/test_ssw_attempt_worker_integration.py
244 passed
```

The new subsequent-step oracle test verifies `K=2` produces exactly two native
candidates and four force evaluations, retaining the existing two-force central
HVP contract per candidate.

The analytic LS-SSW integration baseline changed reproducibly because it no
longer evaluates an extra direction: its completed and fragmented cases move
from 61 to 56 total force evaluations and from 8 to 6 `DIRECTION_ORACLE`
evaluations.  The duplicate case moves from 41 to 45 total force evaluations:
the smaller candidate set selects a different valid trajectory, which requires
four more biased-relaxation evaluations.  Each updated case retains a closed
purpose ledger; this is a bookkeeping-correctness change, not evidence that
the cap improves search quality.

## P1 scientific gate

The previous default-momentum one-dimensional double-well smoke no longer
crosses to the second basin for its fixed seed after enforcing the cap.  This
is an expected consequence of changing the sampled portfolio and its RNG
trajectory, not an accounting defect.  Enlarging `K` from 12 through 20 did
not restore that particular seed's crossing, whereas the explicitly
no-momentum random/bond baseline still finds both wells.  The integration test
therefore explicitly disables momentum and only verifies that baseline
property; it does not silently certify the default momentum portfolio.

No discarded RNG draws are used to reproduce the old trajectory.  Hard-`K`
with momentum is a **paired GPU ablation candidate**, not a claimed default
improvement, until C60/PdO runs compare energy/coverage outcomes against the
pre-cap portfolio at matched total force-evaluation budgets.

## Deliberate scope boundary

This P0 only caps native candidates constructed by
`CandidateDirectionGenerator`.  Archive-replayed directions and optional
synthetic/probe directions remain separate mechanisms and are not reinterpreted
as part of this minimal correction.
