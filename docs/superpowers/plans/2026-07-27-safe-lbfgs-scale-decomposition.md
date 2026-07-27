# Safe L-BFGS inverse-scale decomposition implementation plan

> **Execution:** use subagent-driven development and test-driven development.
> Each task receives implementation review before the next task begins.

**Goal:** Isolate adaptive scalar inverse scaling from the remaining L-BFGS
two-loop corrections on the frozen C60/PdO proposal-relaxation matrix.

**Architecture:** Add one private, fail-closed scale-only experiment seam to the
existing safe-total kernel. Reuse the immutable proposal tasks, trace observer,
and exact evaluator accounting. Do not alter any configured optimizer surface
or production default.

## Task 1: Private scale-only kernel seam

**Modify**

- `pamssw/relax.py`
- `tests/unit/test_relax.py`

### RED

Add tests proving:

1. `False` is a no-op on every existing optimizer path, while `True` is
   rejected before evaluation unless the optimizer is `safe-lbfgs-total` and
   the explicit history limit is zero;
2. every non-boolean flag value is rejected before evaluation;
3. the three modes are identical through the first outer iteration;
4. on an anisotropic quadratic, scale-only changes the second direction while
   the inverse-product history remains empty;
5. scale-only uses
   \(\gamma=(s^\top y)/(y^\top y)\) from the latest accepted pair;
6. rejected curvature does not update the scale pair;
7. a MIC branch change clears the scale pair;
8. default and explicit production history-10 behaviour are unchanged.

Run the focused tests and record the expected failures before implementation.

### GREEN

Extend `_lbfgs_inverse_product` with an optional latest scale pair. Preserve the
existing precedence:

- nonempty history uses its newest pair;
- empty history plus a scale-only pair uses that pair;
- otherwise use the existing fixed \(1/70\).

In the safe loop, keep a separate latest scale pair only for the scale-only
arm. Never append it to two-loop history. Clear it on MIC branch change and
update it only after the existing curvature gate accepts the pair.

Do not change line search, step limiting, telemetry semantics, convergence, or
ordinary safe-total/bias-separated code paths.

### Verify and commit

Run:

```bash
pytest -q tests/unit/test_relax.py
pytest -q tests/unit/test_proposal_energy_trace.py
git diff --check
```

Commit the production seam and tests.

## Task 2: Fail-closed 48-row CUDA runner

**Create**

- `runs/20260727-safe-lbfgs-scale-decomposition/plan.md`
- `runs/20260727-safe-lbfgs-scale-decomposition/run_gpu_ablation.py`
- `runs/20260727-safe-lbfgs-scale-decomposition/.gitignore`
- `tests/unit/test_safe_lbfgs_scale_decomposition.py`

Write runner tests first. Pin:

- exact three-arm order and private arguments;
- the 16 task payload hashes and source-summary hash;
- current source bundle and helper hashes;
- safe-kernel constants and scale formulas;
- model, input, CUDA, imported modules, and repository commit;
- output atomicity and non-overwrite behaviour.

Each arm uses an independent calculator. Record the private flag, resolved
history limit, resolved scale policy, exact trace records, callback state
hashes, endpoint, certificate, telemetry, force evaluations, and wall time.
Fail unless all three accounting totals agree.

The raw output directory remains ignored and is published only after all 48
rows validate.

## Task 3: One approved GPU execution

Before calculator construction:

- run focused tests;
- verify source/task/model/input/helper hashes;
- verify CUDA execution;
- verify the tracked worktree is clean at the expected commit.

Execute the matrix once. Do not retry or retune after physical calls begin.
Protocol failures before PES construction may be repaired; any such failure and
whether physical calls occurred must be documented.

## Task 4: Deterministic evidence and mechanism decision

**Create**

- `runs/20260727-safe-lbfgs-scale-decomposition/analyze_ablation.py`
- `runs/20260727-safe-lbfgs-scale-decomposition/evidence.json`
- `runs/20260727-safe-lbfgs-scale-decomposition/conclusion.md`
- `runs/20260727-safe-lbfgs-scale-decomposition/scale_decomposition_curves.svg`
- `tests/unit/test_safe_lbfgs_scale_analysis.py`

The analyzer fails closed on incomplete rows, schema/type errors, provenance
mismatch, nonfinite values, certificate mismatch, endpoint/trace mismatch, or
accounting mismatch.

Report separately for C60 and PdO:

- certificate counts and termination reasons;
- total and task-paired force evaluations;
- wall time;
- line-search and secant diagnostics;
- endpoint displacement and energy differences;
- exact evaluation trajectories.

Do not compute a weighted composite score.

Decision sequence:

1. compare fixed-scale history-0 with adaptive-scale history-0;
2. compare adaptive-scale history-0 with production history-10;
3. submit the remaining gap for explicit scientific review; no automatic
   numerical threshold promotes a later history-1/history-10 decomposition.

## Task 5: Final verification and stacked PR

Run the full test suite, base-to-head diff check, deterministic artifact
regeneration, ignored-output check, and independent final review.

Push `experiment/safe-lbfgs-scale-decomposition` and create a PR stacked on
`experiment/safe-history-capacity-ablation`. State the exact claim ceiling and
do not promote a new default.
