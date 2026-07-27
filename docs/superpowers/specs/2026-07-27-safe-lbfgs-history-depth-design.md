# Safe L-BFGS history-depth ablation

## Status and scope

This experiment is stacked on commit `31fd257`, after the fixed-scale,
scale-only, and history-10 decomposition.

The preceding fixed 16-task C60/PdO matrix established two bounded facts:

1. latest-secanted adaptive scalar scaling improves the empty-history kernel
   on this matrix but does not recover the existing history-10 kernel;
2. the remaining history-10 two-loop correction stack is a strong positive
   candidate, but the experiment did not separate the newest correction from
   corrections retained from older accepted steps.

The next scientific question is therefore exactly:

> With adaptive inverse scaling held fixed, is the observed history-enabled
> contribution already supplied by the newest two-loop correction, or do older
> retained secants add value on the frozen proposal tasks?

This is an optimizer-kernel ablation. It does not change the SSW outer loop,
starter policy, direction, Gaussian bias, source tasks, force certificate,
model, calculator lifecycle, or production defaults.

## Alternatives considered

### Selected: history 1 versus history 10

Compare the smallest nonempty L-BFGS memory with the existing kernel. This
changes only whether older accepted secants remain in the two-loop recursion.
It is the smallest experiment that answers the open causal question.

### Rejected for this stage: history 1, 2, and 10

Adding history 2 could locate an intermediate response, but it adds a new arm
before the one-versus-many distinction is established. If history 1 and
history 10 differ, a later experiment may examine the depth response.

### Rejected: history sweep

A sweep over 1, 2, 4, 6, 8, and 10 would turn a mechanism ablation into
hyperparameter selection. The present evidence does not justify that cost or
complexity.

## Frozen mathematical kernel

For every accepted same-MIC-branch step,

\[
s_k=x_{k+1}-x_k,\qquad
y_k=g_{k+1}-g_k,\qquad
\rho_k=(s_k^\mathsf{T}y_k)^{-1}.
\]

A pair is usable only when

\[
s_k^\mathsf{T}y_k >
\sqrt{\epsilon_{\rm mach}}\lVert s_k\rVert\lVert y_k\rVert.
\]

Both arms use the newest retained pair to define

\[
\gamma_k =
\frac{s_k^\mathsf{T}y_k}{y_k^\mathsf{T}y_k},
\qquad
H_k^{(0)}=\gamma_k I.
\]

Both arms then apply the standard inverse-L-BFGS two-loop recursion to the
total biased gradient. The only difference is the number of accepted pairs
retained:

1. `adaptive-scale-history1`
   - retain at most the newest accepted pair;
   - that pair supplies both adaptive \(\gamma_k\) and the single two-loop
     rank correction.
2. `adaptive-scale-history10`
   - retain at most the newest ten accepted pairs;
   - the newest pair supplies adaptive \(\gamma_k\);
   - all retained pairs participate in the two-loop corrections.

For deterministic analytic evaluator outputs, a kernel unit test must show
bitwise-identical directions and states before the second usable secant is
accepted. The implementations may diverge only at the first direction for
which history 10 retains a pair that history 1 has evicted. This is a
deterministic kernel identity test, not a cross-arm GPU trajectory gate:
independent float32 MACE evaluations may differ at numerical thresholds even
when the pre-divergence algorithm path is the same.

The following remain identical:

- total biased objective and total-gradient secants;
- empty-history scale \(1/70\);
- maximum movable-atom displacement of 0.2 A;
- monotone Armijo constant, backtracking factor, trial limit, and minimum
  alpha;
- positive-curvature gate;
- descent and finite-value checks;
- convergence and final force certificate;
- PBC wrapping and MIC image-signature reset;
- line-search failure and maximum-iteration semantics;
- evaluator, trace, telemetry, and purpose accounting.

## Private experiment seam

Extend the existing private argument

```python
_safe_lbfgs_history_limit: int | None
```

so that the accepted values are:

```text
None, 0, 1, 10
```

with the following meaning:

- `None`: unchanged production behavior, resolved to 10;
- `0`: existing empty-history experiment behavior;
- `1`: new one-correction experiment behavior;
- `10`: existing explicit history-10 behavior.

The value remains:

- keyword-only;
- unsupported for every optimizer other than `safe-lbfgs-total`;
- absent from `SSWConfig`, YAML, CLI, walker policy, and public presets;
- validated before the first evaluator call;
- restricted to literal integers, with booleans rejected.

No new scale flag, damping rule, rescue optimizer, or tunable parameter is
introduced.

## Frozen task matrix

Reuse the exact 16 source tasks from the reviewed scale-decomposition run:

- systems: C60 and fixed-mask periodic PdO;
- seeds: 42 through 49 for each system;
- one frozen Gaussian-bias proposal objective per task;
- identical initial state, movable mask, cell, PBC, bias center, bias
  direction, width, weight, `fmax`, and `maxiter=400`;
- identical MACE model, precision, device, input structures, task payload
  hashes, and source-summary hash.

Execute two arms for every task:

```text
2 systems x 8 seeds x 2 arms = 32 rows
```

One calculator is created per system and arm, never shared across arms. Tasks
within one arm may reuse that calculator sequentially. Arms and systems run
serially so GPU contention cannot become an uncontrolled treatment.

The run is one-shot. There is no retry, rescue, arm-specific warm-up, or
post-result parameter change.

## Budget and accounting contract

For every row, record and validate:

- source task identity and canonical payload hash;
- requested and resolved history capacity;
- scale policy and secant-gradient policy;
- result energy, final force, iterations, displacement, outcome class,
  convergence certificate, and termination reason;
- evaluator call count and purpose count;
- optimizer telemetry, including line-search evaluations, accepted and
  rejected steps, accepted and rejected secants, finalization calls, and MIC
  resets;
- the complete zero-extra-call energy/force trace.

The summary provenance schema is fixed as:

- `git_provenance`: expected commit, actual commit, repository root, and clean
  worktree boolean;
- `pamssw_source_provenance`: source root, deterministic source-bundle hash,
  imported module paths, and imported symbol-definition paths;
- `runner_helper_provenance`: absolute path and SHA-256 for every imported
  frozen-task, calculator, and trace helper;
- `runtime_versions`: Python, Python implementation, NumPy, SciPy, ASE,
  PyTorch, and MACE versions;
- `platform_provenance`: `sys.platform`, operating-system name and release,
  and machine architecture;
- `cuda_model_input_provenance`: requested device, CUDA device name, CUDA
  runtime version, model path, declared and measured model hashes, and each
  source input path with declared and measured hashes;
- `source_summary_sha256` and `safe_kernel_descriptor_sha256`: declared and
  independently recomputed hashes.

Unknown, missing, mistyped, or non-finite provenance fields fail validation.

The following closure must hold exactly:

```text
trace records
  = evaluator calls
  = telemetry objective calls
  = biased-proposal-relax purpose calls
```

and:

```text
unattributed calls = 0
```

Finite `maxiter` and finite `line_search_failed` rows are valid incomplete
protocol outcomes. They remain in all cost totals and may not be converted
into successful convergence.

Any nonfinite row, source/provenance mismatch, duplicate or missing matrix
cell, open accounting ledger, unknown field, invalid certificate, or partial
output is fatal. Publish the complete output directory atomically only after
all 32 rows and the summary validate.

Raw GPU output remains ignored. Committed analysis artifacts must contain
cryptographic hashes of the raw files and enough reviewed derived data to
audit every stated aggregate. Third-party raw replay requires separately
shipping the ignored ledger.

## Analysis

Use certificate coverage as the first outcome, followed by complete-protocol
evaluator cost and wall time. Do not compare only converged rows.

For each system and for the combined matrix, report:

- certificate-satisfied count;
- termination-reason counts;
- total evaluator calls;
- total wall time;
- accepted/rejected steps and secants;
- MIC reset count.

For every paired task, report history1-minus-history10:

- evaluator calls;
- wall time;
- iterations;
- final force;
- final total biased objective energy;
- exact final-position-hash equality;
- MIC-aware maximum and RMS endpoint displacement.

Final-energy and wall-time differences are descriptive. Unless the endpoints
are proven equivalent, they are not same-minimum speedups.
Exact position-hash equality is a descriptive byte-level identity check. A
nonmatching hash or a small MIC displacement is not a same-basin classifier.

The exact energy/force trace figure uses:

- one panel per system and metric;
- task identity as color;
- history depth as line style;
- exact evaluator positions on the horizontal axis;
- all exact evaluations, callback-observed evaluations,
  callback-nonobserved evaluations, and explicit-finalization annotations.

The figure is an accounting visualization, not evidence that callback records
are optimizer acceptance decisions. `rejected_steps` is reported only as
aggregate optimizer telemetry because the existing zero-extra-call trace does
not identify which individual callback-nonobserved evaluation was a rejected
line-search trial.

## Interpretation rules

No arbitrary scalar pass threshold is introduced.

The result is interpreted as follows:

1. If history 1 preserves certificate coverage and does not increase complete
   protocol cost relative to history 10 on the fixed matrix, the older
   retained corrections are not supported as necessary on these tasks.
2. If history 10 improves certificate coverage, or preserves coverage while
   reducing complete protocol cost, older retained corrections are a positive
   candidate on these tasks.
3. If certificate and cost evidence conflict by system or task, report the
   result as mixed. Do not select a default from an aggregate scalar score.

These are fixed-matrix descriptive conclusions. They are not statistical
noninferiority tests.

## Claim ceiling

This experiment may determine only:

- whether the newest two-loop correction alone recovers the history-10 result
  on the frozen C60/PdO proposal tasks;
- whether older retained corrections add descriptive certificate or
  complete-protocol cost value on those tasks.

It may not establish:

- generic optimal L-BFGS memory;
- a production-default change;
- endpoint or basin equivalence;
- full-SSW search performance;
- canonical or unbiased sampling correctness;
- superiority over ASE or SciPy on general relaxation;
- value from bias-separated secants or analytic Gaussian-bias Hessians;
- a statistically generalized effect.

No further history-depth experiment is entered automatically. Any history 2
or wider depth study requires a new design justified by the observed
one-versus-ten result.
