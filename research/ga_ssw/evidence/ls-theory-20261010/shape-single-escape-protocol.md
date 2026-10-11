# Shape LS paired single-step experiment (prepared, not submitted)

## Question and decision

This experiment asks whether changing the frozen loading geometry from pair
softening to shape softening changes the candidate produced by one complete
SSW outer step, at the same initial LS fraction and the same search budget. It
also retains a no-LS arm to show whether either LS arm differs from the
unbiased baseline. The comparison is developmental evidence only.

The frozen arms are `none`, `pair-0.03`, and `shape-0.03`. The 3% initial
fraction is the existing paper-LS setting. No parameter is selected from these
outcomes. Shape LS is an experimental free-cluster geometry and its controller
response has not been independently qualified.

## Starts, backend, and configuration

The paired starts are:

| Start | Frozen input |
|---|---|
| C4 gauche channel | `channel-v1/run-1728173/plus-0.050.extxyz` |
| C4 trans channel | `channel-v1/run-1728173/minus-0.050.extxyz` |
| C60 finite-response minimum | `c60-finite-response-v1/run-1729152/minimum.extxyz` |
| C60 isomer 3 candidate | `c60-ls-source3-qualification-20261007/qualification-1664282/final-candidate.extxyz` |

Each start uses seeds 701 and 702, paired across all three arms: 4 starts × 2
seeds × 3 arms = 24 one-step runs. Every arm starts from the same saved input;
each arm is a fresh `run_ssw(steps=1)` call, so controller state cannot carry
between outer steps.

All arms use the local MH-1 model at
`/home/gengjianrui/.cache/mace/mace-mh-1.model`, head `omol`, CUDA, float64,
with cuEq/OEQ disabled and deterministic Torch settings. The model SHA256 is
frozen as
`a522eb7f59c7879963d41586528f4980baf33e086c94aa92e3eafdeccad3be47`.
The C4 settings and recovered-rotation settings come from
`c4h6-mh1-lifecycle-20260924/effective_config.json` unchanged (Gaussian width
0.1, 25 Gaussians, 150 K MC, `fmax=0.03`, 400 relaxation steps, safe total
LBFGS with history 500). C60 uses the same settings, with only the final
force threshold set to `fmax=0.05 eV/A` for its per-system qualification.
This difference is fixed before execution and is shared across the three C60
arms.

The C4 LS arms share the existing HC bond-energy table, pair length table plus
0.1 A, target 0.7 eV/atom, `xi=0.2`, learning rate 1.8, and pre-quench
(`fmax=0.1 eV/A`, 50 steps, `force_or_step_limit`). The C60 LS arms use the
prior finite-response values explicitly: C-C bond energy 3.61 eV, cutoff
1.64 A, target 0.02 eV/atom, `xi=0.2`, learning rate 1.8, and the same
pre-quench settings. These are per-system frozen LS configurations, not a
pooled parameterization. Within each family, only `bias_geometry` differs
between pair and shape. The no-LS arm has no LS settings.

## Preflight and records

Before the arms, make one fresh model E/F request per input. Input checks plus
all candidate fresh checks share an aggregate qualification cap of 100
requests; input preflight itself is capped at 120 seconds. Persist each fresh
force result before graph checks so a rejected input retains its qualification
record. Record the input SHA256, energy, maximum force, atom count, sorted
element-labelled graph edges, graph isomorphism to the start, and geometric
center radius. C4 graphs use the existing HC element-dependent cutoffs plus
0.1 A; C60 graphs are retained at 1.64, 1.70, and 1.80 A. C60 isomer 3 must
have fresh maximum force at most 0.05 eV/A and a connected, three-regular graph
at 1.70 A or the campaign is rejected before any search arm. It has no Hessian
qualification; this experiment does not add a Hessian job and cannot establish
stationary-point identity for that start.

For every arm retain its input, request ledger, SSW result, checkpoint, summary,
Gaussian count, LS preparation/update records, and landing geometry when
available. Use the existing driver's valid landing statuses plus a true-surface
landing to identify a generated candidate; retain other landing geometries as
diagnostics. Record the MC decision separately from candidate generation. For
each valid candidate, perform at most one fresh E/F request and report energy,
force qualification, sorted graph edges and element-labelled isomorphism to
the start at the stated cutoffs, and radius diagnostics. Failures,
request denials, partial results, and timeouts remain in the record. A
step-limit pre-quench is not a stationary-point certificate.

## Budget and stop rules

Each arm is capped at 4,000 search E/F requests and 120 seconds, including SSW
initialization, rotation, bias work, and final quench. The campaign is capped
at 60 minutes on one GPU, including model load, input checks, and all arms.
There are at most 24 candidate fresh checks, all included in the aggregate
100-request qualification cap. No cap is extended after truncation. The
complete search-request ceiling is 96,000.

The prepared runner is
`research/ga_ssw/probe_ls_shape_paired_escape.py`. Its default mode only prints
that it is prepared. `--execute` is required to run; this handoff does not
submit a job or invoke the runner.

## Decision signals and limits

Read paired runs by start and seed. Report candidate production, true-surface
force qualification, graph changes and radius changes, MC acceptance, request
cost, and elapsed time as distinct observations. A favorable result requires
repeatable changes in physically force-qualified true-surface candidates
without relying on a larger budget; graph change alone is not a basin or
chemical identity claim.

If pair and shape produce indistinguishable paired outcomes, this sample gives
no evidence that shape loading improves one-step landing behavior. If shape
changes candidate generation but fresh forces fail, the result is a numerical
qualification issue. If all arms are similarly truncated, the run only
qualifies the budget and cannot establish a negative efficacy result. A
consistent positive signal motivates a separately frozen transfer study;
this small panel does not support generality, search success probabilities,
global-minimum claims, production defaults, or physical rate predictions.
