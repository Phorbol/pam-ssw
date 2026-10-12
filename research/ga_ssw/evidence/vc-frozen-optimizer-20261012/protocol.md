# Frozen biased-VC local-optimizer comparison

Status: reviewed; bounded GPU job 1756782 submitted from runner commit 18fa744.
Four contract checks and zero-PES preflight passed with PYTHONNOUSERSITE=1.
Without that launcher environment, root preflight encountered a user-site
Matplotlib/scienceplots import conflict; no PES was evaluated. The question is whether Safe-total and SciPy L-BFGS-B behave
differently when minimizing the same saved biased variable-cell objective from
the same chart point, and whether any inner-stage difference survives an
identical unbiased physical quench. This is a local optimizer diagnostic, not a
global-search or LASP-parity test.

## Frozen cases and evidence boundary

Use two distinct saved 48-atom TiO2 cases, each paired across the two
optimizers:

| Case | Frozen input and objective source | Interpretation |
|---|---|---|
| Phase-87 | `ga-ssw-behavior-parity/research/ga_ssw/evidence/tio2-phase87-vc-pqc-single-step/joint-stage8-memory-compare-v1/frozen-input.json`, its `plan.json`, and upstream `joint-stage8-diagnostic-v1/frozen-stage8.json` plus `provenance.json` | Use its nine saved terms and the plan's width 0.6 A (the first eight frozen records omit the shared width). The stage-8 direction had to be reconstructed and the saved weight reused; this is a newly constructed frozen objective, not an exact replay or repair of the historical stage. |
| Phase-139 | `ga-ssw-behavior-parity/research/ga_ssw/evidence/tio2-phase139-joint-memory-compare/memory400/result.json`, `records[1].chart_reference`, all nine entries of `records[1].frozen_gaussians` in stored order, and `records[1].climb` entry with `index=8` | Use all nine archived terms (source indices 0–8; each contains 150-component center/direction plus weight and width). Its saved index-8 q is provenance only. Start from the stage-8 displaced point constructed by the last term, not from the terminal q: `center_8 + width_8 * direction_8`. This is a newly constructed objective and start, not a claim about an accepted historical SSW point or completed stage. |

For both cases, define the optimizer start from the last archived Gaussian as
`q0 = center_8 + width_8 * direction_8`, matching the saved VC lifecycle's
displacement before local biased relaxation. The phase-87 historical input
already used this construction (`center_q` is the final term center); the
phase-139 saved terminal q is retained only as provenance. These are newly
constructed common starts, not claims of historical replay or accepted-state
recovery. Within each case, serialize one immutable q, chart reference, and
Gaussian collection and give byte-identical copies to both optimizers. Use the existing
`SymmetricLogStrainChart` with strain length 5 A, pressure 0, and the saved
Gaussian widths, centers, and weights. Reconstruct the frozen scalar objective
from the archived case data without adding/removing terms. The objective is
the physical E+pV plus the exact frozen Gaussian sum. Both inner methods
receive the same objective and analytic generalized gradient in this q chart.

The model is the same MACE OMAT-small / OMAT-PBE float64 model, SHA256
`0abfde07862cf1e93b8b4d03cb702f29ce9c344ff2fc4de2ec0d7166d6c113a5`. The
historical phase-87/139 optimizer records were CPU float64; this protocol uses
one V100 and is a new environment-qualified comparison, not a bitwise replay.
The two cases test two starts on the same TiO2/MLIP domain, not transfer across
materials or models.

Before either paired run, independently load each saved q and perform its first
fresh E/F/stress evaluation. Confirm finite values and valid cell geometry,
recording the resulting physical state as start qualification. This is not a
requirement that the biased start
already satisfy the final force/stress thresholds. Stop before optimization
for invalid geometry, nonfinite evaluation, or a source/objective reconstruction
mismatch; retain the failed preflight and do not repair the input.

Runner `--preflight` checks archived inputs, source root, model hash, chart
reconstruction, and optimizer options without making PES calls. `run.sbatch`
is a prepared launcher; this task does not submit it. Outputs are unique by
job ID. The run saves each case's qualified start, each arm's accepted biased
endpoint and unbiased endpoint, and ledgers that separate paid EFS calls,
unpaid cap denials, and actual calculator `calculate` calls.

## Paired inner minimizers

Run four independent arms: two cases times Safe-total and SciPy L-BFGS-B. No
arm inherits optimizer history or a state produced by the other method. Before
either solver starts, evaluate the same initial biased objective and gradient
and compute the shared chart-aware block norm. If it already meets the target,
skip both inner optimizers for that case and send their common initial q to the
physical handoff. Otherwise the inner stopping target is the same block norm
used by the existing VC implementation:

`max(max_i ||g_atom_i||_2, ||g_cell,6||_2) <= 0.1 eV/A`.

Here the six cell coordinates are the orthonormal symmetric log-strain
components scaled by the existing 5 A chart length. This is an inner biased
gradient threshold, not an unbiased physical force or stress certificate. The
historical `.005` generalized-gradient setting is not the target for this
diagnostic.

Safe-total uses the existing flat-coordinate Safe-total implementation,
history 500, max step 0.2 A in the q chart, and at most 300 iterations. SciPy
uses `scipy.optimize.minimize(method="L-BFGS-B", jac=True)` with `maxcor=500`
to match the nominal history, `maxiter=300`, and `maxfun=1000` including the
initial objective evaluation. Its `ftol=2.220446049250313e-9` is the documented
L-BFGS-B default (`factr=1e7` times float64 machine epsilon); report an `ftol`
termination separately and do not count it as reaching the shared 0.1 target.
Set SciPy's componentwise `gtol=0`; the accepted-iterate callback applies the
shared max-block norm and stops at 0.1. The callback must look up the accepted
q and its E/g in the evaluator cache with an explicit exact-coordinate match;
it must not make an uncounted oracle request. A callback/cache mismatch is a
protocol error, not a reason to relax matching.

SciPy documents `gtol` as a maximum projected component norm, which is not the
VC max-block norm above. That is why SciPy's native `gtol` does not implement
the common target. Its `ftol` can also stop before the callback target; that
result remains a valid finite endpoint and is labeled `relative_energy_stop`.
The same applies to max-iteration or max-function-call exits. Any valid
accepted endpoint proceeds to the shared unbiased quench below. The SciPy and
Safe-total update bundles are not otherwise identical: Safe-total bounds the
whole q-step by 0.2 A, while standard L-BFGS-B uses its own line search and
`maxls=20`. Report this as an optimizer-bundle comparison; do not attribute a
difference to line search or curvature algebra alone.

The 300-iteration ceiling is inherited from the saved VC stage protocol; the
1,000-EFS inner cap is a matched allocation for each method within the global
budget below. Enforce it in the paid evaluator as a hard cap; SciPy's `maxfun`
alone is not a hard oracle cap. On a denied next request, return the last
accepted q and mark `request_limit`; never hand off a line-search trial or
unaccepted cached point. SciPy `ftol` is the documented default, and `gtol=0`
deliberately leaves stopping to the shared callback rather than fitting a
componentwise threshold. No optimizer settings may be selected after
examining paired outcomes.

## Common physical handoff and measurements

After every inner method returns, run the same existing unbiased all-DOF
Safe-total cell quench from its final valid accepted q, including endpoints
that stopped early on relative energy change. The quench uses the physical
E+pV objective without Gaussians, the existing 5 A chart, 0.2 A maximum q-step,
history 500, at most 300 iterations and a 400-oracle-call cap. Its physical
landing gate is `max_i ||F_i||_2 <= 0.05 eV/A` and maximum absolute stress
component `<= 0.001 eV/A^3` (pressure is zero). The force threshold follows the user-selected 0.05 setting; the stress
threshold is inherited from the archived VC protocol; the inner biased gradient is not
expected to vanish at the physical endpoint. Perform one independent final
E/F/stress certificate per valid quench result and count it in the same
budget.

Ordinary optimizer exits (`initial_target_reached`, `ftol`, line-search stop,
`maxiter`, or the hard request limit) hand off their last valid accepted q to
the common quench. An EFS/backend exception or nonfinite evaluation stops the
campaign immediately: preserve the original status/error and last accepted
state, perform no further PES calls, and mark all remaining arms not run. Never
overwrite an inner status with the later quench status. Record optimizer-native
termination, whether the shared 0.1 callback criterion was reached, biased
objective and max-block gradient at the accepted endpoint, request and actual
calculator counts, relative-energy exits, line-search failures, quench status,
final fresh force/stress certificate, and total elapsed time. Keep inner cost,
quench cost, and certificate cost separately visible as well as their sum.

## Budget, stops, and decision rule

One V100 allocation is capped at 30 minutes. The aggregate cap is 6,000 paid
E/F/stress oracle requests across both case reconstructions, all four inner
minimizers, all four common quenches, and all independent certificates. Per
arm ceilings are 1,000 inner requests, 400 quench requests, and one final
certificate; both case-level fresh start checks and any other EFS calls are
charged to the same aggregate ceiling. Requests denied at a cap are recorded
separately from paid evaluations. No retries, continuations, budget transfer,
or outcome-dependent setting changes are allowed. Wall or request truncation
is a censored observation, not convergence; preserve partial traces and do not
call it a failed physical minimum.
The runner stops paid work after 28 minutes from process start, reserving two
minutes inside the 30-minute scheduler allocation to write ledgers and results.
Model initialization and qualification consume that 28-minute search budget.
The nominal maximum is 5,606
paid requests: four arms at 1,000 inner + 400 quench + one certificate each,
plus two fresh case-start checks. The remaining global allowance is a ceiling,
not a target.

The comparison is informative only if the frozen objective reconstruction
gate passes for both cases and both optimizers receive byte-identical starts,
charts, and Gaussian terms within each pair. A consistent inner or handoff
cost/qualification difference on both cases would justify a follow-up optimizer
study on additional frozen starts; it would not justify changing the default.
Mixed or budget-censored outcomes retain the current solver pending a better
decision-relevant test. Neither this two-start same-domain panel nor an inner
stationarity result establishes SSW discovery performance, cross-material
generality, or LASP optimizer parity. No submission is authorized by this
protocol; a runner and any calculation require separate plan review.

## References and source of thresholds

The 5 A chart, zero pressure, 0.2 A q-step, 300-iteration ceiling, and
0.001 eV/A^3 stress gate are inherited from the saved local VC protocol, not
fitted to this comparison. History 500 and physical `fmax=0.05 eV/A` are
current user-requested settings for the biased Safe-total inner stage and the
common unbiased quench. The 0.1 eV/A inner target is the current diagnostic
request; it supersedes the
historical `.005` target for this question only. SciPy option meanings and
defaults follow the official [`minimize(method="L-BFGS-B")` reference](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html)
and [`minimize` callback documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html).
The stage-8 memory-only outcomes and their limits are summarized in
`docs/research/vc-optimizer-metric-diagnostic.md`; native optimizer saved-state
limits are in `docs/research/native-vc-optimizer-handoff.md`.
