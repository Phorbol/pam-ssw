# AutoSella / local-optimizer cost readout

**Question.** Do existing SSW runs show that local optimization is an important cost target, and is there measured evidence that its walltime limits SSW throughput? This is a read-only audit of saved records at worktree HEAD `9b8152a0d56b37ade09cba707cad031471510711`; no PES, HPC job, or code-path experiment was run.

**Finding.** Local optimizer stages account for a large share of *evaluation requests* in the examined ledgers, so optimizing that path is a reasonable performance hypothesis. The records do not contain per-stage timers. They therefore do not establish the optimizer's walltime share or show that it is the dominant SSW bottleneck. AutoSella's potential benefit to total runtime cannot be quantified from these data.

## Request-count panels

| Evidence panel | Scope and denominator | Rotation | Biased local optimization | LS pre-quench | True landing quench | Other / unassigned | Failures and timing |
|---|---|---:|---:|---:|---:|---:|---|
| C4H6 MH-1 | 300 selected, correlated climb stages; 7,468 requests | 2,207 (29.6%) | 4,661 (62.4%) | not separable in this stage-only ledger | outside this selected-stage denominator | 600 (8.0%) | All selected stages report `converged`; full-run walltimes exist in the coverage report, but no stage timers. |
| C60 selected paths | 48 selected, correlated climb stages; 3,211 requests | 417 (13.0%) | 2,698 (84.0%) | not separable in this stage-only ledger | outside this selected-stage denominator | 96 (3.0%) | All selected stages report `converged`; no stage timers. |
| C60 direction probe, native LS seed 1101 | 10 outer attempts; 4,090 search requests, including 1 initial/outside-record request | 693 (16.9%) | 2,785 (68.1%) | 75 (1.8%) | 314 (7.7%) | 222 climb-event requests (5.4%) + 1 initial request | 9 `gaussian_limit`, 1 `lower_true_energy`; 10/10 landing quenches converged. Whole-run elapsed 318.727 s; no phase walltime. |
| C60 direction probe, no LS seed 1101 | 10 outer attempts; 4,106 search requests, including 1 initial/outside-record request | 714 (17.4%) | 2,834 (69.0%) | 0 | 317 (7.7%) | 240 climb-event requests (5.8%) + 1 initial request | 10 `gaussian_limit`; 10/10 landing quenches converged. Whole-run elapsed 291.405 s; no phase walltime. |
| Rutile periodic direction, global | 3 outer attempts; 1,259 requests, of which 1,258 reconcile to records | 376 (29.9%) | 685 (54.4%) | 0 | 47 (3.7%) | 150 climb-event requests (11.9%) + 1 initial request | 3/3 attempts stop at `gaussian_limit`; 3/3 landing quenches converged. No walltime field in the saved result/summary. |
| LJ Gaussian-policy diagnostic | 82 saved stages across 12 outer attempts; 5,678 requests after the initial quench | 566 (10.0%) | 4,361 (76.8%) | not separately reported | grouped with other: 751 (13.2%) | Included in the 751 remainder | Selected diagnostic cases only; report gives no phase walltime and warns against extrapolating to a full search. |

The C60 direction-probe record ledger closes exactly: rotation + biased quench + event residual + LS pre-quench + true landing quench equals `record.evaluation_requests` for every record and for the run totals. `result.evaluation_requests` is one higher, matching one initialization/outside-record request. The separate 11 fresh-check requests in each summary are not part of the search-request denominator. The `climb_unassigned` term is a residual inside climb-event requests; it is not claimed to be pure rotation or optimizer work. C60 outer-attempt status and landing convergence are distinct: `gaussian_limit` attempts can still have a converged final landing.

The stage-ablation panels use the fixed selected-path audit's `requests`, `quench_requests`, and raw `rotation_force_requests`; these are correlated subsamples, not complete run budgets. C4H6 full-run reporting separately lists 400 outer records and overall search walltimes of 4,621–4,908 s across the six arms, but the selected-stage ledger cannot be divided into those totals to infer phase shares. That would mix a selected-stage sample with whole-run denominators.

The four complete C60 direction-probe runs together contain 40 outer attempts and 16,298 search requests (two seeds, each with/without LS; correlated configurations, not four independent system classes). Biased quench accounts for 11,254 requests (69.1%), true landing quench 1,162 (7.1%), LS pre-quench 146 (0.9%); all local-quench stages total 12,562 (77.1%). Rotation is 2,804, climb residual 928, initialization/outside-record residual 4. All 40 landing quenches converge; this does not imply 40 useful or distinct minima. The aggregate includes initialization and excludes separate fresh validation. These numbers are reproducible by summing the four `c60_direction_probe_runs` totals in `cost-readout.json`.

## What timing supports

Only whole-run timing is available for the four C60 direction-probe runs: 284.689–318.727 s for 10 attempts each. C4H6 has whole-run walltime in its coverage report. The periodic result files and the LJ diagnostic report do not provide usable phase timers. None records time spent separately in rotation, LS preparation, biased local optimization, true-potential landing optimization, initialization, or Python/control work. Evaluation-request shares are not timing measurements: per-request backend cost, cache behavior, and optimizer/control overhead can differ by phase and system.

Thus the evidence supports investigating local-optimizer runtime, but not the claim that its walltime currently bottlenecks SSW. It also does not predict AutoSella's speedup: that requires a matched benchmark on the same saved inputs and settings, with phase timers and the same E/F backend, followed by complete-run walltime comparison. No such benchmark is part of this audit.

## Conditional Amdahl bound

If a future matched study measures fraction `p` of end-to-end walltime in the local-optimization stage, and that stage becomes `a` times faster at the same scientific outcome distribution, then

\[
S_{\mathrm{total}} = \frac{1}{(1-p)+p/a}.
\]

The local-stage speedup `a` may include fewer E/F requests, faster work per request, or less optimizer/control overhead; request-count and outcome distributions must be reported so the cause and scientific comparability remain clear. The infinite-local-speed limit is `1/(1-p)`. For example, **if** a timer later measured `p = 0.8` and a matched benchmark measured `a = 5`, the conditional total speedup would be `1/(0.2+0.8/5) = 2.78×`, with an infinite-speed ceiling of `5×`. These are algebraic examples, not estimates from the current request counts. If only optimizer algebra is accelerated while E/F request count and calculator cost stay fixed, `p` must instead mean the measured walltime share of that algebra alone; it may be much smaller than the request share.

## Reproduction and source fields

Run `python research/ga_ssw/evidence/autosella-review-20261007/readout.py` from this worktree. It reads saved JSON only and rewrites `cost-readout.json`. The C60 ledger uses `record.evaluation_requests`, each climb event's `rotation_force_requests`, `quench_requests`, and `requests`, `record.ls_preparation.evaluation_requests`, and `record.landing.evaluation_requests`. Whole-run elapsed time comes from the run's `summary.json:elapsed_seconds`. Stage-ablation values come from `stage-cost-audit.json` plus the selected original records named in `inputs.json`. The LJ values are transcribed from the saved diagnostic report, which states its scope and arithmetic.

The raw data distinguishes charged evaluation requests from calculator calls; these are not interchangeable. The report keeps the stage sample, complete direction-probe run, periodic three-attempt probe, and selected LJ diagnostic denominators separate. Failures/stops remain in their attempt denominators.
