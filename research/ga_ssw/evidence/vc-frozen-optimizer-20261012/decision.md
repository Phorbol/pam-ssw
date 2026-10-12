# Frozen VC optimizer decision — 2026-10-12

Development diagnostic, not global-search validation. Runner 18fa744; job 1756782,
one V100, scheduler completed in 24 seconds. Raw evidence is `run-1756782/`.
Frozen protocol: two 48-atom TiO2 objectives, nine Gaussians each, OMAT-small,
inner block target 0.1, identical unbiased quench and fresh force/stress gates.

| Case | Optimizer | Inner EFS | Quench EFS | Fresh | Inner target | Fresh force/stress |
|---|---|---:|---:|---:|---|---|
| phase87 | Safe-total500 | 82 | 59 | 1 | pass | pass |
| phase87 | SciPy L-BFGS-B500 | 81 | 61 | 1 | pass | pass |
| phase139 | Safe-total500 | 107 | 63 | 1 | pass | pass |
| phase139 | SciPy L-BFGS-B500 | 111 | 70 | 1 | pass | pass |

Two case-start qualifications add 2 EFS: 640 paid = 640 actual calculator calls,
independently summed from campaign-ledger.json; no failed paid calls or denied requests.
SciPy native status 99 means the accepted-iterate callback reached the common
criterion, not optimization failure. No relative-energy early termination occurred.
The raw `cost.inner_logical_requests` is null due to a reporting lookup error;
`inner.logical_attempts` and the raw ledger carry the correct counts. Preserve raw
outputs; use those fields in analysis. No rerun is needed for this metadata defect.

The inner request differences are mixed and small. Whole-arm costs are 142/143
and 171/182. All physical force maxima are below 0.05 eV/A and stress maxima
below 0.001 eV/A^3. This is numerical qualification, not evidence of new phases,
better global optimization, or physical model accuracy. First-arm runtime includes
warmup effects; the fixed execution order does not support a timing speedup claim.

Decision: retain current optimizer; do not add a public backend abstraction or
extend this same-domain panel. At practical inner tolerance the selected frozen
objectives do not reproduce historical strict-tolerance failures. This does not
isolate tolerance as their cause: starts, protocol and environment also differ.
The current evidence does not justify prioritizing optimizer replacement over
end-to-end search behavior. No parameter fitting or default changes.

Verification: four targeted contract tests passed; zero-PES preflight passed under
PYTHONNOUSERSITE=1 (the launcher uses it). Omitting it in a root manual preflight
exposed a user-site Matplotlib/scienceplots import conflict before any PES call.
Source, inputs, model and runtime are recorded in the raw manifest. Full accepted
coordinate traces remain raw artifacts; derived summaries must not duplicate them.
