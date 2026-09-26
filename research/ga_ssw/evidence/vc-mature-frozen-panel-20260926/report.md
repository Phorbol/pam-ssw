# VC frozen optimizer comparison: practical 0.05 tolerance

## Decision and limits

Keep Safe-total as a candidate for joint variable-cell tasks. This panel found no
reason to replace it with ASE LBFGSLineSearch or SciPy L-BFGS-B. It does not prove
an end-to-end VC-SSW advantage, universal optimizer superiority, or a causal
line-search advantage. Four fixed objectives from two materials were compared;
no parameters were tuned from these results and no production search was launched.

All three implementations use history500, the same reconstructed target, initial
state, calculator and per-task request cap. Biased joint gradient tolerance is
0.05 eV/Å; unbiased qualification requires both physical fmax <=0.05 eV/Å and
max absolute residual stress <=0.001 eV/Å³. This is a development threshold, not
an assertion that energy differences or phonons are converged. ASE's documented
Optimizer.run default is also fmax=0.05:
https://docs.ase-lib.org/_modules/ase/optimize/optimize.html .

## First accepted point meeting the common criterion (E/F/stress requests)

|Fixed task|Safe-total|ASE LBFGSLineSearch|SciPy L-BFGS-B|
|---|---:|---:|---:|
|AlOH26 biased|26*|55|26|
|AlOH26 unbiased cell-relax|45|80|50|
|TiO2 phase87, 48 atoms, biased|102|179|103|
|TiO2 phase87, 48 atoms, unbiased cell-relax|62|117|64|

Full native termination costs, in the same row order: Safe 26/45/102/62;
ASE 63/96/221/153; SciPy 38/58/121/83. The native stopping norms differ, so these
full costs are descriptive and must not be described as equal-stopping-rule
speedups. The common first-passage comparison removes that specific confound.
Step caps still differ (Safe joint block / ASE pseudo-atom triplets / SciPy no
same cap), hence this is a whole numerical implementation comparison.

## Execution and independent checks

Job1499633 completed in 6m55s, with 11 numerical results and one process-startup
failure (exit124). That failure produced no task directory or PES ledger; stdout
only contains import warnings. It remains in the execution denominator, not
classified as an optimizer failure. *Job1499684 repeated only this missing task,
with a longer external startup allowance but unchanged 100s search deadline,
600-request limit and numerical code, and completed in21s. No further retry.
See startup-recovery.md. Total 13 process attempts, 12 numerical results.

All 12 source physical energy checks passed. All six biased endpoints independently
satisfy the common biased-gradient criterion; their physical forces need not be
small because the Gaussian force balances the true force. All six unbiased
endpoints independently satisfy the physical force and stress certificate.
This is numerical qualification; neither a Hessian check nor a proof of a stable
phase or improved basin discovery. Total paid requests are 1092, including 12
source validations and 12 fresh endpoints. Startup overhead is not hidden;
per-task reported stage timings also contain final calculator setup, so no
universal wall-time speedup is claimed.

The AlOH legacy chart reconstruction was verified against its original input and
post-quench geometry; there is no archived biased-gradient golden value, so the
comparison is of the shared reconstructed formula, not exact LASP numeric parity.
The preserved pathological simple-cubic Cu CPU preflight is separate negative
evidence and is not erased by the successful real-material tasks.

## Provenance and reproduction

Numerical source: commit2f35e91; unmodified during both jobs. Frozen inputs and
model metadata: plan.json and each run's task.json/source-validation.json.
Full logs, ledgers, accepted coordinates, endpoints and environments are retained
locally under run-1499633/ and startup-recovery-1499684/ alongside this report.
Compact machine-readable result: readout.json, including all13 process attempts.
Analysis (zero PES):

```sh
python analyze.py run-1499633 startup-recovery-1499684 --output NEW_READOUT.json
```

The script recomputes each accepted-iterate first passage and asserts agreement
with the recorded summary. CPU1499611 previously passed the four targeted
formula/source reconstruction/accounting tests. Independent read-only code review
confirmed units, norm conversion and common certificate semantics; it is not an
independent physical replication. No core source/defaults changed in this panel.

Next: retain this scoped evidence alongside the completed TiO2 cell-on/off pilot;
any claim of stable VC-SSW efficiency requires a bounded matched end-to-end search
comparison, with identical policy/metric/bias schedule and valid distinct-landings
per total cost. Do not select an optimizer from native iteration counts or add a
new heuristic in response to this panel.
