# Prepared eight-run whole-search history check (not launched)

Executable: `research/ga_ssw/compare_safe_history_e2e.py`. Prepared artifact: `research/ga_ssw/evidence/safe-history-newseed-e2e-prepared/` containing exact raw input structures, full configuration/parameter provenance, fixed8-run order, source snapshot and hashes. Default invocation only prepares; **`--run` is required to calculate**. Source changes after preparation are refused so a revised source needs a newly prepared/reviewed artifact. No PES was invoked during preparation.

Current plan supersedes the earlier hard-C60/seeds29,43 suggestion. Systems are already used in development; only seeds29 and71 are new to this history decision, so this is not unseen-system validation.

| System | Walker | Original input/config sources |
|---|---|---|
|C4H6/GFN2-xTB|paper LS|ASE G2 `butadiene`; `compare_c4h6_native_paper_6000.py`: T150,width .1,NG25,Safe400 iterations,dimerHVP100,fd1e-4,rotation tolerance .02,fmax .01,LS target .7 eV/atom and unchanged frozen-pair table/feedback|
|Cu13/EMT|ordinary SSW|`independent-cu13-surface/quenched.extxyz`, the previously used input, not a newly selected landing; `compare_cu13_safe_total.py`: T300,width .2,NG14,Safe200 iterations, same dimer precision and fmax|

Each system×seed has memory10 and400 arms,2 outer attempts. No other parameter varies and no native-height policy is added. The only intervention is the process-local Safe memory constant, restored after each run. All8 slots are declared before launch and remain in the summary even if the shared wall cap prevents later starts. Order is seed29 then71, each Cu13 pair then C4H6 pair, memory10 then400. The order is explicit; shared-wall censoring must not be interpreted as symmetric per-arm runtime.

Budget: each arm6000 total E/F (5997 search+up to3 independent fresh), campaign48000 E/F/600 elapsed seconds maximum, one CPU thread, no GPU. Deadline includes calculations and fresh checks. Initialization, failed calls, failed quenches and rejected valid landings are retained. No retries, resource extensions or threshold relaxation. Save result before fresh checking. Independent new calculators verify all valid minima including rejected landings; uncompleted validation remains pending. Original atom coordinates and per-request raw E/F are retained for offline diagnosis.

Outputs include all step statuses, MC decisions, valid minima, LS responses, fresh E/F, carbon composition-resolved connectivity/components, minimum distances, radius of gyration, aligned same-atom RMSD and sorted pair-distance differences from initial. Geometric diagnostics are explicitly approximate; they are not exact basin matching. They separate near-recurrence and fragmentation from candidate useful change. Claimed new distinct basins would still require appropriate offline strict geometry qualification; force alone is neither Hessian stability nor global-minimum proof.

Primary comparison: does improved local convergence survive the full deform/quench/MC lifecycle as force-qualified chemically intact non-recurrent candidates per total cost? Report paired seeds, all8 slots, failures, censoring and validation gaps, not acceptance fraction alone. The result cannot upgrade a default on its own.

Reviewable launch command (not executed):

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=/tmp/pam-ssw-tblite-20260909:. \
python research/ga_ssw/compare_safe_history_e2e.py --run
```

This launch operates locally, uses no Slurm or GPU, and does not modify production files.
