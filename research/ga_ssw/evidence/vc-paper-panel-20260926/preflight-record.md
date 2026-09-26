# Runner preflight record

Date: 2026-09-26. No GPU job was submitted by this branch.

## Scheduled preflight

- Slurm job `1499412` on CPU-MISC completed successfully.
- Scope: `py_compile` of the research runner and the one Cu/EMT cell-on/off runner test.
- Result reported by root: **1 test passed in 1.55 s**.
- This is a wiring and lifecycle preflight only. Cu/EMT is not evidence for the TiO₂ mechanism or the MACE model.

## Earlier unplanned login-node check

Before the CPU-MISC job was submitted, the same Cu/EMT test was run locally by mistake, despite the instruction to keep PES evaluation off the login node. It ran on `login-01.mr-sai.ai` in this worktree using the system Python 3.12 and user-site ASE, took 1.06 s, and reported one pass with 358 NumPy/ASE deprecation warnings. A preceding Cu EMT reference-quench probe also ran on that node and reported 5 E/F/stress requests.

The test did not persist or print its per-arm request counts, so its total calculator-call count is **unknown and unrecoverable**. No repeat was made to obtain the missing count. The login-node run is retained here as an execution-policy error and is not substituted for the scheduled preflight result. All subsequent verification and calculation work for this runner stays on scheduled nodes.

## Static checks

- `python -m json.tool research/ga_ssw/evidence/vc-paper-panel-20260926/sources.json`: passed.
- `git diff --check`: passed.
- `bash -n research/ga_ssw/evidence/vc-paper-panel-20260926/run.sbatch`: passed before the parent agent's planned scheduler edits.

## Prepared material screen boundary

The planned GPU task remains unsubmitted pending root review. Per-arm search limits are 6,000 E/F/stress calls and 600 seconds; the separate common reference gate is capped at 300 calls and 300 seconds. There is one paired seed, two requested outer events per arm, up to two independently checked endpoints per arm, no result-dependent extension, and a request or wall cap before a completed outer escape is recorded as censoring.
