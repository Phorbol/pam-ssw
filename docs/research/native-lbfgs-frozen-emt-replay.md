# Actual ELF LBFGS instructions on the 31 frozen Cu13 problems

2026-09-10. The isolated uploaded LBFGS kernel, supplied with consistent EMT + frozen Gaussian energy and gradient, converged **29/31** previously failed biased subproblems within the common budget. Existing PAM `safe-lbfgs-total` converged **31/31** on these same starts. This closes an instruction-level optimizer comparison; it is not a full LASP reproduction or a global-search ranking.

## What actually executed

The uploaded ELF numerical instructions executed under Unicorn, including LBFGS (`0x6e87b0`), DDOT/DAXPY, MCSRCH (`0x6ea9d0`) and MCSTEP (`0x6eb690`). There were **2,779 LBFGS entries, 5,458 MCSRCH entries and 40 MCSTEP entries** in the main comparison. Only `_intel_fast_memcpy` was replaced by host byte copying; no numerical optimization routine was replaced by NumPy, SciPy or ASE optimization. ASE EMT supplied energy and forces through reverse communication. No full native process, protection bypass or production-module change was involved.

The energy was EMT plus every frozen `ProjectedGaussian` and the gradient was its actual negative total force. BFGSDRIVER itself did **not** execute. In particular, the driver multiplies physical force by `force_factor` (default 0.05), whereas this expressly authorized comparison supplies **gradient scale 1** to preserve the common objective/derivative contract. Native Gaussian accumulator behavior, driver restarts, force scaling, allocation and outer convergence logic are outside this experiment. “Native” below means this isolated kernel configuration only.

The decoded 14-argument by-reference ABI is `N, M, X, F, G, DIAGCO, DIAG, IPRINT, EPS, XTOL, W, IFLAG, MAXSTEP, ETOL`. The first six pointers use System V integer registers and the remaining eight use the stack. `IFLAG=1` requests a new E/G; an `INFO=1` return from MCSRCH at `0x6e8dad` identifies an accepted iterate. We stop there upon satisfying the externally imposed raw force certificate. Accordingly, successful files can retain `IFLAG=1`: that flag alone is not our success criterion.

## Frozen parameter provenance

The plan was written before the 31-case run; no parameter was selected from its outcomes.

| Parameter | Value | Source and limit |
|---|---:|---|
| History M | 400 | `readsswpara` default pointer `0x4a49a00`, call setup `0x68903f–0x689046`, member `para+0x2ddc0` |
| FTOL / bfgs_etol | 1e-4 | Non-task-10 initialization `0x688fc1`, member `para+0x2dd58`; task-10 branch instead initializes 1e-10; user overrides remain possible |
| Scalar STPMAX | 0.5 | quick-setting-1 assignment `0x6c4571`, store `0x6c45c0` to `para+0x2ddc8`; init_bfgs passes this to driver field +0x1c0 |
| GTOL | 900 | ELF data `0x5521958`; actual isolated execution uses it; entire full-program write history not exhausted |
| STPMIN | 1e-4 | ELF data `0x5521950` |
| EPS | 1e-5 | bfgs_start literal at `0x5b2255`, field +0x190 |
| XTOL | 1e-16 | bfgs_start literal at `0x5b225f`, field +0x198 |
| MAXFEV | 20 | Native LBFGS initialization `0x6e8c8a` |
| DIAGCO / IPRINT | false / [-1, 0] | Caller-selected automatic native diagonal, quiet output |
| Gradient scale | 1 | Deliberate consistent-gradient comparison condition; differs from driver default 0.05 from parser literal `0x4a49938`, member +0x2ddb0 |

These values instantiate one documented quick-setting configuration. They are not asserted to be universal runtime defaults. STPMAX is a scalar line-search parameter, not necessarily the same quantity as an ASE per-atom displacement cap. GTOL 900 does not satisfy the standard strong-Wolfe premise `c2 < 1`; the formal Python implementation must not inherit it merely to match a symbol name. See [static call-path analysis](native-local-optimizer-linesearch.md).

## Controlled inputs, costs and outcomes

All 31 `biased_quench_failed` records from `cu13-direction-only` were retained. Each start is the last Gaussian center plus its width times direction, with the entire original Gaussian history frozen. The comparison does not recover from the failed endpoint or remove difficult cases. Caps are 201 total E/F requests, 200 accepted iterations and maximum per-atom total force 0.01 eV/Å. Existing Safe-total evidence uses these same conditions; its algorithmic parameters are its own baseline settings, not artificially made identical to native.

| Method | Qualified cases | Total optimization E/F requests | Failure statuses |
|---|---:|---:|---|
| ELF kernel, consistent gradient | 29/31 | 2,779 | Two native failure returns |
| PAM Safe-total, existing replay | 31/31 | 3,058 | None |

On the **29 common successful cases**, costs were native **2,733** versus Safe-total **2,951**. Native used fewer requests in 13 cases, Safe-total in 12, with four ties. The lower total native cost includes early failures and is not evidence of overall superiority. In particular, full success is better supported for Safe-total in this finite selected test set.

The two isolated failures are `17-dimer.json`, step 2 (22 requests, last accepted fmax 1.34091), and `17-ritz.json`, step 2 (24 requests, fmax 1.62276). Both return `IFLAG=-1, INFO=0`. A bounded replay using already saved E/G observes the non-descent return branch `0x6eac23 → 0x6eac29 → 0x6eaf94`, with DGINIT respectively 0.364158 and 0.387627, and compiled LP=6. No claim is made that BFGSDRIVER could not restart or otherwise treat these returns.

## Independent verification

A second instruction replay reused all recorded E/G values and verified exact requested-position agreement throughout. At **all 2,708 accepted-step hooks**, native X equaled the position of the current E/G evaluation. All initial positions match the corresponding Safe-total artifacts exactly, and initial energies agree to 1e-12 eV. This audit uses zero new EMT requests.

Every one of the 29 successful final structures was recomputed with a fresh EMT calculator plus the original frozen Gaussians. All passed raw total fmax ≤ 0.01 eV/Å; energy and force differences from recorded values were below 1e-12. This added **29 separately accounted certificate E/F requests**, outside the optimization budget. Such biased-surface force certificates do not establish true-PES minima, basin changes, physical stability or global-search efficacy.

Main per-case elapsed times sum to 75.48 seconds; deterministic verification took 81.49 seconds including the fresh certificates. These are Unicorn/host elapsed costs, not native machine performance benchmarks. The disposable first-case ABI milestone added 16 EMT requests; it was not another evaluation of a competing parameter setting. Two subsequent failure traces reused stored E/G with zero new requests.

## Artifacts and reproduction

- Script: [probe_native_lbfgs_emt.py](../../research/ga_ssw/probe_native_lbfgs_emt.py)
- Independent checker: [verify_native_lbfgs_replay.py](../../research/ga_ssw/verify_native_lbfgs_replay.py)
- Bounded failure trace: [inspect_native_lbfgs_failures.py](../../research/ga_ssw/inspect_native_lbfgs_failures.py)
- Evidence directory: `research/ga_ssw/evidence/cu13-failed-quench-native-lbfgs/`, containing frozen plan, script snapshots, 31 detailed trajectories, summary, verification and failure trace.
- Comparator: `research/ga_ssw/evidence/cu13-failed-quench-pam/`, including its optimizer-source snapshot.
- ELF: `/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp`; SHA256 `bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704`.

Run from the worktree, with optional Unicorn available in the temporary research environment:

```sh
PYTHONPATH=/tmp/pam-ssw-unicorn-probe:. python -m research.ga_ssw.probe_native_lbfgs_emt --output /tmp/new-native-lbfgs-replay
PYTHONPATH=/tmp/pam-ssw-unicorn-probe:. python -m research.ga_ssw.verify_native_lbfgs_replay
```

The checker intentionally targets the archived repository evidence directory. The formal implementation remains independent Python. This result supports retaining Safe-total as the working local optimizer while full SSW end-to-end behavior is assessed separately; it neither validates the whole original GA-SSW implementation nor settles universal optimizer superiority.
