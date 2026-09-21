# VC mature baseline stopping contract

2026-09-11. Research adapters only; no production optimizer replacement.

## Problem and derivation

The joint chart has `3N + 6` coordinates. Its biased-quench certificate is
`G = max(max_i ||g_atom_i||_2, ||g_cell||_2)`. SciPy L-BFGS-B uses a component
infinity norm (without bounds, the projected gradient is the gradient). The
research ASE adapter packs coordinates into triplets, so its native force norm
is `A = max_j ||g_triplet_j||_2`: the six cell coordinates form two triplets.

Consequently `G <= sqrt(6) * ||g||_infinity` and `G <= sqrt(2) * A`.
Passing `gtol / sqrt(6)` to SciPy or `gtol / sqrt(2)` to ASE is a sufficient,
not necessary, condition for the common biased certificate. These factors
come from norm inequalities, not fitting Cu trajectories. Their conservatism
may incur additional requests and is part of the measured implementation cost.

This conversion only applies to the current joint biased call site. The true
quench uses a dimensionless physical certificate with threshold 1; that value
must never be passed as a native gradient tolerance. The bridge explicitly
passes the configured gradient tolerance there and checks Cartesian force and
pressure-residual stress independently afterwards. A native termination that
fails the common certificate remains `native_stop`, not a successful quench.
SciPy can also stop on its native relative-energy criterion before its native
gradient criterion is met; this remains visible and is not silently retried.

Primary references checked on 2026-09-11:
[SciPy stopping rules](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html),
[ASE implementation](https://docs.ase-lib.org/_modules/ase/optimize/lbfgs.html).
Online documentation can differ from the installed version: executed package
versions must be recorded for each experiment. The norm inequalities above
are project derivations for the explicit adapter packing, not claims from SSW papers.

## Real-system interface evidence

The first complete Cu4/EMT step, using equal numerical native thresholds,
terminated with common biased norms 0.00528479 (SciPy) and 0.00504180 (ASE),
above the requested 0.005. Both returned only their qualified initial state.
This exposed a precision-contract mismatch and is not evidence that Safe-total
has better global-search behavior. Artifact: `cu4-emt-lbfgs-baselines.json`.

With the sufficient norm conversion, all three methods produce one physically
certified landing after the initial quench in this single-step Cu4 check:

| Method | Search EFS | Independent fresh EFS | Total |
| --- | ---: | ---: | ---: |
| Safe-total | 62 | 2 | 64 |
| SciPy L-BFGS-B | 65 | 2 | 67 |
| ASE LBFGSLineSearch | 94 | 2 | 96 |

Artifact: `research/ga_ssw/evidence/cu4-emt-lbfgs-normbound-v2-corrected.json`.
The zero-PES correction separates the initial certificate from landing count;
it does not turn two qualified points into two new basins. Structural identity
has not been established. The v2 minimum summaries lack full coordinates and
force/stress arrays. Subsequent root review recovered initial coordinates from
the first `chart_reference` and the full landing from its stage record, without
PES calls; the earlier correction's blanket non-recoverability statement is
therefore superseded. Initial force/stress vectors are still absent. Recovery:
`cu4-emt-lbfgs-normbound-v2-geometry-recovery.json`. Same-label nonaffine RMS
displacements from initial are only 0.000443/0.000248/0.000196 Angstrom; these
small differences do not establish new basin discovery. Future runners now
serialize complete minima directly. This small real Cu system checks
the lifecycle, not difficult-landscape efficacy or optimizer ranking.

All exploratory Cu baseline costs are retained: 52 EFS for the two-pressure
pure-quench check, 122 for the first complete native-threshold runs, 225 for the
first norm-bound run lacking Safe fresh, and 227 for v2: 626 EFS total. No
additional PES work was used to correct counts or validate native instructions.

The comparison preserves each library's line search, initial Hessian and
native step restrictions. ASE's triplet cap and Safe's joint block cap are not
the same geometry, and SciPy has no identical cap. This is a comparison of
complete numerical implementations, not an isolated line-search ablation.

## Next falsifiable comparison

Use the already qualified Fe7C3-80 input, original non-LS joint settings and
seeds 7/101. Compare Safe-total, SciPy and ASE with memory 10, 300 iterations,
two SSW attempts per arm and the existing 2000-EFS budget including three
fresh checks. Preserve gtol 0.001 rather than tuning again. The question is
whether mature optimizers resolve the same complete biased-quench bottleneck
at comparable cost. Zero qualified new landings is a negative/censored result,
not a reason to add retries or increase caps. Native LASP full-driver parity
and cross-material effectiveness remain separate questions.
