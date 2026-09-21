# VC schedule and biased-stop correction

2026-09-11, following the user's two corrections. The priority is the SSW/VC
algorithm's actual stage semantics, not making every inner artificial-PES
optimization satisfy the final physical convergence contract.

## What was wrong with the preceding interpretation

`vc_reference.py` immediately exits an attempt when `relaxed.converged` is
false, then only allows true quenching for `gaussian_limit` or
`lower_true_enthalpy`. Consequently the preceding six-arm result of zero
landings was conditional on this strict gate. It did not test whether those
finite endpoints were useful starting points for an unbiased quench. Calling
them intrinsically unable to yield physical landings was not supported.

SciPy relative-energy stopping is a legitimate numerical termination rule.
It does not prove a small gradient, but a small biased gradient is not a
mathematical prerequisite for evaluating and quenching the same configuration
on the physical PES. The artificial landscape is a proposal mechanism, not
the final objective. The independent physical force/stress certificate remains
required for a returned candidate. Numerical stop, permission to quench,
certified physical stationary point, distinct basin, and useful global-search
outcome are five different statements.

Native numeric stop handling also must not be confused with a physical success
flag. Lack of complete native parity does not prevent testing an independently
specified release policy. No need to relax final physical tolerances follows
from either of these observations.

## Actual no-bias counterfactual

All ten paid attempts from the preceding comparison had a finite saved accepted
biased point. The two uncharged attempts were excluded. Where the final failure
occurred outside the biased optimizer, the last available accepted biased
point was selected and its stage/status recorded. No last trial was substituted.

From each point: discard Gaussian terms, minimize real E+pV using Safe-total
with the existing strain scale, memory10, maxiter300, maxstep0.2, pressure0,
fmax0.001 eV/Angstrom and full stress tolerance0.0001 eV/Angstrom^3. Budget:
400 EFS/120seconds per point, including two independent checks, one each for
the origin's qualified initial structure and the resulting terminal.

Job1259727 completed on one V100 in27seconds. All10 quenches converged and
their independent physical certificates passed. Total835 EFS includes815
start/quench requests and20fresh checks; CPU preparation added8EMT requests.
The complete source/origins/plan/results are in
`research/ga_ssw/fe7c3-no-bias-counterfactual/`; all EFS ledgers reconcile.

| Origin solver | Terminals | Extra EFS including fresh | Terminal minus initial energy |
| --- | ---: | ---: | --- |
| Safe-total | 3 | 393 | +11.02 to+13.44eV |
| ASE LBFGSLineSearch | 3 | 312 | +9.76 to+11.42eV |
| SciPy L-BFGS-B | 4 | 130 | Three within2.4e-6eV; one+2.94151eV |

Thus the strict gate discarded physically quenchable configurations. None of
these ten results improves the initial energy. Three SciPy terminals have
very small geometry differences from initial; seven other structures require
identity/stability qualification. Root subsequently ran the existing pymatgen
matcher in the isolated mace_env: all three declared tolerance profiles match
three SciPy terminals to initial and do not match the seven other terminals to
initial or to each other. The earlier missing-package claim was an interpreter
selection error, not a required user intervention. These approximate identities
are not exact basin proofs. Continuous same-label nonaffine displacements and
cell changes are recorded separately in `identity-report.json`.
No new terminal Hessian was calculated, so a certificate is not a proof of
strict local-minimum stability. The new diagnostic cost raises the registered
Fe7C3 cost to48449EFS, but cannot be retroactively included inside the old
2000-EFS-per-arm budget or advertised as complete-search efficiency.

## Correct VC reproduction priority

The 2014 paper (§2.2–2.3, journal pp17847–17849) specifies successive blocks:

1. CBD lattice displacement.
2. Fixed-lattice atom relaxation, at most25steps in the reported example.
3. Repeat the prescribed cell cycles.
4. On selected outer steps, fixed-lattice atomic SSW.
5. Remove bias and constraints, then relax atoms and cell together.

The reported H_cell=5 and lambda=2 are example settings, not universal optima.
The prose/flowchart phase and cycle-index discrepancies remain documented in
`vc2014-native-crosscheck.md`. The paper does not require full atom/cell force
and stress convergence after every intermediate operation.

`block_ssw.py` already implements an independent version of these blocks:
five cycles, atomic SSW every second outer move, at most25fixed-cell atom
steps, with intermediate maxiter permitted. It must be audited and validated,
not reimplemented from scratch. Atomic SSW inside that block still has the
strict biased-stop gate, a separate lifecycle issue. Its per-cycle cell-mode
resampling remains an unverified native-policy substitution.

The all-stage joint log-strain walker is an independent alternative. The
previous Fe7C3 optimizer campaign primarily tested that alternative; it did
not establish failure of the paper's VC-SSW. It was a prioritization error to
let this distinction disappear from the active investigation despite the
paper's block schedule already being documented.

For the later uploaded ELF, root located an actual `ratio_atomcell` consumer
in `get_random_mode0` (0x5e8112–0x5e8173): positive ratio tests signed remainder
of `nsswstep` against1 and sets `lcellmove`; the corresponding branches select
cell/noncell mode-pattern routines. `moveds` selects ds_cell or ds_atom from
that flag at0x5ece4f–0x5ece71. This proves mode scheduling, not that all inner
cell coordinates are frozen in the noncell mode. The downstream allowed-DOF
and inner optimization schedule remain to be traced. A 2014 paper schedule,
the later native mode schedule and our joint alternative must not be merged
into one unsupported claim.

Next work: finish the native frequency/allowed-DOF contract, review existing
block timing, and make intentional biased numerical stops eligible for a
budgeted physical quench in a separately recorded full-workflow comparison.
Evaluator failures, invalid geometry and exhausted hard budgets remain distinct
from ordinary numerical stopping. Do not add new heuristic thresholds, increase
caps, or change final physical tolerances to obtain passing counts.

## Implemented controlled release policy (2026-09-11)

`VCSSWConfig.bias_release` now exposes `strict` (unchanged control default)
versus experimental `numerical_stop`. The latter ends climbing and attempts
physical quench when the biased optimizer returns `maxiter` or a normal
`native_stop`, with finite accepted coordinates, energy and gradient and no
error. It adds no force/stress tolerance or adaptive parameter. Failed oracle
calls, request limits and abnormal numerical stops remain failures. The final
physical optimizer and certificate are unchanged; all calls still pass through
the same budget surface. Records preserve the inner stop reason and use outer
`biased_numerical_stop` only if the subsequent true quench certifies a landing.
The material-arm collector includes these landings, including MC rejections.

This tests whether treating the biased objective as a proposal rather than a
physical target can recover useful landings within the existing budget. It is
not yet a default recommendation: retain only if complete-workflow comparisons
show useful physical candidates per total EFS, rather than merely more quenches.
The earlier 835-EFS counterfactual motivated this test but is not its budgeted
performance result. Atomic/block release remains a separate unmodified path.

Regression evidence: real Cu/EMT through joint escape and true quench, controlled
inner maxiter/native_stop versus oracle/budget failure, unchanged independent
physical force/stress checks, and rejection of missing/nonfinite energies,
missing gradients and error-bearing results. The joint reference and mature
bridge regression files pass 20 tests in the isolated mace_env. These establish
lifecycle correctness, not general global-search efficacy.
