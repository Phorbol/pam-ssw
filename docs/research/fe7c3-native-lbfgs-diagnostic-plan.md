# Fe7C3 frozen VC objective: native LBFGS diagnostic

2026-09-11. This plan precedes any new native/MACE numerical run. The question
is whether the isolated released LBFGS configuration can solve the four
previously selected 246-coordinate Fe7C3 LS biased objectives within the same
request and accepted-iteration limits. The existing Safe-total history10/500
results are the comparator, not newly retuned controls.

Selection is exactly the four cases in
`fe7c3-ls-frozen-quench-history/frozen-quench-diagnosis`: first outer attempt
with a completed 300-step biased-quench maxiter, all-pair/filtered LS and
seeds7/101. Use the original fixed-cell preparation source, not select whichever
of the later joint-preparation failures looks easier. Preserve chart reference,
strain length, pressure, complete frozen LS pairs/strengths and all Gaussian
centers/directions/widths/weights. Start at the last Gaussian center plus its
width times direction, identically to the archived Safe-total experiment.

Physical E/F/stress comes from the same float64 MACE-OMAT-0-small model and
source structures. The callback computes the same consistent physical H+LS+B
objective and projected joint gradient. No BFGSDRIVER force scaling, native
Gaussian implementation, native main or protection path is executed.

The already documented isolated release profile is unchanged: history400,
scalar STPMAX0.5, GTOL900, STPMIN1e-4, FTOL1e-4, EPS1e-5, XTOL1e-16,
gradient scale1. These are release/profile observations, not recommended
mathematical defaults. GTOL900 is outside the ordinary strong-Wolfe condition.
The native scalar step limit and Safe-total's joint-block step norm limit0.2
are different quantities. This is a **configuration comparison**, not a clean
line-search-only ablation and not native full-SSW parity.

For each case: at most329 optimizer requests,300 accepted iterations, and one
reserved independent final biased-objective E/F/stress check. Four cases cost
at most1320 EFS; one V100 and15 minutes bound the diagnostic. The accepted
iterate hook must verify that native X matches the E/G evaluation it accepts.
At request exhaustion return the last accepted iterate, not an unevaluated
native proposal. Qualify using the common norm
`max(max_i |g_atom,i|, |g_cell|_2) <= 0.001`; native IFLAG=0 alone does not
qualify. Record emulator errors, unqualified termination and failed fresh checks
separately. All requested cases remain in the denominator. Report accepted
steps, rejected/trial requests, total EFS and wall time; emulation wall time
cannot establish native machine speedup.

Before running MACE, check flat-dimensional ABI/memory allocation and accepted
state with isolated numerical fixtures; these test the harness only. Archive
source/input/model/ELF hashes and compare reconstructed initial E/g and q
against the existing Safe-total ledger. The previous failed-point fresh checks
already validate source reconstruction and remain separately charged in the old
experiment. Do not charge them again implicitly or claim their cost is zero.

Interpretation: convergence can justify further investigation of the numerical
backend, but no biased stationary point is a physical minimum or global-search
success. No optimizer default changes, new heuristics, broader case selection,
or budget extensions follow automatically from this diagnostic.
