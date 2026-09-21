# Native dimer rotation: recovered retry, angular cap and history boundary

Date: 2026-09-09. Static analysis only; no ELF execution, PES run, or scientific validation. This supplements `analysis/native-rotation-spec.md` in the external upload workspace. Existing Python Ritz rotation remains a different numerical solver.

Evidence root: `/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/analysis/`; primary files `kernel-rotate_dimer-lines.asm`, `brions4-lines.asm`, `brzero4-lines.asm`. Addresses below refer to the uploaded `GA-SSW_program/lasp` ELF, not a portable ABI.

## 1. FACT1 first-rotation retry is now resolved

Let `t = F_endpoint - F_base + ((F_base-F_endpoint)·n) n` be the tangent force already recovered. At source line 716 the current evaluated direction is copied to `n00` (0x6e5e1e–0x6e5e33). `n00` is therefore the direction at the start of this rotation call, not necessarily the original random anchor of the whole walk.

At first rotation FACT1 takes the supplied FACT (0x6e5e76–0x6e5e86); later rotations reuse the saved scalar. The internal counter starts at zero (0x6e5e96). Each internal attempt does the following:

1. Reconstruct `r = r0 + dr*n` (scalar 0x6e5fbd–0x6e5fca).
2. Multiply the force workspace by FACT1 (0x6e60a0–0x6e60a8).
3. Increment the internal counter (0x6e60b5), set a control logical true, and call BRIONS4 with outer `rotnum - 1` (0x6e635a–0x6e639a), or use the explicitly selected `rot_sd` branch.
4. Reconstruct the unnormalized candidate `n = (r-r0)/dr` (0x6e6405–0x6e6415), then compute its Euclidean norm.
5. Retry **only if** `rotnum == 1`, `norm(n) > 1.02`, and internal attempt count `< 15` (0x6e66e7–0x6e6705).
6. On retry divide the force workspace by old FACT1 (0x6e677c–0x6e6785), multiply FACT1 by 0.8 and save it (0x6e6827–0x6e682f), restore `n <- n00` (0x6e6893–0x6e68af), and repeat without a new PES call.

Constants read directly with GDB `x/gf` (no process run): 0x4a4cd30 = 1.02; 0x4a4cd18 = 0.8. There are at most 15 internal attempts, hence at most 14 reductions for an ordinary finite nonempty input. They are **not 15 extra force evaluations**. Later outer rotations do not enter this FACT1 adaptation. This is neither a line search with reevaluated PES nor a blanket adaptive dimer separation.

The retry modifies saved FACT1 while BRIONS4 still receives the original FACT argument in r8 (0x6e6396). A port must not silently substitute FACT1 for FACT in that argument.

## 2. Angular cap recovered: 40 degrees

After the candidate coordinate update and first constraint application, `n` is normalized (source 773; 0x6e6e05–0x6e6e36). The routine forms

`theta = acos(dot(n, n00)) * 180 / pi`

(0x6e7083–0x6e70a8) and compares to 40 degrees (0x6e70b0). Constants: 0x4a4cca0 = 180; 0x4a4cd38 = pi; 0x4a4cd40 = 40.

When theta exceeds 40 degrees it computes

```
u = n - dot(n, n00)*n00
n = n00 + 0.83909963117727993 * u / norm(u)
```

Projection subtraction: 0x6e727e–0x6e7291. Tangent norm: 0x6e74e5–0x6e74ff. Reconstruction: 0x6e7728–0x6e773c. The coefficient is stored at 0x4a4cd58 and equals tan(40 degrees). Constraints are applied again (fixed-cell integer mask at 0x6e77a0–0x6e77e8; optional `setconstraints` at 0x6e77f8–0x6e780e), then direction is renormalized (source 791; scalar 0x6e7a4f–0x6e7a6a).

For unconstrained finite unit vectors this construction caps the angle at 40 degrees after normalization. With subsequent constraint projection the geometric interpretation needs separate qualification. No safe zero-tangent handling or acos clipping has been recovered; an independent API must record any explicit domain protection as a divergence from literal native arithmetic.

## 3. Termination exception resolved via Fortran runtime

The previous spec left `for_cpstr` equality semantics unresolved. Static disassembly of `for_cpstr` (0x49a8420) and its tables now resolves this without guessing:

- Caller puts 2 in r8 at 0x6e7cd1.
- Unequal compare mode-2 target, table 0x4ba5ba0, is 0x49a84da: return zero.
- Equal compare table byte at 0x4ba5bc2 is 1.
- Runtime handles trailing spaces as Fortran blank padding (0x49a84eb–0x49a855e).
- Compared literal at 0x4a45fe0 is `CBD_PreRot`, length 10.

Therefore finite ordinary inputs have the exact predicate:

```
stop = reported_rot_force < ftol or rotnum > rotmax
if fortran_blank_padded_equal(infor, 'CBD_PreRot') and curv_real < -1e-6:
    stop = False
if stop:
    n = n00.copy()
rotnum += 1
```

The threshold comes from 0x4a4cd50. Clearing the flag is 0x6e7d03–0x6e7d1c, rollback 0x6e7d42–0x6e7d49, increment 0x6e7d4e. **This exception can override the rotation-budget stop as well as the residual stop.** Preserve a separate external execution budget if providing a bounded ASE API, and report that as an external interruption, not native convergence.

The reported quantity remains `norm(f2_workspace)/FACT1 * 10` (0x6e7b78–0x6e7b84). It is not an HVP residual in eV/Å² unless independently transformed with the actual displacement and workspace history. Budget stop and force-tolerance stop must remain separate diagnostic outcomes.

## 4. BRIONS4 history boundary is clearer; full Broyden update remains open

BRIONS4 (0x6f6440) allocates saved flattened arrays X, F and G0, with dimensions checked against 3*NA (0x6f647a–0x6f6489). Allocation initialization sets saved STEP=1 and ITER=0 (0x6f69fd–0x6f6a13; constant 0x4a4d0d0=1). Its sixth positional argument is the outer `rotnum-1` value passed by rotate_dimer.

At 0x6f6a2d–0x6f6a3c:

```
ITER = (0 if input_step == 0 else saved_ITER) + 1
```

Then BRIONS4 passes `ITER == 1` as the fifth argument to BRZERO4 (0x6f6b3c–0x6f6b8d). This means every retry during outer rotation 1 requests BRZERO4 initialization again: retry is not intended to accumulate additional Broyden secant observations from an unchanged endpoint force. Later outer rotations retain an internal iteration count rather than simply assigning it from outer step number.

Actual update/history matrices reside in BRZERO4 (0x6f6c00), not only BRIONS4. Merely porting the wrapper allocations and counter is insufficient. The existing `brzero4-lines.asm` is the next source for complete secant signs, memory storage/reset, normalization, and step selection. This follow-up does **not** claim those equations have been reconstructed.

## 5. Python port contract and remaining blockers

The rotation state should explicitly own FACT1, evaluated n00, BRIONS4/BRZERO4 history, outer rotation counter, and initial constraint state. Preserve raw endpoint force separately from destructive tangent-force workspace. A reverse-communication response must expose each requested geometry, so FACT1 retries are visible as algebraic retries with zero oracle calls.

The recovered first-rotation adaptation, 40-degree cap and termination predicate are now precise enough for individually named source-level functions. An equivalent native driver remains blocked on BRZERO4 and the exact constraint/curv_real production semantics. No ASE BFGS or generic inverse-Hessian substitute should be called native Broyden parity. Next native oracle should replay every supplied force and state transition, including first-rotation retries and terminal n00 rollback; an end-direction comparison alone misses these differences.

## 6. Paper intent, this executable, and PAM are three distinct specifications

Source cross-check: Shang and Liu, *Constrained Broyden Dimer Method with Bias Potential for Exploring Potential Energy Surface of Multistep Reaction Process*, JCTC **8**, 2215–2222 (2012), DOI [10.1021/ct300250h](https://doi.org/10.1021/ct300250h), section 2.1, equations 5–8; local author PDF/text `literature/65.pdf` and `65.txt`. The paper motivates biased rotation because unrestricted softening near positive-curvature basin regions can drift toward translation/rotation and lose the intended reaction direction. Its rotational force is written with a factor 2, whereas the inspected ELF tangent workspace above has no initial factor 2: overall rescaling must be derived through the complete update, not silently declared identical.

Shang and Liu, *Stochastic Surface Walking Method for Structure Prediction and Pathway Searching*, JCTC **9**, 1838–1845 (2013), DOI [10.1021/ct301010b](https://doi.org/10.1021/ct301010b), equations 3–7 and overall algorithm; local `literature/74.pdf`/`74.txt`. This supports the separation of initial direction, direction-preserving biased rotation, and subsequent modified-PES climb. It does not establish that the particular executable's 1.02/0.8/15 retry constants, 40-degree cap or string-dependent termination exception are universal mathematical requirements. In this report those constants are **executable facts only**, not paper-derived defaults or newly optimized recommendations.

A fresh web search also located the author-hosted LASP methods paper, [LASP author PDF](https://www.lasphub.com/publication/132.pdf), Figure 2: its indexed description explicitly separates NewStart, CBD, moveds, Climb and Allopt states. Direct PDF fetch returned HTTP 502 this turn, and direct BP-CBD author PDF fetch timed out; consequently no new full-text claim is based on those failed fetches. The already archived BP-CBD and SSW full texts were used for the above section-level cross-check.

Live PAM comparison: `/home/gengjianrui/bin/pam-ssw`, HEAD `c798ff7`, `pamssw/walker.py:633` onward. PAM `SoftModeOracle.choose_direction` generates a finite pool, computes each candidate's central-difference directional curvature using two displaced evaluations (`_directional_curvature`, line 743), and picks the largest weighted score. `DirectionScorer` (line 355) combines quadratic energy cost, damage risk, continuity, anchor proximity, optional history push and optional archive novelty. This is **candidate scoring**, not iterative constrained Broyden dimer rotation. A rank-one rotation bias, a direction-angle cap, and a weighted proximity score can share a direction-preservation motivation but are different operators; their trajectories and cost cannot be identified by similar terminology.

The standalone reference's Krylov/Ritz solve is a third operator. Therefore the next informative comparison is native recovered dimer versus standalone Ritz versus PAM finite-pool oracle, holding the true calculator, geometries, initial direction, intended bias surface, displacement accuracy and total E/F budget fixed. Record actual rotational residual, retained-anchor angle, evaluation count and failure causes first; only full basin-to-basin real-system runs can test whether these numerical differences improve search. Do not add the native constants as extra PAM scoring knobs to claim closer reproduction.

## 7. Independent control primitives (subsequent implementation)

`pamssw/standalone/native_rotation_control.py` now implements the unconstrained
fixed-cell angular cap, FACT1 retry predicate and termination/rollback transition.
Nine targeted tests cover strict boundaries, tangent-plane preservation, first-
rotation-only retries, exhausted retries, and the pre-rotation budget override.
They test geometry and recovered predicates, not original execution or efficacy.
The full original-instruction rotation oracle and complete BRZERO4 remain separate
work. The primitives deliberately do not call a calculator or create fictitious
secant history during an algebraic retry. `reported_force` must be the original
workspace norm, not the standalone Ritz residual. Integer counters and invalid
zero directions fail explicitly; no invented native fallback is supplied.

ELF angle denominator at 0x4a4cd38 was read again and is exactly the host double
representation of pi (3.141592653589793). This differs from the nearby height
function's denominator and is not silently shared between those functions.

A fresh author-index lookup located the original CBD 2010 paper as item 50,
https://www.lasphub.com/publication/50.pdf (DOI 10.1021/ct9005147). Web PDF fetch
timed out and bounded curl failed TLS; no new full-text formula is inferred from
that unavailable file. The archived 2012/2013 full texts remain the paper sources.

## 8. Caller stop flag traced through the wrapper (2026-09-17)

The fixed-cell caller's `LCONVERGE` is a stage-stop flag, not solely a
mathematical residual-convergence certificate. A fresh static pointer trace in
the same archived ELF closes the wrapper boundary:

- `ssw_fixlat_mp_unbiasedrot_`: address `0x5c4051` obtains global flag
  `0x78eaea8`; `0x5c405f` passes its pointer.
- `newssw_basics_mp_cbd_rotation_`: `0x6e559e` loads the argument pointer,
  `0x6e55b3` clears the flag, and **`0x6e55cd`** forwards that pointer to
  `rotate_dimer` (call at `0x6e55f6`).
- `rotate_dimer`: `0x6e7cc1` loads the output pointer; the rotation-count and
  force comparisons at `0x6e7cd7` and `0x6e7ce2` are ORed at `0x6e7cf4`, then
  written through that pointer at `0x6e7cf7`. The negative-curvature PreRot
  exception still clears it at `0x6e7d1c`.

Thus reaching the native rotation limit can complete this stage without meeting
its force tolerance. The new Python `force_or_budget` option adopts that
separation of responsibilities, with an explicitly different external HVP
budget. It does not reproduce the native iteration counter, tolerance units,
PreRot exception, or full CBD trajectory. Python retains `converged=False`
and the residual when releasing a budget-limited direction. This trace corrects
an intermediate review hypothesis that the limit flag might not reach the caller;
that hypothesis is not supported by the pointer chain.
