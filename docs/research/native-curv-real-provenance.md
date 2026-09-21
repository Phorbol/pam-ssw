# Native curv_real: recovered upstream force-difference producer

This follow-up closes the curvature producer, its zero-step gate, and LS addition to its current endpoint force. It also locates the stage entrance snapshots. Fresh coordinate/force pairing after every native prequench exit remains unverified; see the final follow-up section for the updated scope. It changes the parity contract: native curv_real is separately produced before rotation dispatch, rather than simply reconstructed inside rotate_dimer by subtracting its rank-one rotation bias. No production kernel was changed.

## Evidence and formula

Uploaded ELF SHA256 `bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704`; function `ssw_fixlat_mp_soften_mode0_` at `0x5c3880`. Archived DWARF names identify object `+0x1788=n0`, `+0x1a60=tf0`; structure force array is `+0x1d0=fa`. Parameter `para+0x2db20=dr`; `control+0x4=rotstep`, `control+0x40=curv_real`.

At `0x5c3af7–0x5c3df0`, the routine contracts the two arrays and current direction, honoring their independent Fortran strides/lower bounds. Scalar arithmetic `0x5c3dbd–0x5c3dcf` is load tf0, subtract fa, multiply n0, accumulate. Vectorized arithmetic has the same sign and operands. Then:

```
kappa = sum((tf0 - fa) * n0) / para.dr
if control.rotstep == 0:
    kappa = 0
object[0x18b0] = kappa
control.curv_real = kappa
```

The division is `0x5c3e02`; the forced zero is `0x5c3e0c–0x5c3e15`; stores are `0x5c3e1b` and `0x5c3e24`. This is a force-difference projection, not an energy-based curvature estimator. Its sign agrees with positive curvature **if** tf0 is the force at a reference point and fa the force displaced by `+dr*n0`. That reference/displacement interpretation is conditional here: producer arithmetic alone does not establish both arrays' lifetime or objective.

Only after both stores does `soften_mode0` dispatch through the biased/unbiased rotation slots at `0x5c3eca` / `0x5c3ed9`. In `biasedrot`, add_rotation_bias is called at `0x5c4c45` and cbd_rotation at `0x5c4cea`, passing the already existing `object+0x18b0`. In `rotate_dimer`, the corresponding passed pointer (`rbp+0x48`) is read for output and the CBD_PreRot negative-curvature stop; its separate computed curvature (`rbp+0x40`) is written from the biased force difference. Therefore the two curvature values have distinct data-flow roles.

A first rotation's forced zero is an algorithmic state convention. It must not be reported as a measured zero eigenvalue of the physical or softened PES.

## Original-instruction check

`research/ga_ssw/probe_curv_real_prefix.py` executes ELF bytes `0x5c3af7` through, but excluding, `0x5c3e29` in Unicorn. There are no function hooks, no force oracle, no native initialization or license/protection execution. It supplies synthetic tf0/fa/n0 arrays, independently sets their descriptors, and stops before dispatch.

Twelve cases passed: array dimensions 3x2 and 12x2 exercise scalar and vectorized loops; rotstep 0/1/4; positive and negative force-difference curvatures. Both object and control outputs agree with the formula within 1e-12; first-step outputs are exactly zero. Full arrays and measured outputs are in `research/ga_ssw/evidence/native-curv-real-prefix/result.json`.

```
PYTHONPATH=/tmp/pam-ssw-unicorn-probe:. python -m research.ga_ssw.probe_curv_real_prefix --elf /home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp --output research/ga_ssw/evidence/native-curv-real-prefix/result.json
```

The first attempt with mace_env lacked Unicorn; the existing isolated `/tmp/pam-ssw-unicorn-probe` installation was reused. No dependency installation or physical PES calls occurred. The excerpt is `native-curvature-evidence/soften-curv-real.asm`; regenerate using objdump with start `0x5c3af7`, stop `0x5c3edf`.

## Consequence and remaining boundary

The independent conservative implementation's analytic removal of the rank-one rotation bias is still a valid estimator of curvature on its declared rotation surface. It is **not yet native curv_real parity**, because native also maintains an independently populated force snapshot and a zero-step gate. Preserve explicit curvature_scope metadata; do not silently replace this estimate with the native field name or copy zero-step behavior into a scientific diagnostic.

The initial arithmetic-block evidence alone could not establish whether tf0 and fa include LS, old Gaussian terms, internal LJ, constraints, or different combinations. The final follow-up section now closes endpoint LS inclusion. tf0 is reused as Gaussian scratch elsewhere, so its name is not evidence that it always means bare physical force. The next minimal closure is to identify the write to tf0 immediately before the same soften_mode0 call and the ordering of LS addition to fa on that path, ideally by a two-evaluation synthetic force callback and two explicit LS states. That is a separate caller-level task, not answered by this block oracle. No new claim about LS scientific efficacy, native end-to-end parity, or PES stability is made.

## Sampling geometry and timing: narrowed by reverse communication

The existing DWARF mapping of `cbd_rotation` is
`rotstep,na,cell,r0,f0,r1,f1,n,n1,ffix,lvcell,fixlat,curv,curv_real,fact,dr,...`.
On the unbiased caller path `0x5c4080–0x5c40ca`, f0 is object.tf0, r0 is object.tstr0 (`+0x1a00`), r1 is object.xa (`+0x170`), f1 is object.fa, and n is object.n0 (`+0x1788`). The wrapper always writes **r1 = r0 + dr*n** on return (`0x6e5671–0x6e5680`, scalar tail `0x6e57a3–0x6e57b5`). Its caller must evaluate that requested point before re-entry; these routines contain no PES call.

Consequently, on a correctly fulfilled reverse-communication cycle with a noninitial rotstep, the intended sampling geometry is one-sided:

```
incoming direction n_old
r1_old = r0 + dr*n_old   # requested by the preceding rotation invocation
kappa_in = dot(f0 - f1_old, n_old)/dr
store kappa_in in object and control
possibly modify f1_old by the negative rank-one rotation bias
rotate using the supplied forces, potentially replacing n_old by n_new
write r1_new = r0 + dr*n_new for the next force request
```

The curvature is therefore the **current incoming request's direction**, before the current rotate update. It is not necessarily curvature of the direction that will be returned by that update. This is iteration timing inside a rotation stage, not evidence of a previous climbing-stage curvature. In particular, if rotation terminates with a direction rollback/change, reuse of kappa_in as curvature of the returned direction requires a separate check.

The Python helper's `lambda_biased + w*(anchor dot n)^2` and native kappa_in are not intrinsically different physical objectives: for the same point, incoming direction, force snapshots, one-sided step, and rotation-bias anchor, the rank-one term can be removed algebraically with no extra PES call. They can differ because of direction timing, finite-difference convention (e.g. a central HVP), or force-surface scope. The separate upstream native scalar does **not** justify adding a second PES evaluation solely for compatibility. Nor does its name prove a bare-PES rather than LS-modified curvature.

Remaining minimal slice: find the last writes to tstr0/tf0 when entering this rotation stage, and the force callback from the previous `r1 = r0 + dr*n` request through LS/constraint updates to the next soften_mode0 entry. The current block plus the wrapper close the intended locations and timing; they do not prove that the caller supplies two forces from the same objective. No further whole-binary scan or long calculation was performed.

## Follow-up: LS addition on the actual CBD force-consumer path is closed

The `status == 'CBD'` branch of `ssw_fixlat::ssw_move` is selected at `0x5bdcc8–0x5bdce9`. The string at `0x4a42980` is literally `CBD`. It tests `bond_info_mp_l_softmode_` at `0x7916168` (`0x5bdcef–0x5bdcf9`). When true, it calls `pot_bond_add` at `0x5bdedd` with **current xa and fa** (xa from `object+0x170`, fa from `object+0x1d0`), plus current cell, stress and energy. It then calls the table's `+0x1b0` slot at `0x5bdf03`.

The actual static method table at `0x53ca680` resolves:

| Slot | Target |
|---|---|
| +0x188 | 0x5c0ac0 set_status |
| +0x1a0 | 0x5c0100 get_random_mode0 |
| +0x1b0 | 0x5c3880 soften_mode0 |
| +0x1b8 | 0x5c3fa0 unbiasedrot |
| +0x1c0 | 0x5c4ae0 biasedrot |

Thus the endpoint-side LS inclusion is no longer speculative: **the force consumed by native curv_real has already received LS when that flag is enabled**. `soften_mode0` does not subtract LS before taking the dot product. The subsequent add_rotation_bias is later still. No addgaussian call occurs in this CBD branch between the LS addition and curvature production. Gaussian scratch reuse in another state is not evidence of Gaussian addition in this branch.

### Stage entrance snapshot

`get_random_mode0` calls `set_status('CBD')` at `0x5c0706`. On that exact branch, `set_status`:

1. resets `control.rotstep=0` at `0x5c0c94`;
2. copies the current coordinate array xa into tstr0 (`0x5c0cae–0x5c0f4b`, destination descriptor +0x1a00, source +0x170);
3. copies current fa into tf0 (`0x5c0f76–0x5c1224`, destination +0x1a60, source +0x1d0);
4. writes `CBD_PreRot`, then invokes unbiasedrot at `0x5c1330`.

These are snapshots of the **rotation stage entrance**, not of the previous Gaussian's force scratch. The snapshot copy neither adds nor removes LS or Gaussians. They persist as base point/force while reverse communication requests displaced endpoint evaluations. The main NewStart path also applies LS (`0x5bd145`) before entering the soft prequench state; the soft-prequench branch applies LS at `0x5bd920`, invokes its optimizer, and on the examined exit calls get_random_mode0 at `0x5bfab1`.

**Remaining precision boundary:** the last sentence traces entry to and exit from the prequench, but this follow-up has not emulated the prequench optimizer's in-place coordinate/force behavior or every entry into get_random_mode0. It therefore does not certify every tf0 snapshot as a freshly paired softened force at tstr0 after every exit. Full backend/update_forcepara constraints, internal LJ and fault/limit paths also remain outside this slice. The now-proven statement is endpoint LS inclusion and no LS subtraction in this curvature producer, not an unconditional claim of a numerically exact Hessian of bare V or even of V+P on all native paths.

### Consequence for the independent implementation

For a consistent base snapshot on the same frozen LS surface, the formula is
`[F_(V+P)(r0)-F_(V+P)(r0+dr*n)] dot n / dr`, i.e. a finite directional curvature of **V+P**, excluding the later rank-one rotation bias. This is the intended counterpart of Python's explicit **LS-modified** curvature scope after analytic rank-one removal, subject to the timing and finite-difference distinctions above. Calling it physical-PES curvature or subtracting the LS Hessian to imitate the native name would be unjustified by the observed consumer path.

The minimal unresolved piece is now smaller: verify that the prequench exit's xa/fa pair used by get_random_mode0 is fresh and paired, and enumerate alternate stage-entrance snapshots. No second physical oracle request or compatibility option is justified by this audit.

The two new static excerpts are `native-curvature-evidence/cbd-ls-dispatch.asm` and `cbd-entry-snapshots.asm`. All follow-up work was static; previous 12-case arithmetic artifact remains unchanged and does not claim to execute these callers.
