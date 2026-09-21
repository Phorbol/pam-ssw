# Native LS cycle state and response eligibility: bounded static closure

2026-09-10. This reads the already archived `ls-bond-info-init.asm` and
`ls-ssw-move.asm` under `research/ga_ssw/evidence/native-ls-response/`. No native
main, protection change, PES call, or scientific experiment was run.

## The default recovered settings disable the periodic branch

At initialization, 0x6c6aef–0x6c6b2f calculates

```
x = float64(softmodecycle) * softratio
nsoftstep = trunc(x + (0.5 if x >= 0 else -0.5))
```

The two original constant pages 0x4a4c150 and 0x4a4c190 contain +0.5/-0.5.
This is Fortran-style nearest-integer rounding, not Python banker's rounding.
The result is stored at `pot_bond_var_def_mp_nsoftstep_`, 0x5520438.

The previously read static settings cycle=100 and ratio=1.100000023841858 give
**nsoftstep=110**. Since periodic operation requires **0<nsoftstep<cycle**, those
explicit settings skip the periodic branch altogether. This closes an important
scope question: faithfully implementing their normal path does not require
inventing a periodic restore policy. It does not prove that input parsing or
some other unexamined runtime write never changes the settings.

## Adaptive branch state transitions

The flag `para+0x178` is `lselfadapt` in existing member-offset evidence. With
LS active and step nonzero, the adaptive branch does the following:

1. Step<=npreselfadapt, nsoftstep<=0, or nsoftstep>=cycle goes directly to normal
   update eligibility (0x6c764f–0x6c767a).
2. Otherwise set phase=(step-npreselfadapt)%cycle (0x6c7680–0x6c7686).
3. If phase==nsoftstep, copy `bond_ener_list` (descriptor 0x55206a0) to
   `bond_ener_list_note` (0x5520640), then zero the live table. Copy the saved
   old bond count into its note slot, and truncate saved response to a signed
   32-bit integer note (0x6c7691–0x6c7ba4).
4. Else if phase==0, restore the note table to the live table, restore the old
   bond count, convert the integer response note back to double, and store it
   as the live response (0x6c7ba9–0x6c7eb8).
5. After either operation, or no phase transition, apply the normal update
   predicate: step%frequency==0 OR step<=npreselfadapt (0x6c7ebc–0x6c7ed1).
   Thus save/restore happens BEFORE that call's possible normal update.

Named scalar state is:

| State | Address |
|---|---:|
| live response, biasperatom_save | 0x1d103350 |
| previous bond count, BONDNUM_SAVE | 0x7916158 |
| note bond count, BONDNUM_SAVE_NOTE | 0x791615c |
| integer response note, BIASPERATOM_SAVE_NOTE | 0x7916160 |

The normal update recomputes current bond count and finally writes it to
BONDNUM_SAVE at 0x6c85cb–0x6c85d1. Zero live B during the off interval remains
zero under the multiplicative normal update. Integer conversion is truncation
of the saved response; it is not a deliberate noise filter inferred from
physical reasoning. Nonfinite/out-of-range conversion cases were not qualified.

A separate **nonadaptive/custom-file** path begins at 0x6c85dc. It tests the
same 0<nsoftstep<cycle condition but uses phase=step%cycle, with no presteps
subtraction. It copies/zeros/restores the table and does not traverse the adaptive
response update. This path must not be conflated with lselfadapt=True.
The allocation/error subbranches have not been executed in this audit.

## Prequench response is recorded even on iteration-limit exit

The soft prequench saves true E_before when STEP_OPT_SOFTPES==0
(0x5bd6fa–0x5bd717). At the analyzed exit (0x5bf699–0x5bf6ae), the alternatives
are `ftol > computed_force_measure` OR `step_opt_softpes >= optsoftmax`.
DWARF member-offset evidence names ftol at para+0x2db28 and optsoftmax at
para+0x2dad0. The scalar reduction feeding that exit uses absolute force
components and a maximum (0x5bdc2c–0x5bdc99); it should not be equated to ASE's
per-atom vector-norm fmax without accounting for that norm difference.

Both exit alternatives write `1000*(E_after-E_before)/N` into the same live
response (0x5bf6b4–0x5bf6f4). Therefore a stored native response is **not proof
of converged prequench**. The displayed bond_info_init normal update predicates
contain step/frequency/cycle checks, with no local prequench-converged test.
This supports allowing an explicitly recorded iteration-limit response in a
release-reference arithmetic controller; it does not qualify numerical backend
errors or prove the entire caller invokes the update after every failure.

## Remaining integration boundary

The static adaptive save/zero/restore state is now substantially recovered, and
its absence under the stated static ratio is explicit. An independent pure `NativeLSCycleState` now implements these known transitions
(see `native-ls-python-component.md`), with caller state explicitly supplied.
Before claiming native full-walk lifecycle parity, the narrow missing checks are:

- isolate the allocated-table scheduler path with stateful calls through save,
  off-period and restore; distinguish descriptor allocation errors from normal
  transitions and verify the full Q reduction in the same chain;
- establish the actual caller's step numbering, initialization of the response
  before the first eligible update, and association with the current accepted
  seed/neighbor count;
- trace numerical optimizer/backend failure exits separately from optsoftmax,
  rather than importing the iteration-limit eligibility into arbitrary failure.

The existing `native_ls.py` now provides arithmetic and explicit cycle state,
not an implicitly completed native full-walk lifecycle. The recovered default ratio gives
a defensible limited normal-path integration target while the remaining stateful
oracle/caller checks are performed.
