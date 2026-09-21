# VC `climb_convg` stage-stop counter contract

2026-09-11. Static extraction from the complete VC callee
`ssw_crystal_basic_mp_climb_convg_` (`0x5f4360`). No PES or main execution.

## Counters and strict comparisons

The function saves `control+0x08` (`climbstep`) through the control pointer
at `0x5f44c2–0x5f44e0`. It reads the trajectory counter `N` from
`object+0x1660` at `0x5f4555` and again at `0x5f4610`, saving the latter as
`[rbp-0x110]`. The stage split is therefore:

```text
if N == 1:
    step_over = (climbstep > para+0x2dd24)  # ngaus_relax_ini
else:
    step_over = (climbstep > para+0x2dd20)  # ngaus_relax
```

Both tests are signed `cmp` followed by `cmovg` (`0x5f48ec–0x5f48fc` and
`0x5f4b7c–0x5f4b8c`), so equality does not set `step_over`. The callee has
no read of `para+0x2dd28` (`ngaus_relax_half`); that parameter cannot be
included in this stage-stop formula from this function.

## Energy/force mask merged into the stage flag

The relevant scalar masks are combined before the counter branch:

1. `energy0=object+0x1b20` is copied to `[rbp-0x108]` at
   `0x5f44e9–0x5f44f0`; `tene0=object+0x1b28` is at `[rbp-0x38]`.
   At `0x5f463b–0x5f4690`, the comparison is
   `tene0 < energy0 - C0`, where `C0` is the rodata scalar at `0x4a46d48`.
   Its mask is ANDed with the `para+0x2db38` comparison and a cap/fix mask,
   then placed in `r13d` (`0x5f4695–0x5f46c2`).
2. At `0x5f477c–0x5f47ac`, `para+0x2dd38` (`e_maxlimit`) is compared with
   the stored `max(tene0-energy0, control+0x58)` and `para+0x2dd40`
   (`e_maxlimit_gm`) with the control scalar at `+0x60`; either true mask is
   ORed into `r13d`.
3. The force producer scans the generalized `sfa` array rooted at
   `object+0x4d8` and accumulates its maximum absolute component into
   `[rbp-0x130]` (`0x5f43d5–0x5f4462`, finalized at `0x5f448c`). This is a
   maximum absolute generalized component, not the largest per-atom vector
   norm. It is also written to `control+0x20` (`climb_maxf`) at `0x5f44e4`.
   The saved value is compared with `para+0x2dd30` (`climb_stopf`) as
   `Fmax < climb_stopf`. The non-debug path executes this at
   `0x5f48b6–0x5f48c7`; the debug/print path executes the duplicate predicate
   at `0x5f4863–0x5f4871` (`[rdi-0x10]` aliases `[rbp-0x180]`, and the mask
   is stored at `[rdi-0x8]` = `[rbp-0x178]`). Both are strict-less-than force
   stop tests and neither clips the force.
4. The indexed trajectory energy branch at `0x5f4970–0x5f49da` compares
   `tene0` with `trajectory[N-lower].+0x230 - C1` (`C1` at
   `0x4a46d30`). Its result is filtered by the stored threshold/fix masks
   and ORed into `r12d`.

The counter flag is then merged into `r12d`: for `N != 1`,
`0x5f48e9–0x5f4903` ORs `(climbstep > ngaus_relax)` with the earlier local
mask; for `N == 1`, `0x5f4b79–0x5f4b93` ORs
`(climbstep > ngaus_relax_ini)` with that mask. This is a bit-mask merge,
not an energy acceptance statement.

## `N`-dependent final stage stop

Only when `r12d & 1` is set (`0x5f4a6b`) does the callee apply the two
trajectory-counter gates:

```text
if para+0xf4 == N and lcellmove == false:
    r13 |= -1                         # 0x5f4a78–0x5f4a98
if para+0xf8 == N and lcellmove == true:
    r13 |= -1                         # 0x5f4a98–0x5f4ab4
```

The equality tests are `jne` fall-throughs, so these gates are exact equality
checks. The `lcellmove` polarity is opposite in the two cases by `cmove` vs
`cmovne`. Finally, `r13` bit 0 causes `r12` to become `-1` at
`0x5f4b4e–0x5f4b5e`; the function writes `r13` to control `+0x78` and `r12`
to control `+0x7c` at `0x5f4b62–0x5f4b66`.

The recovered counter choice and strict predicates were independently covered
by the 108-case bounded result `research/ga_ssw/evidence/native-vc-stop-levels.json`;
the force producer itself was not dynamically emulated in that result. This
closes the callee's portable stage-stop pseudocode and strict comparison
directions. It does not establish the runtime-effective values of the three
`ngaus_*` parameters after parsing, nor which outer caller interprets control
`+0x78/+0x7c` as release, next Gaussian, or Allopt.

Sources: `native-cell-reference-evidence/climb_convg.asm` and
`analysis/kernel-dwarf-member-offsets.txt`.

## Parameter-source boundary

The focused parser slice does not assign defaults: `readsswpara_` passes
`para+0x2dd20` to `readinput_mp_get_int_` at `0x688aaa–0x688aec` and again at
`0x688b18`, passes `para+0x2dd24` at `0x688ba3–0x688be7` (and an alternate
parser branch at `0x690244`), and passes `para+0x2dd28` at
`0x688bf3–0x688c5b`. These are input-key reads for
`SSW.Ngaus_relax`, `SSW.Ngaus_relax_ini`, and `SSW.Ngaus_relax_half`, not
field-specific fallback stores. A file-backed ELF read of the named
`<iii` fields at `para=0x53ed7a0`, `+0x2dd20=0x541b4c0`, gives the compiled
`.data` initial values `(25, 5, 10)`. This establishes compiled initialization,
but not the runtime-effective values after parser overrides or any later
caller writes; the absent-input retention path is outside this focused slice.
The 108-case `native-vc-stop-levels.json` validates the counter/equality
predicates with synthetic parameter values; it does not validate parameter
initialization, and it did not emulate the force producer.
