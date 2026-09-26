# Native moveds: requested displacement versus saved width

Bounded static followup of uploaded `GA-SSW_program/lasp`, using `analysis/kernel-ssw_fixlat_mp_moveds_.asm` and `kernel-dwarf-member-offsets.txt`. No native main, protective-code changes, PES evaluations, or public kernel edits. This closes arithmetic on the straightforward Cartesian trial path, not every move/retry branch or its global initialization.

## Directly recovered arithmetic

`para+0x2db08` is DWARF `ds_atom`, loaded once at `0x5c4ed0` into local `ds_n = [rbp-0x308]`. `para+0x2db10` is `lrandom_ds`: its enabled branch modifies ds_n with a saved random multiplier and lower bound 0.1 Å (`0x5c4efc–5c4f18`; first-index random producer `0x5c977d`). With random disabled, ds_n remains ds_atom, except first Gaussian and run_type==5 (`0x5c97a6`), which applies max(0.1, 0.5 ds_n). Do not silently transfer this run_type-specific exception into the ordinary independent walker.

The direct Cartesian trial loop `0x5c6e83–5c71bd` reads trajectory structure `[ng]` through object+0x1668 (stride0x690), saved direction through object+0x16b0 (stride0x138), and writes object coordinates+0x170:

```
R_trial = R_center[ng] + ds_n * n_saved[ng]
```

The scalar multiply/add is at `0x5c7154/5c719a`; vectorized pair-equivalent loop `0x5c70ac–5c70ce`. There is **no direction max-atom or max-component divisor in this loop or on the inspected ds_n producer chain**. If input n_saved has unit total Cartesian norm, this is a total-norm displacement of ds_n. Input-direction normalization upstream of moveds remains a separate contract; do not claim this conditional as exhaustive all-mode proof.

Following successful move acceptance, `0x5c7863` onwards overwrites the persistent direction record (`object+0x16b0`, element stride documented in `native-gaussian-caller.md`) with Cartesian differences. The subtraction is explicit at `0x5c7b38–0x5c7b69`, followed by `n_normal` at `0x5c7c6a`. The inspected `n_normal` sums all `3N` Cartesian components, takes a square root, and divides each component in-place; it is neither max-atom nor per-atom normalization.

The later width accumulator runs at `0x5c7d72–0x5c805e` and writes the saved width array (`object+0x16f8`, lower bound `object+0x1738`) at `0x5c80c4`. Pointer tracing shows its operands are persistent records: selected coordinates from `object+0x170` (`0x5c7d7b`), center coordinates from `object+0x1668` records at `+0x170` (`0x5c7da4`), and the direction record from `object+0x16b0` (`0x5c7e26`). Thus the weighted difference is consistent with `delta_R · n_out` after the in-place `n_normal`; for a direct unclipped record with a positive displacement this gives `width = ||delta_R||_2`. This identity still requires the selected record/index correspondence and does not cover retries or alternate branches.

## Guards and limits recovered without expanding all fallbacks

After each direct trial, the function calls `present_tooshort` at `0x5c7449`, also checks `disp_perstep` (+0x2e1f0) and `bonddisp_perstep` (+0x2e1f8). A rejected trial multiplies ds_n by **0.95** at `0x5c749c`, retries while ds_n>=0.1 and trial counter<=50 (`0x5c74be–5c74cb`), otherwise sets failure and enters a separate branch. The exact geometry metric for the two displacement guards and the entire fallback trajectory are not closed here; they must not be replaced by guessed chemical thresholds. Normal successful guard path is `0x5c748e -> 0x5ca780 -> 0x5c74e6 -> 0x5c7863`.

Correction to an initial inspection hypothesis: the entry abs-max reduction (`0x5c4dd6–5c4ec4`) is over object+0x1d0, not the saved direction array. Its scalar [rbp-0x48] is passed to later callbacks (`0x5c8fef`, `0x5ca65b`), not used to divide ds_n. It is **not evidence** of a max-atom displacement scale.

Constants were read directly from ELF PT_LOAD file bytes: 0x4a45e08=0.1, 0x4a45e40=0.5, 0x4a45e58=0.95; initial abs-max sentinels are -inf and -DBL_MAX. This is static arithmetic evidence, not an isolated-instruction execution result.

## State boundary and saved-point evidence

The exact excerpts are preserved in `docs/research/native-moveds-evidence/moveds-width-retry-work.asm`, extracted from the absolute source
`/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/analysis/kernel-ssw_fixlat_mp_moveds_.asm`.
After the direct trial, `present_tooshort` returns its logical result through
`rbp-0xdc` at `0x5c7449–0x5c7459`. If that guard is clear, the scalar
`disp_perstep` test is at `0x5c745b–0x5c746c`; the following
`bonddisp_perstep` comparison computes `d_before*(1-bonddisp_perstep)` at
`0x5c746e–0x5c748e`. Only the `jbe 0x5ca780` branch reaches the normal
post-trial path. Otherwise `ds_n *= 0.95` at `0x5c7494–0x5c74a3`, resets the
short flag, and retries while `ds_n >= 0.1` and the counter is at most 50
(`0x5c74be–0x5c74cb`). The detailed class predicate inside `present_tooshort` is only partially resolved below; the fallback after the retry limit remains unresolved.

The complete callee disassembly is preserved in `native-moveds-evidence/present-tooshort.asm`; its core pair/image loop is in `present-tooshort-core.asm`. **2026-09-27 correction:** the second argument is the `iza` array (structure field +0x8); coordinates are passed separately in rcx. The callee computes periodic-image distances multiplied by `short_pres_factor` and returns a minimum-distance scalar and a separate logical short-distance flag. The two integers saved at `0x6dab6e/0x6dab75` are sorted `iza` values, not atom indices. `$XACNA` is a coordinate buffer, not the species/class array. See the [corrected operand audit](2026-09-27-native-moveds-guard-semantics.md) for limits and provenance.

The caller retries for a true short flag, maximum single-atom displacement greater than `disp_perstep`, or `d_before*(1-bonddisp_perstep) > d_after`. Three distinct operands are involved. Equality passes these two scalar tests. The retry loop continues while `ds_n >= 0.1` and counter `<=50`. Earlier text incorrectly substituted `disp_perstep` for `d_before` and conflated the displacement with the returned distance; that interpretation is withdrawn.

On the normal post-trial path, the saved direction record is overwritten by the Cartesian difference between selected trial coordinates and the stored center (`0x5c7b38–0x5c7b69`), then passed in-place to all-3N `n_normal` at `0x5c7c6a`; that callee explicitly sums all components and scales by the reciprocal Euclidean norm. The subsequent width loops use the persistent coordinate, center, and direction records identified above; their weighted difference is accumulated and stored at `0x5c80c4`. Therefore the direct unclipped path supports `n_out = delta_R/||delta_R||` and `width = delta_R·n_out = ||delta_R||` when the selected record/index matches. This remains conditional for retries, clipping, and alternate branches.

The separate climb-completion evidence in `native-gaussian-caller.md` remains
the only recovered release selection: coordinates are copied from `work2`
(`object+0xa28`) into trajectory `ng+1`, while forces are copied from `work1`
(`object+0x9c8`). This is not evidence that every failure exit uses the same
pair; last-trial substitution in the independent Python driver would still be
an unsupported assumption.

## Implication

The proposed explanation “native ds_atom keeps each atom's displacement fixed whereas Python uses total norm” is unsupported by the inspected default direct move. Both use a global Cartesian displacement when their incoming direction is unit normalized. Native's measured width and displacement retry guards are concrete differences; their practical role still requires matching full direction/move conditions and same-PES tests. No new scaling parameter is justified by this finding.

## Parameter provenance and width operand boundary (follow-up)

The parser/member map identifies `para+0x2db40` as `short_pres_factor`
(`analysis/kernel-dwarf-member-offsets.txt:53`), rather than an unnamed
geometric radius. The parser reads the input key `SSW.short_pres_factor` at
`0x687e3d–0x687e96` (`analysis/readsswpara-convergence.asm`) and stores

```
short_pres_factor = 1 / (1 + 0.005 * externaltp)
```

at `0x687e92–0x687e96`; the `0.005` and `1.0` constants are loaded at
`0x687e50` and `0x687e5f`. Thus the distance multiplier in
`present_tooshort_` is pressure dependent. The parser field name and formula
are source-backed; this audit does not assign a physical unit to `externaltp`
or to the resulting scaled distance.

`$XACNA` is a private coordinate buffer in `present_tooshort_`. The comparisons to 1 and 20 read the second argument (`iza`) through r13, not this buffer. The prior classification-array interpretation is withdrawn by the 2026-09-27 pointer audit.

The width write has now been followed to its immediate operands. At
`0x5c7c73` the accumulator `xmm1` is zeroed; the loops beginning
`0x5c7d72` and `0x5c7e63` accumulate the selected-coordinate minus center
coordinate, multiplied by the direction-record component. The result is
stored through the width-array pointer prepared at `0x5c80ae–0x5c80b5`, by
`movsd [r8],xmm1` at `0x5c80c4`. These are persistent native records, so the
weighted projection is compatible with the normalized direction result; the
remaining caveat is selected-record/index correspondence and branch coverage.

The pointer map used for the width conclusion is: `r15` loaded from the
selected trajectory's coordinate field `+0x170` at `0x5c7d7b`; `r14` loaded
from the trajectory-record array `object+0x1668`, record field `+0x170`, at
`0x5c7da4`; and `r10` formed from the direction-record array
`object+0x16b0`, record field `+0`, at `0x5c7e26`. These are persistent native
arrays, not anonymous scratch buffers. The remaining uncertainty is branch
coverage and selected-record indexing, not whether the width loop's operands
can be related to the normalized direction record.

## Retry, clipping, and failure exit table

| Address/path | Verified action | Width/direction consequence |
|---|---|---|
| `0x5c7452–0x5c748e` | Test `present_tooshort`, `disp_perstep`, and `bonddisp_perstep`; only `jbe 0x5ca780` accepts the trial | Accepted trial proceeds to the post-trial record path. |
| `0x5c7494–0x5c74cb` | Multiply local `ds_n` by `0.95`, clear the short flag, and loop to `0x5c6ddc` while `ds_n >= 0.1` and retry counter `<=50` | The rejected trial is not the width record; a later accepted trial is rebuilt and tested. |
| `0x5c74d1–0x5c75d5` | Set the local failure marker to `-1`, restore the current energy/diagnostic state, and enter logging/repair branches | No direct jump to the normal width entry is present on this failure path. Downstream branch-specific handling is not fully closed, so no universal statement is made about every failure caller. |
| `0x5c7863` | Normal post-trial record construction: selected trajectory/direction pointers are formed, displacement is written, then `n_normal` is called at `0x5c7c6a` | This is the only verified entry into the direction normalization and width accumulation slice. |
| `0x5c7b38–0x5c7c2b` | Write selected-coordinate minus center-coordinate differences into the direction record; the tail handles the non-vectorized remainder | Any clipping/repair that changed the selected coordinates is reflected in `delta_R`; requested `ds_n` is not substituted afterward. |
| `0x5c7d72–0x5c80c4` | Use selected coordinates, center coordinates, and direction record to accumulate the projected width and write `width[ng]` | For a matching accepted record, `n_out=delta_R/||delta_R||` and `width=delta_R·n_out`; retries and alternate exits remain conditional. |

The table is an address-level control-flow summary, not a claim that all
post-failure repair branches are semantically equivalent. In particular, the
available slice does not prove that a failure exit later fabricates a width or
reuses the last rejected trial as a Gaussian record.

A compact CFG extract is preserved in `native-moveds-evidence/retry-fallback-cfg.asm`.
The retry backedge at `0x5c6ddc` reinitializes the same `ng` from
`[rbp-0x58]`; the repeated setup at `0x5c6e83–0x5c6ec5` again derives the
trajectory base from `object+0x1668` and direction base from `object+0x16b0`
using the same record offset. Therefore an accepted shortened retry reaches the
same direction/width record contract, with `delta_R` reduced by the new `ds_n`.
The failure marker path is different: `0x5c74d1` sets `-1`, optionally logs
through runtime formatting calls, and branches by outer index to `0x5ca223`,
`0x5ca469`, `0x5ca106`, or `0x5c782e`. Those callees/repair routines were not
ABI-decoded here, so no width or last-trial semantics are assigned to them.

## Root review: failure restores a recorded point before Allopt

The earlier `retry-fallback-cfg.txt` is a human CFG summary, not raw
disassembly. Exact instruction lines are now retained in
`native-moveds-evidence/retry-fallback-raw.asm`. The apparent repair loops
are ordinary Fortran array assignments; they do not construct a new
geometry by an optimization or a heuristic.

At0x5c4f2d, r12=ng*0x690 and it is saved at[rbp-0x38] at0x5c501e.
On retry exhaustion with ng>1,0x5c75e6–0x5c7659 forms the source
trajectory_base+(ng-lower_bound)*0x690-0x520 and destination object+0x170.
Because -0x520=-0x690+0x170, this copies coordinates of trajectory[ng-1]
to the current structure. `for_realloc_lhs` adjusts the destination array;
0x5c77f1 copies contiguous data, and0x5ca106–0x5ca21e handles the
alternative elementwise loop. For ng=1,0x5ca223–0x5ca290 instead selects
trajectory_base-lower_bound*0x690+0x800, i.e. trajectory[1].cart
because0x800=0x690+0x170. Its alternative copy loop starts0x5ca51a.
An unallocated source routes through0x5ca469–0x5ca515 and deallocates
the current cart; this is descriptor behavior, not a scientific fallback.

All these failure branches join0x5c782e. The six-byte string at0x4a45ed4
is exactly `Allopt` (root `objdump -s` read); slot+0x188 corresponds to
`ssw_fixlat_mp_set_status_` in the independently mapped fixed-cell table
(see native-gaussian-caller.md). After that status call,0x5c7854 sets
control+0x120 to1 and jumps to the return at0x5c976e. This caller path
does not enter the successful direction normalization/width construction
at0x5c7863. The effect of Allopt in the outer driver is a separate state
transition, not evidence that the restored point is already quenched.

Thus for this retry-exhaustion path the point released to the next status
is a recorded prior point (or trajectory[1] for the first stage), not the
last rejected displaced trial. Accepted shortened retries retain the same
ng and coordinate/direction records and use their actual displacement to
construct width. This narrows a concrete parity gap; the independent ASE
driver currently reports failed proposals instead of claiming all native
failure-to-Allopt transitions are reproduced. No new default fallback was
added without whole-driver matched evidence.
