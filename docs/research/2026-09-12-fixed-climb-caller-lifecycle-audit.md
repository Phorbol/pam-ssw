# Fixed-cell climb caller lifecycle audit

Date: 2026-09-12. Scope: static inspection of the uploaded fixed-cell ELF and
its complete disassembly for `ssw_fixlat_mp_climb_` and
`ssw_fixlat_mp_climb_convg_`. No LASP main execution, PES call, or production
change.

## Dispatch boundary

The fixed-cell method table is independently resolved in
`docs/research/native-gaussian-caller.md`: slot `+0x110` is
`class_struc_mp_noncrystal_opt_` (`0x5a88c0`), `+0x1e8` is
`ssw_fixlat_mp_climb_convg_` (`0x5cd130`), and `+0x1e0` is `addgaussian_`
(`0x5cda70`). The caller sequence is visible in the complete
`analysis/kernel-ssw_fixlat_mp_climb_.asm`:

| address | operation | lifecycle meaning |
|---|---|---|
| `0x5caf49–0x5caf71` | save `object+0x230` into `object+0x1ac8` | entry energy snapshot (`tene0`) |
| `0x5caf9c` | call vtable `+0x1e0` | add/update the Gaussian field for this segment |
| `0x5caff4–0x5cb023` | construct optimizer descriptor; pass `rdx=control+0x08`, `rcx=control+0x80`, `r8=0` | reverse-communication optimizer state and counter pointer are prepared |
| `0x5cb02b` | call vtable `+0x110` | one local optimizer callback; it may contain internal optimizer work, but this caller does not expose its trials |
| `0x5cb032–0x5cb03d` | call vtable `+0x1e8`, result at `&[rbp-0x34]` | convergence/status decision is called after the optimizer callback returns |
| `0x5cb043–0x5cb047` | test returned low flag bit | selects the subsequent status/release path |

Therefore `climb_convg` is at the optimizer callback boundary. The assembly
proves neither a call for every calculator evaluation nor that the callback's
last internal line-search trial is an accepted physical step. Calling it an
"accepted-step" hook would add an unproved semantic layer; calling it a
line-search hook would also be wrong. The strongest supported statement is
that it consumes the optimizer's returned state once per enclosing
`noncrystal_opt` dispatch.

## Counter and stage selection

The DWARF `ssw_control` layout in
`analysis/selected-dwarf.txt:1570–1585` identifies `+0x08` as `climbstep`
(`+0x04` is `rotstep`, `+0x0c` is `cbdmovestep`). The descriptor setup above
passes the address of this field to `noncrystal_opt`; this is not the
`r8+0x08` descriptor writes inside `climb_convg` (those are output descriptor
metadata, e.g. `0x5cd324`, `0x5cd4b1`, `0x5cd590`, `0x5cd6a0`, and
`0x5cd728`).

Inside `climb_convg_`, `0x5cd434–0x5cd437` loads `control+0x08` into a local
scalar. The stage gate then selects the first-stage budget when
`object+0x1660 == 1` (`0x5cd7a5–0x5cd7a9`, `ngaus_relax_ini` at `para+0x2dd24`)
and the later-stage budget otherwise (`0x5cd7af–0x5cd7cc`, `ngaus_relax` at
`para+0x2dd20`). The comparison is strict `climbstep > budget`. This is an
actual first/later distinction, not a claim that all Gaussian stages use the
same budget.

The full fixed-cell `climb_` disassembly also shows trajectory bookkeeping
around `0x5cbf49–0x5cbf61`: it increments a local trajectory index and copies
the current energy into the next trajectory record. It does not write
`control+0x08`. In the inspected `climb_` and `climb_convg_` bodies, the
counter's reset/increment producer is not a direct store to the global control
object; the counter is supplied by reference to the optimizer and consumed by
convergence. This is an unresolved producer location, not evidence that the
counter is absent. The exact reset point before a first stage and the exact
increment timing inside `class_struc_mp_noncrystal_opt_` require disassembling
that callee's complete reverse-communication loop (or a separately authorized
isolated callee probe).

## Release ordering relevant to lifecycle claims

When `control+0x78` (`lclimb_allstop`) is set at `0x5cb377`, the enclosing
caller restores forces (`0x5cb383–0x5cb629`), coordinates
(`0x5cb647–0x5cb8d8`), and entry energy (`0x5cb8dd–0x5cb8e4`) before status
cleanup. This confirms that the normal release consumes the pre-callback
snapshot. It does not establish that an arbitrary last line-search trial is a
released endpoint. The returned optimizer flag and the two global status words
are separate state; no single `completed` label should be inferred from the
outer function return alone.

## Consequence for the public fixed-cell implementation

The public implementation can reproduce the visible ordering
`addgaussian -> local optimizer -> convergence gate`, and it can report first
versus later budgets. It does not currently have evidence for the native
optimizer's internal counter lifecycle or its accepted-trial semantics. A
minimal future parity probe would target `class_struc_mp_noncrystal_opt_`
(`0x5a88c0`) with its already recovered descriptor ABI, observe writes through
`control+0x08`, and stop at return; it must record optimizer callback count
separately from calculator requests. No default policy change follows from
this audit.
