# Native LS pre-quench counter audit

Date: 2026-09-12. Read-only bounded disassembly; the LASP main program,
protection code, and any PES were not run. ELF:
`/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp`,
SHA256 `bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704`.

The stop comparison at `0x5bf6a7` is:

```asm
cmp r15d,DWORD PTR [rbx+0x2dad0]   ; SSW.LSoptsoftmax
jl  0x5be8aa                       ; continue
```

The relevant object is identified by DWARF as
`ssw_fixlat_mp_ssw_move_$STEP_OPT_SOFTPES.0.7` at `0x78eaea0`. In this
function it has the following bounded lifecycle:

| Evidence | Meaning supported by the instructions |
|---|---|
| `0x5bd309–0x5bd30f`: zero stored to `0x78eaea0` (and `NUM_FAIL` at `0x78eaea4`) | counter is initialized at the start of this soft-PES optimization lifecycle |
| `0x5bd6fa`: load from `0x78eaea0` before the force/energy setup | same counter is carried into the next optimizer cycle |
| `0x5bdb72`: call `bfgs_class_mp_bfgsdriver_` | one BFGS/LBFGS driver dispatch occurs |
| `0x5bdb77–0x5bdb7e`: `inc r15d`, then store back to `0x78eaea0` | counter increments once after the driver returns, unconditionally; no acceptance/status test is visible between return and increment |
| `0x5bf699–0x5bf6ae`: compare and branch | counter participates in the soft-pre-quench budget exit |

The outer cycle also contains analytic LS potential/force work before the driver
dispatch (for example `pot_bond_mod_mp_pot_bond_add_` at `0x5bd918` in the
shown path), while the counter has only one increment at the post-driver
boundary. Therefore the inspected code does not support interpreting this
counter as a PES evaluation count.

The callee is a BFGS driver whose fixed-cell path calls LBFGS and its
reverse-communication line search; the checked-in companion audit
`2026-09-12-fixed-lbfgs-rc-lifecycle.md` establishes that an RC request can
return through one driver dispatch and that dispatch count is not proven to
equal accepted iterations. The LS counter has the same decisive limitation:
the increment follows the `bfgsdriver` return without testing an accepted-step
flag. It is best described as **soft-PES optimizer dispatch/cycle count** (or
an enclosing RC-consumption count), with a likely one-dispatch-per-cycle
relationship. It is not established as accepted iterations or as equal to the number of
physical PES evaluations. The counter is defined by driver returns.

The available slice does not close whether one returned `bfgsdriver` call can
internally complete an accepted iterate, return an unevaluated trial, or report
a line-search failure in every branch. Consequently, do not map
`LSoptsoftmax` to Python `optimizer` iterations or accepted steps. A faithful
implementation should expose a separate `force_or_step_limit`-style budget
whose semantics are explicitly “driver dispatches/cycles”, unless a later
caller/RC audit supplies the missing accepted-iterate mapping.

This conclusion is static evidence only. It does not establish runtime
overrides, effective parameter values, numerical convergence, or scientific
validity.

## Reproduction evidence

The stop slice is retained at
`research/ga_ssw/evidence/native-ls-prequench-stop-20260912/stop.objdump` and
the parser/stop probe result at `.../result.json`. The full local function
slice was obtained with:

```sh
objdump -d -Mintel --start-address=0x5bc760 --stop-address=0x5bffe0 lasp
```

DWARF line mapping places initialization at `Class_ssw.F90:351`, the
pre-cycle load at lines `398–401`, the driver/increment region at lines
`406–410`, and the stop comparison at line `410` (`addr2line`/decoded-line
table against this ELF).
