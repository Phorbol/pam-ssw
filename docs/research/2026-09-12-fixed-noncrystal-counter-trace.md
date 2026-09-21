# Fixed-cell noncrystal optimizer counter trace

Date: 2026-09-12. Static-only trace of the fixed-cell call chain. Source ELF:
`/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp`.
No LASP main, PES, or production code was run.

## Counter alias and increment

At `climb_` entry to the optimizer descriptor (`analysis/kernel-ssw_fixlat_mp_climb_.asm:0x5cafa2–0x5cb02b`), the caller forms:

```text
rdx = control + 0x08
rcx = control + 0x80
r8  = 0
call [object_vtable + 0x110]       # 0x5cb02b
```

`class_struc_mp_noncrystal_opt_` begins at `0x5a88c0` and saves `rdx` as
`r13` (`0x5a88d7–0x5a88dd`). With the fixed caller's `r8=0`, its zero branch
at `0x5a88e3–0x5a88e6` goes directly to `0x5a91b9`. There it loads the
underlying structure data and calls the structure optimizer table slot `+0x10`
at `0x5a91d2`; the table entry is `0x5b2970`,
`bfgs_class_mp_bfgsdriver_` (also documented in
`native-gaussian-consumers.md`). The noncrystal function has no loop around
this dispatch that represents line-search trials.

On return from that optimizer callback, `noncrystal_opt` executes:

```text
0x5a919e  inc DWORD PTR [r13+0x0]
```

then returns at `0x5a91b0`. Since `r13` is the caller's `control+0x08`, this
is direct proof that one `noncrystal_opt` dispatch increments `climbstep` once,
after the downstream BFGS callback returns. It is not a calculator-request
counter, and it is not evidence that the returned trial was accepted as a
physical PES step.

The downstream BFGS path calls `bfgs_basics_mp_lbfgs_` at `0x5b41cd`; the
available disassembly establishes the optimizer arithmetic and reverse
communication interface, but does not expose an accepted-geometry flag to
`noncrystal_opt`. Thus the caller can establish “optimizer callback returned”
before incrementing `climbstep`, while accepted-step versus internal
line-search-trial semantics remain unresolved.

## Reset and first/later budgets

A concrete reset is present in `ssw_fixlat_mp_set_status_`, not in
`climb_convg_`: after the status transition to `climb_new`,
`0x5c26a9` writes `1` to `control+0x08` and `0x5c26b0` writes `0` to
`control+0x80`. The same function increments `object+0x1660` at
`0x5c1a3b–0x5c1a45`. Therefore a new climbing segment starts its optimizer
counter at one, and each later optimizer dispatch adds one at `0x5a919e`.
The caller's `addgaussian -> noncrystal_opt -> climb_convg` order is preserved.

`climb_convg_` reads that counter at `0x5cd434–0x5cd437`. Its strict gate
uses `ngaus_relax_ini` (`para+0x2dd24`) for `object+0x1660 == 1`, and
`ngaus_relax` (`para+0x2dd20`) otherwise (`0x5cd7a5–0x5cd7cc`). This explains
why first and later stages can have different effective optimizer-call
budgets.

The `climb_` trajectory-index increment around `0x5cbf49–0x5cbf61` is a
separate local/trajectory bookkeeping operation; it is not the optimizer
counter increment. The convergence function also writes descriptor output
metadata at `r8+0x08` in several blocks; those writes must not be confused with
the global `control+0x08` counter.

## Remaining lifecycle boundary

The recovered chain is therefore:

```text
set_status(climb_new): control+8 = 1
  -> addgaussian
  -> noncrystal_opt
       -> bfgsdriver -> lbfgs arithmetic
       -> control+8 += 1 on callback return
  -> climb_convg reads control+8 and applies first/later strict budget
```

This closes the counter alias, reset site, and increment site. It does not
close the meaning of an internal LBFGS trial, nor prove that every BFGS
reverse-communication request corresponds to a new accepted geometry. A future
isolated probe of `0x5a88c0`/`0x5b2970` should record callback count and
calculator requests separately; no public stopping-policy change follows from
this trace.
