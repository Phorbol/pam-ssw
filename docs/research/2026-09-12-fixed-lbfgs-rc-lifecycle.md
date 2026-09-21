# Fixed-cell LBFGS reverse-communication boundary

Date: 2026-09-12. Static inspection only; no LASP main, PES, or production
change. Source ELF:
`/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp`.

## Direct call chain

The fixed-cell path is:

```text
climb 0x5cb02b
 -> noncrystal_opt 0x5a88c0
 -> bfgsdriver 0x5b2970
 -> lbfgs 0x6e87b0
 -> mcsrch 0x6ea9d0 / mcstep 0x6eb690
```

The `noncrystal_opt` fixed branch at `0x5a91b9–0x5a91d2` calls the BFGS
method once. In the complete `bfgs_class_mp_bfgsdriver_` disassembly, the
fixed-path LBFGS call is `0x5b41cd`. After that call, control proceeds to
post-call bookkeeping and force/displacement checks (`0x5b41d2` onward), with
no physical energy/force callback and no second LBFGS call before the function
returns through its branch epilogues. The complete saved disassembly is
`research/ga_ssw/evidence/native-noncrystal-opt-20260912/bfgsdriver.full.asm`.

The BFGS driver passes an IFLAG/return slot into LBFGS through its constructed
argument list (`0x5b4168–0x5b41ca`). LBFGS therefore communicates its next
request/result to the caller; it does not itself obtain a new PES evaluation.
The inspected post-call path contains arithmetic, `cal_vol` and status/error
handling, not a physical oracle call.

## What the RC result proves

Existing isolated LBFGS instruction evidence identifies the internal boundary:

- `MCSRCH INFO=-1` returns from LBFGS with `IFLAG=1`, requesting a new E/G
  (`0x6e8db8–0x6e8dc1`, `0x6e9a16–0x6e9a20`).
- `INFO=1` continues the accepted-step/history update; the accepted iterate
  boundary is `0x6e8dad`.
- Other status values take LBFGS failure-return branches.

Combined with the complete BFGS-driver path above, this supports the narrower
lifecycle statement: an `IFLAG=1` result is returned through one
`bfgsdriver`/`noncrystal_opt` dispatch to the enclosing caller, which must
supply the requested E/G on a later dispatch. It does **not** prove that one
`noncrystal_opt` call contains multiple physical oracle requests. LBFGS may
perform internal line-search arithmetic before returning, and its returned X
may be a trial proposal when IFLAG requests E/G; whether that proposal is
accepted is decided only after the subsequent reverse-communication exchange.

The enclosing `noncrystal_opt` increments `control+0x08` exactly once at
`0x5a919e`, after `bfgsdriver` returns. Thus `climbstep` counts these enclosing
optimizer dispatches/RC consumptions. It is not a direct PES-request count and
is not proven to equal accepted iterations.

## Python consequence

The current Python `Safe-total maxiter` can remain a cap on accepted optimizer
iterations for that implementation, but it is not a literal native
`climbstep` translation. A native-parity accounting layer would need separate
counters for optimizer dispatches, physical E/F requests, line-search trials,
and accepted iterates. A request cap must return the last evaluated accepted
point according to the local optimizer contract, while a native-style RC
adapter would need to preserve an unevaluated trial separately. No public
stopping-policy change follows from this static result.
