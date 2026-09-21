# Fixed-cell native local optimizer: line-search semantics

Date: 2026-09-10. Bounded static audit of the uploaded ELF. No native trajectory executed and no production code changed.

**The resolved fixed-cell optimizer really does call a safeguarded line search. Replacing it with ASE LBFGS configured without line search omits a material part of its numerical behavior.** However, its compiled curvature tolerance is unusual, so it must not be described as an unmodified standard strong-Wolfe optimizer.

## Verified call path

Earlier [Gaussian consumer tracing](native-gaussian-consumers.md) resolved `climb → noncrystal_opt → bfgs_class_mp_bfgsdriver_`. This audit continues that actual path:

| Caller | Call site | Callee |
|---|---|---|
| `bfgs_class_mp_bfgsdriver_` | `0x5b2f50`, `0x5b41cd` | `bfgs_basics_mp_lbfgs_`, entry `0x6e87b0` |
| `bfgs_basics_mp_lbfgs_` | `0x6e8da8` | `bfgs_basics_mp_mcsrch_l_`, entry `0x6ea9d0` |
| `bfgs_basics_mp_mcsrch_l_` | `0x6eb4a4`, `0x6eb595` | `bfgs_basics_mp_mcstep_l_`, entry `0x6eb690` |

These are decoded call instructions, not merely matching symbol names. The `_l_` suffix belongs to the symbols. No assumption was made that the other LBFGS implementations also present in the large ELF are used here.

The line search uses reverse communication: MCSRCH `INFO=-1` causes LBFGS to return flag 1 requesting another energy/gradient evaluation (`0x6e8db8–0x6e8dc1`, `0x6e9a16–0x6e9a20`). `INFO=1` continues the accepted-step/history update; other status values take a failure-return branch (`0x6e8dc7–0x6e8dca`, `0x6e9a07`). Thus individual E/F requests cannot simply be equated with completed quasi-Newton iterations.

## Acceptance tests and parameter provenance

Let phi(alpha)=E(x+alpha*p), d0=grad(E)(x) dot p and d=grad(E)(x+alpha*p) dot p.

- MCSRCH forms `DGTEST=FTOL*DGINIT` and `FTEST=FINIT+STP*DGTEST` (`0x6eac7a`, `0x6eb20a–0x6eb22e`).
- Its success branch checks sufficient decrease and `abs(d) <= -GTOL*d0` (`0x6eb2ee–0x6eb322`). The absolute derivative is explicit bitmask arithmetic; this is the strong-Wolfe-shaped test, not just energy-only backtracking.
- Bracketing and safeguarded step interpolation call MCSTEP. Trial STP is bounded with min/max operations at `0x6ead5e–0x6ead62`.
- MAXFEV is initialized to **20** at `0x6e8c8a`, a per-line-search evaluation limit rather than the outer SSW budget.
- FTOL is copied from a caller argument at `0x6e8c78–0x6e8c83`. In BFGSDRIVER this is passed from `para+0x2dd58`, whose DWARF member name is **bfgs_etol**. The actual input-setting value for a production run was not recovered here.
- STPMAX is overwritten on LBFGS entry (`0x6e87dc–0x6e87ec`) from the caller argument. `class_struc_mp_init_bfgs_` loads `para.bfgs_maxstepsize` (`+0x2ddc8`) into the optimizer's `+0x1c0` at `0x5ada36–0x5ada52`; BFGSDRIVER passes that field to LBFGS. Therefore the compiled static STPMAX datum is not necessarily the effective value.

### Important nonstandard tolerance

Reading ELF data through PT_LOAD mappings gives:

| Address | Symbol | Compiled initial value |
|---|---|---:|
| `0x5521958` | `bfgs_basics_mp_gtol_` | **900.0** |
| `0x5521950` | `bfgs_basics_mp_stpmin_` | 0.0001 |
| `0x5521948` | `bfgs_basics_mp_stpmax_` | 0.2, overwritten from caller |

LBFGS checks whether GTOL is too small and can reset it to 0.9 (`0x6e88a4–0x6e8921`); that branch does not reset the value 900 to 0.9. No other GTOL assignment was established on the inspected path. Arbitrary external writes across the entire program were not exhaustively searched, so 900 is a verified compiled initialization plus local-path behavior, not a measured runtime value from a full search.

If GTOL remains 900, the usual mathematical premise `0<c1<c2<1` is absent. In particular this weak derivative test does not provide the standard positive-curvature guarantee. **Do not interpret native line search as proof that every native secant pair has positive y dot s.** Likewise do not copy 900 into the formal Python implementation merely for familiarity with the routine name.

## Hessian and force safeguards: established limits

The default inverse-Hessian scaling branch computes `YS=y dot s` (`0x6e9078–0x6e9087`), `YY=y dot y` (`0x6e90b3`), and fills a diagonal with **YS/YY** (`0x6e911e–0x6e9129`). The observed branch does not take `abs(YS)` or diagonalize/absolute-value a Hessian. A separate caller-supplied diagonal branch checks positivity and can return requesting a diagonal. This is distinct from dense ASE BFGS eigenvalue-absolute-value behavior.

The line-search bounds and caller max-step parameter are confirmed. They are scalar search-step controls; equating them to an identical ASE per-atom Cartesian maxstep needs direction-scaling analysis. BFGSDRIVER additionally contains force/gradient scaling, restart, NaN and displacement branches. The complete arithmetic of these optional safeguards was not reconstructed here. No blanket assertion is made that force clipping or another safeguard is absent everywhere. The evidence does **not** establish an absolute-Hessian safeguard as an explanation for improved native convergence.

## Reproduction

Artifact: `/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp`, SHA256 previously established as `bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704`.

Read-only command pattern:

```sh
objdump -dl -Mintel --disassemble=bfgs_basics_mp_lbfgs_ "$LASP_ELF"
objdump -dl -Mintel --disassemble=bfgs_basics_mp_mcsrch_l_ "$LASP_ELF"
objdump -dl -Mintel --disassemble=bfgs_class_mp_bfgsdriver_ "$LASP_ELF"
objdump -dl -Mintel --disassemble=class_struc_mp_init_bfgs_ "$LASP_ELF"
```

Disposable outputs: `/tmp/pam-native-lbfgs.asm`, `/tmp/pam-native-mcsrch.asm`, `/tmp/pam-native-bfgsdriver.asm`, `/tmp/pam-native-init-bfgs.asm`. Parameter names are in the archive's `analysis/kernel-dwarf-member-offsets.txt`.

## Implication for the current comparison

Adding a principled line search is supported as a correction to the independent implementation's optimizer semantics; it is not an ad hoc landscape escape heuristic. ASE LBFGSLineSearch can serve as a meaningful numerical comparator, but changing to it does not establish native parity: line-search constants, step scaling, history handling and restart conditions still differ.

The separate replay experiment reported by the main agent (31 frozen failed biased stages) must retain its own evidence and E/F budget accounting. This static report neither reproduces those measurements nor attributes remaining failures to negative curvature. It only establishes which previously omitted native numerical mechanism should be controlled in the next comparison. Actual search efficacy still needs complete real-system runs.
