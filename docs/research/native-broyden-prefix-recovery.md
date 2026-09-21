# BRZERO4: verified initialization and secant prefix; non-Euclidean helper discovered

Date: 2026-09-09. This is executable-behavior recovery and numerical verification,
not a completed BRZERO4 port or a scientific validation of Broyden/SSW efficiency.

## Scope and evidence

Uploaded ELF: `/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp`.
SHA256: `bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704`.
Static files in the adjacent `analysis/` directory: `brzero4-lines.asm`,
`brions4-lines.asm`, `broyden-scopes.txt`. Function entry BRZERO4 is `0x6f6c00`.
DWARF associates it with `broyden_module.f90:701–1252`.

New independent implementation: `pamssw/standalone/native_broyden.py` contains
only the three proven arithmetic primitives. It is deliberately not exported
as a complete optimizer or wired into a search driver.

New instruction oracle: `research/ga_ssw/probe_native_broyden_prefix.py`.
Frozen inputs, actual machine-instruction outputs, errors and runtime hook counts
are in `research/ga_ssw/evidence/native-broyden-prefix/result.json`.

The oracle executes the uploaded x86 instructions with Unicorn. For initialization
it returns from the entire BRZERO4 function. For the next call it deliberately
stops at `0x6f9f78`, before matrix construction, history pruning or outgoing step.
It replaces Fortran allocation, memory copy/zeroing and printing only; the
`inproduct_` arithmetic and runtime overflow-check instructions execute unchanged.
This is not execution of a complete native LASP process.

## Caller layout and initial step

BRIONS4 stores consecutive XYZ components directly into flattened X and F:
`0x6f6a80–0x6f6abd` copies byte offsets 0, 8, 16, then advances 24 bytes for the
next atom. `0x6f6b35–0x6f6b39` sets NDIM to `3*NIONS`. It passes these contiguous
arrays to BRZERO4 (`0x6f6b8d`). There is no intervening coordinate transformation
or component-sum representation in this wrapper.

G0 is a diagonal/elementwise array: BRIONS4 fills all entries with saved STEP
(`0x6f6a25`, `0x6f6ac6–0x6f6b33`), initialized to 1 at allocation. The caller's
rotation force has already been scaled by FACT1, so G0 is not a dense Hessian.

BRZERO4 copies input X1, F1 and G01 into saved work arrays (source 784–786).
It resets its internal iteration to 1 on INI, otherwise increments it
(`0x6f8d41–0x6f8d65`). On first iteration it shifts previous X_LAST/F_LAST to
X_LL/F_LL, stores current X/F as X_LAST/F_LAST (source 826–829), saves the supplied
INIANGLE into CURV_LAST (source 833), then returns

`X1 = X + G0 * F`

with elementwise multiplication (source 840; scalar `0x700b74–0x700b89`). It
therefore saves the **evaluated input X** as X_LAST, not the predicted output X1.

## A surprising executable fact: INPRODUCT is a block-sum bilinear form

`inproduct_` at `0x700f20–0x700f80` computes:

```
q(a,b) = sum_i (a[3*i]+a[3*i+1]+a[3*i+2])
               * (b[3*i]+b[3*i+1]+b[3*i+2])
```

Instruction evidence: NDIM is divided by 3 (`0x700f24–0x700f3a`); each of the
three entries in each vector is added (`0x700f46–0x700f65`); the two sums are
multiplied once (`0x700f6b`), accumulated, and pointers advance 24 bytes. It is
**not** `dot(a,b)`. The helper ignores incomplete trailing blocks; BRIONS4 always
passes a multiple of 3, and the independent Python API rejects other sizes.

This is positive semidefinite but degenerate: in XYZ coordinates its per-atom
matrix is `ones((3,3))`, with eigenvalues 3, 0, 0. This matrix representation and
its consequences are mathematical deductions from the recovered arithmetic,
not claims that the source authors intended such a metric.

Three direct executions of the original helper confirm:

| Input vector | Native q(v,v) | Euclidean squared norm |
|---|---:|---:|
| (1,-1,0) | 0 | 2 |
| (1,0,1), a proper rotation of the first vector | 4 | 2 |
| (1,2,3) | 36 | 14 |

Thus this helper is not rotation invariant and can annihilate nonzero vectors.
This does **not** establish that a complete LASP trajectory has already been
measured to violate rotational invariance: full caller/callee compensation would
need a complete rotation or search experiment. The recovered BRZERO4 prefix does
use the helper without compensation for its normalization.

## Recovered noninitial secant preparation

Let F and X be the new evaluated values, and F_LAST/X_LAST the preceding evaluated
values. At iteration k>1 the latest history column is k-1 (Fortran indexing):

```
dF = F - F_LAST                         # source 854
DX[:,k-1] = X - X_LAST                  # source 855, named F_HIST in ELF
s = sqrt(q(dF,dF))                      # source 867, 870
DF[:,k-1] = dF / s                      # source 872
U[:,k-1] = G0 * DF[:,k-1] + DX[:,k-1]/s # source 875
```

Force subtraction is confirmed at `0x6f8ea4–0x6f8eb2`; displacement subtraction
at `0x6f9168–0x6f9176`. INPRODUCT is called at `0x6f96b8`; square root and inverse
at `0x6f97d3–0x6f97e6`. The positive sum in U, not subtraction, is explicit in
`0x6f9b08–0x6f9b19` (also scalar tail `0x6f9e1e–0x6f9e30`).

The script initially assumed Euclidean normalization and every trial failed.
Following that discrepancy into the original INPRODUCT routine revealed the
block-sum operation. After correcting the mathematical specification, **18/18**
initialization-plus-prefix comparisons pass over NDIM 3, 6, 9, 15, 45, 180.
The new regression test compares Python against saved machine outputs, not merely
against a duplicate Python formula.

A separate exact-null secant dF=(1,-1,0) was passed through the original prefix;
DF and U both contain nonfinite values. The independent primitive explicitly
raises on zero s. That is a documented domain-handling divergence, with no hidden
epsilon, Euclidean substitution or automatic fallback. The literal nonfinite
native outcome is recorded in the oracle JSON. This is a diagnostic failure case,
not a recommended physical parameter choice.

## Matrix stage: evidence recovered, still insufficient for full implementation

DWARF and allocation code identify 50-column histories DF, U, Z, T and F_HIST,
and 50x50 matrices GMAT, SMAT, FINF, AMAT, BETA and BETAQ. The executable contains
both generalized-eigenvalue (`DGEGV`) and explicit matrix-inversion (`invers_`)
operations. It is therefore not justified to replace this with a generic rank-one
Broyden formula and claim instruction parity.

Remaining work before a complete port:

1. Recover each matrix entry, including the distinction between explicit
   Euclidean contractions in loops and calls to the block-sum helper. Do not
   globally replace every dot product based on one helper's name.
2. Recover DGEGV inputs, spectral decisions, signs, ordering and singular cases.
3. Recover which histories are removed/shifted and how ITER is adjusted.
4. Recover the final Z/T recurrence and outgoing coordinate update.
5. Recover the last-angle/curvature restart and STEEP flag semantics.
6. Execute a sequence with multiple completed noninitial updates, resets and
   history removal against Python, then integrate into the full rotation oracle.

No unknown matrix branch is implemented in `native_broyden.py`.

## Relation to papers and PAM; research decision

The already archived primary sources are Shang and Liu, JCTC 2012,
DOI [10.1021/ct300250h](https://doi.org/10.1021/ct300250h), section 2.1 equations
5–8, and Shang and Liu, JCTC 2013, DOI
[10.1021/ct301010b](https://doi.org/10.1021/ct301010b), equations 3–7. They motivate
biased direction rotation and SSW, but the block-sum normalization above is an
**executable-specific fact**, not a paper-derived requirement. It must not be
silently given the interpretation of a physical Cartesian inner product.

PAM uses its separately designed finite direction pool and curvature-based
scoring; the standalone paper reference uses a Krylov/Ritz solve. Neither is
this recovered BRZERO4 prefix. The immediately useful consequence is a new
controlled diagnostic for the native-parity route: jointly rotate coordinates,
forces and the original anchor and compare the *full recovered rotation*, when
available, before treating its numerical behavior as a trustworthy improvement
over PAM. This is an implementation audit, not another tunable search component.

Decision: retain the exact legacy primitives for reproducibility and continue
matrix recovery. Keep the proven nonfinite boundary explicit. Do not introduce
this bilinear form as a default PAM metric or as a scientifically justified
preconditioner. Real-system end-to-end efficacy remains untested for these new
primitives, and no long/GPU/HPC job was launched.

## Reproduction

```
PYTHONPATH=/tmp/pam-ssw-unicorn-probe:. python research/ga_ssw/probe_native_broyden_prefix.py --elf /home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp --output research/ga_ssw/evidence/native-broyden-prefix/result.json
python -m pytest tests/reproduction/test_native_broyden_prefix.py -q
```

Current targeted tests: 5 passed. The numerical contract assumes finite
intermediates; merely finite extreme inputs can still overflow subtraction or
multiplication, and that domain was not qualified. Optional Unicorn is needed to rerun the ELF
probe, but not for the independent module or frozen regression.
