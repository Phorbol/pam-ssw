# Native `compress_mode` geometry: bounded ELF evidence

2026-09-11. This is a static disassembly of
`/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp`
(SHA256 `bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704`).
No LASP execution, PES call, or production change was made. The bounded raw
excerpt is [globalcompress.asm](../../research/ga_ssw/evidence/native-cluster-control-generator/globalcompress.asm).

## Facts read directly from the instructions

The function is `newssw_basics_mp_compress_mode_` at `0x6e36f0`. On the
uncached path (`*([rbp+0x10]) == 0`), the pointer in `rdx` is read as nine
consecutive doubles and three Euclidean lengths are formed:

```
r0 = hypot(rdx[0], rdx[1], rdx[2])       # 0x6e3726–0x6e375e
r1 = hypot(rdx[3], rdx[4], rdx[5])       # 0x6e375e–0x6e3783
r2 = hypot(rdx[6], rdx[7], rdx[8])       # 0x6e3787–0x6e37b3
```

They are cached at `0x791b610`, `0x791b618`, and `0x791b620`. The code then
calls `ssw_commsub_mp_rd_numb_` and truncates products with constants 11.0 and
9.0 (`0x4a4ccc0`, `0x4a4ccc8`) into integer `C` and `C2`; a 15.0 gate
(`0x4a4ccd0`) forces `C=1` when the relevant maximum length is at most 15.
The cached path reuses `C`, `C2`, and `IMAX` rather than recomputing them.
The exact random-number and Fortran dummy-argument contract is outside this
bounded function.

The caller at `ssw_fixlat_mp_gen_randommode_` (`0x5d6b94–0x5d6bce`) verifies
the principal argument roles: `rdi` is a local N pointer, `rsi` is the
`object+0x170` Cartesian record stream, `rdx` is `object+0xe0` (nine doubles),
`rcx` is `object+0x8a8` (the 3N integer mask), `r8` is `object+0x1848`
(output), `r9` is `object+0x1ad8`, and stack argument 7 is `object+0x1660`
(cache flag). The same Cartesian and mask roles are independently visible in
the local-pair caller. This cross-check supersedes the earlier generic
description of these pointers.

The active-point accumulation loop at `0x6e3885–0x6e3916` advances the point
array by 24 bytes and a per-point integer array by 12 bytes. For every point
whose tested mask word is positive, it adds a three-double record to
`COM[0:3]` at `0x791b630` and divides all three components by the positive
active count. Thus the instruction-level operation is

```
COM = sum(records[k] for tested_mask[k] > 0) / count(tested_mask[k] > 0).
```

The record meaning and which one of the three mask words is selected are not
named by the machine code; the 24/12-byte strides and tested offsets are
reported in the excerpt.

The selection loop at `0x6e3925–0x6e3a4c` is a compiler-unrolled scan of
successive atoms (two bodies per loop), not evidence of physical paired
groups. It skips non-positive mask entries and retains the smallest selected
Cartesian component (the `minsd` at `0x6e39b0`/`0x6e39fa` and final candidate
comparisons at `0x6e3a26–0x6e3a37`). `IMAX` selects which Cartesian component
is used: the scalar is `coordinates[atom, IMAX-1]`, because the base is
`rsi + 8*IMAX` at `0x6e394f`, rather than an independently supplied radius.
The selected one-based atom index is recorded in `r11`, and the function forms
`NADD = selected_record - COM` at
`0x6e3b41–0x6e3b64` (the analogous branch is `0x6e3e18–0x6e3e3b`). This is
the recovered neighbor/group selection dependency; the parallel scalar is
not proven to be a radial distance, so no radius cutoff is asserted here.

For selected records passing branch-specific scalar gates, the function emits
normalized three-component vectors. The common arithmetic is
`NADD / sqrt(NADD_x^2 + NADD_y^2 + NADD_z^2)`, followed by sign-mask XORs:
the masks at `0x4a4cbf0` are `(-0.0,-0.0)` and the mask at `0x4a4cc00` is
`(-0.0,+0.0)`. Therefore the exact sign pattern depends on the branch and
cannot be replaced by “normalize the displacement” without retaining those
bitwise sign operations. Writes occur to 24-byte records at `0x6e3bbb`,
`0x6e3c10`, `0x6e3c63` and the corresponding second branch at
`0x6e3e92`, `0x6e3ee7`, `0x6e3f3a`.

The selected scalar minus `COM[IMAX-1]` is classified at `0x6e3a62–0x6e3aa5`:
the constants are 3.0 (`0x4a4cce0`) and 6.0 (`0x4a4cce8`), producing mode
codes 1, 2, or 3. Other branches use 5.0 (`0x4a4ccf0`) and 2.5
(`0x4a4ccf8`) as scalar offsets/gates. Values -2.0 (`0x4a4cd00`), 0.6,
0.7, and 0.8 are also present in this code region, but the latter three are
not consumed by the recovered `compress_mode` arithmetic; they belong to
adjacent routines/constants and must not be assigned to this criterion.

## Interpretation and boundary

The bounded evidence supports this description: an active-mask COM is formed,
a masked paired-record scan selects a minimum scalar candidate, its record is
centered at COM, and one of several branch-specific normalized/sign-adjusted
vectors is written to a 3N-style output stream. This is a geometric control
generator, not evidence that the function directly moves the input coordinate
array. The function does write output records through pointers saved from the
call frame, but their Fortran array identity and the downstream consumer are
not closed here.

An isolated instruction probe is saved as
`research/ga_ssw/evidence/native-cluster-control-generator/compress-oracle-v3.json`.
It runs one VM first with cache false and then true, so the cached call reuses
the preceding `C/C2/IMAX/COM` state. N=4 with an oblique nine-double input and
partial 3N mask, N=6 with an oblique input and full mask, and saved C60 initial
coordinates all returned successfully; cached and uncached outputs were
bitwise equal in these cases. v3 also exercises `C=0` and `C=10, C2=8`,
all three `IMAX` axes, a partial mask, and a disabled pair record. The earlier fresh-VM cache rows in
`compress-oracle.json` are retained as preliminary invalid comparisons.

For `C <= 6`, the direct branch at `0x6e4218–0x6e4277` writes the componentwise
negative of the centered record, i.e. `output_i = COM - R_i`, for entries with
the tested `mask[i,0] > 0`; it does not normalize each record. For `C2 <= 5`,
the tail at `0x6e4287–0x6e42e7` selects two positive integer slots and
constructs `v = R[p0] - R[p1]`; the `0x6e433d–0x6e4434` arithmetic then writes
the rank-one projection `output_i = v * (v dot (COM - R_i))`. A non-axis-aligned
N=6 probe verifies both formulas to `4.44e-16` maximum absolute error, while
`C2 > 5` takes the common non-tail path. The reproducible audit is
`research/ga_ssw/audit_globalcompress_pairtail.py` with results in
`compress-pairtail-audit.json`. These are control-record outputs, not runtime
coordinate moves.

The independent C7–C10 matrix is in
`compress-c7-c10-audit.json`, generated by
`research/ga_ssw/audit_globalcompress_c7_c10.py`. It covers all three IMAX
axes with pair disabled, `C=7,8,9,10`, and `C2=8`; cached/uncached outputs
remain bitwise equal. The C10 branch is algebraically confirmed from both
instructions and outputs as `COM-R + 5 e_IMAX`. The C9 branch has a
branch-specific IMAX overwrite by the `-2.0` constant, but its per-atom gate
is not reduced to a single predicate here. C7/C8 retain the raw scalar
overwrite and sign-mask sequence at `0x6e3d03–0x6e3d74`; their complete
conditional algebra remains explicitly open rather than being inferred from
array examples.

Still unknown are the complete Fortran descriptor/layout details of `r9` and
the output stream, the exact mask-word selection for every mode, and the
downstream consumer's coordinate action. Consequently this document does not
claim a global minimum, a physical neighbor radius, an atom-connectivity
operation, or a universal coordinate-compression formula. The direct probe in
`research/ga_ssw/oracle_compress_mode.py` confirms that the isolated entry can
execute with the recovered raw-pointer ABI for N=4, N=6, and a saved C60
geometry (both cache states); its outputs are instruction-oracle records, not
a scientific effect or a complete native trajectory.

The `SSW.globalcompress` parser mapping and random enable probability are
documented separately in
[native-cluster-control-selection.md](native-cluster-control-selection.md).
That mapping does not identify the geometric arrays or prove that every
enabled step reaches this function.

## Root piecewise closure, 2026-09-11

The earlier paragraph incorrectly associates `0x6e3d03–0x6e3d74` with C7/C8;
that is the C10 branch. C7/C8 dispatch at `0x6e3aba` to `0x6e3d8d`.
The full finite-input formulas for C7/C8/C9 are now recovered and exercised
in `audit_globalcompress_piecewise.py`. The retained initial audit has27 cases;
the extended `compress-piecewise-boundaries.json` has57, all with exactly zero
formula error and bitwise equal cached/uncached outputs. These are zero-PES
formula checks, not evidence of better search.

Let a be `IMAX-1`, c the COM of atoms with `mask[i,0]>0`, and
m=min(R[i,a]) over `mask[i,a]>0`. These two masks can differ. Define
region=1 for m-c[a]<3, 2 for 3<m-c[a]<6, 3 for m-c[a]>6, and **0** at the
exact boundaries3 and6. All outputs begin at zero; only atoms with
`mask[i,0]>0` receive updates. For C7/C8/C9, transverse components j!=a are
`-(R[i,j]-c[j])/|R[i]-c|` if `R[i,a]<c[a]+1`, otherwise zero.

For C7 and C8, the axial component is +2 if `m+0.5<R[i,a]<upper`, where
upper is c[a]+1, c[a]-2, or c[a]-3 for regions1,2,3 respectively. Region0
leaves the axial component zero. C7 and C8 use the same recovered geometry.
For C9, the axial component is -2 inside the strict intervals
(c[a],c[a]+2.5), (c[a]-2.5,c[a]), or (c[a]-5,c[a]-2.5) for regions1,2,3;
region0 again leaves it zero. This includes component-mask cases deliberately
reaching all four region codes. Zero centered distances and empty eligible
sets remain outside the finite-output qualification.

The downstream consumer is now also narrowed: `0x5d6bce` calls compress_mode,
optionally applies `setconstraints` at `0x5d6bf3`, unconditionally normalizes
at `0x5d6c07`, then loads coefficient slot2 (zero-based) at `0x5d6c1a` and
accumulates the weighted vector into the aggregate. The aggregate has its own
normalization later. Thus these are raw direction records, not immediate
coordinate displacements. The empirical length constants and Cartesian-axis
dependence do not establish a universally appropriate physical move, and no
production strategy was changed by completing this recovery.
