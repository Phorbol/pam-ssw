# BRZERO4 update block audit (static)

This is a bounded static audit of the recovered BRZERO4 body after the
DGEGV call.  It does not claim a complete native port or a Cartesian metric.
The disassembly artifact is
`research/ga_ssw/evidence/native-broyden-update-brzero4-6fcfb5-700f20.asm`
(SHA256 `9f2a966065f7b689f8e90ed60a21f48f66c0d360472467577ac052d2bb880856`).
Addresses below are virtual addresses in the frozen `GA-SSW_program/lasp`.

## Recovered update ingredients

At `0x6fd03c--0x6fd337` (DWARF source lines 1132--1165), the code constructs
`AMAT` in a 50-by-50 padded layout.  The diagonal path loads `WI_i` and
`FINF_ii`, squares/multiplies the weights, and adds the literal `1.0`
(`0x6fd063--0x6fd096`).  The off-diagonal path at `0x6fd2e0--0x6fd315`
multiplies the corresponding two `WI` entries and `FINF_ij`.  Thus the
recovered matrix is

```
AMAT_ij = delta_ij + WI_i * FINF_ij * WI_j.
```

At `0x6fd33d--0x6fd376`, `broyden_module_mp_invers_` is called with `AMAT`;
its body at `0x701337--0x7013c1` calls `ludcm_`, then repeatedly calls
`lubks_` at `0x7013c5--0x7013ca` with identity right-hand-side columns.  The
result is therefore an explicit LU inverse/solve, not a BLAS `dgemm` path.
The only caller-side diagonal shift visible in this block is the literal
`+1.0` in `AMAT`; pivot handling inside `LUDCM_` remains outside this audit.

The following history algebra is explicit scalar/SIMD arithmetic.  The
post-inverse block reads and writes `BETAQ` while combining weighted history
arrays (`0x6fd5f6--0x6fd621` and `0x6fd8b9--0x6fd8e4`; DWARF source lines
1170--1171 and nearby lines).  The static loop structure is consistent with
forming the old-history columns as

```
Q_old = I - BETA * W * FINF * W
```

with the factors applied in the executable's padded/column-major layout.  The
full-hybrid probe observed `Q = [[beta00, 0], [beta10, 1]]` and reconstructed
`Z_new = U @ (W beta W) + Z_old @ Q.T` to error `2.5e-12`, but its `WI` values
were constant.  Therefore this probe verifies the observed two-history array
convention only; it cannot distinguish the possible orderings of nonconstant
`WI` factors.  The general formula above remains a hypothesis pending a
different-weights probe.  These operations are matrix/history algebra, not
matrix/history algebra; the preceding eigenvalue ratios also feed a separate
scalar modulus/selection path described below.

The runtime-history trace shows the active order changing 4 to 3.  Static
control flow places the spectral modulus test at `0x6fbfea--0x6fbff2`, while
the later branch at `0x6fc064--0x6fc066` tests a separate status/index local;
it must not be labeled the spectral deletion test.  When the shift path is
entered, the helpers receive `[rbp-0x2d8]` as the removal-index address
(`0x6fc12b--0x6fc180`); the helper loads that index at `0x700fc4` and shifts
subsequent columns left (`0x701046--0x70109c`).  The source-level mapping from
the spectral modulus to the selected deletion index remains unresolved.

At `0x6fe320--0x6fe762` (source lines 1213--1227), the active `X` storage is
combined with the history arrays.  The inner loop loads a coefficient from
the temporary coefficient region and performs componentwise

```
X_component <- X_component - coefficient * Z_component
```

(`0x6fe519--0x6fe6e8`); the surrounding loops traverse history columns and
padded Cartesian entries.  The same block forms force-difference work arrays
and calls `inproduct_` at `0x6fe92a`, `0x6fe95a`, and `0x6fe978`; the exact
consumer of those scalars is outside this bounded conclusion.  It then
branches to failure/cleanup or copies `X`/`F` history at
`0x6feb28--0x6fedde`.

## Substitution boundary

A mature multisecant/LU Broyden implementation can reproduce the broad
mathematical skeleton only after the exact meanings and strides of `WI`,
`FINF`, `BETAQ`, `U`, `Z`, `DF`, and the coefficient temporary are fixed:
weighted Gram-like `AMAT`, LU solve, history recurrence, and a weighted
`Z` correction to `X`.  It cannot be treated as drop-in equivalent merely
because it is called “Broyden”.  The recovered code uses a fixed padded
layout, custom Fortran array descriptors, explicit history shifting, and the
non-Cartesian `inproduct_` block-sum already documented by the first-matrix
probe.

The remaining static boundary is the complete source-level `U/DF` recurrence
and every padded stride in the `BETAQ` contraction; the dynamic probe closes
the two-history convention but does not replace that mapping for arbitrary
history lengths.  Also, the
The raw generalized eigenvalues DO control history removal through the
maximum complex modulus `abs(1+alpha/beta)` and the recovered1e7 bound;
the separately stored EIGENVAL array is printed. See the root-corrected
`native-broyden-dgegv-spectrum-audit.md` for exact constants and a dynamic
4→3 history trace. Earlier diagnostic-only and zero-threshold interpretations
were incomplete/incorrect and must not guide implementation.
