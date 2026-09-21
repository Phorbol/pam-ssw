# BRZERO4 first matrix block (read-only audit)

Date: 2026-09-12. This audit extends the verified stop at `0x6f9f78` using the
archived `brzero4-lines.asm` and its DWARF line annotations. No production
source or native LASP process was run.

The stop address is the end of the history copy/shift block (source
`broyden_module.f90:879–881,887`). The next block first constructs the history
matrix values at `0x6fa1db–0x6fa384`, source lines 908–909, and then completes
the weighted matrices at `0x6fa394–0x6fa802`, source lines 914–923. The first
downstream eigensolver is much later: `DGEGV` is called at `0x6fb198` (source
line 985/987 sequence), so this block is complete before eigensystem work.

## Recovered formulas

Let `DF[:,i]` be the normalized force-history columns, `U[:,i]` the stored
secant columns, and `G0` the elementwise diagonal array. For the active history
indices `i,j`, the block computes:

```
FINF[i,j] = q(DF[:,i], DF[:,j])
SMAT[i,j] = sum_l G0[l] * DF[l,i] * DF[l,j]
GMAT[i,j] = sum_l DF[l,i] * U[l,j]
```

`FINF` is written symmetrically at the paired source 908/909 stores. `SMAT` is
written symmetrically at source 920/921. The final source-922 store is the
non-symmetric DF/U contraction. The source-914 loop is explicit componentwise
multiply/add code; it does not call `inproduct_`. In contrast, the source-908/909
value comes from the call at `0x6fa1e5` to `inproduct_`. Address `0x6fa1db` is
only the preceding `mov rdi,rbx` argument setup, not the call instruction. The
helper's recovered definition is the per-XYZ block-sum bilinear form
`q(a,b)=sum_atom(sum_xyz a_xyz)(sum_xyz b_xyz)`.

The loops use Fortran column-major storage with 8-byte scalar elements. The
allocation descriptors show 50-column histories and 50×50 matrices; the
`0x190` byte stride in the copy loops is 50 doubles. Thus the formulas above
are matrix entries, not Cartesian 3-vector reductions. The explicit weighted
contraction is Euclidean in the flattened component index only; it must not be
replaced globally by the degenerate `q` helper.

## Isolated dynamic check

`research/ga_ssw/probe_native_broyden_first_matrix.py` was run against the
archived LASP ELF (SHA256
`bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704`). The
probe performs one initial call and one valid second-call history update for
each of `NDIM=3,6,9`, with nonuniform random `g0`; it stops at `0x6fa815`,
after the GMAT store and before the later eigensolver/DGEGV. Allocation,
memcpy, memset, and printing are hooked only to create isolated native memory
state; the arithmetic and `inproduct_` execute from the ELF. The complete
report is `research/ga_ssw/evidence/native-broyden-first-matrix/result.json`.

All three cases pass. The largest absolute error is `2.22e-16` for FINF and
`1.11e-16` for the prefix/history state; SMAT and GMAT agree to machine
precision (largest GMAT error `5.55e-17`). This dynamically validates the
one-column formulas above, including the distinction between the helper-based
FINF and explicit flattened contractions. The probe did not construct a
two-column saved history, so no two-column recurrence claim is made.

## Evidence boundary

The address/line pairing and pointer targets identify the three output arrays,
and the instruction classes identify the two explicit dot products and the
single helper call. This is sufficient to state the matrix construction above,
but does not recover the later `AMAT/BETA/BETAQ` recurrence, DGEGV spectral
selection, or history removal. A complete port still requires those later
blocks and a multi-update native replay.


Later-stage correction (2026-09-12): the first-matrix stopping point precedes
further GMAT arithmetic/sign changes. A final complete-call GMAT is not simply
DF.T@U. The new mixed-LAPACK full-call probes recover actual history recurrence
and deletion, but do not turn this one-column construction check into a complete
spectral-matrix identity. See native-broyden-dgegv-spectrum-audit.md.
