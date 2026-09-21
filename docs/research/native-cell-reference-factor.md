# Native cell reference factor: Gram matrix and Cholesky

2026-09-11. Narrow static and isolated-prefix evidence; no native main/PES
or production change. This extends the resolved ordinary CSSW dispatch.

DWARF member records identify `factxyz` at object+0x788, `factxyzb` at+0x7d0,
and saved copies at+0x818/+0x860. The previously unknown+0x7d0 field is thus
an explicitly paired transformation field, not a scalar cell-step parameter.
Its producer is `class_struc_mp_set_smat_` (0x5a3b90).

With native physical cell A holding lattice vectors as columns, the prefix
forms G=A.T@A. At0x5a3d3f it calls `matsqrt` with dimension3, the Gram matrix
and output object+0x788. The original-instruction prefix was executed for a
skew cell, a globally rotated copy, and an anisotropic diagonal cell, stopping
before that call. All input matrices match G to1e-12; full errors are in
`research/ga_ssw/evidence/native-stress-producer-review/smat-gram-prefix.json`.
The reproducer is `research/ga_ssw/probe_native_smat_gram.py`. This is not a
claim of having executed native LAPACK or the complete coordinate map.

Despite the name, `matsqrt_` (0x586e80) calls DPOTRF at0x586ee7, with UPLO='U'
(the byte at0x4ab8b7c) and N=3 (integer at0x4a44020). This identifies a
Cholesky-factor construction, not the symmetric spectral square root. The
standard DPOTRF contract is G=U.T@U for its upper-triangular output:
https://www.netlib.org/lapack/explore-html/d2/d09/group__potrf_ga84e90859b02139934b166e579dd211d4.html .
The wrapper's triangle cleanup and complete output require their own arithmetic
check before claiming whole-function equivalence.

At0x5a3d44–0x5a3d5c the producer passes factxyz to the already-recovered
`reci_latt_`, writing its inverse-transpose to factxyzb. `cart2scart`,
`scart2cart` and `update_sstr` consume the paired fields; the full affine
coordinate/force duality and reference-reset lifecycle remain the next narrow
questions. Merely knowing the field names does not establish those formulas.

Inference for Safe-total: a geometry-derived reference transformation is a
more concrete comparison target than guessing a scalar ds_cell from its name.
Cholesky of G preserves the cell metric up to an orthogonal factor; that
identity alone cannot improve a Euclidean optimizer or establish useful
preconditioning. Do not add a cell scaling parameter or replace the validated
log-strain chart until the actual mapping and solver implications are clear.
