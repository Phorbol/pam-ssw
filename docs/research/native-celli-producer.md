# Native celli producer: inverse transpose of the stored cell

2026-09-10. Bounded static closure following `native-cell-force-contract.md`.
No PES, native main, initialization, expiry code or instruction modification was
used. This finding is a static instruction derivation, not a new executed oracle.

`class_struc_mp_recicell_` at **0x59bc00** dereferences the object, passes
`object+0xe0` (cell) in RDI and `object+0x388` (celli) in RSI, then tail-jumps
at **0x59bc19** directly to `ssw_commsub_mp_reci_latt_` (**0x578480**).
Thus this producer is not inferred from a symbol name or an indirect vtable.

Let the nine input doubles, in increasing address order, be `a,b,c,d,e,f,g,h,i`.
With the established Fortran-column-major interpretation the stored matrix is

```
A = [[a,d,g],
     [b,e,h],
     [c,f,i]].
```

The complete call-free routine (0x578480–0x57865c) constructs these output
columns:

```
C[:,0] = (e*i-f*h, f*g-d*i, d*h-e*g) / D
C[:,1] = (c*h-b*i, a*i-c*g, b*g-a*h) / D
C[:,2] = (b*f-c*e, c*d-a*f, a*e-b*d) / D
D = det(A)
```

For the first column, numerators are formed at 0x578484–0x5784e2;
its denominator is assembled at 0x578565, 0x578590–0x5785b2, divided at
0x5785c3, and the scaled values are stored at 0x578601, 0x57860f, 0x578632.
The other two denominators are equivalent determinant expansions in a different
floating-point order. Final column stores end at 0x578653. The numerator of
each reciprocal division is loaded from **0x4a43830**, whose bytes
`00000000 0000f03f` are IEEE754 **1.0**, so no `2*pi` multiplier is present.

These are the cofactor columns, hence **C = A^{-T}**, not `A^{-1}`.
Equivalently `A.T @ C = I` for a nonsingular input. The use of three separately
computed determinant expansions can introduce rounding differences, so this
identity is mathematical, not a claim of bitwise equality under floating point.
No singular-cell guard is visible in this routine.

Combined with the previous original-instruction force oracle, this closes the
stored-array formula to

```
stored_dedlatt = -V * (stored_stress + p*I) @ stored_cell^{-T}.
```

The consequence is that interpreting `celli` as a plain inverse would introduce
a transpose error for a general oblique cell. This does **not** yet fix the
mapping between the native stored cell and ASE row-cell, establish the stress
producer's sign, prove runtime dispatch to this producer at every update, or
resolve the reference block at object+0x7d0. Those remain separate questions;
no production gradient was changed on this evidence.

Evidence: [producer wrapper](native-celli-producer-evidence/recicell.asm),
[complete reciprocal routine](native-celli-producer-evidence/reci_latt.asm),
[constant bytes](native-celli-producer-evidence/constant.txt). ELF and field
provenance are the same as [the preceding contract](native-cell-force-contract.md).
