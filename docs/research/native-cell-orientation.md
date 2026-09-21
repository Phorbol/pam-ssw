# Native physical-cell orientation from frac2cart

2026-09-10. Static-only bounded followup to
[native-celli-producer.md](native-celli-producer.md) and
[native-cell-force-contract.md](native-cell-force-contract.md). No native main,
PES, protection code, or frozen experiment was executed or modified.

The physical `cell` array interpreted in native Fortran column-major order has
**lattice vectors as columns**. Thus for the same Cartesian frame and ordered
lattice basis it corresponds to **A = L.T**, where L is ASE's row-cell matrix.
This is obtained from coordinate multiplication rather than the variable name.

`class_struc_mp_frac2cart_` (entry **0x59c310**) zeroes a Cartesian output
workspace. Its outer loop index RSI starts at zero (0x59c393), runs over three
fractional components, increments the cell base RDX by **24 bytes** each pass
(0x59c88f–0x59c89a). On the scalar atom path (0x59c80f–0x59c88d), it:

- loads cell elements at base+0xe0, +0xe8, +0xf0 into XMM0, XMM1, XMM2;
- loads one fractional scalar per atom/component at 0x59c840;
- multiplies that scalar by all three cell entries (0x59c84f–0x59c85a);
- accumulates into the atom's three consecutive Cartesian slots
  (0x59c85e–0x59c87f).

The outer 24-byte increment selects consecutive **columns** of the stored 3x3
Fortran cell. The arithmetic therefore computes `r = A @ fractional`, with r a
Cartesian column vector. The vectorized paths use the same cell triplets; the
scalar path alone already fixes the orientation. After all three components,
the routine copies the Cartesian workspace into the object's cart array at
+0x170 (0x59c93a–0x59c997). Fractional input uses the field at +0x418 with its
array descriptor. Existing DWARF field evidence identifies these cart/frac/cell
members; the conclusion does not require resolving a caller's virtual table.

## Consequence for the previous force oracle

Combining this orientation with the already established `celli = A^{-T}` gives

```
native_dedlatt = -V (stored_stress+pI) A^{-T}
                = -V (stored_stress+pI) L^{-1}.
```

When converted to the ASE row-cell array layout, the corresponding matrix is

```
native_dedlatt.T = -V L^{-T} (stored_stress+pI).T.
```

**Conditionally**, if stored stress is symmetric and uses the ASE tensile-positive
convention, this is exactly the negative of our positive work-conjugate
row-cell gradient `V L^{-T}(stress+pI)`. Thus there is no matrix-order discrepancy
under that condition. The explicit conditional is essential: this task does
**not** establish the native stress producer's sign, units, or possible
calculator-specific conversions. Nor does the variable name dedlatt establish
whether its downstream consumer expects a gradient or a force.

No extra physical lattice rotation or basis transformation was traced: `A=L.T`
refers to the same ordered vectors and Cartesian frame. Runtime dispatch and
all cell-update paths remain unproven. The scalar branch gives a narrow direct
orientation closure; following stress producer dispatch would require a separate
bounded task and is not needed to claim this result.

Evidence: [complete frac2cart instructions](native-cell-orientation-evidence/frac2cart.asm).
ELF/field provenance is inherited from the two linked contracts above.
