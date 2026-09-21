# Native stress: backend conversion is explicit, final SSW transfer remains open

2026-09-10. Bounded static inspection only. No calculator, native main, expiry
code, MACE, or instruction oracle execution; no shared walker changes.

**The condition in `native-cell-orientation.md` cannot yet be removed.** The
stored-cell orientation and stress-to-cell-force arithmetic are established, but
this inspection does not prove `SSW_object.stress == ASE stress` for the uploaded
NN backend or all external backends. There is direct evidence of backend-specific
sign/unit operations, so inferring that equality from the name `stress` is unsafe.

## What the instructions establish

The legacy `otherpot_` routine at 0x4150a0 has a LAMMPS extraction path:

- 0x415ab2 passes `otherpot_$STRESSP` (0x5d71da0) as output descriptor.
  The constant target loaded at 0x415abc is
  `lammps_mp_lammps_extract_compute_dpa_` (0xd3bba0), called at 0x415af4.
  Its compute-ID string argument at 0x4abae84 is exactly **`peratom`**.
  This is a user/configuration-named compute array, not proof that its entries
  are already a global pressure tensor. The compute definition, component
  ordering, kinetic/virial content and reduction must be traced separately.
- 0x415d89 loads `module_pes_mp_lmpsunit_` (0x1d0c0a88). The loop starting
  0x415dfb flips its sign with the sign-bit constant and multiplies the returned
  array in place. The vectorized loop 0x415e30–0x415e81 likewise multiplies and
  flips the sign. Thus this **intermediate array** obeys
  `STRESSP_after = -lmpsunit * STRESSP_before` on this path.
  The sign-mask bytes at 0x4a312c0/0x4a312d0 confirm arithmetic negation.
  This is not yet a formula for the SSW object's final stress.
- The producer `otherpes_init_` contains string-selected unit branches.
  String constants 0x4a2ffe0 and 0x4a2ffe8 are `real` and `metal`.
  At 0x40d74f the real-unit branch stores the exact encoded double
  **6.32420918e-7** into lmpsunit; at 0x40d7c6 the metal-unit branch stores
  **6.24150919e-7**. Its energy factor `lmpeunit` is respectively
  **0.043364104299378356** and **1.0**, stored at 0x40d746/0x40d7bd.
  Other unit branches exist and are outside this narrow derivation. These
  statements concern initialization on those branches, not all runtime writes.
- A distinct helper, `getstress_` at 0x419738, calls LAMMPS
  `Modify::find_compute` then returns an object pointer at +0xc8. It does **no**
  numeric conversion. It is not the helper called by the extraction sequence
  above; the existence of its symbol is not evidence for an SSW call path.
- Another `otherpot_` branch tests `module_pes_mp_external_dedcell_` at
  0x417734–0x417741. When enabled, it copies the legacy global object's nine
  slots +0x110…+0x150 into +0xc8…+0x108 and directly calls `dedcell2stress_`
  at 0x4177d1 before rejoining at 0x416e47. The block at
  0x416ed9–0x416f3d negates all nine +0x110…+0x150 entries.
  This establishes a further convention adapter; it does not establish the
  external program's supplied derivative convention or complete branch dispatch.

These legacy `module_str_mp_strt_` offsets differ from the SSW derived object's
stress at +0x128. Offset coincidence alone is not a data-transfer proof.

## Exact remaining boundary

The already inspected crystal `update_forcepara` obtains the object descriptor
at +0x38 and calls its slot +0x48 at 0x5e4405. It subsequently consumes the
SSW object's stress at +0x128, +0x148 and +0x168 for the scalar diagnostic.
The initialized concrete descriptor, backend adapter, and transfer/conversion
from backend results into **that object's nine stress entries** remain unclosed.

Accordingly the unresolved tasks are specific: establish the actual initialized
slot +0x48 implementation for the intended backend; identify the final stress
stores and conversions; and for NN or external programs derive the raw tensor's
work-conjugate sign/units. The LAMMPS `peratom` compute also needs its actual
input definition and reduction, so neither its negative multiplier nor the
numeric conversion factor alone proves the final tensile-positive convention.
A bounded `cal_pes_` inspection reached the separate legacy global structure
and indirect calculator calls; it was stopped instead of treating that different
object's similarly numbered members as the target SSW fields.

Until this transfer is established, retain the conditional result:

`native_dedlatt.T = -V L^-T (stored_stress+pI).T`.

It equals the negative ASE row-cell enthalpy gradient **if** the final stored
stress is symmetric and ASE tensile-positive in the same units. No new result
here justifies changing the independently consistent Python E/F/stress math.

Evidence: `native-stress-backend-evidence/{otherpot.asm,otherpes_init.asm,
getstress.asm,constants.json}`. Original ELF/provenance is as recorded in
`native-cell-force-contract.md`; only original disassembly and raw constants
were read. This is static source-grounded evidence, not a runtime backend test.
