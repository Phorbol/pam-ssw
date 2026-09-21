# Native cell force: original-instruction formula and coordinate evidence

2026-09-10. A bounded original-instruction oracle closes one important native formula:

```
dedlatt = -volume * (stored_stress + externaltp*I) @ stored_celli
```

Matrices here are the native Fortran-column-major arrays, not yet ASE row-cell quantities. This is a statement about stored fields. It does not presume that the producer's stress is ASE tensile-positive, or that the name `dedlatt` implies a positive energy gradient. The actual negative sign is now verified independently of the paper's naming/sign conventions.

## Original instructions and numerical discrimination

`class_struc_mp_stress2dedlatt_`, entry 0x59cb10, was executed from entry to its normal return in Unicorn using original uploaded ELF instructions and original constant pages. No function call is present in this routine, no runtime stub was needed, and native main/initialization/expiry/calculator code was never run. Inputs are constructed object data, not patched binary instructions. A 10,000-instruction and one-second per-call ceiling applies.

Four deterministic cases use arbitrary non-diagonal `celli`, symmetric non-diagonal stress, volume 23.7–26.7, and external pressure 0, +0.013, −0.009, +0.04. Three cell-mask integers at para+0x2dde0 are explicitly 1, so the unrestricted branch is tested. These are mathematical oracle fixtures, not physical crystal validation and not a claim about native defaults.

In all four cases the formula above agrees to **1.42e-14 or better**. Competing transpose/order formulas `-V*C@A`, `-V*C.T@A`, and `-V*A@C.T`, with A=stress+pI and C=celli, fail by 52.5–163.9 in at least one case. Thus the multiplication order is discriminated, not inferred from a diagonal-cell test. The raw matrices and every competing residual are retained in `research/ga_ssw/evidence/native-cell-force/result.json`.

Source metadata names stress at +0x128, celli at +0x388, dedlatt at +0x3d0 and volume at +0x608. The routine adds external pressure to diagonal stress entries, multiplies by celli and volume, flips signs with the original sign mask, and stores the nine dedlatt entries. Relevant stores begin at 0x59cd71. A later mask branch can zero selected components; it was not used or generalized in this oracle. No GPa conversion appears in this force formula, unlike the previously identified scalar `maxstress` diagnostic.

Evidence: [routine instructions](native-cell-force-evidence/stress2dedlatt.asm), [selected DWARF fields](native-cell-force-evidence/str-fields.txt). Reproducer: `research/ga_ssw/probe_native_cell_force.py`; run with `PYTHONPATH=/tmp/pam-ssw-unicorn-probe:. python research/ga_ssw/probe_native_cell_force.py`. The script refuses overwriting the evidence directory and checks exact ELF SHA256 before executing address-specific instructions. The ELF is `/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp`.

## Coordinate routine: what is established and what is not

The only other newly disassembled function was `class_struc_mp_scart2cart_` (0x59e610), [saved here](native-cell-force-evidence/scart2cart.asm). At source line 648 / 0x59e635–0x59e706, scart descriptor accesses using atom count N copy three consecutive 3-vectors indexed N+1, N+2, N+3 directly into the object's nine cell entries +0xe0 through +0x120. These stores have no exponential or scalar length multiplier in that block. Therefore **this routine's scart tail represents the cell entries themselves**, not our six logarithmic-strain variables.

For the atomic part, the routine next uses a nine-scalar object block at +0x7d0 through +0x810 in a matrix-vector operation (0x59e7a9–0x59e8a2), fills the `frac` array at +0x418, and later multiplies using the new cell entries and writes `cart` at +0x170. The parent object has additional fields beyond the basic 1680-byte `str`, so +0x7d0 must not be labeled an inverse reference cell without tracing the actual derived type/initializer. The visible sequence is consistent with affine remapping, but this task does **not** claim the complete formula for the atomic mapping or its reference-update lifecycle. The terminal method call at slot +0x40 is also not executed in this static inspection.

The earlier crystal walker has indirect conversion calls at descriptor slots +0x48 and +0xe8. Merely locating a static table containing one function pointer does not prove that this table is the runtime descriptor used there. We did not establish the initialized descriptor and therefore do not claim these caller slots have been resolved by this task. In particular a naive offset inference from one pointer can land on unrelated `arcinit_str`; semantic plausibility is not dispatch evidence.

## Implication for the block-cell implementation

There is now direct native evidence supporting a physical-cell-vector block and a pressure/volume/matrix-aware cell-force conversion. This is stronger than copying NG_cell/ds_cell labels or treating bare stress as a coordinate gradient. It supports deriving a block-cell implementation with explicit affine coordinate mapping and its exact work-conjugate gradient, while keeping the original stored-field force sign separate from a positive mathematical derivative.

Our joint log-strain chart remains an independently derived alternative: six symmetric coordinates with explicit L versus this routine's nine stored physical cell entries. Nine stored entries alone do not prove nine independent native degrees of freedom; rotations/constraints may be projected elsewhere. The next narrow closure, if needed, is the producer definition of celli/stress plus the actual initialized force/coordinate dispatch and the +0x7d0 reference block. No production force, scaling default, rotation treatment or scheduler was changed from the new oracle.

## Available larger TiO2 reference inputs

Already extracted 48-atom Ti16O32 structures, all **reported-phase geometries** from 2017 SSW-NN SI §7, not transition states:

- `literature/benchmark-sources/coordinates/brookite.extxyz`
- `literature/benchmark-sources/coordinates/phase-87.extxyz`
- `literature/benchmark-sources/coordinates/phase-139.extxyz`

Same-name ARC files retain source coordinate precision. Rounded SI energies belong to the original calculation and are not MACE targets. These inputs provide larger reference/starting structures for a separately specified experiment; availability alone is not PES-domain or search validation.

## Subsequent producer closure

The direct `recicell -> reci_latt` producer now establishes
`stored_celli = stored_cell^{-T}`, with no 2*pi factor. See
[narrow producer evidence](native-celli-producer.md). Combined with the original
instruction oracle, the stored formula is `-V*(stored_stress+pI)*stored_cell^{-T}`.
The native-to-ASE cell orientation and stress sign remain distinct unresolved
contracts; this update does not change the independently derived ASE gradient.
