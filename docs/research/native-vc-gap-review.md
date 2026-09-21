# Native VC parity gap review: stress transfer remains the blocker

**Later evidence update (2026-09-11):** the unresolved descriptor statements
in the original audit below are historical. Ordinary CSSW table/refresh/
stress2dedlatt dispatch is resolved in `native-vc-consumer-followup.md`;
the VC convergence boolean and scalar pressure producer are verified by
isolated original instructions in `native-vc-convergence-contract.md`.
Backend-specific stress provenance and full coordinate/reference lifecycle
remain separate questions. Do not reopen the already resolved descriptor slots.

2026-09-11. Bounded static audit of the independent ASE VC implementation and
the supplied native `lasp` binary. No native execution, PES evaluation, job
submission, or production code change was performed.

## Priority finding

The highest-consequence unresolved VC detail is the conversion of the backend
stress into the native crystal object's `str_basic.stress` field: its physical
sign, units, tensor ordering, and whether it is the same quantity consumed by
the cell-force routine. This directly controls the direction of every native
cell move and the meaning of its stress stopping criteria. The independent
implementation currently requires a different, explicit contract: the
callback returns symmetric ASE tensile-positive stress in eV/Å³
(`pamssw/standalone/vc_geometry.py:140-160`), and the cell gradient is formed as

```
gf = solve(deformation.T, volume * (stress + pressure*I))
```

at `pamssw/standalone/vc_geometry.py:161-166`, followed by the exact adjoint
of the matrix exponential. This is mathematically consistent for that ASE
contract, but native parity cannot be claimed until the producer is identified.

## What the existing native evidence closes

The direct routine `class_struc_mp_stress2dedlatt_` at **0x59cb10** was already
executed as an original-instruction oracle. Its stores beginning at
**0x59cd71** discriminate the stored-field formula

```
dedlatt = -volume * (stress + externaltp*I) @ celli
```

in native Fortran-column-major storage. The direct producer
`class_struc_mp_recicell_` at **0x59bc00** passes `cell` and `celli` to
`ssw_commsub_mp_reci_latt_` at **0x578480**, whose arithmetic establishes
`celli = cell^{-T}` for that storage convention. The `scart2cart` routine at
**0x59e610** also shows direct physical cell-entry writes, with no logarithmic
strain or scalar metric factor in the inspected block. These facts rule out
calling the Python six-component log-strain chart a recovered native formula.

## What remains open despite DWARF/disassembly

The crystal walker calls an initialized descriptor indirectly at
`ssw_crystal_basic_mp_update_forcepara_` **0x5e4405**, descriptor slot `+0x48`,
after which it reads the stress diagonal at object offsets `+0x128`, `+0x148`,
and `+0x168`. The displacement walker calls another unresolved descriptor at
`ssw_crystal_basic_mp_moveds_` **0x5edea3**, slot `+0xe8`. Existing DWARF names
the fields (`stress` +0x128, `cell` +0xe0, `celli` +0x388, `dedlatt` +0x3d0,
`scart` +0x478), but does not identify the initialized descriptor target or
the backend stores feeding `stress`.

The separate `maxstress` diagnostic at **0x5e451b–0x5e4596** is known to be
`abs(stress_xx + stress_yy + stress_zz + 3*externaltp) * eva3togpa / 3`.
That closes a scalar diagnostic conversion, not the tensor producer or the
cell-gradient consumer. Likewise, the inspected LAMMPS extraction path has a
backend-specific intermediate sign/unit operation, but it does not prove the
final `str_basic.stress` convention for the intended NN or external backend.

Therefore the current evidence does **not** close the one parity-critical
question: whether native `stress` is ASE tensile-positive eV/Å³ (and hence
whether the Python pressure term and cell-gradient sign can be compared
directly). It also does not recover the native cell-coordinate metric or
`ds_cell` normalization. The next smallest useful native task is to resolve
one initialized descriptor for slot `+0x48` or `+0xe8`, then trace only the
stress stores and coordinate conversion to that target. Until that is done,
keep the Python `strain_length` and ASE stress contract as an explicitly
independent experimental design; do not copy native `ds_cell` or `strtol`
values into the Python configuration.


## Root follow-up: input transfer closed; distinguish NPT field reuse

The caller-side boundary is now narrower. `move_` loads r12=module_str_mp_strt_
(0x53e88e0) at0x4ed4c7. At0x4ed887–0x4ed894 it passes strt+0x110 as
the first stack argument to run_ssw_ (target0x51cba0, call0x4ed997).
Existing DWARF declares this seventh argument `stress`, located at
[rbp+0x10] in run_ssw_.

For the ordinary CSSW branch,0x53565c–0x535717 copies the input3×3
tensor elementwise into CSSW_A+0x128, without a sign flip, volume factor,
unit conversion or transpose. The corresponding crystal step is called at
0x535ddf. Therefore the remaining producer question is upstream of
`module_str_mp_strt_+0x110`; it is no longer necessary to guess the
stress pointer from an indirect force method inside the crystal walker.

A different use of the same CSSW_A field occurs in the NPT path:
setup calls npt_v::set_up_md at0x538495;0x5386ac loads CSSW_A.volume;
0x53872a–0x538774 stores -volume*input_stress into CSSW_A.stress.
The later npt_v2 call is at0x53904c. This is a separate MD-path tensor
conversion. Applying its sign/volume factor to ordinary VC-SSW would
conflate two uses of the field.

Raw routine/transfer evidence is in
`research/ga_ssw/evidence/native-stress-producer-review/`.
`cal_pes_` accesses the global strt tensor, not a class_struc object;
offsets with the same numerical value must not be confused across types.
A direct relative-call scan found no call to getstress_; its function
returns the vector pointer of compute `c_press`, but presence of that
helper alone does not prove it is the active backend producer.
This pass makes no new claim about the upstream stress sign or unit,
and changes no Python stress calculation.
