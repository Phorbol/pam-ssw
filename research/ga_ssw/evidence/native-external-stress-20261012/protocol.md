# Native external stress-transfer diagnostic (prepared only)

## Question and scope

This one-case CPU diagnostic asks whether an ASE full stress tensor returned
through LASP's external potential reaches the native `class_struc` stress
consumer with the expected cell and component layout. It is an interface and
geometry-gradient handoff check, not a variable-cell search benchmark, a Cu
materials result, or evidence for native/Python search parity.

Use one periodic four-atom conventional fcc Cu cell from ASE (`a=3.6 Å`),
deformed once by the preregistered matrix below. The diagonal entries impose
three distinct small strains and the off-diagonal entries add a small shear.
This is a fixed diagnostic starting geometry; it is not fitted to an outcome.

```
F = [[1.0080,  0.0020,  0.0000],
     [0.0000,  0.9930, -0.0015],
     [0.0005,  0.0000,  1.0120]]
cell_new = cell_fcc @ F.T
positions_new = positions_fcc @ F.T
```

The potential is ASE EMT. Each external request is evaluated at its current
cell and positions, with periodicity on all axes and no constraints. The
private callback returns `E`, Cartesian forces, and all nine components of
`-ASE_stress` in Fortran column-major order in `external.ene`. The negative
sign is an explicit diagnostic hypothesis, not an established convention:
the native external-potential path subsequently applies its own sign/conversion
steps. Runtime observations must decide whether this input reaches the native
consumer as expected. The key `external.dedcell F` is frozen in `lasp.in`; the
GDB observer records its parsed global value and the later reset/write boundary.
If the effective flag is not false, classify the tensor-path question as
unresolved; do not infer it from the input text.

## Observations

The socket callback logs the exact current cell, positions, fresh calculator
evaluation count, `E/F`, ASE full stress, and the signed nine-value tensor
written to `external.ene`. A read-only GDB observer records the effective
`external_dedcell` flag at the parser return and reset boundary, the nine
legacy tensor values at `otherpot_` return, and the exact class object tensor,
cell, inverse cell, volume, and existing `dedlatt` at entry to
`class_struc_mp_stress2dedlatt_` (`0x59cb10`). Its ABI uses the already verified
Fortran descriptor in RDI, with the object pointer at `*(uint64_t *)RDI`;
known object fields are read only at offsets documented by the existing
`probe_native_cell_force.py` oracle. No unknown descriptor layout is guessed.
The observer exits after the third stress-consumer entry. If that entry is not
reached inside the limits, the result is incomplete, not a reason to change
parameters or retry.

Correlate callback and native events by the latest flushed callback request
captured at `otherpot_` return and the immediately following class event;
require the request IDs to agree. The class cell/celli/volume are recorded as
cell-state context, but class atom coordinates are not read, so equality of
full geometries is not claimed. Report raw tensors and both row-/column-major
matrix views before attributing sign, units, or component order. A match of the class tensor to
`+ASE_stress` at the same cell would support only this external/EMT handoff;
it would not qualify the native stress producer generally or prove a cell
gradient. Because a physical stress tensor is symmetric, this one case cannot
distinguish a pure row/column-major transpose; retain that limitation even if
the nine scalar values agree. A mismatch, absent parsed flag, missing callback stress, or
unreliable object read leaves this prerequisite open.

## Bounds and stop conditions

- One arm; at most 20 external EMT energy/force/stress calculations total.
- One CPU and a 5-minute Slurm allocation; native process timeout is 120 s.
- No retry. The callback denies requests after the 20-call cap and preserves
  the denial in its log. The supervisor reaps descendants.
- The GDB run terminates at three class stress-consumer entries, or earlier on
  a timeout, bad descriptor, absent/incorrectly parsed tensor state, callback
  failure, or resource cap. Preserve the incomplete artifact and stop.
- No binary patch, debugger memory write, initialization/date alteration,
  GPU use, new public bridge/API, or full finite-strain sweep.

## Evidence basis and limits

The frozen binary is `/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp`, SHA256
`bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704`.
Existing `native-stress-backend-boundary.md` documents the nine-double
`external.ene` tensor read, `external.dedcell` conversion branch, and remaining
uncertainty about final stress sign/units and transfer to the class object.
Existing `native-cell-force-contract.md` and
`research/ga_ssw/probe_native_cell_force.py` establish the
`stress2dedlatt` entry and descriptor/object field offsets. The read-key and
reset instruction slices at coordination commit `deb93e4` identify the parser
key and flag reset/write instructions. The legacy `otherpot_` tensor address
uses `module_str_mp_strt_` base `0x53e88e0` and offset `+0x110`, as recorded in
`native-vc-gap-review.md`; it is not a same-offset read from the class object.
These are static/runtime-contract references; they do not predetermine this
diagnostic's result.

The ASE EMT calculation is a small real calculator-backed geometry check, but
the four-atom strained cell is not a scientific Cu benchmark. No claim is made
about convergence, physical accuracy, finite-strain consistency, or production
cell relaxation. Those require a separately reviewed experiment after this
interface boundary is closed.

Preparation canonicalizes the cell and atom positions together before ARC
serialization, because ARC stores only lengths/angles. The observer also
records the existing external-pressure field; it is not inferred from omitted
input keys. This probe does not yet validate the full enthalpy gradient.
