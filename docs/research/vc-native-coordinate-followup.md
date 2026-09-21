# Bounded native VC followup: a stress diagnostic is now identified

2026-09-10. This static followup closes one formula left unidentified in
[the native VC gap review](vc-native-gap.md). It does not close the native
cell coordinate map, conjugate force, or stopping predicate. No native code
was executed, patched, or run through initialization/expiry checks.

## Closed assignment

In `ssw_crystal_basic_mp_update_forcepara_`, addresses
`0x5e451b–0x5e4596` (compiler line metadata: `Class_ssw_crystal.F90:385–386`)
compute the following assignment in terms of native stored fields:

```
control.maxstress = abs(stress(1,1) + stress(2,2) + stress(3,3)
                        + 3 * para.externaltp) * eva3togpa / 3
```

`eva3togpa` is 160.2176565. The scalar stored under the name `maxstress`
in this block is therefore an absolute mean normal-stress/external-pressure
mismatch in the converted units, not a maximum over tensor components.
The internal stress producer's sign convention is not established here;
the plus sign above is the actual stored-field arithmetic.

The evidence chain is compact and independently inspectable:

- [Field metadata](vc-native-coordinate-evidence/selected-fields.txt):
  `str_basic.stress` has offset 296 (`0x128`), followed by `cart` at 368;
  the three accesses at `0x128`, `0x148`, and `0x168` select the diagonal
  elements of this 3×3 double matrix. `externaltp` has offset 187192
  (`0x2db38`), and `maxstress` has offset 40 (`0x28`).
- [Register context](vc-native-coordinate-evidence/register-context.asm)
  identifies the parameter base and object load;
  [assignment instructions](vc-native-coordinate-evidence/maxstress.asm)
  show the three additions, parameter term, absolute-value mask,
  multiplication, division and store. The control base and the named
  `constants_ssw_mp_eva3togpa_` symbol are visible in that block.
- [ELF constants](vc-native-coordinate-evidence/constants.txt) preserve
  section address/file offsets, the values 3.0 and 160.2176565, and the
  absolute-value mask whose low 64 bits are `0x7fffffffffffffff`.

This does **not** show that the walker ignores shear or that `strtol` uses
this scalar alone. DWARF also contains separate `maxlattf` and `maxf_scaled`
fields. Their consumers were not traced in this bounded task. Our fresh
physical full-tensor stress certificate cannot be equated to this diagnostic.

## What DWARF adds, and where it stops

DWARF gives `str` a size of 1680 bytes and names additional fields:
`dedlatt` at `0x3d0`, `frac` at `0x418`, `scart` at `0x478`, and `sfa` at
`0x4d8`. These names are useful anchors, but do not by themselves specify
cell-gradient sign, volume factor, lattice inverse, normalization, or finite
coordinate update.

The unresolved dispatches remain visible in the bounded excerpts:

- [Force update](vc-native-coordinate-evidence/force-dispatch.asm):
  descriptor load at `0x5e43fe`, call through slot `+0x48` at `0x5e4405`.
- [Displacement update](vc-native-coordinate-evidence/move-dispatch.asm):
  descriptor load at `0x5ede9f`, call through slot `+0xe8` at `0x5edea3`.

The available DWARF entries for methods such as `stress2dedlatt` and
`scart2cart` report member location zero. They do not resolve those dispatch
table slots. Assigning a method name from semantic plausibility would not
constitute recovered dataflow.

The next smallest native investigation, if needed, is to resolve **one actual
initialized class descriptor**, identify the function address in either
slot, and inspect only that target's coordinate/force conversion. A subsequent
coordinate/gradient oracle would need to exercise the resolved routine; no
such oracle was run here. Reading the 2014 VC-SSW paper and SI remains a
separate primary-source requirement before attributing a coordinate or
metric definition to the authors.

## Reproduction and boundaries

Binary:
`/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp`.
The original selected debug metadata is the sibling
`analysis/selected-dwarf.txt`; saved excerpts preserve its line numbers.
Each bounded disassembly artifact starts with its exact `objdump` command.
The register context comes from
`objdump -dl -Mintel --disassemble=ssw_crystal_basic_mp_update_forcepara_`.
Constants can be reproduced with Python `struct.unpack('<d', ...)` at file
offsets `0x4a46cf8-0x400000` and `0x54c0420-0x600000`; `objdump -h` verifies
those section mappings. The mask is the 16 bytes at
`0x4a46c60-0x400000`.

A static GDB type query encountered an internal `gdbtypes.c:2918` assertion;
existing DWARF excerpts were sufficient to finish the assignment above.
This debugger failure is not evidence of native protection or anti-debugging.

**Decision:** preserve the independent Python VC chart and its explicit
metric contract. This new diagnostic formula narrows the native gap without
justifying copying a native tolerance, changing the Python cell force, or
claiming native VC parity. No additional formula was pursued this round.
