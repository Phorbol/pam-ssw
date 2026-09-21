# Native unbiasedrot anchor and pre-rotation parameters

Static trace only; no LASP/PES execution.

## Anchor copy

The post-assignment path at `0x5c45fe` does not directly write the saved
anchor. If the flag at `object+0x17a0` is set, `0x5c4616` forms the destination
`object+0x17e8`, while `0x5c4624` forms the source descriptor address
`object+0x1788` (`n0`, per `analysis/kernel-dwarf-member-offsets.txt:302-307`).
The earlier call at `0x5c40ca` passes this `+0x1788` storage by reference as
the `cbd_rotation` direction argument `n`; the companion `+0x1848` storage is
the `n1` argument. `rotate_dimer` updates `n` through that reference during
the presweep and its terminal handling can roll it back to evaluated `n00`.
Therefore the `n0` label does not mean an untouched random vector here.

`0x5c4650`–`0x5c4654` call `for_realloc_lhs`, passing the destination and
source descriptors. The following loop uses the reallocated pointer at
`object+0x17e8` as destination (`0x5c469f`) and the current `n` data pointer
at `object+0x1788` as source (`0x5c46ae`), with the byte count derived from
the saved dimensions. The loop repeats at `0x5c471e`–`0x5c472c`.

Thus the saved anchor supplied later to `biasedrot` (`object+0x17e8`, as
documented in `analysis/native-rotation-spec.md`) is a copy of the direction
state present after the presweep writeback/rollback path, not demonstrably the
original random `N0` and not necessarily the instantaneous final direction
unless that is the state at this copy point.

## CBD_PreRot controls

The DWARF member map identifies:

| para offset | DWARF field | use in `unbiasedrot` |
|---|---|---|
| `+0x2dc5c` | `rotmaxstep_prerot` | loaded at `0x5c3ff1` |
| `+0x2dc70` | `rotftol_prerot` | loaded at `0x5c3ff9` |

`readsswpara` passes these exact addresses to the input reader at
`0x6883ce` and `0x6884d1`; the associated input strings are
`SSW.RotMaxStep_preRot` and `SSW.Rotftol_preRot` (ELF literals
`0x4a49fcc` and `0x4a49ff4`). The archived `allkeys.log` records values 5 and
1.0000 respectively (`runs/water-archive-validation/2/allkeys.log:83,85`),
but this is parsed-run evidence, not proof that those are compiled fallback
defaults. No separate compiled-default assignment was recovered in the
available static material. CBD has corresponding controls in the output
(`CBD.RotMaxStep_preRot=5`, `CBD.Rotftol_preRot=1.0000`), but that does not
establish a different source or override for the SSW fields.
