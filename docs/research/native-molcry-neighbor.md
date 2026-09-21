# Native half-bin neighbor dispatch and molecule image handling

2026-09-11. Static audit of the archived LASP/stock LAMMPS ELF
`/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp`,
SHA256 `bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704`.
No LAMMPS or PES execution was performed.

## Dispatch

The ordinary implementation is
`Neighbor::half_bin_newton_tri(NeighList*)` at `0x1011f30` (size `0x7f0`);
the custom implementation is
`Neighbor::half_bin_newton_tri_molcry_only(NeighList*)` at `0x1012720` (size
`0x9b0`). Both are assigned as function pointers by
`Neighbor::choose_build(int,NeighRequest*)`.

The relevant dispatch is `0x1004040–0x10040ab` and
`0x100413b–0x100414d`. Under the preceding request/neighbor flags, the value
at `neighbor_object+0x39c` is tested: value 1 stores `0x1012720` (custom),
while the corresponding value-1 path stores `0x1011f30` (ordinary). The
machine code does not expose the source-level name of `+0x39c`; it is therefore
reported as a dispatch field, not called a “molcry” or image flag
without the surrounding source contract. Two other observed sites (`0x454bc4`
and `0x4ad9b4`) pass the address of this field to an indirect input/parser
call; this bounded audit found no direct assignment establishing its runtime
value.

## Shared image-boundary test and special-bond encoding

In the ordinary routine, the image candidate reaches
`0x1012405–0x1012448`. For each enabled domain component at neighbor-object
offsets `+0xb4,+0xb8,+0xbc`, it masks a displacement component with the
sign-mask constant at `0x4aa1800`, then compares it with the corresponding
neighbor/domain values at `+0x108,+0x110,+0x118`. A strict `ja` diverts to the
non-image output path. If the image selector is positive, the output index is
encoded as `atom_index XOR (image_selector << 30)` at `0x101244a–0x1012458`;
otherwise the atom index is written unchanged (`0x1012462`).

The high bits here must not be called image tags. The local LAMMPS header
defines `SBBITS=30`, reserves the two highest neighbor-list bits for special
bonds, and defines `NEIGHMASK=0x1fffffff` (`/tmp/pam-ssw-lammps-20260910/lammps/include/lammps/lmtype.h`).
Thus this XOR is the ordinary special-neighbor class encoding; the image
boundary comparisons and special-bond class are separate mechanisms.

The custom routine has the same three component checks at
`0x1012d85–0x1012dd0`, using the same `0x4aa1800` sign mask and the same
neighbor offsets `+0x108,+0x110,+0x118`. This establishes that the custom
variant did not remove the ordinary three-axis boundary comparison.

## Custom molecule-specific special-bond branch

Before the shared three-axis checks, the custom routine tests additional
state at `0x1012d52–0x1012e78`: fields `neighbor+0x3fc`, `+0x400`, and
`+0x404` gate lists of atom IDs. A matching atom ID is detected by the loop at
`0x1012e15–0x1012e30` (and the analogous list loop immediately before it).
Matching entries are written with explicit special-bond class bits:

* `atom XOR 0x80000000`: `0x1012e7d–0x1012e90`;
* `atom XOR 0xc0000000`: `0x1012e95–0x1012ea8`;
* another accepted branch writes `atom XOR 0x40000000`: `0x1012dd6–0x1012de9`.

The custom equality path at `0x1012ead–0x1012ef2` additionally requires
componentwise equalities/inequalities against saved boundary values before
accepting the special-bond classification. The ordinary routine has a different equality
path at `0x10126bb–0x1012711`, comparing two displacement components and the
source index, but has no corresponding molecule-ID list/class-bit sequence in this
bounded excerpt.

## Scientific boundary

The ELF proves that the custom variant is selectable and has extra
molecule-ID-dependent special-bond classification while retaining three-axis
boundary checks.
It does not, by itself, prove which `+0x39c` value the XXXII stock run used,
nor does it reduce the observed 1–4 image crossing to a specific z-half rule.
The exact units and construction of the saved boundary values and molecule
lists remain unresolved. Thus the stock `+0.2403769 eV` jump cannot be called
a native bug from this disassembly alone. A valid follow-up must capture the
runtime `NeighRequest`/neighbor object fields and emitted encoded neighbor
indices for the special pair, then compare both selected implementations on
the same geometry.

Evidence excerpts:

* [choose-build-dispatch.asm](../../research/ga_ssw/evidence/native-molcry-neighbor/choose-build-dispatch.asm)
* [half-bin-functions.asm](../../research/ga_ssw/evidence/native-molcry-neighbor/half-bin-functions.asm)
* [half-bin-molcry.asm](../../research/ga_ssw/evidence/native-molcry-neighbor/half-bin-molcry.asm)
