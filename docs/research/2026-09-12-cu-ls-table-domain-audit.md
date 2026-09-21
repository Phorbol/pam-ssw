# Cu LS table domain audit

This is a read-only, zero-PES audit of the release ELF
`/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp`
(SHA-256 `bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704`).
The question is why the Cu55 native-LS arm has no initialized Cu--Cu bond.
No input was changed and the LASP main program was not run.

## What initializes the native LS table

`bond_info_init_` is present in
`research/ga_ssw/evidence/native-ls-response/ls-bond-info-init.asm` at
`0x6c6a30`. Its no-custom-file path builds the element list, calls the leaf
energy lookup at `0x6c9dd2 -> bondeneval_ (0x6cb660)`, and calls the length
lookup at `0x6c9fd3 -> bondlenval_ (0x6cc0b0)`. The returned length is then
multiplied at `0x6ca023` by a per-element-pair entry reached through
`para+0x16e50` (the recovered field name is `bond_lengthen`). The corresponding
energy path multiplies an energy filter at `para+0x188` (`0x6c9e3b`) and
other normalization factors before storing the runtime energy list. Thus a
leaf `bondlenval_` return is not itself the final `bond_len_list` value.

The initialization audit records the default `bond_lengthen` H/C entries as
1 and the default `bond_filter` entries as 1. The runtime length tolerance is
`bond_info_mp_len_toller_` at `0x5520568`, initialized to `0.1` in the normal
path (`0x6c9efb..0x6c9f0c`); the custom-file path can replace it
(`0x6c74c6..0x6c74fd`).

### Bounded audit of `bond_lengthen` initialization

The DWARF member map identifies `para+0x16e50` as `bond_lengthen`
(`analysis/kernel-dwarf-member-offsets.txt`, line 32). In the available
`readsswpara_` disassembly, the pair-table loop at `0x68625d..0x686303`
writes the table at `para+0x188` (the `bond_filter` field); the surrounding
parser code has no store to `para+0x16e50`. The same bounded search of
`ssw_options_` found no such store. The only `0x16e50` reference in the
audited initializer is the consumer `lea 0x16e50(%rdi,%r8,8)` at `0x6ca01b`,
followed by the multiplier load at `0x6ca023`.

The static release image gives a stronger default-state result than the
earlier H/C-only wording: `para` is `0x53ed7a0`, so the field begins at
`0x54045f0` (file offset `0x4e045f0`). Reading the 108 x 108 double region
there (11,664 entries, the same pair-table extent used by the initializer)
returns exactly `1.0` for every entry. This includes the Cu slot in the
static default image, so “Cu default is 1” is supported as an image-level
initializer fact. It still does not prove the post-parser runtime state.

This closes a narrower negative result: no element-specific Cu setter or
parser keyword for `bond_lengthen` was found in the inspected parser path.
`pot_bond_input.txt` is a
separate custom pair-table path (`0x6c6c13..0x6c6cc2`); the audited code
shows it replacing the energy/length lists, but does not show it populating
`bond_lengthen` or expose a complete custom-file schema. Thus Cu=1 remains
the static default; only post-parser mutation remains unknown.

`bond_info_init_` first obtains the element list and maps pair indices through
`ssw_commsub_mp_symbol_` (`0x6c909f..0x6c909a`). The element list is therefore
not a universal H/C-only table. The recovered extended isolated leaf oracle
(`research/ga_ssw/probe_native_ls_pair_table_extended.py`) gives Cu--Cu raw
length `1.8750000000` and raw energy `3.6298000813`. The audit of that oracle
shows these are the generic fallback branch for Cu, not a Cu-fitted energy or
length table. The same fallback length is returned for several non-dedicated
metal pairs. The dedicated species-radius function is only a leaf source for
this lookup; it does not prove an LS neighbor is created.

The only user pair-table override visible in this initializer is
`pot_bond_input.txt`: the filename literal and open/read sequence occur at
`0x6c6c13..0x6c6cba`, followed by reading `num_atom_type` and allocating the
energy/length lists (`0x6c6cc2` onward). No such file is present in the
archived metal examples or the Cu55 campaign inputs. The example
`TYPE0-Au10Ag10/configure.non` sets `BLLimit=default`, but that is the GA
structure-generation bond-length setting; it is not proof of a native LS
pair-table override. Its `lasp.in` selects `Run_type 5` and does not provide
`pot_bond_input.txt`.

## Which cutoff is actually used

The native bond counting path is independently documented in
`docs/research/native-ls-fixed-atom-audit.md`. In `pot_bond_mod_mp_bond_counter_`
and `selfadapt_nbondcounter_`, after the fixed-endpoint checks, the geometric
condition is strict `distance < bond_len_list(type_i,type_j) + len_toller`.
The corresponding initialization calls are visible at
`0x6c7ef2` and `0x6c9d1e`, with the returned bond count saved at
`0x6c9d35`. This is the LS `actual_bonds`/normalization denominator path; the
energy filter is separate and does not remove a pair from the distance count.

For the raw Cu fallback, the resulting default-image cutoff is
`1.875 + 0.1 = 1.975 Å`. The Cu55 campaign input has nearest Cu--Cu distance
about `2.5526 Å`, so zero Cu--Cu neighbors follows from the static default
image and strict cutoff, subject to any unobserved post-parser mutation. This
is not a basis for changing the table to 2.9 Å. The campaign evidence labels
`3.6298/1.875` as generic fallback, and the absence of a Cu-specific table is
the actual domain limitation.

## Evidence boundary

Static evidence closes the following points: element-pair lookup is used to
populate the runtime lists; the normal tolerance is 0.1; the length list is
scaled by `para+0x16e50`; custom `pot_bond_input.txt` can supply table data;
and actual bond counting uses strict `distance < length+tolerance` after fixed
endpoint checks. It closes the static default `bond_lengthen` image as all
ones, including Cu, but does not close the post-parser runtime value, the
complete parser schema/format for a custom file, or whether an unobserved
caller mutates the lists after initialization. The
archived input's `BLLimit=default` cannot fill those gaps. Therefore the Cu55
native-LS failure is a valid “no eligible Cu LS pair under the supplied raw
fallback” observation, not evidence of a native Cu table or of a search
algorithm disadvantage.

Root independent byte-read confirmation is saved in
`research/ga_ssw/evidence/cu-ls-lengthen-static-20260912.json`
(11,664 doubles, unique value 1.0; zero PES calls).

## Consequence for the comparison matrix

Keep Cu55 as a metallic SSW/optimizer/GA case, but do not treat the present
raw-fallback LS arm as an eligible LS performance baseline. The local full
LS supporting information `literature/ct4c01081_si_001.txt`, section 7.7,
explicitly includes an Fe--Fe `SSW.soft.bondFilter` block for its Fe7C3
example. That is source evidence that pair selection is an intentional part
of a material-specific LS input, not evidence for importing that input to Cu.
The strong-bond LS question remains testable on the already sourced H/C/O
cases without expanding metallic pair distances to fit a failed run.
