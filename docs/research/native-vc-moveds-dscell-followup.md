# Native VC `moveds_`: `ds_cell` data chain

2026-09-11. Static, zero-PES audit of
`ssw_crystal_basic_mp_moveds_` (`0x5eccf0`) in the uploaded ELF. This is a
bounded address/formula audit; it does not run the native main loop and does
not identify `ds_cell` with the paper's `.15 ||L||_F` rule.

## Scalar selection and immediate scaling

The entry first obtains a scalar from the preceding mode-dependent reduction
(`0x5ecd56--0x5ece2d`) and saves it as the local move scale at
`0x5ece36`. For the ordinary VC path (`object+0x1660 != 1`), the decisive
selection is:

```text
if object.lcellmove bit 0 at 0x5ece4f:
    s = para.ds_cell      # load at 0x5ece5f, offset +0x2db18
else:
    s = para.ds_atom      # load at 0x5ece69, offset +0x2db08
```

If `para+0x2db10` (`lrandom_ds`) is enabled, the selected scalar is multiplied
by the saved random factor at `0x5ece82--0x5ece91` and clamped by `max(...,
constant)` at `0x5ece92--0x5ece9e`. No lattice length, cell-volume, square-root
of the number of coordinates, or Frobenius norm occurs in this selected
scalar path.

## Coordinate update arithmetic

The trajectory-record descriptors are loaded at `0x5edb49--0x5edb8c`; their
data pointers/lengths and record strides control the loops. In the scalar
remainder of the first movement loop (`0x5edd81--0x5edd93` and
`0x5edd99--0x5eddaa`), the native operation is componentwise:

```text
trial_component = source_component + s * direction_component
```

The vectorized and later equivalent loops use the same multiply/add pattern,
including `0x5ef1bf--0x5ef1cf`, `0x5ef35e--0x5ef36e`, and
`0x5ef886--0x5ef896`. The inspected arithmetic applies the same scalar `s` to
whatever vector descriptor the selected trajectory branch supplies. It does
not itself normalize a 3N+9 vector or rescale cell components by a lattice
norm. The available slice does not prove that every VC branch supplies a
single combined 3N+9 vector, so the exact dimension coverage remains a
descriptor/branch question.

## Guards and limits

The visible pre-move reduction uses absolute values of descriptor components
at `0x5ecd8c--0x5ecdfc`; this is a scale/diagnostic reduction and is not a
proof of a cell-norm conversion. The subsequent movement code contains
trajectory validity and copy/repair branches, including descriptor-size
conditions at `0x5ed048`, `0x5ed320`, and their corresponding later paths.
Those branches lead into repair and communication callbacks; no closed
VC-specific volume or strain bound was identified in this data chain. The
`present_tooshort` and retry contract documented for fixed-cell `moveds_`
cannot be transferred here without following these VC callback paths.

Consequently the strongest supported formula is `trial = source + s*d`, with
`s=ds_cell` when `lcellmove` is set and `s=ds_atom` otherwise, optionally
modified by the native random-ds factor/floor. This is a native scalar-step
result, not evidence for `.15 ||L||_F`, and not evidence that the scalar is a
norm of the nine cell components.

## Upstream cell-pattern and random-mode boundary

The CSSW table resolves the cell-pattern helper at `+0x210` to
`findmodepattern_cell_` (`0x608520`). In the inspected body it reads only
`para+0x2db7c` (`0x60852b`) and writes a fixed coefficient pattern into the
output descriptor: the two branches place `0.8/0.2` or `0.2/0.8` at offsets
`+0x18` and `+0x28`, while clearing the intervening vector slots
(`0x608538--0x60858f`). There is no cell matrix, lattice length, volume,
square-root, or norm operation in this helper. Thus these constants are mode
pattern coefficients, not evidence of a physical cell-length normalization.

The subsequent generator is the table slot `+0x1b0`,
`gen_randommode_` (`0x5ff750`). Its inspected entry saves the incoming
descriptors (`r15`/`r14` at `0x5ff75c--0x5ff76a`), reads descriptor metadata
including `[r14+0x48]` (`0x5ff83d`), obtains a random scalar through
`rd_numb_` (`0x5ff835`), builds a local vector descriptor
(`0x5ff873--0x5ff8c8`), and performs the random-vector writes/scales in
`0x5ff941--0x5ffa3a`. A later direct RNG call is at `0x5ffbf2`.
Within this inspected entry slice there is no explicit multiply by a lattice
matrix or division by a lattice norm before the vector reaches the storage
loops. The ELF also contains calls to `generate_ncell_local_` at
`0x6076ef` and `0x607bda`, but their semantic connection to this generated
vector is outside this bounded chain.

Therefore the two chains provide no evidence that `ds_cell` is pre-multiplied
by a lattice norm, or that the cell direction is divided by such a factor
before `moveds_`. The supported downstream equation remains
`trial = source + ds_cell * d_cell` on the cell branch, with the direction
metric and any upstream Cartesian/cell conversion unresolved. This must not be
identified with the paper's `.15 ||L||_F` rule. The finding also does not prove
that `gen_randommode_` completely replaces a prior direction: `update_mode0_`
assembles persistent-state-dependent input before this call, as recorded in
`native-cell-direction-lifecycle-followup.md`.

Sources: the uploaded ELF disassembly at the addresses above,
`analysis/selected-dwarf.txt`, and `docs/research/native-moveds-scale.md` for
the separately audited fixed-cell path. No production source, MAINLINE, PES,
GPU job, or native main execution was changed or run.
