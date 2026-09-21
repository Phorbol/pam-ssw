# Native VC allowed-DOF and lcellmove consumer audit

2026-09-11. Static audit of the uploaded ELF only; no LASP main or PES.

`get_random_mode0_` stores the scheduling result in `object+0x2260`, which
DWARF names `lcellmove` (offset 8800). The first concrete consumer is
`ssw_crystal_basic_mp_set_status_`:

- At `0x5e8d6a`, it tests bit 0 of `object+0x2260`.
- If set, `0x5e8d77–0x5e8dfc` writes the status string beginning `CBD_Unb...`.
- If clear, execution takes `0x5e8e49` and the alternative status path.

Thus `lcellmove` selects a status branch; it is not evidence that the biased
optimizer itself freezes or releases particular coordinates.

The first explicit atom/cell termination split is in
`ssw_crystal_basic_mp_climb_convg_` (`0x5f4360`). `r14` is materialized as
`para` at `0x5f44c9`, and the current counter is held at `[rbp-0x110]`.
When the control result has bit 0 (`0x5f4a6b–0x5f4a72`):

- `0x5f4a78–0x5f4a94` compares `para+0xf4` (`ng`) with the current counter;
  equality plus `lcellmove == 0` (`test object+0x2260`, `0x5f4a8d`) sets the
  stop mask to `-1` via `cmove` at `0x5f4a94`.
- `0x5f4a98–0x5f4ab8` compares `para+0xf8` (`ng_cell`) with the current
  counter; equality plus `lcellmove != 0` sets the same stop mask via
  `cmovne` at `0x5f4ab4`.

This closes the native distinction between the atom count `ng` and cell count
`ng_cell`: the former terminates when the selected mode is atom-only, while
the latter terminates when the selected mode is cell-moving. It still does not
prove that all optimizer coordinates are constrained during biased rotation.

`ssw_commsub_mp_setconstraints_crystal_` (`0x5816b0`) is a separate constraint
construction routine. Its caller in `newssw_basics_mp_rotate_dimer_` passes
`rdi=r14`, `rsi=[rbp-0xa0]`, `rdx=r12`, and `rcx=[rbp+0x38]` at
`0x6e7de5–0x6e7df6` (and a second call at `0x6e7eba–0x6e7ecb`). The routine
reads the crystal axis globals `NX/NY/NZ` and rotation mode globals, packs
axis/mode bits, and stores the resulting constrained vectors in its static
`AXISATOM`, `COM`, and `V` arrays (`0x583307–0x58361c`). It also reallocates
working arrays at `0x583af2` and `0x584049`. This proves where crystal
constraints are constructed and consumed by the dimer rotation path, but the
available static slice does not establish a complete per-coordinate mask for
the VC biased optimizer.

`lcellmove` therefore has a verified scheduling/status and termination role;
the claim “biased VC relax freezes all atomic DOF” remains unsupported without
the downstream rotation/optimizer coordinate map.
