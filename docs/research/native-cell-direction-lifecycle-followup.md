# Native CSSW cell-mode lifecycle: resolved dispatch slice

2026-09-11. Zero-PES static audit of the uploaded `lasp` ELF
(`bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704`). This
follow-up resolves the previously unknown type-bound slots; it does not run
the LASP main loop or claim complete trajectory parity.

## Descriptor and slot identity

The ordinary CSSW descriptor constructed by `run_ssw_` stores
`0x53cacc0` as its type-bound procedure table at `0x535d28--0x535d2f`, then
passes that descriptor to `ssw_crystal_basic_mp_ssw_move_` at `0x535dd5--0x535ddf`.
The VC-DESW constructor independently supplies `0x53cd440`; both tables carry
the same CSSW entries in the relevant range. Decoding either table gives:

| slot | target |
|---:|---:|
| `+0x1a8` | `get_random_mode0_` `0x5e80d0` |
| `+0x1b0` | `gen_randommode_` `0x5ff750` |
| `+0x1f0` | `update_mode0_` `0x5ff2e0` |
| `+0x210` | `findmodepattern_cell_` `0x608520` |
| `+0x218` | `findmodepattern_` `0x608370` |

This is table-byte evidence, rather than a name inferred from adjacency.

## First mode selection and the post-climb update

`ssw_move_` calls slot `+0x1a8` at `0x5e65dc` and again at `0x5e7c66` on
its two mode-entry branches. In `get_random_mode0_`, the ratio gate reads
`para+0x2db58` at `0x5e8112`, reads `object+0x2a74` (`nsswstep`) at
`0x5e814b`, and writes `object+0x2260` (`lcellmove`) at `0x5e8159` or
`0x5e8167`. It then dispatches slot `+0x210` for cell mode or `+0x218` for
atomic mode at `0x5e8173--0x5e8195`. Therefore this entry both selects the
atom/cell branch and invokes the corresponding pattern helper; it is not the
random-vector generator itself.

The enclosing climb has a concrete post-processing dispatch. After the
selected trajectory record is handled (`0x5f3470--0x5f3711`), it reads
`object+0x100` (`run_type`, DWARF offset 256) at `0x5f3750`:

* `run_type == 0xf` reaches `0x5f3762`, the type-bound slot `+0x1f0`, hence
  `update_mode0_`, then writes the `CBD_Unb...` status through slot `+0x190`;
* `run_type == 0x10` reaches `0x5f378d`, the type-bound slot `+0x1a8`, hence
  `get_random_mode0_`;
* other values skip both calls at `0x5f379a`.

The separate convergence routine supplies the atom/cell stopping distinction:
`climb_convg_` compares `para+0xf4` (`ng`) with its current counter at
`0x5f4a78--0x5f4a94`, and `para+0xf8` (`ng_cell`) at
`0x5f4a98--0x5f4ab8`, with `lcellmove` selecting which comparison can set the
stop mask. This establishes the termination inputs, while the `run_type`
branch above establishes which mode-update routine follows the climb result.

## What the update does and does not prove about redraw

Every invocation of the recovered `update_mode0_` reaches a new random input
stage. At entry it dispatches the existing `lcellmove` choice to `+0x210` or
`+0x218` at `0x5ff330--0x5ff35d`, then reads persistent trajectory/state
records while constructing its local descriptor (`0x5ff447--0x5ff4a3`). It
calls `setconstraints_crystal_` at `0x5ff6bc` and `n_normal_` at `0x5ff6e5`,
and finally passes a local descriptor to slot `+0x1b0` at
`0x5ff721--0x5ff72f`. The table resolves this slot to `gen_randommode_`
(`0x5ff750`), whose direct random-number calls are at `0x5ff835` and
`0x5ffbf2`.

The stronger statement that this completely replaces the previous direction
is not established. `update_mode0_` demonstrably reads persistent records
before the generator, and `gen_randommode_` receives the local descriptor
assembled by the update. The generator's inspected entry consumes descriptor
metadata (including `r14+0x48` at `0x5ff83d`) and writes/generated-vector
storage; the slice does not provide a complete semantic proof that no old
direction component is mixed into the generated result or used as an anchor.
Therefore the supported claim is: `run_type==0xf` invokes a path that
introduces fresh random numbers after persistent-state-dependent preparation;
it cannot be labeled either “independent redraw every cell cycle” or “pure
retention” from this evidence alone.

The exact paper-level meaning of `run_type` values 0xf/0x10 and whether every
individual NG inner iteration reaches this post-processing branch remains
outside this static slice. It also does not map all paper `cell_cycles`
counters to native state transitions. The independent implementation must
keep its cycle frequency and direction-lifecycle policy explicit until that
outer state machine and the generator's data-flow semantics are separately
traced.

Sources: `research/ga_ssw/evidence/native-stress-producer-review/cssw-type-table.json`,
`docs/research/native-cell-direction-evidence/ssw_move.asm`,
`analysis/selected-dwarf.txt`, and the uploaded ELF disassembly at the addresses
listed above. No production source, MAINLINE, GPU job, PES call, or native main
execution was changed or run.
