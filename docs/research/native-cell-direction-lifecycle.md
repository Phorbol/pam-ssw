# Cell direction lifecycle: bounded native audit remains inconclusive

2026-09-10. **This audit does not establish whether each native CBD-cell cycle redraws a random direction, or whether one direction is retained and softened throughout a complete cell block.** The present Python per-cycle redraw must remain an explicit experimental design choice, not a recovered native behavior. Conversely, the continuation interpretation is not established merely by the phrase “one particular lattice mode direction.”

Only `ssw_crystal_basic_mp_ssw_move_` and `ssw_crystal_basic_mp_gen_randommode_` were disassembled. No native execution, oracle, binary changes, initialization, expiry path or PES calculation was performed. Source binary and exact address provenance are the same as [the cell-force contract](native-cell-force-contract.md).

## Observable call order

`ssw_move_` (0x5e5de0) dispatches work through object method slots and state-dependent branches. It is not a self-contained, visibly labeled `for each cell cycle` loop. The following actual instruction sequences are established without assigning unresolved slot names:

| Driver branch | Observed calls / continuation |
|---|---|
| Source line 565 | 0x5e65dc: slot +0x1a8; initialize an array from parameter +0x2db08; optionally slot +0x190 then +0x1d0 at 0x5e66c3; join 0x5e7267 |
| Source line 608 | 0x5e7c66: slot +0x1a8; initialize array from same parameter; optionally slot +0x190 then +0x1d0 at 0x5e7d45; join 0x5e7267 |
| Source lines 624–627 | Write state text `CBD_PreRot`; slot +0x1c0 at 0x5e7186; join 0x5e7267 |
| Source lines 629–631 | Slot +0x1b8 at 0x5e7198; compare a status string; conditionally slot +0x1d0 at 0x5e71d3; join 0x5e7267 |

The initializer-related slots occur on separate branches from the displayed pre-rotation/rotation-like state branches. This refutes treating every entry to this driver as an unconditional random draw. It does not determine how often an outer **cell cycle** enters an initialization branch: that requires resolving state/counter transitions and the real runtime descriptor.

`gen_randommode_` (0x5ff750) contains direct calls to `ssw_commsub_mp_rd_numb_` at 0x5ff835 and 0x5ffbf2 and a direct call to `setconstraints_crystal_` at 0x5ffc7b. This verifies that this routine can generate random data and apply crystal constraints. Its modes and entry branches differ; inspecting those direct calls alone does not prove that every invocation redraws all nine cell components or that a particular driver slot invokes this routine.

The static binary contains repeated adjacent function-pointer sequences with `update_forcepara`, `ssw_move`, `get_random_mode0`, `gen_randommode`, `soften_mode0`, `unbiasedrot`, `biasedrot`, `moveds`, and `climb`. These identify useful class-table candidates, but adjacency does not establish the absolute slot base or the initialized runtime descriptor. An off-by-one pointer interpretation can map the driver's early slot to its own `ssw_move`, illustrating why names must not be assigned from adjacency alone. This task did not execute initialization to resolve those tables.

## What would close the question

The missing evidence is a concrete lifecycle connection: runtime descriptor → actual random-mode target → state/counter controlling entry → return path after one CBD-cell displacement and atomic relaxation. Then the same block's second cell cycle can be checked for re-entry to initialization or reuse of its saved direction. We have not established that path inside the allowed one-driver/one-initializer scope, and do not infer it from unrelated array-copy loops.

Keep results of the current per-cycle redraw experiment under their existing configuration. If the implementation changes to retained/softened direction on an independently justified continuation design, label that intervention and preserve the old results; neither version is native parity on this evidence. This unresolved question does not undo the recovered stored cell-force formula or the physical cell-vector representation.

Reproducible static evidence is saved in [ssw_move.asm](native-cell-direction-evidence/ssw_move.asm) and [gen_randommode.asm](native-cell-direction-evidence/gen_randommode.asm), generated with `objdump -dl -Mintel --disassemble=SYMBOL <uploaded lasp>`. The output carries original compiler file/line mappings. No further functions were disassembled in this task.
