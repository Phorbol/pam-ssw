# Native VC frequency and partial-relaxation follow-up

2026-09-11. This is a zero-PES static audit of the uploaded native binary. It
closes one real parameter gate in `ssw_crystal_basic_mp_ssw_move_`; it does not
claim recovery of the complete VC scheduler.

Latest root correction: the later section resolves the actual modulo consumer
in `get_random_mode0` and35isolated instruction cases have passed. Earlier
statements below that the modulo schedule is unknown refer only to the initial
`ssw_move`-only inspection and are superseded. Negative ratio reverses the
positive-ratio selection; it does not merely substitute its absolute value.

## Parameter identity

DWARF for `ssw_parameters_mp_para_` identifies the following absolute offsets
(the base is `para`, not the object descriptor):

| field | offset | evidence |
|---|---:|---|
| `ng_cell` | `0xf8` | `selected-dwarf.txt`, `ssw_para` member location 248; `readsswpara` passes `para+0xf8` to `readinput_get_int_` at `0x686d89` |
| `sswsteps` | `0x2daf8` | `selected-dwarf.txt`, member location 187128 |
| `ds_cell` | `0x2db18` | `selected-dwarf.txt`, member location 187160; `readsswpara` passes `para+0x2db18` to `readinput_get_real_` at `0x687b87` |
| `ratio_atomcell` | `0x2db58` | `selected-dwarf.txt`, member location 187224; `readsswpara` passes `para+0x2db58` to `readinput_get_int_` at `0x6881cb` |

The option initializer writes a real value to `para+0x2db18` at `0x6c445b`
and integer values 5/2 to `para+0x2db58` at `0x6c449a`/`0x6c46e3`.
Therefore the offset/name mapping is established. The parser type and these
writes do not by themselves prove that `ratio_atomcell` is the paper's
lambda schedule, nor that `ds_cell` is consumed by every short relaxation.

`sswsteps` is the native field at `0x2daf8`. It must not be called paper
`N_cell`: the latter is a cell-direction vector in the 2014 description,
whereas this native field is an integer used as a count/status input.
`ng_cell` is a separate integer at `para+0xf8`; the `+0xf8` loads at
`0x5e72af` and `0x5e78ad` are from an object, not from `para`, and therefore
must not be relabeled as `ng_cell`.

## Closed native gate

In `ssw_crystal_basic_mp_ssw_move_` (`0x5e5de0`), the parameter pointer is
materialized as `r14 = &para` at `0x5e6148`. The sequence at
`0x5e6551–0x5e656c` is:

```text
cmp  dword ptr [r14+0x2daf8], 1       ; sswsteps
jle  0x5e656e
cmp  dword ptr [rdi+0x2a74], 0
jne  0x5e659c
test byte ptr [r14+0x2db4c], 1
je   0x5e659c
```

The `sswsteps <= 1` case joins at `0x5e656e`. For `sswsteps > 1`, a nonzero
`object+0x2a74` takes the `jne 0x5e659c` path; it does **not** join
`0x5e656e`. Only when that object value is zero and `para+0x2db4c` bit 0 is
clear does execution reach `0x5e656e`. That branch clears `object+0x2b08` at
`0x5e6574`, calls descriptor slot `+0x190` at `0x5e6590`, and returns to the
common continuation `0x5e725a`. The alternative path checks
`object+0x2b08 == -1` at `0x5e659c` and has its own state handling.

This is a genuine `sswsteps`-dependent state/status gate. It is the strongest
closed count-related branch currently available in the crystal driver, but it
does not by itself prove a partial relaxation or cell-frequency policy. The
separate `get_random_mode0` consumer below is the actual ratio gate. The
`ssw_move` slice does not show a modulo operation involving
`ratio_atomcell`, and this driver slice has no direct load of `para+0x2db18`
or `para+0x2db58`; nor does it use the separately parsed `para+0xf8`
`ng_cell`; therefore no atom/cell alternation ratio is claimed here.

The same driver increments `ssw_parameters_mp_control_%step_perssw` at
`0x5e61cb` and resets it at `0x5e73db`. DWARF identifies control offset
`+0x6c` as `step_perssw` (`ssw_control`, structure size 472). The increment
occurs after the state setup and before the later mode-dependent work; the
available slice does not compare this counter to `ratio_atomcell` or
`sswsteps`. Thus it proves a per-entry counter update, not a frequency rule.

## Boundary for implementation

The independent implementation task supported by this evidence is limited to
preserving an integer SSW-step gate and a per-entry `step_perssw` counter at
the corresponding driver boundary, with explicit state inputs
(`object+0x2a74`, `object+0x2b08`, and `para+0x2db4c`). Implementing a
`ratio_atomcell` modulo scheduler from these observations would be an
unsupported inference. `block_ssw` already supplies an explicit cell-cycle
implementation and is not reopened by this audit.

## Ratio consumer recovered after the driver

The next function, `ssw_crystal_basic_mp_get_random_mode0_` (`0x5e80d0`),
contains the actual atom/cell selection gate. At `0x5e8112` it loads
`para+0x2db58` (`ratio_atomcell`); at `0x5e814b` it loads
`object+0x2a74` (`nsswstep`), then executes signed `idiv` at `0x5e8152` and
tests the remainder at `0x5e8154`. For positive ratios, remainder 1 writes
`object+0x2260` (`lcellmove`) = `-1` at `0x5e8167`; other remainders write
zero at `0x5e8159`. Ratio zero takes `0x5e8137` and writes zero without
division. A negative ratio takes `0x5e82ea–0x5e82f8`, computes its absolute
value, and reverses the final predicate: `0x5e82fd` sends remainder != 1 to
the true write (`0x5e8167`), while remainder 1 reaches the false write
(`0x5e8159`).

For the observed arithmetic slice the executable formula is therefore:

```python
if ratio == 0:
    lcellmove = 0
elif ratio > 0:
    lcellmove = -1 if nsswstep % ratio == 1 else 0
else:
    lcellmove = -1 if nsswstep % abs(ratio) != 1 else 0
```

The selected virtual call is slot `+0x210` when the result is true and
`+0x218` otherwise (`0x5e8173–0x5e8195`). The available class-table evidence
maps the cell pattern helpers to `+0x210`/`+0x218`; this audit does not infer
their internal mode generation. `moveds` later consumes `ds_cell` and uses the
false `lcellmove` path to select the atomic step, but the complete caller chain
from that result into every relaxation budget remains outside this isolated
slice.

`research/ga_ssw/probe_native_vc_ratio_mode.py` is the corresponding Unicorn
probe with only descriptor, object, parameter, vtable, and four return stubs
provided. Root corrected the negative-ratio branch: at0x5e82fd a
non-equal remainder branches to the true setter; the negative case reverses
the positive selection. Root executed35cases for ratios0/1/2/5/-1/-2/-5 and
nonnegative counters0/1/2/5/6. All original-instruction assertions passed,
including return PC and the selected virtual slot. The helper bodies are
return stubs, so this is a scheduling gate test, not a full mode or quench test.
Evidence: `research/ga_ssw/evidence/native-vc-ratio-mode.json`.
The existing `/tmp/pam-ssw-unicorn-probe` dependency path was used; the earlier
claim that Unicorn was unavailable was an environment-selection error.

Static sources: `native-cell-direction-evidence/ssw_move.asm`,
`analysis/selected-dwarf.txt`, `analysis/readsswpara-convergence.asm`, and
`analysis/ssw-options-convergence.asm` under the 2026-09-09 research bundle.
No LASP main/protection path, PES call, GPU job, or production source was run
or changed.
