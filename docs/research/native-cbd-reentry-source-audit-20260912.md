# Native CBD re-entry source trace

> 2026-09-17 correction: the earlier Allopt equality interpretation was reversed.
> `for_cpstr` operator 3 is inequality. The text below is corrected; see
> [verification and decision](2026-09-17-direction-lifecycle.md).

Static audit only; no LASP main/PES execution. Source assembly is
`/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/analysis/`.

| address | direct evidence | conclusion |
|---|---|---|
| `0x5cc4ac` | `lea rbx,... # 53ed7a0 <ssw_parameters_mp_para_>` | `rbx` is the global parameter object |
| `0x5cc4c3` | `mov eax,DWORD PTR [rbx+0x100]` | re-entry guard reads `para+0x100` |
| `0x5cc4c9–0x5cc4cc` | `cmp eax,0x5; je 0x5ccb97` | value 5 enters the mode-update branch |
| `0x5cc4d2–0x5cc4d5` | `cmp eax,0x6; jne 0x5cc4e4` | value 6 has a separate callback path; other values skip both |
| `0x5ccb9e` | `call QWORD PTR [rax+0x1f0]` with `rdi=r13` | type-bound `update_mode0` callback |
| `0x5ccba4–0x5ccbc6` | `rdi=[r13] + 0x1b34`, `for_cpstr(..., len=6)` | object status at `+0x1b34` is compared with `Allopt` |
| `0x5ccbe7` | indirect call `[rax+0x188]`, `rsi=CBD`, `ecx=3` | status unequal to `Allopt` dispatches `set_status(CBD)` |

## Resolved callback body

The fixed-cell method table at `0x53cc8c0`, slot `+0x1f0`, contains
`0x5d55f0` (see `native-gaussian-caller.md`, type-bound call table).
The ELF symbol table names that target as
`ssw_fixlat_mp_update_mode0_` at `0x5d55f0`; therefore the earlier statement
that the body was absent from the inspected slice is superseded. The body is
not a status setter. Its directly visible stages are:

- `0x5d5607–0x5d5667` loads a 0x58-byte static work descriptor from
  `0x54cbe00` and object fields `+0x8f0`, `+0x900`.
- `0x5d5677–0x5d5788` accumulates integer coordinate/descriptor data over the
  object arrays.
- `0x5d57a7` reads `control+0x140`; `0x5d5790–0x5d5990` constructs the update
  record, including object direction pointer `+0x1788`.
- `0x5d5a70–0x5d5bc4` rebuilds pair/history differences. At
  `0x5d5bf4–0x5d5c15` it calls `ssw_commsub_mp_n_normal_` and indirect record
  slot `+0x1a8`; when the saved integer differs, the alternate path at
  `0x5d5c2a–0x5d5c38` first calls `ssw_commsub_mp_setconstraints_`.

No direct store to object `+0x1b34` occurs in this body. The indirect record
slot and its caller-owned descriptor are still unresolved, so these
instructions establish numerical/state update work but do not establish an
Allopt transition frequency or an E/F evaluation.

## Parser correction

`readsswpara-convergence.asm` shows two adjacent input records. The call at
`0x68694a` is the preceding `readinput_mp_get_int_` call. Then:

```
0x68694f  lea rsi,[rip+...]  # 0x4a308cc
0x686956  lea rdx,[rbx+0x100]       # para+0x100
...
0x6869be  call 0x60eeb0             # readinput_mp_get_int_
```

Static ELF bytes at `0x4a308cc` decode to the exact key `Run_type\0`. Thus the
`0x6869be` call, not the `0x68694a` call, owns the `para+0x100` destination.
This closes the key/destination/read chain, while later runtime writes and the
semantic conditions selecting values 5 versus 6 remain unknown.

## Allopt status and counters

The `Allopt` write/transition is conditional and directly visible in the
outer climb path. `0x5cb377` tests `control+0x78`, whose DWARF member name is
`lclimb_allstop`; `0x5cb9ae` tests `control+0x7c`, named `lclimbstop`.
On the all-stop path, `0x5cb8dd–0x5cb8e4` copies `[object+0x1ac8]` to
`object+0x230`, optional cleanup runs at `0x5cb8ee–0x5cb904`, and
`0x5cb910–0x5cb928` calls the status setter with `Allopt`. The convergence
routine writes the accumulated mask and derived status at `0x5cd9b8` and
`0x5cd9bc`, respectively. It also compares the Gaussian index against
`object+0xf4` (`ng`) at `0x5cd5fe–0x5cd60c`, and compares counters against
`para+0x2dd20`/`+0x2dd24` at `0x5cd7bc–0x5cd7cc` and
`0x5cd9d5–0x5cd9eb`.

The fixed-cell allopt routine is separately symbolized at `0x5d46e0`; its
`alloptstep` reset is `0x5d52cb–0x5d52d3`. These counters and status writes
are evidence of state transitions, not proof of a physical landing or of a
per-Gaussian re-entry schedule. The indirect record descriptor and the later
post-setter E/F path remain the concrete unresolved boundary.

The resulting proven control-flow fact is therefore:

```
para+0x100 == 5
  -> fixed-cell update_mode0 at 0x5d55f0
  -> if object+0x1b34 != Allopt
       -> set_status(CBD)
```

No production rule or frequency is inferred from this trace.
