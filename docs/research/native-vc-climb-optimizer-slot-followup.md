# VC climb optimizer slot `+0x108`

2026-09-11. Bounded static trace from VC `ssw_crystal_basic_mp_climb_`
(`0x5f1d60`). No main/PES execution.

At `0x5f25d1`, `r15` is loaded from `[r13+0x38]`, the CSSW object method
table. The table bytes at `0x53cacc0+0x108` resolve to `0x5a7f90`,
`class_struc_mp_crystal_opt_` (the pointer is the 33rd 8-byte table entry).
Thus the actual call at `0x5f2642` is:

```text
class_struc_mp_crystal_opt_(
    rdi = r13,                 # object wrapper
    rsi = &stack_descriptor,   # rbp-0x150
    rdx = control + 0x08,
    rcx = control + 0x80,
    r8  = 0
)
```

The descriptor is constructed at `0x5f25c0–0x5f263a`: element length
`0x1d0`, callback table `53cd3c8`, work/info tables, and the relevant pointer
fields. It is not the fixed-cell `+0x110` slot.

## One-call behavior

`class_struc_mp_crystal_opt_` saves `rdi` and obtains the underlying object
from `[rdi]` (`0x5a7fa4–0x5a7fad`). With caller `r8=0`, its entry branch
`0x5a7faf–0x5a7fb3` reaches the callback path at `0x5a8889`. There it passes
the following object arrays to callback table `53cd3c8` slot `+0x10`:

```text
rdi = stack_descriptor
rsi = object + 0x350
rdx = [object + 0x478]  # scart data pointer
rcx = [object + 0x4d8]  # sfa data pointer
r8  = object + 0x230
```

The table entry resolves to `0x5b2970`,
`bfgs_class_mp_bfgsdriver_`. The callee reads the coordinate-row count (DWARF sna at object+0x350)
from `[rsi]` at `0x5b29a1–0x5b29af`, forms `3*count`, and selects its cell
branch from `object+0x1a8` (`0x5b29b9–0x5b29c1`). In the ordinary branch it
updates BFGS work arrays with vector loads, multiplies and stores
(`0x5b2aa2–0x5b2bfa`); in the cell branch it handles the final cell entries
(`0x5b2bff` onward). This is the actual BFGS driver call, not an unresolved
slot name.

No PES/calculator call occurs in the inspected `5a7f90` callback path or in
the shown `5b2970` arithmetic body. After the callback returns,
`class_struc_mp_crystal_opt_` increments the caller-provided counter
`[control+0x08]` at `0x5a886e–0x5a8880`. This establishes a reverse-communication
style bookkeeping boundary: the slot call performs the BFGS array update and
returns; it does not itself provide evidence of an energy/force oracle call.

Back in `climb_`, the next calls are method-table slots `+0x1e8` and `+0xe8`
at `0x5f2649–0x5f265d`, followed by a branch on `control+0x78` at
`0x5f2663`. Their enclosing ordering is recorded here only to delimit the
optimizer call; their calculator semantics are outside this bounded slice.

Sources: `native-cell-reference-evidence/climb.asm`,
`analysis/kernel-dwarf-member-offsets.txt`, and ELF table bytes at
`53cacc0`/`53cd3c8`.
