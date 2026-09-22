# LS caller selected geometry: bounded static trace (2026-09-23)

Static ELF inspection only; no native execution, numerical tests, or PES runs.

## Evidence

`ssw_fixlat_mp_make_decision_` is at `0x5cfe20`. In its MC branch,
`0x5d195c` passes `&0x1ac0(%r15)` (candidate energy) and `&0x230(%r15)`
(current energy) to `ssw_commsub_mp_metropolismc_` (`0x57e820`); its scalar
result is written through the fourth argument at `-0x7c(%rbp)` and copied to
`-0x288(%rbp)`. The nearby object write is only `0x2298(%r15) = -9999 (0xffffd8f1)` at
`0x5d199e`. No coordinate-array store is present on this branch.

There is no direct call instruction to `0x5cfe20` in the ELF. A read-only table
at `0x53ca800` contains the address `0x5cfe20` at `+0x88` (`0x53ca888`), so
one possible dispatch is through a Fortran-derived-type vtable. The runtime
object's vtable pointer is loaded from `0x38(%r12)`/`0x38(%r13)`, but the
relevant runtime alias is not recoverable from these static references alone.

At the next move (`ssw_fixlat_mp_ssw_move_`, `0x5bc760`), the exact sequence
before bond counting is:

```text
0x5bcf0f  %rdi = (%r12)                 # current object base
0x5bcf1c  %rax = 0x230(%rdi)
0x5bcf23  0x1ac0(%rdi) = %rax            # only energy snapshot here
0x5bcf2a  %rdx = 0x170(%rdi)
0x5bcf31  %rsi = &0xe0(%rdi)
0x5bcf38  %rcx = 0x8(%rdi)
0x5bcf3c  %r8  = &0x1b30(%rdi)
0x5bcf43  call pot_bond_mod_mp_bond_counter_
```

The bond counter therefore consumes the `+0xe0`, `+8`, and `+0x170` fields as
already present on entry. The conditional at `0x5bcf1a` only suppresses the
energy snapshot when `0x2298(%rdi) == -1`; it does not restore or select a
geometry.

## Boundary

The accept/reject geometry producer remains unresolved. The missing producer is
the dynamic dispatch through the object vtable at `+0x38`, including caller
slots around `+0x28`, `+0x150`, `+0x190`, and `+0x238`, plus the exact runtime
object alias used by the move loop. The table entry proving that `+0x88` can
point to `make_decision_` does not identify the instance or continuation that
updates the geometry fields.

This does not justify changing the Python MC-selected-current update and does
not establish native parity for accepted or rejected coordinates. A next
discriminator would need the initialized `+0x38` vtable pointer and concrete
stores to the `+8`/`+e0` geometry fields before `0x5bcf43`.

Reproduction:

```bash
objdump -d -C --no-show-raw-insn --start-address=0x5cfe20 --stop-address=0x5d47c0 lasp
objdump -d -C --no-show-raw-insn --start-address=0x5bc760 --stop-address=0x5be6d2 lasp
objdump -s --start-address=0x53ca800 --stop-address=0x53ca8c8 lasp
```

Root independently re-read the three address ranges on 2026-09-23 and corrected the draft scalar constant from -8559 to -9999. This added a concrete table entry and field offsets, but did not close the geometry alias. Stop this static branch without changing the Python convention; additional native execution is not justified solely to relabel the existing implementation as parity.
