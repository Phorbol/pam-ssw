# Native BRZERO4 `WI` assignment audit

This is a bounded static audit of the frozen LASP ELF. It does not execute LASP,
its protection logic, or a Unicorn probe.

## Frozen source and names

- ELF: `/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp`
- SHA256: `bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704`
- The module routine is `broyden_module_mp_brzero4_` at `0x6f6c00` (high PC
  `0x79cf20` in DWARF), with the DWARF source name `brzero4`, declaration line
  701 (`readelf --debug-dump=info`, DIE `<2cce02>`).
- Its DWARF formal names are, in register/stack order: `ndim` (RDI), `x1`
  (RSI), `f1` (RDX), `g01` (RCX), `ini` (R8), `iu6` (R9), then `iout`,
  `iniangle`, `norder`, `langle`, and `rotmode` on the stack
  (`<2cce1b>–<2ccebf`). The caller `broyden_module_mp_brions4_` at `0x6f6440`
  has the same leading names and passes its local `ini` at `0x6f6b51–0x6f6b8d`.
  At that call site `ini` is set to `-1` when the caller's preceding test at
  `0x6f6b3c` sees `EAX == 1`, and to zero otherwise. This establishes a caller
  condition for initialization; it does not by itself assign a physical meaning
  to `WI`.

The allocatable descriptor for module `WI` is
`broyden_module_mp_brzero4_$WI.0.9` at `0x5522780`. The older non-module
`brzero4_` has a separate descriptor at `0x53ecee0`; the two must not be merged.
The older routine's DWARF independently names its formal parameters (`ndim`,
`x1`, `f1`, `g01`, `ini`, `iu6`, `iout`, `iniangle`, `norder`, `langle`,
`rotmode`) at DIE `<3e55c>` and is useful only as a source-name cross-check.

## Explicit 1000 assignment

The module routine allocates the `WI` array with element count `0x190` at
`0x6f8534–0x6f8625`. After the surrounding state setup, it obtains the array
pointer from the descriptor at `0x6f9f78–0x6f9f84` and fills it:

- `0x6f9f9a–0x6f9fbf` broadcasts the 16-byte constant at
  `0x4a4d070` over blocks of eight elements;
- `0x6f9fd3` loads immediate `0x408f400000000000`, which is IEEE-754 double
  `1000.0`, and `0x6f9fdd–0x6f9fe7` writes the tail elements.

Thus the recovered assignment is an explicit full-array initialization to
`1000.0` (the array length/descriptor rules determine the actual active extent).
The old routine repeats the same code shape at `0x50f744–0x50f7c0`, with the
same immediate `0x408f400000000000` at `0x50f7ac`; this is corroboration, not a
reason to combine the separate static arrays.

The module descriptor is deallocated/reset on the initialization path at
`0x6f7508–0x6f751a` and `0x6f75b0–0x6f75bb`; these are descriptor operations,
not writes of numeric array entries. The caller's initialization flag therefore
controls broader BRZERO4 state lifetime, while the numeric 1000 fill is the
visible array initialization slice above.

## Later use and dynamic-weight boundary

The module code subsequently reads the `WI` data pointer for the matrix/history
work, including pointer setup at `0x6fd404` and weighted arithmetic beginning
at `0x6fd5fc–0x6fd653`. In the inspected direct references to the module
`WI` descriptor and the surrounding pointer-derived slice, the only recovered
numeric writes to `WI` entries are the broadcast/tail initialization at
`0x6f9fa5–0x6f9fe1`; later references are loads or address arithmetic. No
separate assignment from `fact`, `iniangle`, `rotmode`, a force norm, or a
history-dependent scalar was recovered.

This is a static boundary, not proof against an untracked alias inside the
large optimized routine. It supports the narrow statement that native BRZERO4
starts this `WI` workspace at 1000.0 on the inspected path. It does **not** prove
that 1000.0 is a universal damping/weight parameter, its units, or that the
whole native driver always reaches this path. The direct caller evidence only
closes the `ini` condition above; branch-specific dispatch and any external
writes through aliases remain unknown.
