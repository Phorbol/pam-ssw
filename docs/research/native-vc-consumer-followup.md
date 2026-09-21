# Native VC descriptor and stress conversion dispatch

2026-09-11. Static closure from the uploaded ELF, SHA256
`bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704`.
No native main, PES, initialization or protection path was executed.

The previously unresolved indirect call at `update_forcepara+0x485`
(`0x5e4405`) is now resolved for the explicitly constructed ordinary CSSW
path. The essential distinction is between a Fortran polymorphic descriptor
and its underlying object: at `0x5e3f94`, R13 receives the descriptor RDI;
`[R13]` is the object pointer, while `[R13+0x38]` is the type-bound procedure
table. Descriptor `+0x48` allocation metadata is a different address from
`[[R13+0x38]+0x48]`. The previous attempt conflated these levels.

## Actual caller and table

Ordinary `run_ssw_` constructs a descriptor at `[rbp-0x17a8]`: it writes
`0x53cacc0` (`_TBPLIST_PACK_13`) to its `+0x38` at `0x535d28–0x535d2f`,
passes that descriptor in RDI at `0x535dd5–0x535ddc`, and directly calls
`ssw_crystal_basic_mp_ssw_move_` at `0x535ddf`. In the walker the same
descriptor is passed through table slot `+0x198` at `0x5e6374`, resolving to
`ssw_crystal_basic_mp_update_forcepara_` (`0x5e3f80`).

The separate VC-DESW constructor caller at `0x614481` supplies table
`0x53cd440`; it is corroborating table evidence, not the basis for identifying
the ordinary SSW path. Both concrete tables have the following entries:

| Slot | Address | Function |
|---|---|---|
| +0x40 | 0x59b420 | i_update_str |
| +0x48 | 0x59d930 | refresh_cellstr |
| +0x60 | 0x59f0b0 | cart2scart |
| +0x98 | 0x59cb10 | stress2dedlatt |
| +0x198 | 0x5e3f80 | update_forcepara |

`refresh_cellstr_` calls slots `+0x40`, `+0x98`, then tail-jumps to `+0x60`
(at `0x59d93d`, `0x59d947`, `0x59d965`). Thus the actual resolved sequence is
`update_forcepara -> refresh_cellstr -> i_update_str, stress2dedlatt, cart2scart`.
In particular the stress-to-cell-force conversion **is** called by refresh;
it is not an unknown target or merely an unrelated function in the table.

Machine-decoded complete tables are in
`research/ga_ssw/evidence/native-stress-producer-review/cssw-type-table.json`,
reproduced with `research/ga_ssw/inspect_native_cssw_dispatch.py ELF OUTPUT`.
The script only reads ELF program segments and the symbol table. Addresses
are verified against file bytes rather than manually counting table entries.

## Formula and remaining scientific boundary

The already isolated original-instruction `stress2dedlatt` oracle establishes
`dedlatt = -V * (stored_stress + pI) @ stored_celli`.
The producer and orientation records establish `stored_celli=A^{-T}` and
`A=L.T`, where ASE stores lattice vectors as rows in L. The new dispatch
closure connects that existing formula to this ordinary CSSW update path;
it does not introduce a new strain metric or alter the independent Python
log-strain derivative.

The existing LJ producer evidence also proves that `cal_pes_` passes
`strt+0x110` to `ljperipot_` at `0x49a996`. Ordinary SSW copies its stress
argument into CSSW `+0x128` at `0x5356fc–0x535700`. The isolated LJ stress
is an energy-strain derivative divided by absolute volume. Backend mixing
through `pesfact` and postprocessing remain separate from this arithmetic;
other NN/external backends and the complete native coordinate schedule are
not covered by the present closure. No full LASP trajectory, native metric
parity or search improvement follows.

The prior constructor scans identified four direct `new_cssw_` callers in
VC-DESW. They are no longer an unresolved blocker for identifying the ordinary
`+0x48` target. Further work should address the actual remaining coordinate
scaling/schedule or physical effectiveness, rather than repeat table discovery.
