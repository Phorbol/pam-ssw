# Native LS pre-quench exit: caller-order boundary

This static audit covers the archived ELF
`/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp`.
It does not run LASP or add a PES calculation. The actual instruction excerpts
are saved in `native-curvature-evidence/native-ls-prequench-caller.asm`.

The caller-level chain is directly observed as `pot_bond_add` at `0x5bd920`,
then `bfgs_class_mp_bfgsdriver_` at `0x5bdb72`, followed on the inspected path
by the indirect call at `0x5bfab1`, which existing provenance identifies as
`get_random_mode0`. Existing provenance further records `set_status('CBD')` at
`0x5c0706`, copying `xa` (`+0x170`) to `tstr0` (`+0x1a00`) and `fa` (`+0x1d0`)
to `tf0` (`+0x1a60`).

This confirms caller ordering only:

```
BFGS/LS call -> get_random_mode0 -> set_status copies xa/fa
```

The later common continuation contains an indirect callback site at
`0x5be8c7`; this audit did not establish its target's complete E/F semantics.
I also did not follow BFGSDRIVER's exit-state writes or its `iflag` and
reverse-communication branches through max-iteration and line-search-failure
returns. The precise status is therefore: **caller order confirmed; driver
exit remains unaudited**.

No inconsistency of `xa/fa` has been demonstrated, and no freshness failure has
been demonstrated. Existing Python same-point evaluation caching already keeps
the coordinate and force paired, so this evidence does not justify a redundant
PES evaluation or a new algorithmic mechanism. It provides no evidence of
algorithmic gain and is limited to the traced `0x5bd920 -> 0x5bdb72 ->
0x5bfab1` path; alternate entrances and native termination paths remain open.
