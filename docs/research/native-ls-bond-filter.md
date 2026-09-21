# Native LS `bondFilter` audit

This is a read-only audit of the research ELF, not a production implementation or a
PES run.  The binary examined was
`/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp`,
SHA-256
`bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704`.

## Parser spelling and write

The SI input writes the block as `SSW.soft.bondFilter` (`ct4c01081_si_001.txt`,
pp. S15--S16, Fe7C3 example).  The parser's ELF string is
`SSW.LS_bondFilter` at `0x4a49cd8`.  These are the user-facing and internal
spellings respectively; this audit does not claim that the generic input
normalizer accepts one spelling in every release.

In `readsswpara_` (`0x685ef0`), the block lookup starts at `0x6861b5` and the
entry loop is `0x68625d--0x6862fb`.  The two one-based indices are multiplied
by `0x360` and `8`, then stored symmetrically relative to `para-0x1e0`:

```text
offset = 0x188 + 0x360*(a-1) + 8*(b-1)
value  = 0.0
```

The zero comes from the `r13` initialization at `0x685f65`.  Thus the SI entry
`26 26` sets the Fe--Fe entry of the `108 x 108` pair matrix to zero.  This is
a pair energy factor/table entry.  It is distinct from `atom_filter`
(`para+0x16e08`), `bond_lengthen` (`para+0x16e50`), and `fixatom`
(`para+0x2df30`).

## Consumer and counting evidence

The recovered bond-potential path reads the pair filter at `0x6c9e3b` and
multiplies it into `B_ab`; the separate length factor is read at `0x6ca023`.
The inspected `bond_counter/selfadapt_nbondcounter` path counts geometric,
distance-eligible pairs with its fixed-atom rules and does not directly read
`para+0x188`.  Therefore the supported semantic statement is “zero this pair's
softening energy contribution.”  It is not evidence that the pair is removed
from `N_b` or from a neighbor list.  Whether another uninspected caller changes
that denominator remains outside this audit.

This agrees with the independent native-derived implementation: `N_b` is
computed before applying `energy_filter` in
`pamssw/standalone/native_ls.py:108-117`, while the table receives the filter
factor.  That implementation is an independent reconstruction, not proof of
all native callers.

## Difference from `LSResponseState`

`initialize_native_ls(..., energy_filter=...)` preserves a zero table entry,
and `update_native_table` updates each existing entry multiplicatively
(`native_ls.py:160-167`), so a zero-filtered pair remains zero.  In contrast,
`LSResponseState.update` computes one total strength and rebuilds the next
potential from the next geometry (`softening.py:220-234`).  `_build` includes
every distance-eligible pair and redistributes total strength proportional to
the supplied standard bond energies (`softening.py:142-162`).  Consequently,
using `LSResponseState` after a native-style Fe--Fe filter would reintroduce the
filtered pair unless an explicit filtered bond-energy table is carried through.

This is a lifecycle/representation mismatch, not evidence for changing the
paper controller or for adding separate `pair_eligibility`, amplitude, and
normalization parameters.  A faithful future adapter needs one explicit
pair-energy filter/table propagated through initialization and response rebuild;
the denominator semantics must remain separately documented until a direct
native read proves otherwise.
