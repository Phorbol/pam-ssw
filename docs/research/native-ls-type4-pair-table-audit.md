# Native LS pair values for TYPE4/CuO elements

2026-09-11. This is a zero-PES, research-only lookup audit. The uploaded
TYPE4-TiO2@Au24O4 inputs contain composition, fixed-atom and backend metadata,
but no explicit `bond_energy`, `bond_length` or `ls_target` table. The public
`pamssw.standalone.native_ls` table likewise intentionally supplies only H/H,
H/C and C/C values. No source input in the checkout provides a TYPE4-specific
LS chemical table or target value.

## Isolated lookup evidence

`research/ga_ssw/probe_native_ls_pair_table_extended.py` executes only the
release `bondeneval_` and `bondlenval_` functions under Unicorn. It preserves
the existing H/C rows and queries O(8), Ti(22), Cu(29) and Au(79), including
reverse pairs. The length fallback calls the recovered `species_radius_` ABI
using the frozen radius table in
`evidence/native-cluster-control-generator/species-radius-table.json`. It does
not enter LS initialization, neighbor counting, input parsing or a PES.

Results are in
`research/ga_ssw/evidence/native-ls-pair-table-extended/result.json` with the
ELF SHA256 and every raw return. Selected raw values are:

| pair | raw bond-energy return (eV-like table value) | raw bond-length return (Å-like value) | interpretation |
|---|---:|---:|---|
| O–O | 1.5157799721 | 1.4800000191 | dedicated branch |
| O–Ti | 3.6298000813 | 1.9429999589 | energy generic fallback; length dedicated branch |
| Ti–Ti | 3.6298000813 | 1.8999999762 | energy generic fallback; length dedicated branch |
| O–Cu | 3.6298000813 | 1.4325000000 | generic energy and species-radius length fallback |
| Ti–Cu | 3.6298000813 | 1.8750000000 | generic energy and species-radius length fallback |
| O–Au | 3.6298000813 | 1.4325000000 | generic energy and species-radius length fallback |
| Ti–Au | 3.6298000813 | 1.8750000000 | generic energy and species-radius length fallback |
| Cu–Cu | 3.6298000813 | 1.8750000000 | generic energy and species-radius length fallback |
| Au–Au | 3.6298000813 | 1.8750000000 | generic energy and species-radius length fallback |

Reverse pairs returned the same values in all completed queries. The repeated
3.6298000813 energy value is the release generic fallback branch, not evidence
of an independently fitted Cu, Ti or Au bond energy. The non-dedicated length
values are produced by the species-radius fallback and a release multiplier;
they are not a standard chemical bond-length table.

## Consequence for TYPE4 LS qualification

The recovered normal LS initialization still requires explicit pair energies
and lengths, then applies atom-count/bond-count normalization and later
amplitude/filter factors. A raw lookup cannot be substituted for that full
contract. In particular, no `ls_target` for Ti/O/Au/Cu was found, so a future
fixed-substrate LS qualification must register its pair table and target as
explicit experiment inputs with provenance. This audit supplies no recommended
chemical defaults and does not enable new elements in production.

The existing H/C baseline remains at
`evidence/native-ls-pair-table/result.json`; this extension does not overwrite
it. The lookup command was:

```bash
PYTHONPATH=/tmp/pam-ssw-unicorn-probe:. python -m research.ga_ssw.probe_native_ls_pair_table_extended \
  --elf /home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp \
  --output research/ga_ssw/evidence/native-ls-pair-table-extended/result.json
```
