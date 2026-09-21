# Native LS element-table recovery audit

Read-only audit of `GA-SSW_program/lasp` from the 2026-09-09 research bundle.
No LASP main program, PES, or expiry bypass was used.

## Static evidence

The ELF symbol table identifies the runtime arrays:

| object | ELF address | size | observed file contents |
|---|---:|---:|---|
| `bond_info_mp_bond_len_list_` | `0x55205e0` | 96 bytes | zero-initialized `.data` storage |
| `bond_info_mp_bond_ener_list_` | `0x55206a0` | 96 bytes | zero-initialized `.data` storage |
| `bond_info_mp_len_toller_` | `0x5520568` | 8 bytes | `0.1` |

The 96-byte objects are Fortran allocatable descriptors (metadata), rather
than twelve 8-byte table slots. Their zero bytes therefore do not establish
that the pair values are absent. The corresponding symbols and addresses are also recorded in
`analysis/lasp-symbols.txt:54029-54034`.

The two leaf lookup functions are directly executable in isolation. Existing
HC probing used `bondeneval_` at `0x6cb660` and `bondlenval_` at `0x6cc0b0`.
The expanded H/C/O oracle is
`research/ga_ssw/probe_native_ls_pair_table_hco.py`, with results in
`research/ga_ssw/evidence/native-ls-pair-table-hco-20260912/result.json`.
It maps only the two function bodies and their referenced `.rodata`; it does
not initialize LASP, call the main program, or invoke external routines.

For the nine ordered H/C/O pairs, the raw returns are:

| pair | energy (`bondeneval_`) | length (`bondlenval_`) |
|---|---:|---:|
| H-H | 4.5265798569 | 0.7400000095 |
| H-C | 4.2981700897 | 1.0900000334 |
| H-O | 4.8172798157 | 0.9599999785 |
| C-H | 4.2981700897 | 1.0900000334 |
| C-C | 3.4468400478 | 1.5399999619 |
| C-O | 3.3845500946 | 1.4299999475 |
| O-H | 4.8172798157 | 0.9599999785 |
| O-C | 3.3845500946 | 1.4299999475 |
| O-O | 1.5157799721 | 1.4800000191 |

The static assembly source is the ELF `objdump` of the `bondeneval_` body
`0x6cb660..0x6cc0b0` and `bondlenval_` body `0x6cc0b0..0x6cc3f1`; callers and
symbol references are in `analysis/ls-bond-info-init.asm`. The result JSON
records their `.rodata` reference ranges. These are raw leaf returns.

`bond_info_init_` reads the optional custom file `pot_bond_input.txt`: the
filename is referenced at `analysis/ls-bond-info-init.asm:33`, opened/read in
the sequence at `:115-150`, and the resulting atom-type count and pair data
are used to allocate/populate the runtime lists around `:150-220`. No such
file is present in the archived examples or run directories. The assembly
also shows runtime save/restore storage for the energy list, but that is state,
not a static default table.

The ELF `.data` section alone does not expose these values; the leaf oracle
recovers them from the lookup code and constants. The caller's scaling,
filtering, initialization, and semantic units remain separate questions.

## Result and boundary

The recovered code-level values establish lookup returns and the element
index convention for H=1, C=6, O=8. They do not by themselves establish the
caller-level table units, any `pot_bond_input.txt` override, or whether a
production run uses a different initialized array. Chemical bond tables and
`H2O_pf.pot` remain inappropriate substitutes for this native lookup.
