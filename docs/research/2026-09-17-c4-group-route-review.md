# C4 group route review

This is one bounded static review of the archived LASP ELF (SHA256
`bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704`).
It corrects the method-slot interpretation in
`2026-09-17-native-c4-generator-branch.md`; it did not execute LASP or a PES.

## Corrected dispatch and member transfer

The generator receives an Intel Fortran polymorphic descriptor in `r12`:
`[r12]` is the fixed-cell object and `[r12+0x38]` is its type-bound procedure
table (`0x5d5c64–0x5d5c73`, `0x5d7833–0x5d7844`). `run_ssw_` explicitly writes
`0x53ca680` to descriptor offset `+0x38` at `0x51e42b–0x51e432` (and for the
following fixed-cell descriptors at `0x51e4e3–0x51e4ea`). Thus `0x53ca680` is
the actual fixed-cell table, not merely a plausible base table, and
`table+0x160 = 0x5a61a0`, `class_struc_mp_find_atom_in_group_`, is confirmed.

The previous report is nevertheless wrong at the nested slot. Arithmetic on
the same table gives

```text
table+0xc8 = 0x5a67f0  class_struc_mp_group_atoms_
table+0xd0 = 0x5a4300  class_struc_mp_get_shortestdist_
table+0xd8 = 0x5a4500  class_struc_mp_centralize_
```

The call at `0x5a6583` is therefore `group_atoms`, not `centralize`. This is
the missing transfer: `find_atom_in_group` seeds the private visited mask at
object `+0xb48`, then `group_atoms` grows the component and writes every newly
reached atom to both `+0xb48` (`0x5a6b0f–0x5a6b2c`) and the generator-visible
mask at `+0x50` (`0x5a6b30–0x5a6b43`). The apparent `+0xb48`/`+0x50`
disconnect is disproved.

## Implementable selection rule

For a seed atom `s`, `find_atom_in_group` first clears the visited mask and
constructs a symmetric pair cutoff matrix. For each pair `(i,j)`, it obtains
`b_ij = fastbond(Z_i,Z_j)` (`0x5a633e–0x5a63a2`). If `b_ij < 0.05`, it replaces
that missing/small table value by `species_radius(Z_i)+species_radius(Z_j)`
(`0x5a63c6–0x5a6431`; literal `0.05` at `0x4a44648`). It marks `s` visited at
`0x5a6554–0x5a657c` and invokes `group_atoms` through slot `+0xc8`.

`group_atoms` implements a recursive connected-component traversal. From the
current atom, among unvisited atoms it chooses the smallest periodic
`get_shortestdist` value satisfying

```text
d_periodic(current, j) < 1.3 * b_current,j .
```

The strict comparison is at `0x5a6a35–0x5a6a7c`; `1.3` is the literal at
`0x4a44658`. `100.0` at `0x4a44650` is only the initial no-candidate sentinel.
The selected atom is marked in both masks and recursion continues from that
atom (`0x5a6aa4–0x5a6b50`). Repeating nearest eligible additions yields the
same connected component as the graph defined by the strict cutoff; the
nearest order matters only for exact native traversal order. The seed itself
is visited but is not written into the `+0x50` output mask.

The generator applies this rule twice when `object+0x2204` is true. It clears
`G1=object+0x50`, finds the component from pair endpoint `p1`, and tests whether
`p2` is in `G1` (`0x5d7816–0x5d7877`). If so, it clears `+0x2204` and falls back
to `localatompair_mode_` at `0x5d8753–0x5d8794`. Otherwise it copies `G1` to
`object+0x98`, clears `object+0x50`, finds `G2` from `p2`, and calls
`localatompairgroup_mode_` with `(G1,G2)` at `0x5d7893–0x5d7a58`. That callee
assigns the normalized `p1-p2` axis to `G1` and its negative to `G2`, followed
by constraints and normalization before c4 mixing.

This closes the C4 group-routing algorithm sufficiently for implementation.
The `fastbond` element table is native empirical data (routine
`0x58f2a0–0x58f671`), while the radius fallback and factors `0.05/1.3` are
binary-specific rules; they should be represented explicitly rather than
described as universal chemical cutoffs.

## Bounded reference implementation

`pamssw/standalone/native_bond_groups.py` now implements this rule for an
explicitly verified probe domain: H/C/O fastbond pairs and the Cu/Au missing
fastbond pair with its native 1.25 Å-per-element radius fallback. It accepts
only nonperiodic, unconstrained structures and returns either two endpoint
group masks with status `separate_groups`, or the first mask plus status
`connected_pair_fallback`. Unsupported element pairs raise rather than
silently applying a guessed cutoff. This is a probe-facing pure function and
is not wired into a walker.

`research/ga_ssw/probe_native_bond_groups.py` executes the original
`fastbond_` instructions at `0x58f2a0` for the seven supported pair entries,
then checks three canonical-box group cases. Its evidence artifact is
`research/ga_ssw/evidence/native-bond-groups-20260917.json`: 7/7 lookup values
and 3/3 pure-reference group cases pass. Those three cases alone are not a
native traversal comparison.

The dynamic traversal comparison is
`research/ga_ssw/probe_native_c4_group_generator.py`. It executes the complete
generator with the original `check_forbiden`, `find_atom_in_group`,
`group_atoms`, `get_shortestdist`, local producer, rigid projection and C4
mixing instructions. The probe replaces only allocation/memory runtime calls
and deterministic RNG. On a non-collinear pair of separated CH fragments, the
two native endpoint masks equal the Python masks and the group producer is
called once. With the same geometry and a C-H endpoint pair, the first native
mask contains the second endpoint, the group producer is bypassed and the
pair producer is called once. Both final direction outputs match the
independent references, with maximum absolute error `1.67e-16`. Evidence:
`research/ga_ssw/evidence/native-c4-group-generator-20260917.json` (2/2 pass).
This still does not establish periodic support or scientific search benefit.
