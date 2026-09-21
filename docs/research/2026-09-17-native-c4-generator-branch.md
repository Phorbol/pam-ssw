# Native `gen_randommode` c4 local branch

This is a bounded static audit of the archived LASP ELF (SHA256
`bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704`). It does
not run LASP or a PES. The evidence is
`research/ga_ssw/evidence/native-cluster-control-generator/gen_randommode.asm`
and `localatompair_mode.asm`.

## Entry and gates

The c4 slot is the **fifth** double in the incoming coefficient array (zero-based
index 4).
At `0x5d7634` the generator loads `[rbx+0x20]` and compares it with the common
small-value constant at `0x4a45e90`; `jbe 0x5d80aa` at `0x5d7641` skips this
local branch. Thus the branch is entered only for a c4 value above that
threshold. The earlier jumps into `0x5d7634` (`0x5d6ee5`, `0x5d720d`,
`0x5d72c3`, `0x5d72e2`, and `0x5d7411`) are exits from preceding workspace
construction, not additional c4 geometry formulas.

On entry, `0x5d7647–0x5d778a` sizes and clears the coordinate workspace
(`object+0x1848`) using the object dimensions at `+0x1890`, `+0x18a0`,
`+0x1878`, `+0x1898`, and `+0x17d8`. The `N<=0`/zero-sized cases converge at
`0x5d77a8`. This block establishes storage and does not select an axis or
compute a local vector.

`0x5d77a8–0x5d77e6` calls `check_forbiden_` with the exact argument order
`N`, `cell` (`object+0xe0`), `coords` (`object+0x170`), `&pairfirst`
(`object+0x1ad8`), `&pairsecond` (`object+0x1adc`), `&control+0x144`, and
`&result` (`[rbp-0x50]`). If its result at `[rbp-0x50]`
is zero, `0x5d77ef–0x5d77f3` jumps directly to `0x5d80a6/0x5d80aa`; no local
helper is called. This is a c4 branch gate and must be retained in an
integration, rather than treating c4 as unconditional pair/group generation.

## Group path

After a nonzero forbidden check, `0x5d7804` tests the object flag at
`+0x2204`. The path then calls an indirect object method through vtable slot
`+0x160` at `0x5d7844`. The fixed-cell table at `0x53ca680` has
`[0x53ca680+0x160] = 0x5a61a0`, symbol
`class_struc_mp_find_atom_in_group_` (verified from the ELF table and `nm`).
The call receives the structure/control object in `rdi` and `object+0x1ad8` in
`rsi`; the latter is an input descriptor at this call site, so the evidence
does not show that the method updates the axis. Its result is then inspected
at `object+0x1adc` and against the integer array reached from `object+0x50`
(`0x5d785a–0x5d7881`). This is the group/pair routing state; an automatic
axis update should not be inferred here.

The callee body is `0x5a61a0–0x5a67ea`. It saves `rsi` as an input pointer
(`0x5a61b4–0x5a61c0`), reads structure-owned arrays and counts, and clears
`structure+0xb98` (`0x5a625c`). For eligible indexed pairs it calls
`fastbond_` (`0x5a639a`); when the resulting value passes the radius gate, it
calls `species_radius_` for both atoms (`0x5a63e7`, `0x5a6406`), sums those
radii, and writes the derived value to a temporary array
(`0x5a641d–0x5a6431`). It copies a selected temporary record and marks an
entry in a structure-owned integer array (`0x5a657c–0x5a6583`). The small
size path zeroes that array (`0x5a6766–0x5a67e1`). The body contains no
coordinate cross product and the inspected stores do not write the caller's
pair descriptor. The two mask pointers consumed later by
`localatompairgroup_mode_` therefore remain caller/object fields
(`object+0x98`, `object+0x50`); their complete semantic mapping is not proven
by this method alone.

The nested indirect call is resolved by the corrected table arithmetic in
`2026-09-17-c4-group-route-review.md`. At the actual fixed-cell table
`0x53ca680`, slot `+0xc8` is `0x5a67f0`,
`class_struc_mp_group_atoms_`; `centralize` is slot `+0xd8`, not `+0xc8`.
`group_atoms` writes each newly reached atom to both the private visited mask
at object `+0xb48` (`0x5a6b0f–0x5a6b2c`) and the generator-visible group mask
at object `+0x50` (`0x5a6b30–0x5a6b43`). There is one receiver/object view and
the member transfer is therefore explicit, rather than an unresolved mapping.

The group-producing call is `localatompairgroup_mode_` at `0x5d7a58`, whose
callee is `0x6e4aa0`. Its call setup is explicit at `0x5d7a38–0x5d7a54`:
the receiver state is `object+0x1ad8`, geometry workspace is `object+0x1848`,
and the freedom/group arrays are taken from `object+0x98` and `object+0x50`.
The callee first rejects a zero atom count or zero pair descriptor
(`0x6e4ab0–0x6e4abb`). It loads the two endpoint coordinates, subtracts them,
and divides the resulting vector by its Euclidean norm
(`0x6e4ad1–0x6e4b3c`). In the per-atom loop, rows selected by the first mask
receive this normalized vector and rows selected by the second mask receive
its sign-reversed form (`0x6e4b45–0x6e4ba9`). The tail also writes the two
endpoint rows with fixed packed/scalar factors (`0x6e4bb5–0x6e4bee`). This is
a normalized pair-axis with signed group masks; it is **not** the cross-product
formula of `localatomgroup_mode_` at `0x6e4c00`. No distance-list or RNG
operation occurs inside this callee.

Immediately after the call, `0x5d7a5d–0x5d7a79` conditionally calls
`setconstraints_` when the saved flag `[rbp-0x48]` is set. It then always calls
`n_normal_` at `0x5d7a8d`. Only after this projection/normalization sequence
does the caller load c4 at `0x5d7aa0` and enter the coefficient-scaled
accumulation. The aggregate direction is normalized later at the separate
`0x5d865a` site.

## Pair path and distinction from the group path

If the group state does not select the group route, control reaches the pair
call at `0x5d7d9e`, callee `localatompair_mode_` at `0x6e4490`. The call setup
(`0x5d7d6b–0x5d7d97`) supplies the current descriptor, coordinates,
atom-fix/control data, and the `object+0x1848` workspace. The pair callee is
the neighbor-aware routine documented in `localatompair_mode.asm`; it builds
and samples admissible local vectors. It is not the cross-product group
formula above.

The post-call order is the same and address-backed: conditional
`setconstraints_` at `0x5d7dc3`, unconditional `n_normal_` at `0x5d7dd7`, c4
load at `0x5d7dea`, then accumulation. Consequently both local alternatives
are normalized before c4 mixing; the final aggregate normalization is later.

## What this means for the existing Python helpers

`pamssw/standalone/native_local_group.py` matches the verified cross-product
geometry and freedom mask of the separate `localatomgroup_mode_` helper, and
`native_local_pair.py` matches the isolated neighbor-aware pair producer.
Neither helper contains the c4 gate, forbidden
check, vtable state update, or branch selection. The group helper’s direct
formula is therefore reusable as the numerical producer, but calling it for
every c4 step would omit the native control path. The pair/group selection and
mapping of the indirect state remain the minimum integration work.

No new checkpoint fields are implied: this branch is entered and completed at
the existing outer generator call boundary. The evidence supports one shared
ASE driver with the two already-authorized local producers; it does not require
a second walker or a new persistence architecture.
