# Native local-pair direction: current closure and remaining probe

This note supersedes the stale “isolated local-pair oracle unresolved” wording
in `native-cluster-control-selection.md`. It is a static/isolated-oracle audit
only: no LASP main execution, protection path, PES call, or production change.

## What is already established

The local component itself is closed by
`research/ga_ssw/evidence/native-cluster-control-generator/localatompair_mode.asm`
and `localpair-oracle-v7.json`:

- `neighboringlist_` at `0x580c50` builds the endpoint list from the species
  radius sum plus the ELF constant `0.5` at `0x4a43890`, with a strict distance
  comparison and the freedom mask.
- `localatompair_mode_` at `0x6e4490` normalizes the selected pair vector,
  samples endpoint list slots without replacement, accepts neighbor vectors only
  for norm `>3.0` (`0x4a4cce0`), applies factor `0.8` (`0x4a4cd18`), and caps
  accepted neighbors at four. Pair admissibility uses `0.6` and `0.7`
  (`0x4a4cd08`, `0x4a4cd10`).
- The v7 isolated runner executes the native helper with raw contiguous `3*N`
  output and reports maximum native/reference error `5.56e-17` across its
  twelve zero-PES cases, including repeated slots, endpoint asymmetry, masks,
  cap behavior, and the archived neighbor-list call.

The consumer path is also partly closed. In
`gen_randommode_local_calls.asm`, `gen_randommode` calls the helper at
`0x5d7d9e` (and a second branch at `0x5d8794`), optionally calls
`setconstraints` at `0x5d7dc3`, then calls `n_normal` at `0x5d7dd7`. The local
coefficient is loaded at `0x5d7dea`; the surrounding accumulation multiplies the
already normalized local workspace before adding it to the aggregate. The
aggregate is normalized later at `0x5d865a`. Therefore native local-vector
normalization-before-mix is established, rather than the paper-level raw-pair
formula.

## Actual remaining gap

The unresolved part is the *selection and mapping of the ten coefficient slots*
and the complete runtime branch that combines global, local-pair, local-group,
compression, and mass-weighted workspaces. The static consumer at
`ssw_fixlat_mp_gen_randommode_` (`0x5d5c50`) reaches the local branches through
object state and an indirect method table. The known facts do not establish
which runtime state selects the pair branch on a normal step, nor the final
semantic mapping of every coefficient slot. This is a control/mix contract gap,
not a missing neighborhood cutoff or an unresolved normalize-before-mix rule.

The paper reference in `literature/74.txt` can document the paper-level raw-pair
description, but cannot settle this native runtime mapping.

## Coefficient-slot trace (static)

The incoming coefficient pointer is copied from `rsi` to `rbx` at `0x5d5c67`
and saved/restored through `[rbp-0x278]` around nested calls. The following
table records only instruction-backed consumers. “Workspace” names describe
the directly called native routine or operation; they are not claims about the
user-facing meaning of the coefficient.

| zero-based slot | load / gate | directly observed consumer | status |
|---:|---|---|---|
| 0 | `0x5d5ff9`, gate to `0x5d6005` | random workspace, `n_normal` at `0x5d620a` | workspace known; global label not proven |
| 1 | `0x5d64f4`, gate to `0x5d6501` | `atom_neighbor_radius_` at `0x5d66cf`, then `vmb2_` at `0x5d66fe` | consumer known; physical label/mass meaning unknown |
| 2 | `0x5d6a1e`, gate to `0x5d6a2b` | `compress_mode_` at `0x5d6bce`, normalization at `0x5d6c07` | compression branch known |
| 3 | `0x5d6ed8`, gate to `0x5d6ee5` | `vmb2_` at `0x5d7342`, optional constraints and normalization | workspace known; semantic label unknown |
| 4 | `0x5d7634`, gate to `0x5d7641` | local-pair/group dispatch: `localatompairgroup_mode_` at `0x5d7a58` and `localatompair_mode_` at `0x5d7d9e` | local branch known; exact state selection unresolved |
| 5 | `0x5d8553`, gate to `0x5d8558` | integer/coordinate workspace assembled at `0x5d85e0–0x5d8648`, then aggregate `n_normal` at `0x5d865a` | workspace known; semantic label unknown |
| 6 | `0x5d80aa`, gate to `0x5d80b7` | `localatomgroup_mode_` at `0x5d8255`, optional constraints and normalization | local-group branch known |
| 7–9 | no corresponding coefficient load found in the inspected `0x5d5c50` path | none established | unknown; may belong to other run-type/dispatch paths |

The `localatompair_mode_` call therefore uses slot 4 in this path, but this
does not prove that slot 4 is selected on every normal step: the surrounding
state tests at `0x5d7860–0x5d7891`, the indirect method call at `0x5d7844`, and
the pair/group branches determine reachability. Slot 1 must not be called a
mass-weighted branch from the symbol name alone; the inspected instructions
show neighbor-radius and `vmb2` operations, not a mass lookup.

## Smallest feasible next probe

The one-function entry is `ssw_fixlat_mp_gen_randommode_` at `0x5d5c50`, with a
synthetic fixed-cell object and no outer caller. A bounded Unicorn probe should
stop after the local branch and final aggregate normalization, recording the
ten coefficient input, branch flags, local-pair output, aggregate output, and
the selected workspace scales. It may reuse the already validated v7 helper
ABI and hook only known numerical/runtime callees:

1. `localatompair_mode_` at `0x6e4490` (or the direct v7 implementation),
   `setconstraints_` at `0x5790e0`, and `n_normal_` at `0x578e20`;
2. the already mapped random-number helper `0x580640`;
3. `for_realloc_lhs` only for the allocatable workspace requested by this
   function; and
4. the object method-table target at `vtable+0x160`, whose receiver and
   descriptors are visible in the call setup around `0x5d799a–0x5d7a58`.

The probe must provide the object fields/descriptors used at
`+0x170`, `+0x1788`, `+0x1848`, `+0x18a0`, `+0x1ad8`, and the ten-slot
coefficient buffer at `control+0x158..0x19f`, then compare output under one
coefficient slot at a time and paired local/global slots. It must retain the
actual indirect method-table dependency as a blocked result if the receiver
cannot be reconstructed. Feeding guessed descriptors or claiming a complete
direction from the direct local helper would not close this gap.

This probe would close only native coefficient-to-workspace mapping and branch
selection. It would not justify changing the paper implementation, choosing a
new default, or claiming full native trajectory parity.
