# Pool adapter: optional element-permutation identity

## Decision scope

The question is whether the research-only `PoolStarterAdapter` should expose a non-default molecular identity matcher for LS-plus-pool experiments. This proposal is scoped to that adapter and its explicit checkpoint contract. It does not change `MinimaArchive`, public `run_ssw`, or default matching, and it does not assert that a bond graph or one geometric match proves two physical minima are the same.

## Current branch behavior

This review is against `c60-local-defect-qualification` at `8a37f30`.

`MinimaArchive` documents that it is approximate geometry-first matching and does not resolve atom permutations ([`archive.py`](../../pamssw/archive.py), lines 43–49). `find_match` compares all boundary-compatible archive entries and uses the RMSD threshold only; energy is validated as finite but is not an identity gate (lines 90–98). `_rmsd` rejects different ordered atomic-number arrays, then uses ordered Kabsch for nonperiodic multi-atom structures (lines 291–314). Thus same-element relabelings with unchanged `numbers` arrays, including any permutation in homonuclear C60, are not rejected at the composition check; instead their ordered Kabsch RMSD can exceed the threshold. The adapter creates a `State` from ASE `Atoms` including its cell, but archive molecular routing depends on PBC, so a nonperiodic `Atoms` with a populated cell still takes the nonperiodic Kabsch branch.

`PoolStarterAdapter` is research-only and rejects ASE constraints at insertion ([`pool_starter_adapter.py`](../../research/ga_ssw/pool_starter_adapter.py), lines 229–242). It currently uses `archive.find_match` to classify observations and then `archive.add` to insert/merge. Its checkpoint is implemented: version 1 stores the adapter `mode`, archive tolerances (`energy_tol`, `rmsd_tol`, `max_prototypes`, `cell_tol`), scorer and selector policy in `checkpoint_contract()` (lines 20–35, 67–85); `export_state()` persists that contract with entries, prototypes, mappings, outcomes and decisions (lines 87–125); `restore_state()` requires exact contract equality before restoring and validates references atomically (lines 133–227). It does not currently include an identity-matcher name because the matcher is implicitly the core archive implementation.

The pool's RDF/distance descriptor is a routing/novelty signal, not its geometry identity test; this proposal leaves that descriptor and all scoring weights unchanged. The existing zero-PES probe reported by the parent found that the archived 30 C4H6 geometries passed ordered matching against themselves, all 90 rigid transforms passed, while all 90 same-element permutations were missed despite unchanged RDF. This supports a concrete identity-mode question only; it does not show distinct physical minima are being merged or establish a search-quality effect.

## Frozen zero-PES matcher comparison

The archived comparison is [`matcher-qualification-20260926/result.json`](https://github.com/Phorbol/pam-ssw/blob/c51dc1c/research/ga_ssw/evidence/matcher-qualification-20260926/result.json), analyzed in `ls-mechanism-panel-20260926@c51dc1c` against source commit `8a37f30`. It uses 33 saved geometries and 330 same-geometry queries (33 exact, 99 rigid transforms, 99 same-element permutations, 99 rigid-plus-permutation transforms), at RMSD threshold 0.1 Å. It made zero PES evaluations. In that environment ASE was 3.26.0, pymatgen 2026.5.4, NumPy 2.0.2, SciPy 1.16.0.

`ase.geometry.distance(..., permute=True) / sqrt(N)` passed all 330 queries. This gives a usable bounded development matcher for the tested saved geometries, including C60, without a new dependency. Its local implementation centers structures, aligns principal inertia axes through four sign combinations, and greedily assigns same-element atoms. It is not a global minimization over all allowed rotations and permutations. Therefore call it `ase_permute_v1` (or similarly explicit), not exact/minimum RMSD. A score below the threshold witnesses a concrete matching alignment; a score above threshold may be a false nonmatch.

The alternative `pymatgen.core.molecule_matcher.HungarianOrderMatcher` passed exact/permutation transformations but only 45/99 rigid transforms and 45/99 rigid-plus-permutation transforms. Its source uses two inertia-axis orientations, same-species Hungarian assignment, and Kabsch, and explicitly says it cannot guarantee the best match. The source result preserves all misses and lists C60 examples with large RMSD after known exact rigid/permutations. It is therefore not the selected matcher for the current saved-geometry domain. `BruteForceOrderMatcher` enumerates within-species permutations and is unsuitable for C60; `MoleculeMatcher` defaults to `InchiMolAtomMapper` and requires Open Babel Python bindings, adding topology assumptions not carried by `State`. The local ASE source is available at [ASE `distance` implementation](https://wiki.fysik.dtu.dk/ase/_modules/ase/geometry/distance.html); [pymatgen matcher API](https://pymatgen.org/pymatgen.core.html#pymatgen.core.molecule_matcher.HungarianOrderMatcher).

## Minimal adapter option

If enabled after the offline comparison is reviewed, add one constructor setting to `PoolStarterAdapter`, defaulting to the historical mode:

| Setting | Behavior |
|---|---|
| `identity_matcher="ordered_v1"` | Current `MinimaArchive.find_match/add` behavior, unchanged. |
| `identity_matcher="ase_permute_v1"` | Adapter-local molecular comparison using `ase.geometry.distance(..., permute=True) / sqrt(N)`; no change to the core archive or `run_ssw`. |

The opt-in matcher should apply only when both compared states have all PBC false, equal atom count, and equal elemental composition. Reject the opt-in mode for periodic states rather than use a molecular matcher that ignores cell/basis identity. The adapter already rejects `Atoms.constraints`; keep that rule, so there is no fixed-mask mapping problem in this mode. A cell carried by nonperiodic ASE `Atoms` is not used to define this molecular equivalence, consistent with the current archive's PBC-based route.

Implement the adapter-local path as a small archive subclass or equivalent local wrapper so `add`, visits, duplicate counts, prototypes, and representative retention still use one code path. Do not call the current ordered matcher first and then merge separately, which could double count or disagree between `find_match` and `add`. Keep stored `State` and ASE geometries in their original order; the matcher is used only to compute the comparison distance, never to reorder calculator inputs, direction state, or checkpoints.

Keep the existing `rmsd_tol` as the sole geometric threshold, with the same per-atom RMSD convention and energy mismatch accounting. No graph filter, topology threshold, extra restart count, or new matcher tuning parameter is proposed. Name the matcher heuristic in outputs and report unresolved/nonmatched cases explicitly. A low returned RMSD gives a valid element-preserving correspondence; a higher result does not prove no lower global RMSD exists.

## Checkpoint identity contract

The matcher choice changes observation-to-entry mapping, entry count, prototype references, and pool starts, so it must become part of the existing adapter checkpoint contract before any resumed use. Record at least:

```json
{
  "identity_matcher": "ordered_v1",
  "matcher_implementation": "pamssw.archive.MinimaArchive._rmsd",
  "matcher_version": 1,
  "rmsd_tolerance_angstrom": 0.1,
  "periodic_policy": "ordered_core_archive",
  "molecular_labels": "atomic_number"
}
```

For the optional mode, record `identity_matcher="ase_permute_v1"`, the resolved ASE package version (the qualification run used 3.26.0), callable path `ase.geometry.distance`, `permute=true`, normalization `divide_by_sqrt_n_atoms`, same RMSD tolerance, and the nonperiodic-only policy. The existing contract equality check should reject a checkpoint created under another matcher before any new observations or PES work. Preserve version-1 historical checkpoint restoration under `ordered_v1`; either keep its legacy contract shape on that path or implement an explicit v1-to-v2 compatibility branch. Any opt-in matcher checkpoint should use the new explicit contract/version. Do not migrate old archive entries silently: rematching changes IDs and parent/prototype statistics. If an offline rebuild is later authorized, preserve the source payload and write a new run identity.

## Minimal offline validation before use

- The completed qualification covers saved-geometry exact copies, rigid transforms, same-element permutations, and their combination for C4H6 and C60. Before any search interpretation, confirm the adapter wrapper preserves `MinimaArchive` visit/duplicate/prototype semantics and that no matcher fallback occurs. Do not call the result a chemically qualified minimum census.
- Verify the adapter's representative IDs, duplicate counts, prototype references, and `mapping` on matched and unmatched observations.
- Check that default mode reproduces existing serialized results and that checkpoint restore rejects a different matcher contract while legacy version-1 ordered checkpoints remain restorable.
- The completed 330-query screen supports development use of ASE permutation matching on these geometries only. Its wall times are method-call timings, not search-efficiency evidence. The method remains approximate; retain misses and test outcomes as observations. Do not add custom restarts or matcher parameters if later controls expose failures; restrict use to the domain actually qualified or defer it.

The zero-PES matcher qualification is complete. The remaining discussion is whether to implement the adapter-only `ase_permute_v1` identity mode and its checkpoint contract. No core change, dependency change, default switch, or PES run is proposed.
