# Existing full-direction bundle enables local C60 repair in this probe

## Result

GPU1484974 completed20m12, exit0; CPU1484988 completed exit0. All four ten-attempt trajectories completed:16298 search requests+44 fresh checks;14832 actual search Calculator calls+44 fresh calls. All40 landings passed independent force≤0.03eV/Å and composition/cell/PBC checks; all request ledgers close. This compares the existing full recovered direction bundle with parent global+same CBD, not a newly invented algorithm.

| Full-direction arm | Search requests | Force-qualified / cage landings | Ih observations | First Ih cost |
|---|---:|---:|---:|---:|
| SSW1101 |4106|10 /2|0|—|
| NativeLS1101 |4090|10 /3|2|2748|
| SSW1102 |4301|10 /3|3|1614|
| NativeLS1102 |3801|10 /3|0|—|

Two of four trajectories found Ih, versus none of the four parent trajectories. This is not an estimated50% general success probability. The five Ih observations contain repeated visits; only two first-hit trajectories are counted. Both first hits were accepted and met the0.01eV reference window. All four parent/full comparisons retain this0-vs-2 distinction at their common search prefixes; the parent arms have5–6 completed landings at those prefixes. Full ten-attempt totals are26509(parent) versus16298(full) search requests, not proof of universal speedup.

First-hit fresh checks:
- NativeLS1101 landing6: fmax0.02928eV/Å, ΔIh0.000582eV.
- SSW1102 landing3: fmax0.01778eV/Å, ΔIh0.000375eV.

Independent saved-geometry alignment enumerates120 graph-compatible mappings with proper rotations: RMS0.002384/0.001790Å to the qualified Ih reference, shortest pairs1.37908/1.37910Å. These corroborate recovery of the reference geometry, not new Hessian/TS/barrier certificates. No attempt is made to optimize sub-meV differences. [Geometry evidence](hit-geometry.json), [full cost and common-prefix comparison](comparison.json), [per-arm analysis](analysis.json).

## Interpretation and competing evidence

This establishes useful behavior of an already implemented mechanism on this specific local-defect task. It does not establish LS alone as beneficial: one seed succeeds with LS and the other without. Full direction changes pair/group selection, initial local/global mixture, within-climb displacement continuation and outer state together. Matching CBD5/15 and tolerances does not make this a one-component selector ablation.

The two hit attempts use3 and5 Gaussian stages, respectively, with `pair_fallback` then `pair` direction routes and recovered CBD. Selected/refreshed pairs differ; no single pair or group is established as the causal repair mechanism. Earlier randomC60 full-direction results were mixed-to-negative and remain valid counterevidence: [prior comparison](../../../../../docs/research/2026-09-21-direction-multistep-results.md). This experiment neither satisfies random-cloud C60 global acceptance nor supports switching universal defaults.

## Confirmed startup-order boundary

All four actual input-to-initial-quench displacements are exactly zero. In `select_native_local_group`, first-index `argmin` resolves the tie; its subsequent first axis is deterministic. Five cyclic reorderings of identical coordinates yield different source-index first axes59,0,6,30,40 and group sizes22,17,19,17,22. This is a physical site-selection dependence on atom ordering, not coordinate roundoff. Pair refresh and later state evolution also use randomness; this audit does NOT establish that initial ordering caused the two successes.

[Zero-PES audit](startup-order.json) motivates explicit treatment of startup policy before a robustness claim. Preserve the native-recovered primitive and old results. No new search, parameter sweep, default switch or automatic successor was submitted. A proposal separating legacy startup from permutation-randomized startup is provided for discussion; the public direction/state contract should not change silently.

## Provenance

Frozen experiment runner3f35a98; runtime docs/analyzer HEAD a618cc7. Original parent runner and plan snapshots remain in parent `runs/`; full run snapshots/checkpoints/trajectories and ledgers remain in `direction-probe/runs/`. Raw data are not all inGit; preserve this worktree. Shared analysis/script edits do not rewrite frozen run snapshots. CPU geometry1485232 initially failed on a glob that included a JSON file;1485242 reran after a focused file-selection fix. CPU1485250 audited startup order. These diagnostics have zero PES calls.
