# Geometry matcher qualification result

This zero-PES probe qualifies geometry-matcher invariance on a small panel. It does not demonstrate contamination of the historical pool, and it does not establish that distinct structures are the same minimum.

## Result

Job 1500506 completed on CPU-MISC (`sjtu-caoxiaoming`, `rush-cpu`) in 12 seconds with exit code 0. It evaluated 33 geometries (30 C4H6 archived initial/first-topology representatives, one ASE-built symmetric C60, and two archived C60 structures), each with exact, three rigid, three species-preserving permutation, and three rigid-plus-permutation queries. Threshold remained 0.1 Å. The JSON records zero PES calls, source hashes, individual RMSDs/errors, and timings.

| Method | Exact | Rigid | Same-species permutation | Rigid + permutation |
|---|---:|---:|---:|---:|
| PAM ordered `MinimaArchive` | 33/33 | 99/99 | 0/99 | 0/99 |
| ASE `geometry.distance(..., permute=True)/sqrt(N)` | 33/33 | 99/99 | 99/99 | 99/99 |
| pymatgen `HungarianOrderMatcher` | 33/33 | 45/99 | 99/99 | 45/99 |

No calls raised exceptions. Hungarian's rigid-transform pass count separates into C4H6 44/90 and C60 1/9; its combined-transform counts are the same. Thus the assignment step handled atom permutations in this panel, but this matcher implementation did not preserve rigid-body invariance for many geometries, including the highly symmetric C60 case. The ASE procedure passed all known same-geometry transformations here. These results qualify only this fixed panel and implementation; matcher timing does not imply search efficiency.

Matcher-call totals across the whole panel (exact/rigid/permutation/combined) were: ordered PAM 0.057 s; ASE 1.529 s; Hungarian 0.466 s. These are environment-specific and exclude input loading and transformation construction. They are included as implementation cost context, not as a search-performance ranking.

The three pairwise comparisons among the distinct/ideal C60 references all exceeded 0.1 Å for all methods. They are retained as raw diagnostics with `ground_truth_assigned=false`; no pair is assigned basin truth from its graph or provenance. The ASE-built C60 is a geometry-only symmetric test object, not PES-qualified evidence.

## Meaning and boundary

The controlled self-pair transforms show that exact atom-order matching is not invariant to same-species reindexing, while the ASE matcher was invariant to all tested transformations. This makes permutation-aware geometry matching a plausible way to address that specific identity limitation, subject to broader validation of false merges and distinct-minimum separation. HungarianOrderMatcher is not qualified as a drop-in replacement by this panel because its rigid-transform misses were substantial.

Stable atom indices in historical trajectories do not rule out a real atom exchange or reorientation followed by the same unlabeled geometry. Conversely, this probe does not show that such a relabeling occurred in the historical pool or that its novelty/acceptance decisions were polluted. Historical-pool contamination remains untested; this is a matcher-invariance qualification only.

No threshold, reward, pool rule, API, or core behavior was changed. The exact inputs, per-query outcomes, timing, and provenance are in [result.json](result.json); the fixed protocol and stop rule are in [protocol.md](protocol.md).
