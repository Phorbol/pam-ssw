# Standalone TYPE3 molecular geometry operators

Implementation: `pamssw/standalone/ga_operators.py`. This independent Python/ASE module executes no Java process, reads no model files, and invokes no calculator. It reconstructs selected uploaded TYPE3 GA operations; it supplies complete TYPE3 proposal batches for fixed-internal-monomer groups (all changeTypes=0), but not a GA-SSW controller or the mutable-monomer TYPE3 branch. Scientific effectiveness remains unvalidated.

## Sources and recovered semantics

Original uploaded `sgn.jar`, decompiled under the external research root `decompiled/sgn/`:

- `other/CooHandle.java:56–88`: row-vector x/y/z rotations with angle `atan(tan(3.14159*(U-.5)))`; not uniform SO(3).
- `ga_cluster_cell/Cut.java` and `CutBasicAbstract.java`: geometric centering, repeated rotation and plane cuts until group counts differ by less than two.
- `ga_molecular_crystal/CutMC.java`: cut virtual monomer centers, cancel the ±0.3 Å separation, then restore full monomers using only final plane y rotation. The initial virtual-center random rotation is not applied to restored atoms, faithfully retaining the source behavior.
- `ga_molecular_crystal/CrossMC.java:84`: actual butt chain, `fitBinding(1.5, ...,5)`, monomer identity reorder, cell with coordinate extents +0.5 Å.
- `ga_molecular_crystal/MC_Base.java:97–209`: deterministic geometric docking and maxDock selection. The trial direction has length10 and translate() does not normalize it. A distance decrement of0.1 therefore displaces atoms by1 Å. Candidate0 is excluded, the ten smallest bounding-radius candidates remain, then minimum total inter-fragment pair distance wins. These values are compatibility facts, not recommended physical defaults.
- `nna/BasicInfo.getMonR`: half bounding-box diagonal +1.5 Å.
- `ga_monomer/MutateMonomer.monomerRotation`: select lowest-energy parent, rejection-sample a non-singleton monomer, rotate its absolute coordinates about origin, and concatenate groups in group order. This operation moves its center; it is not rotation about the monomer center.

## Public API

- `rotate_coordinates(positions, rng) -> ndarray`: original rotation distribution, no input mutation.
- `mutate_single_monomer(parents, energies, groups, rng) -> MolecularMutation`: `.atoms`, `.groups`, `.parent_index`, `.group_index`; explicit energies avoid calculator calls.
- `cut_monomers(atoms, groups, rng, max_attempts=...) -> MolecularCut`: `.son`, `.daughter`, `.plane_slope`, `.attempts`; gametes contain original `.group_ids` and complete `.fragments` (ASE Atoms).
- `gametes_match(son, daughter, n_groups) -> bool`: original monomer identity matching only, not geometry qualification.
- `fit_binding(subject, object_, min_distance=1.5, accuracy=5) -> DockingResult`: combined `.atoms` and selected `.candidate_index`.
- `dock_gametes(son, daughter, n_groups) -> MolecularChild`: complete actual CrossMC.butt geometry, `.atoms`, `.groups`, `.candidate_index`.

Groups must explicitly partition all input atoms with zero-based, nonoverlapping indices. Only unconstrained, nonperiodic clusters are supported. Shared rigid-chain joint groups are outside this contract. `Atoms.copy`/slicing keeps parents immutable and avoids inheriting cached energies. `pbc=False` is retained even when legacy geneCell dimensions are produced; a file-format PBC record does not authorize changing the physics.

`max_attempts` is a caller-specified execution budget, not a scientific parameter of the Java algorithm. The original balanced-cut loop is unbounded. On exhaustion the Python module raises `SamplingExhausted`; it never fabricates a fallback child. All-singleton mutation and malformed groups similarly report explicit errors rather than reproducing infinite loops. These failure contracts are deliberate differences from the legacy executable.

- `mutate_type3(parents, energies, groups, change_types, counts, rng, max_selection_attempts=...) -> list[GeneticCandidate]`: three real operations for all-zero changeTypes: rotate all monomers then recombine; fixed-monomer reconstruction from parent0 through shuffled sequential docking; rotate one monomer in the best parent. AtomUtil docking preserves cached radii, centers after each docking, and uses10N random swaps for combination order. The single-group selection retry is bounded.
- `propose_type3(parents, energies, groups, change_types, rng, min_ga=..., bond_limits=..., max_batches=..., max_cut_attempts=..., max_pair_attempts=...) -> ProposalResult`: caller selects/flattens regions; original Compete weights choose gamete parents, both random parent-index columns are consumed though only column0 is used;100*n_parent cuts per batch. Each batch has G//4 crossovers plus three mutation modes each (G-G//4)//4, followed by explicit pairwise cutoff filtering. Entire passing batches are appended, never truncated/padded. Return statuses distinguish target_reached, empty_batch, and budget_exhausted.

Each GeneticCandidate includes actual `.atoms`, `.groups`, `.operation`, `.group_parent_indices` (one per monomer in original monomer order), and `.details` including chosen rotation group, docking order or selected gamete indices, and proposal batch index. Parent indices always refer to caller-supplied order. The caller is responsible for attaching durable archive IDs. BLLimit is an explicit mapping from element-number pairs to Angstrom distances; no guessed elemental radius table is used. An empty mapping explicitly disables this legacy geometric filter; it does not certify chemistry.

Private parent copies preserve the original CutMC centering effect on later mutation in the same proposal run, while leaving caller archive Atoms untouched. This distinction matters because the single-monomer operation rotates about the global origin.

## Verification and evidence

`tests/standalone/test_ga_operators.py` uses an existing 45-atom water archive at `research/ga_ssw/evidence/staged-water/final-arc/0.arc`; no new all-atom model was constructed. It checks parent immutability, selection of the lowest-energy parent, unchanged unselected molecules, rigid internal distances, origin-based mutation, recovered CutMC restore frame, whole-monomer conservation, identity matching, failure behavior, and full cut→dock molecule identity/cell behavior.

`tests/standalone/fixtures/type3_docking.json` contains original-JAR output for deterministic MC_Base.fitBinding on the first two molecules of that real archive, centered as one six-atom structure. Python matches all returned coordinates with absolute tolerance2e-12 Å. The fixture records JAR provenance; `research/ga_ssw/java/Type3DockingProbe.java` calls the uploaded class directly, not a recompiled decompilation. The oracle ran only a small deterministic geometric operation, with no force backend, optimization, or search job.

Test command:

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q -p no:cacheprovider tests/standalone/test_ga_operators.py
```

Observed: **17 passed**, using ASE3.29.0; ASE emits NumPy2.5 array-shape deprecation warnings. Both implementation steps followed test-first failure then passing execution. This is geometry/behavior verification, not an energy/force or scientific end-to-end certificate.

A second fixture `tests/standalone/fixtures/type3_reconstruction.json` checks actual all-fixed-monomer reconstruction against the original JAR with a frozen Java random-draw replay. Both original-JAR coordinate oracles pass at2e-12 Å. A whole-batch test exercises all four operators: G=8 produces five candidates per batch and returns ten after two batches. Tests also reject pbc, constraints, empty/nonfinite fragment data before ASE concatenation can discard metadata; a forced singleton-selection replay confirms finite failure.

## Remaining TYPE3 work

Mutable-monomer reconstruction (changeType=1), population initialization, and the quick/fine controller are not implemented here. The changeType1 atom-level interMu branch has a verified problematic source condition: CooHandle.collisionDetection returns true when all distances exceed the threshold, but interMu continues while true, so it seeks a too-close pair before its original fallback. This standalone entry rejects changeType1 with NotImplementedError and does not silently manufacture a modified GA branch or label such geometries qualified. Fixed-internal water monomers do not use that branch. Compete n<=2 or zero energy span is similarly rejected explicitly instead of propagating legacy NaNs.

Proposal budget statuses are operational outcomes, not evidence of global search convergence. Full proposal generation without energy calls still does not establish GA-SSW or scientific end-to-end completion.

A common NumPy Generator enables deterministic Python replay; its stream is not the Java Math.random stream, so equal integer seeds do not establish cross-language random trajectory identity. Random CutMC/mutation have source-based invariant checks; deterministic docking and fixed-monomer reconstruction have original-JAR numerical oracles; the random cut and parent-pool stages do not yet have full cross-language draw replay.
