# TYPE0 atomic/alloy mutation contract

2026-09-10. Scope: uploaded `sgn.jar` TYPE0 bytecode and independent ASE proposal integration. Source root: `/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/decompiled/sgn/`. This is a code-derived compatibility contract, not a mathematical justification of the original empirical mutation constants or evidence of global-search efficiency.

## Actual source behavior

`ga_Interface/TYPE0.java:getGA` flattens all supplied parent regions for crossover, then mutates each region separately. Each batch constructs a fresh `Cross` pool even when the requested crossover count is zero. For target G:

1. Generate `G//4` cut-and-splice offspring from all parents.
2. Region zero: call pure/alloy mutation with n=`G//2`.
3. Each later region in supplied order: call mutation with n=`G//8`.
4. Return the whole batch, with no TYPE0-local collision/bond-table filter or deduplication. `getFinalGAStructure` appends entire batches until size≥G, without truncation, stopping if a batch is empty.

Region zero is the first supplied region; TYPE0 does not rank regions itself. `SSWGaSupport`'s parent preparation sorts regions by their lowest energy; caller provenance matters. Each mutation method stably sorts its own parents by ascending energy (`Model.compareTo`). Random mutation parents are uniform among that region's sorted parents; best means index zero after sorting. Crossover competition follows its separate energy weighting.

### Integer quotas, not independent operator probabilities

The original code uses Java integer division; n is not the returned count.

| Ordered operation | Pure count | Alloy count | Parent | Number of changed selections / width |
|---|---:|---:|---|---|
| Species exchange (alloy only, comes first) | 0 | `3*(n//4)` | uniform | `10*N` independent index-pair swaps |
| Small disturbance | `n//4+1` | `n//8` | best | `N//10`, width 0.3 Å |
| Medium disturbance | `n//4+1` | `n//8` | best | `N//2`, width 0.5 Å |
| Broad disturbance | `n//2` | `n//8` | uniform | `N//2`, width 0.7 Å |
| Sparse broad disturbance | `n//2` | `n//8` | uniform | `N//10`, width 0.7 Å |
| Undercoordinated reinsertion | `n//4+1` | `n//8` | uniform | remove/reinsert 5 atoms |
| Undercoordinated reinsertion, best | 1 | 0 | best | remove/reinsert 10 atoms |

Totals: pure `3*(n//4+1)+2*(n//2)+1`; alloy `3*(n//4)+5*(n//8)`. Thus pure n=0 still produces four candidates, while alloy n<4 can produce none. Do not replace these with an invented 50/50 crossover/mutation probability.

`disturbance` selects an atom **with replacement** on each of m iterations, then consumes three independent draws and adds `width*(u-0.5)` to its coordinates. Repeated selections accumulate. A zero move count is a real no-op. `exchange` swaps species labels at two independently selected sites `10*N` times; equal indices/species are allowed. Coordinates are unchanged; atom count and exact composition are preserved. Neither is a uniform whole-structure displacement or a guaranteed nontrivial mutation.

## interMu reconstruction and measured released-code anomaly

`Mutate.java:interMu` computes coordination numbers using all nonperiodic Euclidean pairs and strict distance `<3.2 Å`, excluding self. Stable ascending sorting chooses the lowest-CN indices, preserving original index order at ties. Remove the first k atoms, retain survivor order, and insert removed species sequentially in that sorted order. Before **each** insertion, recenter the current partial cluster to its unweighted centroid and calculate

`R = 0.5*sqrt(dx²+dy²+dz²) + 1.5 Å`,

where dx,dy,dz are its bounding-box spans (`nna/BasicInfo.java:getMonR`). Draw `r=(0.1+0.5*u1)*R`, elevation=`(u2-0.5)*2π`, azimuth=`u3*2π`, then Cartesian spherical coordinates. This samples neither a uniform ball nor an isotropic sphere. The code is reproduced as an empirical geometric proposal, without a claim that its density is physically privileged.

**Released-code anomaly was executed, not just inferred:** `CooHandle.collisionDetection(list,0.3)` returns true iff all pair distances are ≥0.3 Å. `Mutate.interMu` bytecode offset 374 invokes it, and offset 455 `ifne 267` repeats while it is true. The routine therefore stops on a collision; after 10,001 attempts it inserts the origin unconditionally. Original JAR calls on Cu/Ag13 at seeds 17 and 71 both return structures with a pair below 0.3 Å and the original predicate false.

By the user's/root's explicit implementation decision, the independent production helper **corrects this acceptance direction**: accept all-pairs distance≥0.3 Å, and report exhaustion after the explicit per-insertion budget. It never pads with an origin atom. Operation name and telemetry say `atomic_reinsertion_corrected`; exact JAR parity is not claimed for this operation. This change is a basic geometric-domain correction, not a new mutation operator. Existing collision pairs among survivors also cause explicit exhaustion; no hidden repair is performed.

## Implemented ASE interface

`pamssw/standalone/atomic_ga.py` now provides:

- `disturb_atoms(atoms,n,width,rng)` → copied atoms and selected site indices.
- `exchange_atoms(atoms,rng)` → copied atoms and species-source site permutation.
- `reinsert_undercoordinated_atoms(atoms,count,rng,max_insertion_attempts=...)` → copied atoms and removed indices, output source ordering, attempted draws and correction flag.
- `mutate_type0(parents,energies,rng,n=...,max_insertion_attempts=...)` → existing `GeneticCandidate` sequence with full original pure/alloy quota schedule.
- `propose_type0(parents,energies,rng,min_ga=...,bond_limits=...,max_batches=...,max_cut_attempts=...,max_pair_attempts=...,parent_regions=None,max_insertion_attempts=10000)` → existing `ProposalResult`.

`parent_regions=None` means one region containing every parent; otherwise regions must form a nonempty exact partition of supplied parent indices. Groups in each candidate are singletons; `group_parent_indices` stores a parent index for each output atom. Mutation indices refer to caller order even after internal energy sorting. Crossover uses its existing two-parent per-atom lineage. Mutation/interMu preserve count and composition while site ordering may change. In the exchange operation, site geometry lineage and species lineage differ; its `species_source_atom_indices` records the latter.

Inputs are finite positive-Z, unconstrained nonperiodic ASE atoms, common composition (not necessarily common ordered species), and one finite energy per parent. Full pure mutation requires N>10; full alloy mutation is conservatively restricted to N>5 so at least one survivor exists for reinsertion. Crossover retains the native competition requirement of >2 total parents and positive energy span. Unsupported/degenerate inputs raise explicitly rather than manufacturing parents. Calulators and index-bound arbitrary metadata are not inherited by offspring.

The added `bond_limits` filtering is an **explicit independent caller filter**, not present in released TYPE0. An empty table adds no cutoff. It uses existing per-species pair-cutoff semantics; passing whole batches are appended, no truncation. Cut/pair/insertion exhaustion returns the existing partial-result status and reason. The failing batch is not partially accepted. Parent coordinates remain private copies; selected parents undergo the original crossover centering effect before subsequent mutation, including across successive batches.

No energy evaluation or quench happens in these proposal helpers. Controller integration must preserve candidate→true-surface quench→certified observation/archive→short SSW order. It must pass the region index partition rather than flattening it away, and allow alloy species ordering to change. `paper_ga.py` previously hard-coded TYPE3, contiguous rigid monomer groups and identical ordered species: that path cannot represent atomic mutation without explicit TYPE0 routing. Root owns that integration. Its existing three-stage schedule is paper-inspired and remains distinct from exact uploaded Java controller scheduling.

## Evidence and validation boundaries

Original-bytecode probe: `research/ga_ssw/java/Type0MutationProbe.java`. Only the probe is compiled; it invokes original `Mutate` bytecode using seeded Java `Math.random`, with reflection selecting the private disturbance/exchange primitives. `research/ga_ssw/run_type0_mutation_reference.py` regenerates `tests/standalone/fixtures/type0_mutation.json`, including JAR SHA256 and six cases. Four disturbance/exchange cases match Python to ≤2e-14 Å; two reinsertion cases confirm the released predicate anomaly. Existing cut/crossover bytecode fixtures remain intact.

`tests/standalone/test_atomic_mutation.py` additionally checks exact integer counts, parent mapping, composition, private copies, explicit exhaustion, region allocation, and whole-batch overshoot. Together with existing TYPE0 tests: **10 passed**. These are component/bytecode checks, not end-to-end scientific validation. Real Cu/alloy calculator runs through the full GA/quench/walk/archive pipeline are required before claiming useful global exploration; the empirical length scales need cross-system assessment and are not promoted as universal physical defaults.
