# Historical GA / RC gap inventory

**Superseded implementation status:** the inventory below predates the RC forest,
RC-VC, constrained-substrate and TYPE1/TYPE2 controller implementations. Consult
[current family status](ssw-family-current-status.md) for live implementation
boundaries. Retained below as the original task rationale.

# GA / RC core completeness against recovered behavior

2026-09-10. Read-only live-code inventory of the independent worktree. No PES, new implementation, frozen experiment changes or test run. **GA has an independent working cluster controller with reconstructed TYPE0/TYPE3 operators; RC-SSW's generalized-coordinate kernel is not implemented. Neither is complete release parity.** This supersedes the early port specification's implementation-status paragraph, which said crossover, mutation and KMeans were not yet ported; that historical source specification remains useful for the recovered JAR behavior.

## GA: implemented operations and exact boundaries

| Layer | Live implementation evidence | Status and limitation |
|---|---|---|
| Atomic cuts/pool/crossover | `atomic_ga.py:72 cut_atoms`, `:113 build_atomic_pool`, `:158 cross_atomic_pool` | Reconstructed TYPE0 near-balanced cuts, gamete pools, composition-compatible pairing; explicit RNG, copied parents, finite attempt budgets |
| Atomic mutations/proposal | `atomic_ga.py:184 disturb_atoms`, `:200 exchange_atoms`, `:212 reinsert_undercoordinated_atoms`, `:252 mutate_type0`, `:296 propose_type0` | Actual independent geometries, not binary invocation. Reinsertion deliberately corrects the released JAR collision predicate and removes origin fallback; source-derived empirical scales are not universal physical defaults |
| Molecular cuts/docking | `ga_operators.py:92 cut_monomers`, `:142 gametes_match`, `:233 _fit_binding`, `:291 dock_gametes` | Whole fixed monomer units with identity and docking; actual JAR docking fixture referenced by `test_ga_operators.py:82` |
| Molecular mutations/proposal | `ga_operators.py:386 mutate_type3`, `:495 propose_type3` | Fixed-internal TYPE3 supported; tests explicitly reject the known bad mutable-monomer branch. Not a general molecule-chain coordinate implementation |
| Descriptor/archive | `legacy_descriptor.py:14 cluster_descriptor`, `:66 descriptor_similarity`, `:85 remove_duplicates`, `:96 merge_archive`, `:112 energy_window` | Reconstructed nonperiodic legacy NNA/projection behavior. It is not a newly established universal DCCD, periodic crystal identity, PES connectivity, or kinetic network |
| Parent regions | `population.py:12 partition` | Recovered first-three-projection Lloyd iteration rules, 100 iterations, 20-member cap and source energy scale; injected NumPy RNG differs from Java stream. Operational rejection cap prevents unbounded sampling |
| Region score | `population.py:89 rank_regions` | Recovered empirical energy/variance scoring. Mixed energy/energy-squared terms retain unit sensitivity; not a first-principles acquisition rule |
| Full independent cluster flow | `paper_ga.py:130 run_ga_ssw` | Supplied seeds → true quench → quick walks → GA offspring quench/short walks → ranked fine walks. Stores observations, stage costs/failures and optional structure validation; uses the independent SSW walker, not the JAR/LASP process |

`PaperGAConfig.proposal_type` accepts only **0 or 3**. TYPE0 rejects PBC/constraints and requires shared composition; its full mutation rejects undersized pure/alloy clusters. TYPE3 requires explicit disjoint, contiguous monomer groups and fixed ordered topology. `run_ga_ssw` passes `(0,)*len(groups)` to the TYPE3 proposal, deliberately limiting the active controller to fixed internal monomers. A monomer cut/mutation is a GA candidate generator, **not** a rigid-body SSW kernel.

The controller accepts an optional `ls` and passes the supplied SSW configuration to independent walks. That hook is not evidence of native LS scheduling parity. Initial/offspring quench has its own explicit BFGS call in `paper_ga.py:235 relax`; it is not automatically the Safe-total quench backend selected for the inner SSW. This is an independent numerical choice and a concrete consistency/documentation boundary, not an absent GA mechanism.

## GA: what is absent versus deliberately substituted

**Deliberate independent substitutions:** `paper_ga.py` explicitly does not reproduce Java's hidden 3/4/5 multipliers, forced first fine iteration, carry-tail behavior or divisor nine. It accepts explicit stage budgets and final-region count. These features are not accidentally missing from a controller claiming exact scheduling: the current controller intentionally implements the algorithmic three-stage architecture. Exact release scheduling would need a separately named execution policy if requested; it should not silently replace the independent controller. NumPy RNG and bounded failure behavior likewise prevent whole-trajectory Java equivalence, even where individual geometry fixtures agree.

**Actual missing coverage:** TYPE1 crystal GA operators/controller integration, TYPE2 molecular-crystal/chain GA integration, and TYPE4 supported-system operators are absent from `PaperGAConfig`. No periodic descriptor/identity backend exists in this GA path. There is no declared walker-selection interface for GA invoking block VC, joint VC or RC proposals. The controller imports only independent `paper_reference.run_ssw`; the existence of independent VC modules does not integrate them into GA.

**Initialization is supplied, not reconstructed:** `initial` and frozen descriptor references must be passed explicitly. The original TYPE0 collection of packing/ring/cage/random generators, TYPE3 initialization pool and original periodic system-specific seed generation are not implemented by this controller. This is a valid explicit-input interface but incomplete coverage of the supplied program's autonomous generation behavior.

**Not scientifically established:** exact DCCD semantics across all systems, truly structural archive uniqueness, reference-insensitive region discovery, GA cost advantage, and full multi-type coverage. Existing water/atomic development evidence and original-JAR fixtures do not imply those claims. Tests were inspected, not rerun here. Stage completion or a force certificate cannot certify chemical stability or a new phase.

Recovered source details remain in [Java port specification](ga-ssw-audit/java-ga-ase-port-spec.md). In particular, the Java quick/fine carry is a task-tail structure, not simply the minimum or MC current state; reconstructing it requires preserving task histories separately from archive deduplication. None of that should be guessed from the independent controller's current return schema.

## RC: core missing, source evidence available

No standalone rigid-body/rigid-chain coordinate module, RC SSW driver, chain topology state, generalized force pullback or RC calculator adapter is present in the inspected `pamssw/standalone/` tree. `cluster_frame.py` removes whole-cluster gauge motion; it does not constrain each molecular fragment, build a connected chain, or define independent molecular rotations. PAM's older `rigid.py` has the same zero-mode/projection role and must not be relabeled RC.

The available original RC paper DOI is **10.1021/acs.jctc.5c00350**. Full-paper/SI findings were already audited in [LS/rigid comparison sections 5–6](ga-ssw-audit/ls-rigid-comparison.md). Local official SI is `/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/literature/ct5c00350_si_001.pdf` and its text extraction. This inventory uses those archived source findings and does not infer new formula details from an abstract.

Missing RC operations, in dependency order:

1. Parse and retain explicit rigid groups, linking bonds, parent/child topology and atom ordering. Connected groups may share bond-endpoint atoms; the GA `_partition` disjoint-group validator cannot be reused as a general RC topology definition.
2. Forward map chain/generalized coordinates to Cartesian atoms, including reference conformations, angle-axis/Rodrigues conventions, zero-angle limits and periodic image bookkeeping. Independent free rigid bodies are an intermediate subset, not a complete connected-chain implementation.
3. Exact force/torque pullback, chain force transmission, and consistent allowed cell coupling. Paper transmit parameter and its role must be distinguished from LS response parameters; no universal value should be inferred from one system.
4. Generalized-coordinate direction/HVP, bias energy and gradient, climbing and local optimization sharing the same coordinate map.
5. Explicit release of rigid constraints and full-atom final quench before true-PES MC and minimum certification. A rigid-surface stationary point is not a full atomistic minimum.

Native symbols already identified include `rigidssw_` (0x4c0a00), `rigid_f_c2r_` (0x8132a0), `rigid_x_r2c_` (0x81c0f0), `rigidreset_` (0x818880), and `super_rigid_` (0x80bc10). They establish a native module and useful reverse-engineering anchors, not recovered transform/derivative parity. This task did not disassemble them. TYPE2-XXXII really supplies rigidbody/blist and movecell settings; water TYPE3 uses ordinary fixed-cell SSW despite GA moving molecular units. The two examples therefore test different inner kernels.

## Highest-value bounded next core task

For completing variant **mechanics**, prioritize a single RC geometry/derivative slice: close the exact input/output field contract of `rigid_x_r2c_` and `rigid_f_c2r_` using the existing TYPE2 rigidbody/blist topology and SI transform definitions. Implement only the independently specified forward map and work-conjugate pullback for an explicitly supported topology, retaining shared-endpoint semantics. Verify original-instruction cases if feasible and exact finite differences; do not expose a full RC walker until these agree. This targets a missing mathematical core rather than adding population heuristics. A full connected-chain E2E and unconstrained final certificate remain subsequent, explicit work.

For GA, the smallest separate coverage task is a walker/stage contract that preserves the existing GA archive/proposal semantics while admitting an explicitly chosen SSW variant with exact stage cost and returned candidate/current distinctions. It must not silently feed periodic structures into the present nonperiodic descriptor, or promise TYPE1/2/4 support before their operators and identity model exist. Exact Java schedule transcription is a bounded compatibility task, but ranks below RC mechanics and periodic GA correctness when the user's objective is a complete independent family of algorithms rather than literal control-flow mimicry.

These are prioritized implementation specifications only. No new module, experiment, performance claim or complete-parity claim is delivered by this inventory.
