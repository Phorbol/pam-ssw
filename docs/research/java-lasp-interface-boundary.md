# Uploaded Java GA ↔ LASP interface boundary

2026-09-10, read-only audit. No bridge or new feature implemented. Paths below are relative to `/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/` unless explicitly stated otherwise.

**The uploaded Java GA invokes LASP as a structure-search executable through files, not as a callable energy/force evaluator. It cannot directly accept an ASE Calculator object. This does not mean the underlying LASP energy evaluator is restricted to NN `.pot` files.**

## Concrete call and file protocol

| Stage | Exact source location under `decompiled/sgn/` | Behavior |
|---|---|---|
| Request a search | `app_ssw_ga/SSWGaSupport.java:129` (`SSWExplore`), `:174` (`SSWExploreInitial`), `:190` (`SSWOpt`) | Instantiate `LaspCalculation` using configured executable/template/potential paths, temperatures and SSW steps; call `runLasp()` |
| Create per-structure inputs | `lasp/LaspCalculation.java:59` | One numbered directory per input Model; serialize `input.arc` at :69; copy template to `lasp.in` at :81, replacing existing lines containing `ssw.sswsteps` and `ssw.temp`; copy configured potential file at :83 |
| Generate launch script | `lasp/LaspCalculation.java:88`, `:97`, `:104` | Default command `mpirun -np CPU/runNumPer LaspPath > output 2>&1`; Gaussian/external mode invokes `LaspPath output > output 2>&1`; writes `lasp.bash` |
| Start process | `lasp/LaspProcess.java:21` (`startLasp`) | `ProcessBuilder` runs `/bin/bash <taskdir>/lasp.bash` in task directory |
| Dispatch and monitor | `lasp/LaspCalculation.java:46`; `lasp/LaspDynamicLoadProcess.java:25`, `:44` | Fill a bounded number of concurrent slots; timer checks process liveness and replaces finished tasks. The inspected path counts process exit as task completion, not a force-convergence certificate; configured `endCondition` is not applied in that loop |
| Parse search output | `app_ssw_ga/SSWGaSupport.java:140`, `:181`, `:195`; `other/ArcFile.java:86` | Read `all.arc`. Explore collects minima and applies the carry rule; Opt sorts each file's records and takes its lowest-energy structure |
| Parse each structure | `other/ArcFile.java:106` (`getModel`) | Retains lines containing `Energy`, `PBC ` or `CORE`, split by `end`; energy from token index 3; cell lengths/angles and element/Cartesian coordinates. No force/stress certificate parsed here |

The Gaussian-mode helper additionally copies `lasp.external.sh`, `gaussian.inp.pre`, `gaussian.inp.after` (`LaspCalculation.java:115`). Its template-adjustment methods modify Gaussian executable, charge, multiplicity and resource entries (:138). Native script completion text is appended to the in-memory list **after** `FileUtil.write` at :104, so that appended string should not be assumed to be in the written script. This audit did not alter this behavior.

`SSWExplore` additionally selects the terminal structure from the trajectory containing a new best if the improvement exceeds 1e-4, otherwise from task zero, and appends that terminal model to its aggregated output (`SSWGaSupport.java:153–170`). Replacing the process therefore requires meaningful trajectory ordering, not merely an arbitrary bag of optimized structures.

## Energy backend: evidence does not support “NN-only”

Uploaded `GA-SSW_examples_run/global_exploration/README.md`, sections 1.2 and 3.8, says `CalSoft=lasp` is the supported search backend and explicitly instructs retaining an empty `lj.pot` when an NN potential is not used. That is a file-initialization requirement, not proof that calculations use an NN model.

Concrete uploaded templates demonstrate distinct LASP potential routes:

- `input-templates/TYPE0-LJ75/lasp.in:1`: `potential lj`.
- `input-templates/TYPE0-Au10Ag10/lasp.in:1`: `potential NN`.
- `input-templates/TYPE2-XXXII/lasp.in:1`: `potential lammps`; line 2 `explore_type rigidssw`; Java also copies its LAMMPS/rigid-body files.
- `GA-SSW/input/ssw_gaussian/lasp.in:1`: `potential external`.
- `GA-SSW/input/ssw_gaussian/lasp.external.sh:27–43`: read LASP `external.coord`, run Gaussian, then write energy and converted Cartesian gradients as forces to `external.ene`. Its stress-extraction section is empty, so this concrete script does **not** demonstrate a complete variable-cell external interface.

Thus ab initio external energy evaluation is explicitly anticipated by the upload. This bounded audit does not enumerate every DFT code or potential supported by the LASP binary, nor establish that arbitrary DFT backends work without extra configuration. The Java path leaves the LASP `potential` setting in the copied template and delegates actual energy evaluation to LASP.

## Dated update: external MACE bridge

2026-09-17: The fixed-cell nonperiodic external E/F boundary below has now been exercised with the actual LASP binary and persistent ASE/MACE on C60. This supersedes its earlier future-work status only for that tested scope; see [evidence and limitations](2026-09-17-c60-shared-mace-acceptance.md). No complete global-search or VC validation is implied.

## Two possible replacement boundaries, neither implemented at the original audit

1. **Whole-search executable boundary.** A replacement configured as `LaspPath` would need to understand the task inputs and SSW step/temperature semantics and write correctly ordered LASP-style `all.arc`, plus respect the generated MPI/direct launch convention. It could internally use independent Python SSW + ASE, but a bare ASE Calculator cannot replace an entire search executable. Protocol compatibility would not establish algorithmic parity.
2. **LASP external E/F boundary.** A future `lasp.external.sh` implementation could call an ASE-backed evaluator and write the native response format. That would retain native LASP SSW/optimizer behavior. Units, atom order, failure signaling and—for VC—stress/strain conventions require explicit validation. The supplied Gaussian script establishes this boundary's existence but is not an ASE adapter.

For the user's current goal of a fully independent Python SSW/VC-SSW, an ASE surface interface underneath the Python kernel is the direct architecture; neither original-Java bridge is necessary. GA additions are paused while that kernel is completed.

## What differs from a generic ASE GA setup

The Java package contains specific choices, not a uniquely general GA principle: type-dependent crossover/mutation (TYPE0 atomic, TYPE1 crystal, TYPE2 molecular crystal, TYPE3 molecular cluster, TYPE4 supported cluster), NNA descriptor-based archival/partition, per-region parent selection, explicit short/fine SSW scheduling and a trajectory-terminal carry rule. TYPE0's detailed integer quotas and corrected-versus-released reinsertion behavior are documented separately in `docs/research/type0-mutation-contract.md` of the PAM worktree. They must not be summarized as a fixed 50/50 operator probability.

The inspected Java schedule includes initial `3*OPTSSWStep` at `5*T` (`SSW_GA.java:43`), offspring optimization at `5*T` (:88), alternating quick-walk multipliers 1 and 4 (:116), and a later fine-stage step formula using `TaskNum/9` (:258). These are implementation-specific scheduling choices and not intrinsic requirements of GA+SSW.

Locally installed ASE GA components under `/home/gengjianrui/.local/lib/python3.12/site-packages/ase_ga/` include configurable `Population`/comparator (`population.py:25`), `CutAndSplicePairing` (`cutandsplicepairing.py:50`), rattle/permutation/mirror/strain/rotational operators (`standardmutations.py`) and an explicitly weighted `OperationSelector` (`offspring_creator.py:62`). They are reusable building blocks; saying “ASE GA” alone does not specify its operators, parent selection, relaxation backend or SSW schedule. A meaningful comparison must state those choices and match search budgets and input systems. No relative quality or efficiency follows from Java versus Python or package identity.
