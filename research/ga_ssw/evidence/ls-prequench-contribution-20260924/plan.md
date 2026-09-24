# C4H6 same-state LS prequench contribution ablation

Purpose: distinguish the effect of the LS softened-surface prequench from the
subsequent Gaussian walk at twelve already-saved paper-LS/native-LS states.
This reuses the existing 400-step MH-1 coverage trajectories; it does not rerun
SSW, create new trajectories, rank LS methods, or claim a global search effect.

Cases are `paper_ls` and `native_ls`, seeds 61 and 67, record indices 0, 199,
and 399. CPU preflight reconstructs each actual step start from the run's
`initial` state and only advances it after accepted landings. It exports that
step start, the saved `ls_preparation.soft_quench.atoms`, and the full SSW
landing to one JSON input per case. Geometry comparison remains for downstream
analysis: element-marked graph identity alone does not establish same basin;
paired geometry with atom permutation, energy, and butadiene torsion are also
needed. No Hessian or stability claim is included.

For each saved LS-prepared geometry, run one unmodified, fixed-cell true-PES
quench with the current Safe-total implementation, `fmax=0.03 eV/Å`, 400
optimizer iterations, and history 500. Record every actual E/F request and
failure, with a hard ceiling of 1,000 requests for that quench. Then make one
fresh true-PES E/F evaluation of the original saved SSW landing and one of the
prequench-only terminal geometry when that quench produced a terminal state.
Save all three inputs, the quenched terminal when available, reported and fresh
energies/forces, optimizer telemetry, and per-stage request accounting.

The fixed oracle is the qualified MACE-MH-1 model, `omol` head, CUDA float64;
no model switch or precision change is allowed. The upper request bound is
12 × (1,000 quench + 2 fresh) = 12,024 E/F requests. One V100 allocation is
capped at 20 minutes. There are no retries or automatic continuation. Failed
calls, denied calls, and wall/request-censored cases remain in the ledger and
summary. No job is submitted by this preparation task.

Run `python run.py --preflight` for the CPU-only twelve-case export and static
checks. GPU work is opt-in through `python run.py --execute` after review.

Execution: CPU1477850 completed in10s with12 inputs verified; root independently checked shell syntax, AST and cap. GPU1477853 submitted under the above20-minute limit. Main core/quench implementation is unchanged from the original coverage run. This runner and inputs are preserved in this experiment directory; only these bounded additions are uncommitted at submission. Earlier helper execution of the same preflight is not independent verification evidence.

Readout uses element-labeled graph isomorphism and minimum proper-rotation RMS displacement over graph-compatible permutations, plus paired energies and carbon-chain torsion when defined. A rigidly transformed/reordered saved molecule checks the geometry implementation. Same graph does not imply same basin; no new similarity acceptance threshold or Hessian qualification is introduced. The cost of original LS preparation remains part of the prequench-only route.

Literature context: Guan, Shang and Liu, JCTC2024, DOI https://doi.org/10.1021/acs.jctc.4c01081 describes pairwise penalties transforming mode space. This counterfactual is a project diagnostic, not an assertion that the paper proposes skipping Gaussian climbing.

Failure record: GPU1477853 exited signal11 after4s, before runs/ existed and before any PES call. No Python traceback. CPU1477882 completed identical imports in14s, zero PES. Slurm node4v100n06 reports NOT_RESPONDING and the failed job remains COMPLETING; this supports an environment-level recovery, not a proven crash cause. One manually reviewed replacement excludes this node and enables Python faulthandler, capped19minutes (with the first4s still within original20-minute total). No request budget increase or scientific setting change. No further automatic retry. Pending analysis dependency is moved to that replacement to avoid a stuck cleanup dependency.
Replacement job1477892; readout CPU1477876 now afterany:1477892. Failed job/raw logs remain archived separately.

## Current state: blocked by unexplained system cancellation

Replacement1477892 was CANCELLED by UID0 after2s on4v100n15, with empty Reason/AdminComment/SystemComment and no program stdout/stderr. CPU1477876 completed geometry invariant checks and explicitly reports all12 cases missing; zero new PES calls and no scientific ablation result. The first4s failure plus replacement2s allocation is6 GPU seconds. No further GPU submission is planned until this system cancellation is understood. The code does not modify the algorithm; execution checks the frozen core tree and clean core, allowing subsequent documentation-only commits. The12-case inputs remain prepared. CPU imports/geometry checks passed, but GPU quench execution is not validated by this attempt.
