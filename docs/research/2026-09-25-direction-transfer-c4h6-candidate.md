# Candidate: bounded recovered-direction transfer check on C4H6/MH-1

**Status: deferred candidate; not submitted.** Routine bounded resources are already authorized; scientific decision value, not permission, is the remaining condition. No runner, settings change, API change, or default change is proposed.

## Question and decision value

If the C60 result leaves open whether the observed full-direction behavior is specific to that local-defect cage, ask whether the already implemented complete recovered-direction bundle produces a materially different early trajectory on the existing C4H6/MH-1 input than global direction with the same recovered-CBD limits. This is a small, outcome-informed development transfer check, not independent validation, an efficiency ranking, or evidence of general PES coverage.

The old C4H6 coverage analysis already provides a useful early comparator: at 20,000 charged requests, SSW61/native-LS61/SSW67/native-LS67 contain 31/30/32/31 frames including the initial structure, with 3/4/2/3 connected graph classes, no fragmented frames, fresh force qualification for all, and both torsion-sign regions represented in every arm. That is enough to make an early-trajectory comparison interpretable, but too little to rank methods: the counts are small, class counts include the initial structure, and the complete-direction bundle changes pair/group selection, local-mode mixing, and displacement continuation together. The full fixed-protocol study's 200,000-request common prefix remains the scale of its coverage analysis.

There is also a reason to defer or stop: if the C60 order probe makes the current bundle question irrelevant, or if only an efficiency/generalization conclusion would change the decision, this 20k panel cannot provide that evidence. Do not run it merely to add another system to a table.

## Fixed candidate protocol

- Input: the qualified ASE G2 trans-butadiene structure already used by `research/ga_ssw/evidence/c4h6-mh1-coverage-20260924/plan.json`, unchanged bytewise and with its recorded SHA256. Model: the recorded MACE-MH-1 model SHA, `head=omol`, CUDA, float64.
- Four new arms: `ssw` and `native_ls`, each with seeds 61 and 67. Compare against the existing same-seed SSW/native-LS records at their 20,000-request prefixes; do not rerun paper-LS or the old controls.
- Hold the C4H6 plan's SSW, optimizer, MC, LS/native-LS, prequench, input, model, runtime and all non-direction settings fixed. For global controls the old run used `RecoveredRotationSettings(pre_rotmax=5, rotmax=15, pre_ftol=0.2, ftol=0.02, metric='euclidean', max_force_calls=40)`. The complete-bundle arm uses the same CBD values and the already frozen `RecoveredDirectionSettings(ratio_local=50, local_probability=0.5, group_threshold=0.5, pre_rotmax=5, rotmax=15, pre_ftol=0.2, ftol=0.02, metric='euclidean', max_force_calls=40, c1_radius_policy='restricted')`. No parameter is selected from these C4H6 or C60 outcomes.
- Per arm: at most 20,000 charged search requests, 100 outer attempts, and 101 fresh checks (initial structure plus at most 100 landings). Four-arm total: at most 80,000 search requests and 404 fresh requests. One GPU, 40 minutes total wall ceiling, no retry, resume, or extension. Preserve failed initialization, failed attempts, and censored prefixes.
- Read out each arm at the fixed 20,000-request prefix: request accounting, completed attempts/landings, connected graph classes, fragmentation, CCCC torsion regions, and independent fresh force/energy checks. Compare with the archived same-seed prefixes; note explicitly that the added direction state changes RNG consumption, so identical seed does not imply matched random draws. A graph/torsion difference is an observed trajectory difference, not a causal contribution of one bundle component.
- Stop at the cap. Do not extend because few events occurred, tune settings, add seeds, or promote a default. A null or mixed result closes this bounded transfer question without a parameter sweep.

## Source and compatibility choices

The archived C4H6 controls used commit `57e5097c2a8546ccb6cf60a4396af662f19c58a1`, with core tree `46f0049ed02c773621ac614093386220227e3d3a`. That commit already contains the complete recovered-direction entry point and the same direction-bundle component files; a new arm on that frozen core is the preferred source-matched option. It avoids mixing the old controls with later `paper_reference.py` changes.

The current checkout is commit `1933208fe9face15765044b995e0227fa748aa8c`, core tree `a040dcffe6d029ded010da341d7cd29a99afae9a`. The core differs from the archived C4H6 core in six files, including `paper_reference.py`. The compact observer has an archived equivalence check on four Cu13/EMT trajectories, but that is not a direct C4H6/MH-1 equivalence result. If the current compact source is chosen instead, record the source mismatch and restrict interpretation accordingly; do not describe the comparison as same-core replication.

## Domain limits

C4H6 is a 10-atom, mixed C/H, nonperiodic, unconstrained molecular input and meets the present controller's basic finite-coordinate, at-least-two-atom, free-cluster entry contract. The result would still be limited to one small molecule, one qualified starting geometry, this ML potential and four short trajectories. It would not validate periodic/constraint paths, native LASP caller parity, Hessian stability, chemical accuracy, or global-search success rates.

Source anchors: `research/ga_ssw/evidence/c4h6-mh1-coverage-20260924/{plan.json,plan.md,run.py,report.md}`; `research/ga_ssw/evidence/c60-local-defect-20260925/direction-probe/plan.json`; `docs/research/2026-09-17-run5-direction-integration-gap.md`; and the source/core hashes above. The original C4H6 raw ledgers and full `analysis.json` are in the `ga-ssw-behavior-parity` worktree; the current worktree has the compact reports, not that untracked analysis artifact.

## Parent decision: do not launch this short panel now

The parent reproduced the four 20k-prefix counts directly from the original analysis artifact and verified the direction-component files are unchanged against the frozen source. However, a different trajectory after a bundled direction change is expected and by itself resolves no performance or mechanism question. The proposed small class counts cannot support retaining/promoting the bundle, while the controller already has executed mixed-species coverage at component level. Thus this protocol is retained as a scoped candidate but not selected for execution. Do not create its runner or submit it merely because the current GPU job is running. Reopen only with a concrete retain/delete decision and a discriminating outcome beyond trajectory difference; a full comparative protocol may be needed instead of this short probe. This decision does not assert the bundle is ineffective.
