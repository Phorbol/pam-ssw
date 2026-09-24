# Saved-path depth ablation

Question: are useful landing changes already present early in an LS-SSW climb, or does the remaining path create them? Prior12-case C4H6 prequench ablation returned close to the start, so deleting all climbing is unsupported. This diagnostic localizes where changes arise; it does not tune a global Gaussian cap or claim an adaptive stopping policy.

Reuse the same12 C4H6 saved outer steps (paper/native LS, seeds61/67, indices0/199/399) and four C60 native-LS steps: seed17093 indices5/69, seed17094 indices0/82 (first and last complete gaussian_limit records). Selection is by sequence/completion, not energy or success; these are development trajectories. C4H6 NG25/width0.1 and C60 NG12/width0.6 are separate inherited protocols, not interchangeable size scaling. Both use MH1/omol, true fmax0.03, Safe-total500; preserve original maximum400/1000 optimizer steps respectively.

From each saved path, directly true-quench the center of the second Gaussian (after1 biased stage) and the center after floor(NG/2) completed stages (12 or6). The saved next-stage center is the preceding relaxed endpoint; direction-only refinement evaluates copies and does not move it. No new biased trajectory, LS update, direction generation, MC, or model change.32 counterfactual quenches total; each max1000 requests plus2 fresh endpoint checks (new and original full). Total cap32064 E/F, oneV100 allocation30min including load/fresh, cooperative29min deadline. No automatic retries or expansion. Original failed/missing/truncated cases retained; fresh failure and force-nonconvergence excluded from qualified structural comparisons but charged.

Prefix cost includes saved stage requests plus original nonstage overhead; this is conservative attribution, not a measured rerun of the truncated outer algorithm. Add new true-quench cost to that prefix; keep diagnostic fresh cost separate. Reuse original full path cost. Compare graph identity, proper-rotation/permutation geometry where tractable, and energy, retaining failed outcomes. C4H6 conformational differences must not be merged solely by graph; C60 graph comparison uses1.8Å with1.64Å sensitivity, not a newly fitted cutoff. No Hessian/stability claim.

If early and full landings agree at much lower cost across both classes, this justifies a separately designed end-to-end stopping-policy study. If the full path changes topology/geometry or lower-energy access, retain depth and stop trying to remove it uniformly. Mixed outcomes imply path-dependent value, not authorization to add heuristic thresholds. The possible next decisions are explicit; no parameter selected from this diagnostic will be labeled independent validation.

CPU1483075 COMPLETED0 in11s;32case export/schema, selected C60 record indices and prefix nonnegativity verified. Core unchanged; execution runner and inputs committed979c65c before GPU1483081 submission. One GPU30min cap, no submission export override.

## Completed execution and next decision

GPU1483081 completed on 4v100n19 in100s, after153s queueing; account/QOS/partition were valid. The32 new quenches used1352 requests plus64 fresh checks=1416 total. All new and full endpoints passed independent fmax<=0.03; this is not a Hessian minimum certificate. Analysis explicitly compares against the starting structure because a lower-energy early landing may simply be a return, not productive exploration.

Decision: keep full climbing, do not promote an early-quench rule. Saved first-stage C4H6 endpoints all retain the start graph and small continuous RMS; later points create distinct structures. The32 rows reuse16 outer paths, with two depths per path and correlated trajectory states; they are not32 independent trials.

A zero-PES read of exactly the frozen source records gives300 C4H6 stages (median10 accepted biased-optimizer steps, max79) and48 C60 stages (median47.5, max186). All were force-converged. Biased quenches account for4661/7468 and2698/3211 stage requests respectively; see stage-cost-audit.json. These are selected-path diagnostics, not population averages.

The native counter semantics are already closed in docs/research/2026-09-12-fixed-noncrystal-counter-trace.md and 2026-09-12-fixed-lbfgs-rc-lifecycle.md: native climbstep counts optimizer RC dispatch returns, whereas Python bias_stage_steps caps Safe-total iterations. Do not copy integer limits as equal effort or reopen this reverse-engineering branch.

Next bounded question: preserve Gaussian depth and compare biased-force tolerance0.1 against0.2 eV/A, the upper end of the user's previously suggested inner range. Fixed paired first escapes with identical per-system settings can test sensitivity/cost, not establish long-run efficiency. Preparation is separate from execution and cannot change this frozen experiment.

CPU1483479 completed6s; expanded start/full/truncated readout and proper rotation/permutation self-check passed, zero ledger issues. Main-agent independent raw-result checks confirmed32 converged quenches,1352 search+64 fresh calls.
