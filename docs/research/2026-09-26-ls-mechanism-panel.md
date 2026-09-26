# LS mechanism panel: existing toggle evidence, zero PES

**Current outcome.** Existing toggle runs allow a zero-PES check of refined direction support and accepted/rejected graph-changing landings. Refined Gaussian directions differ under LS, but their effective atom support shifts inconsistently by seed and is nearly unchanged in each whole-panel median. Connected, force-qualified graph-change events rise in the C4H6 panel and remain neutral/mixed for C60. The evidence is system-dependent and does not support a universal advantage. The LS-specific force contribution needed to explain *why* is absent from saved records. No new PES experiment is justified by this audit.

The offline reader is [audit_existing.py](../../research/ga_ssw/evidence/ls-climb-depth-panel-20260926/audit_existing.py), and its machine-readable readout is [audit.json](../../research/ga_ssw/evidence/ls-climb-depth-panel-20260926/audit.json). It used only archived JSON, made zero calculator/PES calls, and verified `initial + records == result.evaluation_requests` for each of eight arms.

## Evidence

| Paired environment | Equal-input toggle | All qualified connected graph-change events / accepted subset | LS preparation cost |
|---|---|---:|---:|
| C4H6 / MH-1-omol, seed 61 | SSW vs NativeLS; 400 records/arm | 39 / 1 vs 73 / 8 | 6,766 / 260,522 = 2.60% |
| C4H6 / MH-1-omol, seed 67 | SSW vs NativeLS; 400 records/arm | 50 / 0 vs 85 / 7 | 6,737 / 259,049 = 2.60% |
| C60 / MH-1-omol, seed 17093 | baseline vs NativeLS; 12,000 requests/arm | 11 / 8 vs 10 / 8 | 197 / 12,000 = 1.64% |
| C60 / MH-1-omol, seed 17094 | baseline vs NativeLS; 12,000 requests/arm | 11 / 7 vs 9 / 6 | 154 / 12,000 = 1.28% |

`initial_direction` is the sampled random anchor, not the solver-refined direction. In the frozen archived caller (`paper_reference.py` at lines 655–659, then lines 716–730 and 894), LS soft-prequench finishes first, the random anchor is sampled on that prepared geometry, and the anchor is passed to mode refinement; each Gaussian record's `climb[i].direction` stores the returned refined mode. Matched first anchors are exactly equal for the same seed and atom masses, which controls the random input and says nothing about LS changing the mode spectrum. The proper support readout is `climb[i].direction`: median effective atom count across recorded Gaussian modes is 5.108 (SSW) vs 5.236 (NativeLS) of 10 atoms for C4H6, and 28.489 (baseline) vs 28.285 (NativeLS) of 60 atoms for C60. First-attempt refined-mode support also splits by seed: C4H6 6.189 vs 6.161 and 6.233 vs 6.263; C60 20.254 vs 25.459 and 31.634 vs 21.302. Thus LS changes refined directions, but this support metric shows no consistent localization effect. Later same-index attempts are not paired physical states after the trajectories diverge.

The graph-change count is an event diagnostic using the archived fixed cutoffs and atom ordering, not a count of unique minima. The table separates all force-qualified connected candidates from their MC-accepted subset; MC rejection therefore remains visible. C4H6 exact distinct connected graph classes at the common 200,000-request prefix are already reported as SSW 5/5, paper-LS 6/8, NativeLS 10/8 across seeds 61/67; the corresponding fragmented classes are 2/4, 7/7, and 3/6. These existing data support a C4H6 coverage signal for NativeLS, with fragmentation and force/stability qualifications retained. C60's two matched inputs still produced 0/2 target cages and 0/2 reference-energy hits under NativeLS, with opposite best-energy changes (+2.20 and −10.19 eV relative to baseline). A changed graph event is not a C60 success criterion.

The same-state C4H6 ablation (`ls-prequench-contribution-20260924`) further separates the phases: all 12 selected prequench-only outputs remained near their outer-step starting geometry and retained its graph; structural changes appeared in completed LS landings. Its fixed sample was selected by record index and contains repeated/correlated points, so it shows where those particular changes appeared, not a generic causal guarantee.

The periodic OMAT-small TiO2 lifecycle data add a different physical environment but no LS-off control: two cells, three LS attempts each, with force-qualified chain execution. Observed responses were 0.48–3.53 meV/atom, far below the 20-meV/atom target. This is evidence that the existing controller ran and how weakly it responded in that short lifecycle; it is not evidence that LS helped or failed relative to SSW. The mainline documents that AlOH has no dedicated LS table, so it is not a valid NativeLS toggle control.

## Mechanism boundary and next action

The audited outputs save the sampled anchor, solver-refined modes, event displacement/landing, LS preparation cost, scalar response, neighbor count, and species-pair amplitude table. They do **not** save per-pair LS force vectors or a matched true-PES/LS force decomposition at the same geometry. Consequently, the exact mechanism question “did the pairwise softened terms redirect force into the bonds that later changed?” is not answerable from current trajectories, even though refined-mode support and graph-change outcomes are measurable. Missing force decomposition does not establish LS has no value: the C4H6 toggle data show a coverage signal while C60 remains mixed. A decisive attribution would require a separate future measurement decision. This task adds no probe, sample, compute budget, or job.

Verification: CPU-MISC job 1499456 independently reran the reader (zero PES); its `audit-root-check.json` is byte-identical to `audit.json` (`cmp`, exit 0). The reader now requires `--output NEW_PATH` and rejects existing outputs. The legacy `complete_attempts` field only excludes three named errors and is not a completion or qualification count; scientific denominators here use all records and the explicit landing certificate. No trajectory, source input, model, or core API was changed.

---

## Deferred draft: climb-depth panel

> **主审暂缓。** The following draft initially treated “how far must climbing proceed?” as the next mechanism experiment, but that reopens the existing climb-depth / early-quench branch. It is not the current execution recommendation. `prepare.py` is an offline extractor for a possible later review; it was not run, and no PES job was submitted.

# LS climb-depth mechanism panel (prepared; no PES run)

> **主审暂缓。** 本草稿最初把“爬升多深才出现重排”作为下一机制实验，但这重开了已有的爬升深度/早期淬火支线。当前主审要求先完成已授权 LS 开关轨迹的零 PES 跨环境读出；本草稿不是当前执行建议。`prepare.py` 仅为后续留档的离线输入提取器，没有运行，也没有提交 PES 作业。

**Question.** Do frozen LS searches need most of their Gaussian climbing sequence before the first useful, force-qualified structural rearrangement appears, or is the apparent benefit already present after the soft prequench / early climbing? This tests one mechanism dimension: the depth reached along an already-recorded LS outer attempt. It does not add an early-stop policy to PAM-SSW or compare the paper and native LS controllers.

## Why this gap remains

The 2026-09-24 C4H6 same-state ablation already quenched the soft-prequench output for 12 fixed saved states. All 12 prequench-only minima retained the starting bond graph and near-start geometry (aligned RMS displacement 0.00184–0.01956 Å); the complete LS landings changed graph in 2 cases and had substantial same-graph geometry changes in 7 more. This separates prequench from the completed walk, but it cannot locate when within the walk a useful displacement first appears. It also has only one physical environment.

The approved MH-1/omol C60 NativeLS trajectory records each Gaussian center and cost, while the OMAT-small TiO2 lifecycle run records 25-center attempts in two fixed periodic cells. These provide a second, periodic material environment without another LS search. The existing C60 full-path comparison had mixed best-energy outcomes, and the TiO2 lifecycle was explicitly not a search-performance test. Neither answers the depth question.

## Hypothesis and alternatives

**H1:** substantial useful rearrangement is present by the midpoint of the recorded Gaussian climb. A midpoint true-PES quench should yield force-qualified endpoints with measurable structural change at lower cumulative cost than the completed LS attempt.

**H2:** change emerges late, or only after the final true-PES landing/quench. Midpoint quenches will mostly return the source basin, fail qualification, or fragment; full attempts will remain the only level with useful structural change.

**H3 (confound check):** the selected centers can look displaced while their direct true-PES quenches return to the starting basin. This would mean the LS surface moves coordinates without yet crossing a physical basin boundary; raw LS displacement / response is not sufficient evidence of effective escape.

The inference is conditional on these recorded attempts. A positive midpoint result would locate a candidate mechanism within archived paths; it would not establish that truncating future searches improves an end-to-end search policy.

## Fixed sample and one changed factor

Use only completed archived NativeLS attempts; do not generate a new search trajectory.

| Environment | Frozen archive | Attempts selected before readout |
|---|---|---|
| C60 molecular cluster, MACE-MH-1/omol | `mh1-native-ls-equal-budget-20260920` | seeds 17093 and 17094; records 0, 5, 10 |
| Periodic TiO2, MACE-OMAT-0-small/omat_pbe | `native-ls-tio2-lifecycle-20260920` | rutile and anatase; records 0, 1, 2 |

For each fixed attempt, prepare three endpoints: (a) its saved LS soft-prequench terminal geometry, (b) the saved Gaussian center at index `floor((n_centers-1)/2)`, and (c) the already-saved completed physical landing. The depth is the only within-attempt contrast. Preserve each system's original input, model/head, periodicity, NativeLS table, rotation settings, and LS geometry settings. Do not compare absolute energies or raw request rates between model families/cells.

For (a) and (b), run a direct physical-potential quench with the same frozen true-quench tolerance, optimizer, history, and iteration ceiling used by that source trajectory. For (c), reuse the archived completed landing; make one fresh energy/force call for each of the three endpoint geometries using that system's original model. No LS retuning, new trajectory, continuation, or result-dependent resampling.

The middle Gaussian center is a saved center before its corresponding biased relaxation, not a claimed completed half-run minimum. Therefore a midpoint success is evidence that an early recorded center can seed a useful physical rearrangement. It does not simulate a live “stop at midpoint” branch byte-for-byte.

## Readouts, accounting, and decision

Keep execution status, numerical qualification, physical geometry, and search interpretation separate. Retain every failed or capped quench. The primary event-level readout is whether the physical quench is force-qualified at the existing `fmax=0.03 eV/Å` and remains in the intended connected structure class. For C60, report carbon graph components and graph change under the existing 1.64/1.70/1.80 Å cutoffs; cage/icosahedral success and MH-1 reference-energy success remain distinct and are not expected from this selected small panel. For TiO2, retain the fixed cell/PBC and report saved-order MIC displacement from the attempt start plus Ti–O/O–O/Ti–Ti neighbor counts using the existing source analysis convention. Do not claim a new minimum or phase identity from displacement or neighbor counts alone.

Report per attempt: number of climb centers, selected center index, archived preparation requests, prefix sum of archived Gaussian-center requests, archived complete-attempt requests, new quench requests, fresh checks, terminal energy/force, qualification, and geometry outcome. Plot event-level cumulative requests against first force-qualified rearrangement. The effectiveness metric is qualified changed endpoints per cumulative E/F request within each physical environment, with connected/fragmented and unchanged outcomes shown separately. Also report the paired difference in source-relative energy; a lower energy alone is not a successful rearrangement.

Hard new-work ceiling: 12 attempts × 2 new quenches (prequench and midpoint) × 1,000 E/F requests + 12 attempts × 3 fresh endpoint checks = **24,036 E/F requests maximum**. Existing source-path requests are separately included in each counterfactual cumulative-cost curve, not charged again as new work. No retries or extensions. Proposed allocation after review: one V100, 60 min hard wall, with a runner deadline at 55 min for ledger flush. Stop when all 36 endpoint rows are resolved or either cap is reached; censored rows stay in the denominator and ledger. This is a bounded development panel, not success-rate validation.

**Decision rule.** If midpoint quenches produce no additional force-qualified changed endpoints in either environment, abandon the “early climb already suffices” explanation and retain full-depth LS as the supported current behavior. If midpoint success appears in both environments at lower cumulative cost, keep early-center reuse as a mechanism finding only and require a separate live-prefix end-to-end ablation before changing any default. If outcomes split by environment, retain system-scoped conclusions and do not add a universal rule. Stop after this panel; no threshold search, extra seeds, or long-run extension is authorized by this protocol.

## Reuse and reproducibility

The zero-PES extractor/preflight is `research/ga_ssw/evidence/ls-climb-depth-panel-20260926/prepare.py`; run it from this checkout with `python research/ga_ssw/evidence/ls-climb-depth-panel-20260926/prepare.py`. It verifies that all 12 selected source attempts contain the required geometry and accounting fields, writes a compact `prepared.json` manifest, and records source hashes. It does not load a calculator or evaluate energy/forces. Existing quench-runner patterns are in `ls-prequench-contribution-20260924/run.py`; the MH-1 source adapter and immutable model settings are in `mh1-native-ls-equal-budget-20260920/runner.py`; the OMAT model adapter and periodic lifecycle convention are in `native-ls-tio2-lifecycle-20260920/runner.py`. The archived full result formats differ, so execution must dispatch by source/model family and preserve the corresponding original quench settings. No execution runner or job is included in this preparation deliverable.

Source directories and exact experiment contracts are under `/home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity/research/ga_ssw/evidence/`. Large trajectories remain in those directories; the preparation script reads them in place and does not copy them. Prepared manifest provenance lists exact SHA256 values so a later execution can reject changed inputs.

## Limits and exit criteria

Two C60 starting paths and two TiO2 cells with three attempts apiece are a mechanism panel, not an independent method ranking. Different models, sizes, and geometry classes mean only within-environment paired effects can be interpreted. The TiO2 lifecycle records only three outer attempts per cell; a null result there says nothing about long-run search efficacy. A center quench can return to its source basin, and force qualification does not imply a positive Hessian or chemical stability. No work proceeds if any source artifact is incomplete, input ordering cannot be reconstructed, the relevant true-quench configuration cannot be recovered, or the budget cannot be enforced in the eventual runner.
