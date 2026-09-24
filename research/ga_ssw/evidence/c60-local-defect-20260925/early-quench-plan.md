# Saved-stage true-quench diagnostic

Purpose: distinguish useful early minima lost with later climbing from no repair in the selected directions. This is a post-hoc diagnostic of the completed local-defect experiment; not a search-policy comparison or independent evaluation.

Select all four method/seed arms, FIRST outer attempt only, endpoints after stages2,3,6 (last consistently cage-like stage, onset of distortion, and halfway). Selection uses a common stage rule, not per-path minima. Twelve endpoints. Input coordinates are next-stage centers; source current and final landing retained. Both hypotheses remain possible outside this sparse subset.

Reuse existing `climb-depth-ablation-20260925/run.py` execution unchanged except its fixed32 completion denominator now reads actual input count. Safe-total/history500, true fmax.03eV/Å,1000 iteration ceiling,1000 request cap each; MH-1/omol CUDAfloat64 unchanged. Fresh truncated AND saved full landing checked separately,2 calls/case. Total ceiling12024requests, singleV10030min (existing harness1740s deadline). No retry/resume/cap extension/full SSW run. CPU preparation verifies finite nonperiodic C60 geometry and creates fixed inputs/snapshots before submission.

Judge force convergence, three-cutoff Ih/cage/defect graph, energy relative to qualified Ih and starting defect, and prefix+quench cost. Preserve failed/censored cases. An early Ih or other qualified cage would motivate review of an exit rule across existing C4H6 and randomC60 evidence before changing anything. Returns to original defect would close this early-endpoint explanation within the sampled subset and prioritize existing direction/LS mechanism evidence. Different non-cage minima would show path sensitivity, not successful repair. No automatic new search follows.

Preparation1484891 failed before producing inputs or PES: wrapper import path omitted repository root. Corrected the import path; failed log preserved. No scientific configuration changed.

Corrected CPU preparation1484901 passed (12 geometries, zero PES). GPU1484902 submitted on8V100V0/rush-1o2gpu/group, singleV10030min, frozen8ff1251. No automatic search successor.

## Result and decision

GPU1484902 completed51s exit0; CPU1484915 completed6s exit0.264 quench requests+24 fresh=288,12/12 quenches converged and24/24 fresh force checks qualified. At depths2/3 all8 endpoints return to the source defect graph (three cutoffs), ΔIh1.22250–1.22310eV; at depth6 all4 are non-cages, ΔIh13.9074–17.5901eV. No Ih recovery. Under this selected subset there is no evidence of useful early repair subsequently lost; no further depth scan or early-stop default change. Same-graph identity is not a global basin proof. Raw and derived outputs in `early-quench/`;[analysis](early-quench/analysis.json).

Next: one existing full pair/group direction-bundle comparison at the same frozen inputs/seeds/limits, omitting separate recovered_rotation. This is justified by a mechanism absent from the present comparison, not by assuming it improves. Prior randomC60 evidence is adverse/mixed and remains part of the decision.
