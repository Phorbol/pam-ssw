# Saved-stage true-quench diagnostic

Purpose: distinguish useful early minima lost with later climbing from no repair in the selected directions. This is a post-hoc diagnostic of the completed local-defect experiment; not a search-policy comparison or independent evaluation.

Select all four method/seed arms, FIRST outer attempt only, endpoints after stages2,3,6 (last consistently cage-like stage, onset of distortion, and halfway). Selection uses a common stage rule, not per-path minima. Twelve endpoints. Input coordinates are next-stage centers; source current and final landing retained. Both hypotheses remain possible outside this sparse subset.

Reuse existing `climb-depth-ablation-20260925/run.py` execution unchanged except its fixed32 completion denominator now reads actual input count. Safe-total/history500, true fmax.03eV/Å,1000 iteration ceiling,1000 request cap each; MH-1/omol CUDAfloat64 unchanged. Fresh truncated AND saved full landing checked separately,2 calls/case. Total ceiling12024requests, singleV10030min (existing harness1740s deadline). No retry/resume/cap extension/full SSW run. CPU preparation verifies finite nonperiodic C60 geometry and creates fixed inputs/snapshots before submission.

Judge force convergence, three-cutoff Ih/cage/defect graph, energy relative to qualified Ih and starting defect, and prefix+quench cost. Preserve failed/censored cases. An early Ih or other qualified cage would motivate review of an exit rule across existing C4H6 and randomC60 evidence before changing anything. Returns to original defect would close this early-endpoint explanation within the sampled subset and prioritize existing direction/LS mechanism evidence. Different non-cage minima would show path sensitivity, not successful repair. No automatic new search follows.

Preparation1484891 failed before producing inputs or PES: wrapper import path omitted repository root. Corrected the import path; failed log preserved. No scientific configuration changed.
