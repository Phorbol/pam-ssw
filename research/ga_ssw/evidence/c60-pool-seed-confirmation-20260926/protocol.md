# Frozen-seed confirmation of C60 pool routing

Question: does the single-seed Ih repair signal persist under a new main/pool
RNG pair, or was it a one-trajectory outcome? Parent series ls-pool-routing-20260926
showed C60 PAM at21055 paid search requests with fresh fmax0.033914 and Ih graph
at all three frozen cutoffs, within0.000480eV of the reference energy. At the
common23877-request prefix MC/uniform had no Ih hit. C4H6 showed no pool advantage.
These mixed observations justify one seed confirmation, not tuning or promotion.

Same C60 isomer2 input, MH-1/omol model, Native-LS/rotation/full direction, MC,
ASE pool matching, descriptor/scorer, fmax0.05 and bias_fmax0.1 settings. Only
main seed changes181->197 and selector seed193->199, frozen before these runs.
Within this new seed, compare MC/uniform/PAM with the same input/main RNG. This
is independent RNG evidence on a development geometry, not a new-geometry test.

Keep parent per-arm100 outer attempts/80000 search requests/101 fresh cap and
1700s search deadline/1800s process cap. Three tasks, at most2 V100 concurrently:
maximum240000search+303fresh and1.5GPU-hours. Original search settings unchanged.
Expected actual request counts may be lower due the frozen time limit. Compare
only complete qualified observations at the common paid-search prefix; preserve
partial attempts and every failed/missing/censored endpoint/cost.

Acceptance diagnostics: independent fresh force; complete cage, Ih graph at
1.64/1.7/1.8 Angstrom and reference-energy window0.01eV reported separately
and jointly. Record first qualified joint hit cost, best true energy and
fragmentation. No metric based on pool size alone.

Decision: if PAM's advantage does not repeat or controls also repair at comparable
cost, retain a mixed result and do not promote/retune/automatically extend. If
PAM again repairs earlier, retain it as a candidate for later genuinely distinct
inputs; two seeds on one defect still do not establish general success rates.
No automatic third seed or random-C60 production campaign.

Engineering-only difference: worker removes optional MC identity regrouping
from the GPU completion path; CPU1501023 confirms exact unchanged search and
fresh result JSON. Active pool identity/selection remains unchanged.
