# 2026-10-07: finite-budget atomic proposal can reach a different TiO2 endpoint

This completes a saved-stage development check, not a global-search acceptance
or native LASP reproduction. Both sizes start from the original block panel's
first failed atomic stage after five cell/partial-relax cycles. They are not
force/stress qualified minima; no initial structure is replaced.

GPU1663813, source04f4aa2 (core change6129c5f), and CPU1663814 completed.
Strict rotation exits after40 requests/no Gaussian in both sizes. The existing
opt-in force_or_budget path completes10 Gaussian stages in both, with every
rotation still explicitly unconverged/budget-released. All20 biased quenches
pass the unchanged .1eV/A gate. Both final cell quenches pass .05eV/A and
.001eV/A³ and independent cold checks. No budget denial/model error occurs.

| Size | Released stage search requests | Cold fmax | Cold max stress | Anatase target |
|---|---:|---:|---:|---|
|12|688|.0169213|.000211482|yes, both matchers and energy window|
|48|789|.0371634|.000192650|no|

Rotation cost is400/400, biased quench242/243, other climb20/20,
true joint quench26/126. Strict cost40/40 is retained, not dropped.
Total1557search/1497actual calculate plus6cold,24sec GPU allocation.
The opted-in path has not become a converged low-mode solver.

Competing explanation: the pre-existing cell disturbance already enters the
anatase attraction region, making Gaussians unnecessary. The follow-up control
GPU1663900/source3743147 runs exactly the same true-quench settings directly
on each identical saved input; CPU1663927/source19122e4 compares endpoints.
Both direct results cold-qualify; neither matches anatase and neither matches
the released endpoint under either frozen identity tolerance.

| Size | Direct requests | Direct E/eV | Released E/eV | Interpretation |
|---|---:|---:|---:|---|
|12|31|-105.62131025959208|-107.03384266014662|released finds qualified anatase; direct does not|
|48|27|-426.27185417112986|-420.5879172057339|released changes endpoint but raises energy|

Direct cost58search/56calculate plus2cold. Complete stage/control series:
1615search+8cold=1623paid requests,1553actual search calculate; no failed/
denied request. The earlier cell-panel and its qualification/probe costs are
separate series, not erased by this cheap saved-state control. Input cold
measurements are reused for the direct control, not new independent evidence.

Decision: retain the existing finite-budget option as an experimentally useful
proposal capability; default strict mode and all physical gates remain. A
precise mode certificate is not necessary for a physically qualified endpoint
in these two examples. This does not show every approximate mode is good,
stable global efficiency, Safe-total superiority or native numerical parity.
Approximate endpoint matching does not prove dynamical basin/TS connectivity.
Only12atom hits the predefined anatase target;48atom is a counterexample to
uniform energy improvement. No temperature, height, threshold or cell metric
is tuned; no long panel is automatically appended. Close this component check
and prioritize new-input fixed-cell C60 direction transfer.

Evidence: `run-1663813-{0,1,2,3}/result.json`, `requests.jsonl`,
`gaussians.jsonl`, saved centers and certified endpoints;
`readout-1663814/analysis.json`; `direct-1663900-{0,1}/result.json`,
full final E/F/stress and raw ledgers; `direct-readout-1663927/analysis.json`.
Source and model/input identity are in each provenance/effective config.
Numerical semantics regression: CPU1663773,32 real Cu/EMT/block/checkpoint tests.
Paper2014 Eq1–2 and CBD-cell motivate approximate directional escape and
cell/atomic alternation; the present OMAT-small PES/phase templates differ
from its empirical MA model. Native later-rotation Gaussian callback lifetime
is still unresolved, as documented in the protocol, and is not inferred here.
