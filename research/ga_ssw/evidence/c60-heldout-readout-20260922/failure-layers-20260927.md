# Random C60: failure-layer readout, 2026-09-27

Question: does the new LJ fragmentation observation justify prioritizing a
confinement/short-distance component for random C60? This is a zero-PES,
post-hoc analysis of the **existing** four held-out runs, not new trials.
No input, acceptance threshold, search parameter or algorithm changed.

`failure_layers.py` joins original per-attempt results with the archived graph
qualification using exact landing-energy order and complete counts. It also
closes all four 60000-request budgets including initialization and interrupted
last attempts. Final output: `failure-cost-layers-20260927.json`; the earlier
count-only output is retained as `failure-layers-20260927.json`.

| Input / rotation | Qualified landings | Fragmented | Accepted fragmented | Requests spent on fragmented landings | Best improvement in second 30000 requests / eV |
|---|---:|---:|---:|---:|---:|
|17095 / Broyden|60|5|0|5261|21.3411|
|17095 / recovered|86|4|0|2517|4.4399|
|17096 / Broyden|62|5|0|5628|10.2412|
|17096 / recovered|83|13|0|9415|0|

All 291 saved successful landings have numerical force certificates in the
original readout; these are not 291 new independent E/F evaluations.
Connectivity classifications agree at the existing 1.64/1.7/1.8 Å thresholds.
27 fragmented landings consume 22821/240000 requests (9.51%); none enters the
accepted chain. All four chains start connected. The four other attempts end
at the request boundary, not an unexplained optimizer/SCF failure.

## Interpretation and decision

The accepted chains do not become trapped as disconnected fragments in this
sample. Removing fragmentation could save some wasted attempts, but it is not
an evidenced primary fix for failure to reach C60's cage/energy target. The
9.51% is retrospective cost, not a counterfactual speedup guarantee.

Three arms continue to lower their best energy in the second budget half;
one does not. Thus a universal stagnation diagnosis is also unsupported.
These runs contain only 60–86 successful outer steps each; they do not settle
whether the method can eventually meet the random-C60 acceptance criteria.
Keep the prior prohibition on tuning/resuming these held-out seeds. No new
Hookean, bond guard, width scan or global controller is promoted.

Core-source followup independently confirms the inspected LS/Gaussian-to-bare
quench path removes temporary terms and hands off returned coordinates
(`paper_reference.py` around 1140,1208,1238–1249; `surface.py` default terms=()).
No demonstrated bug was found; no speculative new regression test was added.

Next useful evidence must address basin-changing search efficiency on fixed
protocols, rather than treating isolated LJ repulsion or small forces as the
universal blocker. Existing local-defect and C4H6 coverage results remain
separate from random assembly. Larger random-C60 work requires an explicit
predeclared decision signal; this analysis does not launch a production run.
