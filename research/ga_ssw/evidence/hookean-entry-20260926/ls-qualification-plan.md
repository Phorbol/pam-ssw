# LS + native pair restraint state qualification

2026-09-26 development/interface evidence, not efficacy. Existing core a108def.
Question: do both LS response controllers retain V+U (excluding transient LS and
Gaussian terms) and exactly restore frozen pairs/strengths/runtime across an
ordinary step and an actual pool restart? A mismatch triggers focused debugging;
a qualified result closes this combination gap without promoting a policy.

Reuse qualify.py and exact preceding Cu13/EMT Hookean fixture; optional --ls paper
or native selects settings copied from tests/standalone/test_ls_pool_restart.py.
Native: Cu pair3 eV,2.8A,scale.1; paper:1eV,3A,target.001eV/atom; remaining defaults
saved in protocol. These are interface fixtures, not physical Cu parameter fits.
No search parameter sweep. Each mode four paths, two outer steps; assert response
state/frozen potential identity as well as all preceding trajectory/RNG/cost checks.
Each mode <=6000 search+4 fresh,270s wall; combined <=12000+8, one CPU task10min,
no GPU. On failure preserve output; no threshold or physics changes to pass.
New runner and script are archived by this commit; old source remains at a108def.

Execution correction: CPU1498315 failed on the resume call because the research
runner omitted ls=ls; the existing core correctly rejected changed settings
before further PES. Preserved653 search+1 fresh and failed runner under
runs-ls-paper/. Fix only forwarding; retry runs-ls-paper-v2, native retains first
run directory. Reduce each retry mode cap to5000 so cumulative hard search cap
is10653, below original12000. Fresh cumulative cap9 (one already spent plus8),
explicit bounded diagnostic retry, same scientific thresholds and parameters.

After CPU1498322: paper four paths1400+4, native weak fixture1366+4 pass.
Native scale.1 produces zero prequench steps/response, though normal table updates
execute. This is insufficient coverage of nonzero displacement response.
Single next probe uses existing NativeLSSettings default scale5, all else fixed,
not a fitted new value; <=5000+4/5min. Cumulative actual3419 plus cap5000=8419
search below original12000. Expected nonzero prequench motion; if absent or
numerically unqualified, record limitation instead of tuning. No efficacy claim.
Saved LS prequench geometries will also be independently recomputed (<=24 E/F)
to check base augmented energies vs stored true_energy_after; no extra search.
