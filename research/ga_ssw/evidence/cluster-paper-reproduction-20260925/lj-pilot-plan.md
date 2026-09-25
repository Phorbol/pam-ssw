# Frozen LJ pilot: existing global SSW with recovered CBD

2026-09-25, development experiment; not exact reproduction of SSW2013 success rates.

Question: does the current independent kernel reach published full-pair LJ minima on a tractable positive control (55 atoms) and a difficult landscape (38 atoms)? This precedes any increase in C60/GPU search. Competing explanations for a miss include short budget versus unsuccessful/inefficient escapes; retain first-hit, completed steps, qualified landings and full request cost to distinguish them. Do not tune after seeing one case.

Two independent random inputs per size, seeds 25092501/25092502. Uniform volume sampling inside radius 5.5 sigma follows the radius in Wales/Doye1997; uniformity, no overlap rejection, and no confining potential are explicitly our operational choices, not claimed SSW2013 settings. Raw inputs are shared by all subsequent controls; target coordinates never enter proposals. Ar is a label only, full pair LJ ignores it.

Potential: epsilon1 eV, sigma2.7 Angstrom, no cutoff/PBC; published SSW LJ scales. Gaussian width0.6, maximum14, kBT0.8 eV. Current Safe-total/history500, true fmax0.01 eV/A, bias fmax0.1, quench1000steps, fd0.001, rotation_bias1; recovered CBD pre5/rot15, tolerances0.2/0.02, euclidean metric, at most40 force requests. All numerical settings inherited independent implementation; no claim they equal paper defaults. Standard MC; no LS, pool, native pair/group or chemistry-specific radius. Core/defaults unchanged.

LJ55: at most500 attempts/200000 search requests/600 seconds per seed. LJ38: at most2000 attempts/800000 requests/1800seconds per seed. One CPU task, total <=2 million search requests, <=80 CPU minutes, Slurm90min. Fresh validation at most one candidate per trajectory plus final best (<=8 requests). No auto-retry/extension. This is 4 development trajectories, not a success-rate estimate. LJ75/M80 not submitted by this plan.

Stop at first force-qualified energy candidate within0.001epsilon of held-out published energy; save geometry for independent structural verification. An energy candidate is not yet structural success. Outside this stop, retain whole search until one frozen cap. Report starting-quench failure rather than resampling. On completion compare first hit and common cost to paper scale; do not equate paper average-to-success columns to our censored median. If LJ55 fails across both seeds, inspect the stage ledger before larger cases or parameter changes. A cap does not justify automatic continuation.

Reference qualification: CPU1487995 completed exit0 in2s. Independent FullPairLJ energies of Cambridge LJ38/55/75 all within1e-5epsilon of their table; max force components below0.04epsilon/sigma. The paper PDF Table1 LJ55 value -297.24847 contradicts both coordinates and Cambridge -279.248470; use the latter verified target, retaining the contradiction.
