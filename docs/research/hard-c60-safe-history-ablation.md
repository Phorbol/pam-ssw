# One-factor history ablation changes convergence on the selected C60 LS stage

Runner `research/ga_ssw/probe_safe_memory400_c60_stage8.py`; evidence `research/ga_ssw/evidence/hard-c60-safe-memory400-stage8/`. The pre-PES plan asks whether changing only Safe-total history10 to native-derived400 is sufficient to change this selected stage's convergence. Only the research process temporarily assigns `pamssw.relax._SAFE_LBFGS_MEMORY=400` and restores it afterwards. Armijo, curvature admission, step cap, initial inverse scale, tolerance and all other algorithm constants are unchanged. Production files/defaults remain unchanged.

Same original eighth-stage start,90 frozen LS pairs,8 frozen Gaussian terms, GFN2-xTB/tblite0.7 accuracy .001. Bound:423 optimizer E/F plus1 independent fresh evaluation,400 accepted steps,600 seconds, one CPU thread. Actual **264 optimizer E/F+1 fresh=265 physical E/F,248 accepted steps,137.09 seconds**. All248 s·y pairs are positive and retained. No retry or parameter sweep.

| Kernel | Optimization E/F | Accepted steps | Modified max force eV/Å | Modified energy eV |
|---|---:|---:|---:|---:|
| Safe history10 (archived) |423|400|.0113545810|−3477.832151868307|
| Original ELF history400 (archived) |314|309|.0097104135 fresh|−3477.825698664686|
| Safe history400 |264|248|.0098578890 fresh|−3477.819618536613|

The fresh energy differs by0 and force components by at most5.809e-7 eV/Å from the final optimizer evaluation; the fresh maximum force passes .01. All three solve the same *modified* objective, not the true PES after bias removal. Safe400 stops at a higher energy than both other endpoints; passing the chosen force threshold with fewer calls does not certify a better minimum.

Additional **zero-PES** cached-prefix check: feed the original history10 logged energies/forces into Safe400 and halt at the first coordinate mismatch. The first12 requested coordinates are exact matches; request13 (original global request1011), after11 accepted curvature pairs, differs by0.000578908 Å in its largest coordinate component. This is the point at which retaining more than10 pairs can first change the generated direction. It confirms the implementation change becomes active through history retention, rather than a hidden change to the initial conditions or bias. The fully archived history10 replay already matched all423 coordinates exactly. Cached-prefix failure after divergence is intentional, since the old oracle values cannot be used at new coordinates.

Fresh physical repeats have tiny GFN2 numerical differences, so this is not a bitwise physical-calculator determinism proof. In conjunction with the unchanged algorithm and exact cached-prefix agreement, the result supports **history retention being sufficient to resolve the selected step-limit failure under this protocol**. It does not show that memory alone explains every difference between native and Safe, that400 is optimal, or that global SSW/LS search improves. Native's unusual GTOL900 is not necessary for this particular threshold crossing. No public knob/default or new heuristic was added.

The stage was selected after observing failure. Original hard-C60 search remains failed and is not retroactively reclassified; no complete walker rerun or final bias-free quench occurs here. Accounting stays separate: this ablation265 new E/F; preceding native diagnostic315; hard-C60 comparison1806; qualification39; previous Ih study1292. All trajectories and failure/reference denominators remain available.
