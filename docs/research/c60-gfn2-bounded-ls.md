# C60/GFN2 ordinary versus paper LS: bounded stress test

Evidence: `research/ga_ssw/evidence/c60-gfn2-two-step-4000/`; runner `research/ga_ssw/compare_c60_gfn2_paper_ls.py`. Full frozen source, input, preregistered plan, every charged evaluation, quench observations, completed and failed stage records are retained. No public kernel changed.

The input is ASE's closed C60 cage, not a random carbon cluster. LS SI (`ct4c01081_si_001.txt`, section 3) specifies C60 response target **0.02 eV/atom**, distinct from C4H6 0.7. SI sections 7.3/7.4 supply T=150 K, displacement/width parameter 0.6 Å and 12 Gaussians. We retain the existing stricter numerical baseline: fmax .01 eV/Å, Safe-total 400 iterations, dimer HVP budget 100, finite difference 1e-4 Å, rotation tolerance .02, direction-only cluster frame. Thus this is an independent implementation test, not literal LASP trajectory reproduction. LS uses existing paper feedback (.03 initial fraction, xi .2, eta 1.8) and recovered CC energy 3.4468400478363037 eV / cutoff 1.6399999618530273 Å. GFN2-xTB via tblite 0.7, accuracy .001, differs from the original paper's NN/DFT PES.

Before PES: one seed (3), two attempts per arm, maximum 4000 E/F requests each (3997 search + at most 3 fresh), shared 600 seconds, one CPU thread, no retries. Arms execute sequentially ordinary then LS; consequently LS receives the remaining wall budget, not an independent 600 seconds. No performance comparison can ignore this censoring.

| Arm | Completed valid new landings | Search E/F | Fresh E/F | Wall s | Outcome |
|---|---:|---:|---:|---:|---|
| Ordinary | 2 | 319 | 3 | 131.69 | Two lower-energy acceptances |
| Paper LS | 0 | 970 | 0 | 468.53 | Shared wall deadline during Gaussian index 6 of first attempt |

Total 1292 requests; recorded wall 600.39 s includes shutdown/serialization. Both initial true quenches cost 7 requests. LS completed six biased Gaussian stages before interruption, retaining a seventh failed event with 45 charged requests. The failed outer record costs 963 plus 7 initialization = 970. Pre-soft-quench response was 0.00194410 eV/atom over 90 frozen CC pairs. Biased-stage convergence does not certify a true-PES minimum; no final LS landing or MC outcome was obtained. Its second outer attempt did not start.

Ordinary's initial and two landed geometries independently pass fresh GFN2 force checks (max forces .00959346, .00822855, .00748199 eV/Å). All remain one connected three-coordinated C60 network, graph-isomorphic under the explicit CC cutoff, with radius of gyration 3.52422–3.52434 Å. Same-atom aligned RMSDs to the initial relaxed cage are 0.000819 and 0.001091 Å; maximum pair-distance changes .002842 and .003503 Å. Total energy reduction is only 2.756e-5 eV. These observations are consistent with repeated relaxation of the initial cage, not evidence of a newly discovered basin. Neither Hessian stability nor a C60 global minimum was established.

LS has no fresh request because the wall cap also applies to validation. Its sole recorded minimum is byte-identical in positions/species/cell/PBC to ordinary's independently checked initial minimum (`offline-geometry.json`); this shared evidence does not become an additional LS fresh calculation. There are no valid rejected landings omitted: ordinary accepted both, LS generated none. Unfinished biased structures remain in the trace and must not enter a minimum archive.

This test exposes the cost of a sustained LS biased climb on a strong covalent cage, but cannot determine eventual LS success, relative global-search efficiency, or general effectiveness. The bound was retained without increasing budgets or tuning parameters after seeing results.
