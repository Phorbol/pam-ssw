# Fixed-cell LS threshold separation: four-arm interface qualification

One controlled repeat of the existing reference-Gaussian constraint campaign is complete. The only search-parameter change was an explicit soft-prequench force tolerance0.1 eV/Å; prequench budget300, outer certificate0.03, seed11, one outer step and all other values were retained. SSW-only arms were not rerun. This is an interface/state-flow qualification, not a search-efficiency ranking or a parameter optimization.

New frozen evidence: `research/ga_ssw/evidence/constrained-ls-prequench-decoupled-20260912/`. Baseline: `constrained-gaussian-reference-20260912/`. Four planned arms, four completed; no censored/unrun arms. Supervisor wall time5.0201s, below the total180s and per-arm90s limits. Each arm retained3998 search +2 fresh request caps. Source snapshot, hash manifest, plan, generated arm runner, supervisor and raw evaluation streams are saved before/alongside results.

| Case / LS | Baseline status; search+fresh | Explicit prequench status; search+fresh |
|---|---|---|
| water dimer / paper | completed;119+2 | completed;106+2 |
| water dimer / native-derived | completed;120+2 | completed;118+2 |
| Cu111 / paper | completed;56+2 | completed;55+2 |
| Cu111 / native-derived | ls_prequench_failed;152+1 | completed;53+2 |

New total340 requests =332 search +8 fresh. Baseline corresponding four arms454 =447 search +7 fresh. Every baseline failure/cost remains in comparison.json. All8 new fresh checks pass the unchanged outer active-force0.03 criterion, energy consistency, exact fixed coordinates/cell, and preserved constraints. Soft preparations converged in2,3,2,3 optimizer steps respectively. A valid landing is not a Hessian-qualified new minimum or evidence of basin novelty.

The Cu native-derived arm now exits soft preparation by satisfying the explicitly looser threshold. The baseline had a genuine failed soft-quench state; that historical failure is not rewritten as budget exhaustion or hidden by the new result. Both runs still require converged preparation, so this comparison demonstrates separation of soft and true-force criteria. Fewer requests in this single-seed check are not a general efficacy claim.

Inputs retain the saved water dimer with O–O Hookean restraint and the13-atom Cu111/adatom case with fixed support and laboratory-frame Hookean plane. Water uses tblite0.7 GFN2-xTB accuracy0.001; Cu uses ASE EMT. Cu LS bond values remain explicit interface-test coefficients. This small mixed-backend comparison does not validate those coefficients as a material model.
