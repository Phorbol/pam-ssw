# GA03 complementary-gamete failure: exact pure-geometry replay

Conclusion: the first proposal failure was the explicit100-pair sampling budget, not a parent-ID or molecular-type matching bug. No algorithm or fallback was changed. This diagnostic makes zero calculator requests.

`diagnose_pool.py` reconstructs the RNG stream from seed20260909 by replaying the three quick-walk initial directions on their saved starting Atoms, then the one-region partition. All three reconstructed directions have exactly zero maximum difference from the saved vectors. Thus the subsequently rebuilt pool and pair draws reproduce the original failing proposal, rather than merely using a fresh arbitrary seed.

The actual selected parent observation IDs were1,0,5: the third initial-quench parent (ID2) had been replaced by its slightly lower-energy quick entry (ID5). The300 son and300 daughter gametes contain1218 valid ordered pairings out of90000. Of these1002 are same-parent and216 different-parent pairs. With independent uniform draws from the two fixed pools:

- Per-draw success probability =1218/90000 =0.0135333333333333.
- Expected number of draws until first match =73.8916.
- Probability of zero matches in100 draws =(1-p)^100 =0.25600088994.
- Exact replay first match occurs on draw128, son22/daughter186 (both local parent index0, which the source permits).

Source correspondence: `decompiled/sgn/ga_molecular_crystal/CrossMC.java:53–65` concatenates the two gametes' monomer sequence IDs, sorts, and requires exactly0..n_monomer-1. `CutMC` assigns these IDs from each parent's local monomer order; it does not include a parent identifier. Python `gametes_match` uses precisely the same rule. Different parents with complementary local-ID subsets are accepted; using only chemical monomer type or allowing duplicate/missing monomer IDs would change the uploaded algorithm. Same-parent combinations are also permitted.

Pure-GA budget comparisons from the identical pre-pool RNG state:

| max pair attempts | max batches | Returned candidates | Result |
|---:|---:|---:|---|
|100|2|0|No match before the first pairing budget expires|
|10000|2|2|Pairing succeeds; batch quota still cannot reach4|
|10000|4|4|target_reached; four crossover candidates, zero BLLimit rejections|

G=4 gives floor(G/4)=1 crossover and floor((G-floor(G/4))/4)=0 of each mutation per batch. Therefore two batches cannot reach four candidates. No padding, relaxed identity criterion, changed offspring weights, or substitute mutation was used.

An additional independently seeded pool built only from the three **initial_quench** structures (IDs1,0,2 in energy order) has1088/90000 valid pairings, confirming that the conclusion does not depend on the tiny quick-entry replacement.

The10000-pair diagnostic cap is an explicit computational cap, not an optimized scientific parameter or a universal guarantee. The four returned geometries are not independently evaluated minima and are not evidence of different basins; this diagnostic only checks actual candidate generation, matching and the existing pair cutoff. Summary values and exact paths are in`summary.json`.

Reproduce from the worktree root:

```sh
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python -W ignore::DeprecationWarning research/ga_ssw/evidence/independent-water-gfn2-ga-03-ga-diagnostic/diagnose_pool.py
```
