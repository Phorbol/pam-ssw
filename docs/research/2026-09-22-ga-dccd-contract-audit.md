# GA descriptor and routing contract audit

Date: 2026-09-22
Scope: `legacy_descriptor.py`, `population.py`, and the descriptor/routing calls in
`paper_ga.py`. No PES or HPC execution was used.

## Decision

No descriptor-local behavior change is justified on the evidence currently
available. The current implementation is a bounded reconstruction of the
uploaded Java NNA contract, while the paper's DCCD is a different descriptor.
Changing the implementation to the paper equations would be a semantic
replacement, not a parity fix, and would alter the existing `paper_ga` routing
and archive identities.

The finite evidence gap is an original-JAR numerical fixture for the same
coordinates, element-pair table, and reference rows. The current hand-coded
tests therefore verify the recovered formulas and routing decisions, but do
not establish byte-for-byte Java RNG or floating-point parity.

## Formula and identity comparison

| Component | Paper/SI DCCD | Uploaded Java NNA | Current Python | Finding |
|---|---|---|---|---|
| First-shell discrete term | `N_i^{alpha,beta}` count | `getConfigureInfo`: per-species count | `n1` per-species count | Matches Java |
| First-shell continuous term | Printed Eq. 1: `D_i^{alpha,beta} = sqrt(sum_j sin[2*pi*(dist_ij/cut_ij - 1/2)])` (the PDF's line breaks/parentheses require verification) | RMS of `dist - 0.5*bond` | `d1` RMS of `dist - 0.5*bond` | Does not match Java; the printed paper expression can have a negative radicand for a single neighbor, so no square/absolute/mean correction is inferred |
| Higher-shell graph | hierarchical nonduplicate neighbors | Java `getAll2CooNum`/`getAll3CooNum`, excluding earlier levels and center | set unions with the same exclusions | Matches nonperiodic Java graph semantics |
| Higher-shell continuous term | per-level DCCD continuous vectors | `getBaseDI`: pooled RMS of `dist - 3*bond` | `d2`, `d3` pooled RMS of `dist - 3*bond` | Matches Java; not paper DCCD |
| Atom-row ordering | lexicographic fingerprint ordering | active `SortNeighbourInfo.sortCNS`: concatenated count vectors only | stable key `n1+n2+n3` | Matches active Java path; tie behavior is preserved |
| Similarity | Printed Eqs. 2/3 sum over species inside each level, then Eq. 4 sums levels with `W_N^beta`, `W_D^beta` normalization | `Sim.getSim`: six normalized component weights, max-denominator similarity | six components with the same zero/max rule and weight normalization | Paper aggregation is not interchangeable with Java; exact printed Eq. 1 semantics remain unresolved |
| Reference identity | fixed reference fingerprints `F_R` and projection `C_A={S_AR}` | supplied `baseConInfo`, one similarity per base | frozen `references`, one similarity per reference | Same role; no reference synthesis |
| Duplicate identity | paper similarity is a coordinate, not an identity predicate | `Classify.isSim`: every projection coordinate within `dv` | `same_projection`: same componentwise threshold | Matches Java; not RMSD/chemical identity |
| Population routing | paper describes grid/regions conceptually | `selectParentsByKMeans`: first 3 sims, random centers, 100 Lloyd rounds, cap 20 | same finite rules, injected NumPy RNG | Routing parity except RNG implementation |
| Scheduling | paper stage architecture | Java quick/fine multipliers, carry, and integer quotas | explicit paper-oriented replacement | Deliberate replacement; outside descriptor fix |

The paper equations and defaults are in `literature/GA-SSW-user.txt:61-165`
and the rendered uploaded PDF page 2; `literature/244-SI.txt:S12` supplies
weighting context but does not resolve the Eq. 1 print ambiguity. The Java
equations and control flow are in
`decompiled/nna/nna/Neighbour.java`, `decompiled/nna/nna/Sim.java`,
`decompiled/nna/nna/SortNeighbourInfo.java`, and
`decompiled/sgn/app_ssw_ga/SSWGaSupport.java:529-665`.

## Routing notes

`population.partition` intentionally preserves the Java early return when
`n <= k`, before filtering missing projections. It also preserves first-three
projection routing, tie-to-earlier-center assignment, 100 Lloyd iterations,
the 20-member elite-plus-weighted cap, and lowest-energy region ordering.
`max_draws` is an explicit finite resource guard; it raises instead of
silently changing the sampling distribution. NumPy's RNG is injectable for
reproducible Python tests but cannot claim Java stream parity.

`paper_ga.py` computes descriptors before any surface request and projects
every landing against the supplied frozen references. This is a valid routing
contract for the legacy descriptor. It must not be described as canonical
paper DCCD until a separate implementation is authorized with its own
reference construction and comparison fixtures.

## Disposition

Retain the legacy implementation and label it as such. Do not alter
`legacy_descriptor.py` to the paper sine descriptor in this parity task, and
do not alter scheduler multipliers or stage carry behavior under a descriptor
change. A future exact-DCCD task should introduce a separately named,
paper-equation implementation only after the Eq. 1 transcription is resolved
from an authoritative source or executable fixture; it must then be compared
against fixed numerical fixtures before switching any GA routing call sites.

The earlier draft of this audit described Eq. 1 as a mean `sin^2` expression.
That transcription was unsupported and is intentionally corrected here; no
implementation or test relies on it.

## Offline permutation follow-up

`research/ga_ssw/audit_legacy_permutation_invariance.py` is an opt-in,
zero-PES analysis script. Its concrete manifest currently contains the saved
water15 frames and the existing GA config; C60 and Cu55 are explicitly marked
unavailable because this checkout has no complete saved GA table/reference/
weight configuration for either case. The manifest sources water15's
neighbor range from `tests/fixtures/ga_ssw/water.json` and weights from the
archived runner, rather than falling back to guessed values. The script also
rejects periodic inputs and reports whether raw or full-sort projection drift
crosses the configured `0.0001` identity tolerance. It reverses atom order,
computes the legacy descriptor and projection, then repeats the comparison
after an experiment-only complete `n1,n2,n3,d1,d2,d3` row sort applied to both
candidates and frozen references. It does not modify controller behavior or
write new defaults. No script execution was performed here because the
delegated numerical check is reserved for root CPU and both accounts were
currently blocked by `AssocGrpBilling`.

Root input review found the original ARC frames carry `PBC=ON`. The
archived GA runner explicitly converted these to nonperiodic structures and
saved all three in `independent-water-gfn2-ga-02/initial.extxyz`. The reviewed
manifest uses those three explicit frame indices, rather than silently
changing boundary conditions in the new audit. No xTB calculation is rerun.

The prior login-host check remains provenance only: on `login-01`,
`pytest -q tests/standalone/test_ga_dccd_contract.py tests/standalone/test_population.py`
returned `12 passed in 1.04s`, and the three targeted modules passed
`py_compile`. No new numerical test or PES run was executed in this follow-up.
