# Cu13: independent SSW workflow passes, structural escape does not

## Evidence and scope

The independent Ritz and plane-minimizing dimer solvers were each run for 15
outer SSW steps at seeds 3 and 17, with at most 14 Gaussians per escape, on the
same cached Cu13/ASE EMT input. Full plans, input, script snapshot, step traces,
and independent force checks are in
`research/ga_ssw/evidence/dimer-ritz-cu13-escape/`.
This is a bounded diagnostic of one initial structure and potential, not a
cross-system efficacy campaign or a claim about published SSW performance.

| Seed | Solver | Search E/F requests | Failed outer moves | Stored initial/landings |
|---|---|---:|---:|---:|
| 3 | Ritz | 7692 | 1 | 15 |
| 3 | dimer | 8443 | 1 | 15 |
| 17 | Ritz | 8185 | 0 | 16 |
| 17 | dimer | 6777 | 0 | 16 |

Both failures are `biased_quench_failed`; the summary status `completed` means
execution finished, not that every move succeeded. Search cost totals 31,097
surface requests. Every stored structure passed a fresh EMT force check at
0.01 eV/Angstrom. These 62 records include four initial records and do not mean
62 distinct minima. The 62 fresh original-tolerance checks are additional to
the search-request column.

## Independent strict validation

`research/ga_ssw/validate_cu13_escape.py` requenches every stored structure at
1e-5 eV/Angstrom with a 300-step limit, then independently evaluates EMT again.
All 62 pass; this validation costs 1,762 additional E/F requests. The combined
energy span is 6.6191e-11 eV. Proper-rotation Kabsch alignment, retaining original
atom labels and allowing translation but no reflection or permutation search,
gives a maximum RMSD of 2.6736e-6 Angstrom to the first requenched reference.
Even before strict requenching, the maximum aligned RMSD is 0.002567 Angstrom.

This provides strong numerical evidence of repeated return to the same
structure. It is not a positive-Hessian certificate or a mathematical
classification of every possible basin. No new structural minimum is
established by these runs.

## Where the apparent walking goes

`research/ga_ssw/diagnose_cu13_rigid_motion.py` compares consecutive stored
Gaussian centers. Terminal displacements and failed stages without a following
center are excluded. Across the four trajectories, median unaligned RMS
movement is 0.619–1.046 Angstrom, while median aligned RMS movement is only
0.001773–0.002084 Angstrom. The per-pair squared-displacement reduction from
rigid alignment has trajectory medians 0.9999953–0.9999976.
This is a geometric measure, not a fraction of force, energy or work.

The implemented Cartesian projected Gaussian depends on
`(R - center) dot direction`, and its biased quench allows unconstrained rigid
motion. Even when a proposed direction contains internal deformation, a free
cluster can change this projection through rigid motion at nearly constant
true energy. The present evidence identifies rigid-motion leakage as a concrete
failure mechanism to investigate; it does not quantify all causal contributions
or establish that projecting just the initial random direction would fix it.

BP-CBD 2012 itself discusses unwanted translation/rotation convergence during
unbiased dimer rotation (uploaded BP-CBD.pdf, Section 2.1). That observation is
related but different: our traces also implicate the subsequent biased quench.
The original SSW article is DOI 10.1021/ct301010b,
https://pubmed.ncbi.nlm.nih.gov/26587640/ . Its published efficacy is not evidence
that this partial independent implementation already reproduces it.

## Consequence for development

Do not promote either solver or increase the Gaussian cap based on these runs.
Before integrating more global-controller operations, recover the original
cluster coordinate alignment/force projection at the native climbing and
optimizer interfaces and compare it with paper definitions and PAM's treatment.
For an isolated rotation/translation-invariant cluster, a prospective internal
coordinate or gauge-fixed formulation must treat the Gaussian energy, its
force, mode rotation and quench consistently. A force-only projection with an
unchanged reported objective is not an acceptable fix.

A follow-up must preserve this failing input as a diagnostic, distinguish
original behavior from a new formulation, and repeat end-to-end comparisons
with reported cost and structural identity. Surface/field/periodic systems
require their actual symmetries; removing all cluster rigid motions universally
would be physically incorrect. No such new projection is implemented here.

PAM comparison lead: this checkout already has
`pamssw/rigid.py:project_out_rigid_body_modes`, used for candidate directions and
Hessian-vector products in `pamssw/walker.py`. The independent Ritz/dimer
modules explicitly omit this projection. These source differences are useful
leads, not proof that PAM's entire biased quench already has the desired gauge
consistency or that copying the direction projection alone suffices.

Verification after this diagnostic and the bounded TYPE0 addition:
`PAMSSW_NATIVE_MC_ELF=<uploaded lasp> python -m pytest tests/standalone tests/reproduction -q`
returned 135 passed (ASE/NumPy shape deprecation warnings). This checks interface
and reproduction regressions, not structural-search success.
