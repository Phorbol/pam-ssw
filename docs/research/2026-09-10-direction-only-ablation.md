# Direction-only rigid projection versus a complete cluster section

## Outcome and decision

For the same Cu13/EMT input, seeds 3/17, two solvers, 15 steps and 14-Gaussian
cap, **direction-only projection also produces 12 fingerprint-distinguishable
structures after strict true requenching**. All 12 representatives have positive
internal Hessians at both finite-difference spacings. A full fixed Eckart section
is therefore NOT necessary for nontrivial escape in this experiment.

However, the two approaches have different failure patterns and candidate sets:

| Geometry | Search requests | Failed moves / 60 | Stored records incl. four initials | Strict groups incl. initial |
|---|---:|---:|---:|---:|
| Unrestricted Cartesian directions | 31,097 | 2 | 62 | 1 |
| Direction-only rigid projection | 57,671 | 31 | 33 | 12 |
| Full fixed section | 65,585 | 10 | 54 | 12 |

Only five fingerprint groups are shared between the latter two sets; each has
seven exclusive groups (19 in the combined set, including initial). This is
relative coverage, not ground truth, and does not show either method dominates.
Same outer-step and per-stage limits do NOT mean equal force-evaluation cost.
No lower energy than the initial minimum is established.

Retain both as explicitly named alternatives. Do not promote the full section
on the premise that it is required for escape, and do not discard it solely
because the smaller intervention reaches the same group count. Next explain
biased-quench convergence and compare at genuinely matched E/F cost on multiple
initial structures before choosing a default.

## Intervention and independent API

The research launcher `research/ga_ssw/compare_cu13_direction_only.py` used the
existing Cartesian SSW driver with process-local solver wrappers. At each
Gaussian center, it builds the rigid tangent space, projects the original anchor,
and solves the projected rotation problem. HVP trial points lie in that local
linear section; the complete Gaussian-biased quench is unrestricted Cartesian.
No random stream or scalar search setting changes. Full-section runs instead
keep a reference for the entire escape. Local versus fixed reference is therefore
part of this comparison; this is not a one-line force-only ablation.

The same behavior is now available through the public independent API:

```python
config = SSWConfig(..., cluster_frame='direction_only')
```

The other values are `cartesian` (unchanged default) and `eckart`. Both cluster
options require an explicitly assumed isolated, nonlinear, unconstrained,
nonperiodic system with rigid-motion-invariant physical energy. No new empirical
scalar is introduced. The native rigid subspace is the provenance, while SVD,
Ritz/dimer, ASE optimizer and paper-height/MC rules still differ from the original
release. This is not complete native CBD parity or RC-SSW.

The public integration replays the first real Cu13 escape with both solvers,
including the failed-quench outcome, through regression tests. The complete
four-run scientific artifacts predate that integration and use the explicitly
archived research wrapper; do not relabel them full public-driver replay.

## Validation and costs

Evidence: `research/ga_ssw/evidence/cu13-direction-only/`. Input, full config,
seeds, source snapshots, all step results and initial/landing checks are saved.
The per-run direction-only request counts are 13,727 / 14,036 (seed 3 Ritz/dimer)
and 14,411 / 15,497 (seed 17 Ritz/dimer). All 31 failures are biased quenches.

All 33 stored records passed fresh 0.01-eV/Angstrom checks, then strict 1e-5
true requenching within 300 steps. The common validator uses sorted pair-distance
fingerprints at 1e-4 Angstrom tolerance. This representation is not injective;
the reported classes are distinguishable structures, not exhaustive certified
basin labels. Strict requenching is additional computation and can move structures;
these group counts apply after it, not to the loose-tolerance raw landings alone.

Each representative is checked with central Cartesian force-difference Hessians
at 1e-4 and 5e-5 Angstrom, symmetrized and projected into its 33-dimensional
internal space. The smallest internal eigenvalue across groups is approximately
0.03880049 eV/Angstrom² and remains positive at both spacings. This is local
numerical stability under EMT, not DFT or global-optimality certification.

Additional requests: 33 fresh loose-tolerance checks; 3,879 strict-quench/fresh/
Hessian requests; 31 failed-force reconstruction requests. Total for this
experiment plus these diagnostics: 61,614. Unit/regression and native-oracle
checks are separate development costs. Previous full-section and Cartesian
campaigns are not pooled into this denominator.

## What the failures show, and what they do not

Consecutive stored climbing centers now have median aligned internal motion
0.178–0.209 Angstrom across runs, rather than the approximately 0.002 Angstrom
of the original Cartesian experiment. This suppresses the previous almost-pure
rigid-motion behavior but is not a universal no-leakage proof.

`research/ga_ssw/analyze_direction_only_failures.py` reconstructs the COMPLETE
modified forces at all 31 failed endpoints from the stored Gaussian histories
and fresh EMT forces. It exactly reproduces each recorded maximum force.
Maximum force ranges from 0.010337 to 0.536874 eV/Angstrom; after instantaneous
rigid projection, every failed endpoint still exceeds 0.01 eV/Angstrom. Median
rigid squared-force fraction is only 0.005452. Thus these failures cannot simply
be dismissed as a pure rigid residual obstructing the force certificate.
This diagnostic is not a causal proof of why LBFGS failed within its budget.

Do not repair the count post hoc by enlarging the step cap or moving failed
structures into the archive. Next inspect optimizer trajectories/curvature and
prospectively compare quench policies at fixed E/F budgets. The physical target,
height rule and finite-temperature acceptance should remain separately identified.

## Native evidence changes the interpretation

[The new setconstraints oracle](native-setconstraints-instruction-oracle.md)
executes 94 original warm-cache function calls. For asymmetric nonlinear N=4/7/13
geometries, the full native map matches orthogonal rigid projection to 5e-16.
Inline Gram-Schmidt was located; the previous static concern about missing
orthogonalization was incorrect. A two-atom degeneracy remains problematic.
The root independently reran the probe to `/tmp/pam-setconstraints-root-check.json`.
Cold cache initialization and complete native walker execution are still untested.

The source evidence now supports direction projection as a recovered missing
component, while the full fixed section remains an independent geometry design.

Final software verification: the combined standalone/reproduction suite returned
146 passed (existing ASE/NumPy deprecation warnings). New public-driver tests
match each solver's first failed Cu13 escape, including final coordinates and
request counts, against the archived research-wrapper run. Three new reference
checks compare the Python nonlinear rigid projector with the complete archived
native matrices for N=4/7/13. No scientific-success claim follows from test counts.

LS preparation also recovered original H/C pair-table lookup functions; see
`ls-validation-next-step.md`. The raw native C-C value differs from the paper's
example, and caller scaling remains to be resolved. Do not mix those parameter
sources in a purported paper-vs-native comparison or block on requesting more
papers when the needed native evidence is already locally recoverable.
