# Isolated-cluster section: first qualified Cu13 exploration result

The explicit fixed-section experiment eliminates the previously observed
same-structure outcome on the tested Cu13/EMT input: after independent strict
requenching, 54 stored initial/landing records occupy 12 distinguishable
sorted-distance fingerprint groups, including the starting structure. Each
representative has positive internal finite-difference Hessian spectra at two
step sizes. This is local exploration evidence for one model/system, not a
complete SSW reproduction, an efficiency result, or a new global minimum.

## What changed and why

`SSWConfig.cluster_frame='eckart'` selects an experimental equal-weight linear
Eckart section. Default `cartesian` behavior is retained for explicit controls
and calculators whose external environment breaks overall rigid symmetry.
For a fixed reference X0, thin orthonormal rigid basis B and P=I-B B^T,
all Gaussian-era coordinates use X=X0+P(z-X0), while the full modified force
uses P F(X). Thus the optimization sees the exact pullback gradient, not an
unconstrained coordinate update with an unrelated force projection.

Both Ritz and dimer use the same section for force-difference trial points and
height probes. The section remains fixed through Gaussian accumulation. Pair
LS prequenching happens before establishing X0; after withdrawing the bias,
true quenching is unrestricted and always checks complete Cartesian forces.
The filter projects the complete modified gradient, including LS and every
Gaussian. Its force certificate is named `modified_cluster_section`.

The positive alignment branch of trace(X0^T X) I - sym(X0^T X), with centered
coordinates, must remain numerically positive definite. Branch loss is recorded
as `cluster_frame_failed`; no trial repair or hidden penalty is applied. Such a
record retains the last completed geometry. Linear/degenerate references and
zero internal anchors explicitly fail; the latter is an input-level error,
not a silently resampled direction. No branch loss occurred in the four runs.

These are local coordinate charts, not global nonsingular structural coordinates.
The caller must declare a free, nonlinear, nonperiodic, unconstrained cluster.
This does not implement RC-SSW or variable-cell SSW. LS+frame is supported in
code but not physically validated by the non-LS experiment below.

Mathematical sources, review and limitations:
[fixed Eckart review](cluster-eckart-design-review.md), including
[Szalay 2017, II.1–II.2](https://arxiv.org/html/1701.01823v1).
A separate [native audit](native-cluster-rigid-treatment.md) found explicit
rigid-mode treatment in the original random-mode and dimer paths. Its precise
arithmetic is not this SVD section, and no whole native biased-quench section
has been established. Do not label this option exact LASP parity.

## Fixed experiment and costs

Same Cu13/ASE EMT initial geometry, seeds 3/17, 15 outer steps per run, Gaussian
cap 14, and all scalar configuration values as the prior Cartesian failure.
Only the coordinate option changed. Final-code artifacts and source snapshots:
`research/ga_ssw/evidence/dimer-ritz-cu13-eckart-final/`.
The reproducible launcher is `research/ga_ssw/compare_dimer_ritz_cu13_eckart_final.py`.
The original Cartesian results remain in `dimer-ritz-cu13-escape/`.

| Seed | Solver | Search E/F requests | Failed moves | Stored initial/landings | Strict fingerprint groups |
|---|---|---:|---:|---:|---:|
| 3 | Ritz | 16005 | 1 biased quench | 15 | 9 |
| 3 | dimer | 16052 | 1 true quench | 15 | 5 |
| 17 | Ritz | 16917 | 5 true quenches | 11 | 7 |
| 17 | dimer | 16611 | 1 biased + 2 true quenches | 13 | 6 |

Union: 60 attempted moves, 10 failed, 50 force-qualified landings plus four
initial records. Search cost 65,585 requests (previous Cartesian 31,097), about
2.11 times as many requests at the same outer-step/cap limits. Search wall time
was 23.0–24.5 seconds per run in this environment; this is not a speedup claim.
Do not compare the solvers as though two seeds establish a ranking.

A preceding developmental section run cost 66,961 search requests and 55 fresh
checks. Review then added explicit chart checks to HVP/height trials and failure
recording; final code was rerun in full. The preliminary run is archived under
`dimer-ritz-cu13-eckart/`, marked superseded, and is not pooled with final results.
These are numerical chaotic trajectories: even roundoff-size projection changes
can lead to different later outcomes. Only final-code artifacts support the table.

## What independent validation establishes

`research/ga_ssw/validate_cu13_eckart.py` uses EMT, fmax=1e-5 eV/Angstrom and a
300-step quench limit, followed by a fresh force evaluation. All 54 records pass.
This stricter postprocessing can change structures substantially: one stored
landing near 10.291879 eV relaxes further to 10.174237 eV. Therefore the 12 groups
are explicitly **after strict requenching**, not a claim that all original
0.01-eV/Angstrom landings were already positive-Hessian minima.

Classification uses sorted pair distances with maximum absolute difference
1e-4 Angstrom. It is invariant to rigid motions and identical-atom permutations
but is not injective (homometric structures can coincide). Thus these are
fingerprint-distinguishable structures, not an exhaustive basin enumeration or
an exact permutation-alignment algorithm. The original Cartesian records all
strictly return to the initial structure within 2.7e-6 Angstrom aligned RMSD.

For all 12 group representatives, a central Cartesian force-difference Hessian
is computed at h=1e-4 and 5e-5 Angstrom, symmetrized, and restricted to the
instantaneous 33-dimensional internal subspace. All eigenvalues are positive
at both steps; the smallest across representatives is 0.07901966 eV/Angstrom².
This supports numerical local stability under EMT, not DFT stability or global
optimality. Full spectra, asymmetry measures and coordinates are saved.

Additional final-run validation costs: 54 original-tolerance fresh checks;
4,991 strict-quench/fresh/Hessian requests; 2 symmetry-probe requests. Final
search plus these checks totals 70,632 requests. The symmetry probe rotates and
translates the input, giving energy difference 5.33e-15 eV and maximum force
covariance error 1.40e-14 eV/Angstrom; it does not certify arbitrary calculators.

## What remains unsolved

The initial structure remains lowest at approximately 9.361357882 eV. Other
strict representatives range from 10.078311053 to 10.938343536 eV. New higher
energy proposals are rejected by the specified 300 K Metropolis rule; accepted
landings remain near the original structure. The result validates proposal
exploration, not accepted-trajectory interbasin transport or improved GM search.
Do not increase temperature post hoc and label the resulting campaign independent.

The experiment combines direction projection and a consistent biased-quench
section. It does not isolate how much comes from each. Next compare an explicit
direction-only control inspired by the native call path, recover native
setconstraints arithmetic with an instruction oracle, and then test additional
initial geometries and real systems. This could show the full section is
unnecessary for some systems; do not retain extra geometry complexity solely
because this first demonstration succeeded. LS, GA integration and broad
calculator compatibility still require their own end-to-end evidence.

## Software verification

`PAMSSW_NATIVE_MC_ELF=<uploaded lasp> python -m pytest tests/standalone tests/reproduction -q`
returned **141 passed**, with existing ASE/NumPy shape deprecation warnings.
Six new geometry tests cover derivative consistency of the complete EMT+bias
objective, rigid directions, invalid geometry, section quenching, the antipodal
branch, and failed-step recording. These checks support implementation contracts;
the actual Cu13 experiments above supply the structural-search evidence.
Stable PAM code and branch refs were not changed; this remains the research
worktree with preserved uncommitted reproduction work.
