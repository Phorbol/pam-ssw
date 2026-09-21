# P0: distinguish lattice-vector displacement from physical strain

2026-09-11. Zero-PES audit of the frozen Fe7C3 block-baseline ledgers and
`cell_before/cell_after` matrices. No algorithm or parameter changed.

The implementation uses the reported example rule
`||Delta L||_F = 0.15 ||L||_F` with a normalized nine-component cell direction.
That does not imply15% physical strain. With ASE row-cell convention,
`F = inv(L_before) @ L_after` maps the old Cartesian positions affinely to
the new ones. Singular values of F are the principal stretches. In particular,
`||F-I||_2 <= ||inv(L)||_2 ||Delta L||_F`.
Thus a fixed lattice Frobenius fraction can produce much larger relative
changes along short directions of an anisotropic cell. This is linear algebra,
not a fitted explanation or evidence that the paper's equation was miscoded.

The initial qualified Fe7C3 cell has volume686.104437A^3 and cell singular
values16.55405,9.55749,4.33653A. The twenty stored cell steps use the specified
0.15 Frobenius fraction. The largest absolute principal-stretch changes among
each five-cycle block are:

| Seed | Outer cell-only block | Outer cell+atomic block |
| --- | ---: | ---: |
| 7 | 0.308347 | 0.403901 |
| 101 | 0.350483 | 0.384455 |

These are finite single-step deformation magnitudes, not force/stress errors.
For seed101's combined block, the cell preparation ends at371.9020A^3,
about54.2% of the initial volume; its stored stress max is5.590eV/A^3 and
physical energy about529.17eV above initial. The subsequent fixed-cell atomic
climbing lowers some atomic energy but retains the high-stress cell, with
completed energies508–517eV above the outer reference. No new potential
validity boundary or failed-structure classifier is inferred from these values.

Evidence: `research/ga_ssw/fe7c3-block-baseline/cell-deformation-audit.json`,
`cell-energy-diagnosis.json`, the immutable source ledger and input.
The latter is an observed path; correlation alone does not prove that a smaller
step, more accurate soft mode, or retained direction will improve search.

## Priority decision

Before further fixed-cell atomic optimizer tuning, close the VC cell-displacement
contract: ds_cell/normalization, per-cycle state and allowed directions in the
uploaded native version, and the exact paper-vs-independent coordinate metric.
Check whether the existing6-EFS cell direction approximation produces useful
modes on this system. Do not label budget-limited modes as converged.
Only then specify one bounded controlled comparison. A strain-based alternative
must have a stated metric and objective-consistent derivatives; it is an
independent design, not native parity. Do not silently replace0.15 with a value
selected to rescue these same test trajectories.

## Direction diagnostic and interpretation fixed before evaluation

The next diagnostic uses four saved cycle starts (outer index1, cycles0/4,
seeds7/101), with central cell-Hessian differences at0.005 and0.0025A.
One center and36 displaced evaluations per point give148 additional EFS;
the hard cap is160. This is a fixed-fractional local cell-chart calculation,
not a search rerun or a full-system stability certificate. The same center
rotation projector is used throughout. Compare the saved direction with all
six nonrotational eigenmodes, its Rayleigh curvature, residual, and soft-half
subspace weight; check step-size sensitivity before interpreting them.

The2014 paper, section2.2/Fig.2 (journal17848), explicitly reports hybrid
modes after its six-evaluation rotation budget:57.8% in the three soft modes,
4.7% in the three hard modes and37.5% hybrid, for5000 quartz trials.
These are paper results, not our Fe7C3 results. Therefore failure to reach
the exact lowest eigenvector is not itself an implementation failure. A
saved direction concentrated in low-curvature modes would weaken the claim
that inaccurate mode solving is the primary cause of these trajectories;
large hard-mode weight would motivate a controlled direction-only comparison.
Neither outcome establishes end-to-end search improvement.

The native scalar move audit confirms ds_cell selection and componentwise
`trial = source + s*direction` in the inspected path, but upstream direction
normalization remains unresolved. See `native-vc-moveds-dscell-followup.md`.
Absence of a lattice norm at this scalar load does not exclude upstream scaling.


## Related coordinate-conditioning literature checked

Gubler, Krummenacher, Huber and Goedecker, *Efficient variable cell shape
geometry optimization*, J. Comput. Phys. X17 (2023)100131,
DOI10.1016/j.jcpx.2023.100131, https://arxiv.org/abs/2206.07339.
The checked abstract explicitly relates lattice-Hessian conditioning to cell
shape and particle count and proposes a coordinate transformation. It supports
investigating geometry before optimizer parameters; its detailed transformation
has not been imported from the abstract or equated to our proposed step rule.

Souza and Martins, *Metric tensor as the dynamical variable for variable
cell-shape molecular dynamics*, Phys. Rev. B55,8733 (1997),
DOI10.1103/PhysRevB.55.8733, https://arxiv.org/abs/cond-mat/9701085.
The checked abstract uses six metric components to remove orientation and
addresses cell-edge invariance. This is background, not evidence that our
unchanged entry-space direction solver is basis invariant.

The next proposed comparison changes only displacement length to
||L^-1 DeltaL||F/sqrt(3)=fraction. The sqrt(3) makes its numeric fraction
agree with the original rule in a cubic cell for every unit direction.
It is an independently derived proposal normalization, not either paper's
optimizer and not a change to the objective gradient. Its effect requires
an end-to-end comparison; full walker basis invariance is not claimed.
