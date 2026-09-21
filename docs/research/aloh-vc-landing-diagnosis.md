# Independent diagnosis of the AlOH joint-VC landing

2026-09-10. Input: completed original-chart one-step run
`research/ga_ssw/evidence/joint-vc-aloh26-l5-clean/result.json`.
This analysis performs no walker and no further optimization. The main result
is a structurally changed, higher-enthalpy landing with positive approximate
local curvature under the same MACE model, correctly retained as an observed
landing but rejected as the next Metropolis state.

## Metropolis and composition

The ordered elements remain H4Al8O14 (26 atoms). Enthalpy increases from
-177.07279145463974 to -176.48120292091858 eV, delta=+0.5915885337211648 eV.
At p=0 and 300 K, ordinary Metropolis probability is 1.152780629e-10.
The saved record says accepted=false; current equals initial. Rejection is
consistent, but positive delta does not logically force rejection in a
probabilistic algorithm. The random variate was not recorded, so this is not a
bit-for-bit replay of that draw.

## Periodic structural diagnostics

ASE neighbor_list includes neighbor-cell images, not just unique atom indices.
All directed pairs below 3.3 Angstrom and the underlying species-resolved
minimum distances are saved in geometry.json. A 6 Angstrom enumeration radius
is used for minima; distances beyond it would be reported missing. The quoted
coordination cutoffs are explicit descriptive thresholds, not fitted bond orders.

| Quantity | Initial | Landing |
|---|---:|---:|
| Volume (Angstrom^3) | 273.7493599 | 283.2072509 |
| Density (g/cm^3) | 2.6924775 | 2.6025604 |
| Closest O-H (Angstrom) | 0.9814700 | 0.9720091 |
| Closest Al-O (Angstrom) | 1.6895794 | 1.7001285 |
| Closest O-O (Angstrom) | 2.4286652 | 2.3681931 |
| Closest Al-Al (Angstrom) | 2.7239491 | 2.6914116 |
| Closest H-H (Angstrom) | 3.0341387 | 2.1292250 |

Every H has exactly one neighboring O for all four tested distance thresholds
1.1, 1.2, 1.3 and 1.5 Angstrom, in both structures. The nearest O identities
change for two H atoms (zero-based indices, full periodic images in data):
H8: O2 -> O15; H18: O15 -> O24. H4 remains associated with O1 and H20 with O16.
This is evidence of changed proton attachment geometry, rather than merely
uniform elastic strain. It does not establish a physical reaction trajectory
or barrier: biased SSW coordinates are not dynamics.

For Al indices [3,6,9,10,14,21,22,25], Al-O coordination at 2.3 Angstrom changes
[5,5,4,4,6,4,4,6] -> [4,5,4,5,6,4,4,5]. At 2.5 Angstrom these counts are unchanged;
at 2.1 the initial last Al is counted as 4 instead of 6, whereas landing counts
are unchanged. Thus some initial contacts are threshold-dependent; the full
distances are retained rather than promoting one threshold to exact chemistry.
These checks show no near-zero overlap or extreme volume explosion in this
landing. They do not certify the applicability/accuracy of the MACE model.

## Joint local curvature diagnostic

A new reference chart is built at the landing, with a common Cartesian
translation placing its mean atomic position at the origin. This leaves the
periodic physical state unchanged and reduces origin-dependent numerical
coupling. L=5 Angstrom is explicit. The 84 coordinates comprise 78 atomic and
6 orthonormal symmetric logarithmic strains; removal of three uniform atomic
translations leaves an 81-dimensional basis. No atomic rotations are removed.

Central gradient differences with h=1e-4 Angstrom build 81 Hessian columns
(162 E/F/stress calls), plus one fresh central call. The lowest two eigenmodes
are rechecked with h/2 and 2h (8 further calls). Total **171 calls in 44.921 s**,
within the 200-call/90-second cap, CPU float64 and one PyTorch/BLAS thread.
The local checkpoint SHA and all software versions, original and evaluated
coordinates, raw E/F/stress, exact analysis source, basis, raw Hessian,
symmetrized Hessian and eigensystem are archived in
`research/ga_ssw/evidence/joint-vc-aloh26-diagnosis/`.

All 81 symmetrized Hessian eigenvalues are positive: smallest 0.1076503899,
second 0.1371704902 eV/Angstrom^2. The antisymmetric Frobenius norm is
6.3662e-6, compared with symmetric Frobenius norm 277.2809. The lowest-mode
fresh directional gradient curvatures are 0.1076503945 and 0.1076504020 at
h/2 and 2h respectively; second-mode values are 0.1371705118 and 0.1371705135.
The positivity is therefore well separated from this finite-difference
sensitivity. Fresh central force/stress also reproduce the saved values:
fmax=0.0041468012 eV/Angstrom and max stress=3.6301819e-5 eV/Angstrom^3.

This is stronger local evidence than a small force alone. It remains an
approximate near-minimum diagnosis, not a strict stationary-point theorem:
the generalized gradient norm is still 0.015259 eV/Angstrom, curvature away
from exact stationarity depends on coordinates, and no tighter optimization
was performed. Only perturbations periodic in this 26-atom cell plus uniform
strain are tested; larger-supercell phonon instabilities are untested. It is
not a new-phase identification, thermodynamic stability result, global optimum,
DFT validation or efficiency comparison.

## Initial endpoint qualification and combined interpretation

A separate analysis now applies the identical landing protocol to the **initial
quenched endpoint**, also centered by a uniform physical translation, in its own
reference chart at L=5 Angstrom. Evidence is
`research/ga_ssw/evidence/joint-vc-aloh26-initial-diagnosis/`.
It used **171 requests in 45.798 s**, with no relaxation. All 81 internal
atomic/strain curvatures are positive, with minimum 0.0755075112 and second
0.2745618463 eV/Angstrom^2. Hessian antisymmetry norm is 6.7395e-6 versus
symmetric norm 271.1383. The lowest directional curvature is 0.0755075210 at
h/2 and 0.0755075493 at 2h, again robustly positive. Fresh initial fmax is
0.0031515887 eV/Angstrom, stress maximum 1.2291099e-5 eV/Angstrom^3 and
full generalized gradient norm 0.0075796 eV/Angstrom.

The initial analysis reused the diagnostic runner's 200-call emergency guard,
but its deterministic planned work was exactly 171 and no retry loop exists;
it performed exactly those 171, within this subtask's 171-call/90-second budget.
This metadata distinction is preserved in the frozen script and plan, rather
than altering their recorded values retrospectively.

Together, the two finite-cell, near-stationary positive-curvature checks and
the changed proton/Al-O coordination provide evidence consistent with a
basin-to-basin structural landing under this MACE model. They are not a rigorous
distinct-minima proof: the endpoints were not optimized to exact stationarity,
full permutation/symmetry equivalence has not been solved, and larger-cell
instabilities and DFT energetics are not tested. No new-phase label is assigned.

For PES exploration, the MC selection result and archive eligibility have
different meanings: this higher-enthalpy, physically force/stress-qualified
landing should remain an observation/archive candidate even though MC keeps
the original current state. The saved walker already retains it in `minima`.
Any later deduplicating archive should apply its declared structural identity
criterion; it must not erase the landing simply because `accepted=false`.

Total independent endpoint Hessian cost is **342 E/F/stress requests** in
90.719 s across two separately bounded CPU runs. This is diagnostic cost in
addition to the original 1,748 search plus 2 fresh endpoint requests, not a
search speed or efficiency improvement.
