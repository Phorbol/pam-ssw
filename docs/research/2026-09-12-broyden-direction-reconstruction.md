# Recovered Broyden direction: independent research comparison

The scientific question is whether the original history-based local rotation
root solver adds useful directions or lowers direction-solve cost relative to
our Ritz and plane-minimizing dimer alternatives. A zero tangent force does not
certify the lowest Hessian mode. The initial comparison must record curvature,
anchor overlap, residual, actual oracle requests and failures separately.

Implementation: `research/ga_ssw/broyden_direction_reconstruction.py`. This is
independent Python/ASE callback code and does not invoke the LASP binary.
History equations and spectral prefix removal/reset are supported by the
isolated instruction probes in `2026-09-12-broyden-full-update-recovery.md`.
The caller uses physical endpoint coordinates R0+dr*n and scaled tangent force
`-FACT1*dr*(Hb*n-(n.Hb*n)*n)`. Fixed G0=1, first-rotation algebraic retries,
40-degree cap and fixed-radius restoration follow the recovered numerical
path. History is retained between evaluated endpoints, never accumulated from
multiple algebraic retries at an unchanged force.

Two metric choices remain explicitly different: Cartesian Euclidean and the
executable's degenerate XYZ block-sum form. The latter is compatibility research,
not a proposed physical geometry. Weight1000, history bound50 and spectral
limit1e7 come from ELF; the input FACT remains explicit. Initial FACT=1 in the
planned screen is provisional, not an optimized or proven universal value.
FACT has inverse-curvature units (Angstrom squared per eV) when G0 is unitless
and its product with tangent force is added to positions.

The shared rank-one curvature bias is anchored to the initial direction. The
runner's termination deliberately uses directly evaluated finite-secant
residual in eV/Angstrom squared and a hard endpoint request budget, matching
the other independent solvers. Native ftol and CBD_PreRot's termination override
are not claimed to be replicated by this comparison API. Constraints are not
implemented. Every returned direction has its own evaluated endpoint.

Root verification: 14 targeted tests passed under mace_env, including nonzero
center, independent residual, first-only retries, null-seminorm failure and
spectral restart. On an independent six-dimensional diagonal H=(1,...,6),
seed11 anchor, both metrics reached a root near curvature2 rather than1; this
is a direct illustration of local-root selection, not an error corrected by
forcing lowest-mode behavior. Matrix checks are not evidence of search gain.

Next bounded test: shared qualified Cu13/EMT, fixed-cell Cu31-vacancy/EMT and
bicyclobutane/GFN2 geometries, two anchors each, four direction solvers and
common residual/endpoint budgets. Keep backend failures, costs and independent
certificates. Only if the implementation survives this screen should the
same intervention enter complete SSW trajectories; no default promotion from
fixed-geometry convergence rates alone.

## Fixed-geometry real-system screen completed

Evidence: `broyden-direction-multicase-20260912`, root audit
`audit_broyden_direction_multicase.py`. Three qualified initial geometries,
two shared anchors each; same physical calculator, fixed frame projection,
finite separation1e-4, residual tolerance.02 and endpoint bound100. All source
hashes, initial structure identities, paid requests and final certificates
were checked offline. All initial forces remained below.01 eV/Angstrom.

| Solver | Qualified direct certificates | Direction requests | Fresh certificate requests |
|---|---:|---:|---:|
| Ritz | 6/6 | 101 | 12 |
| Plane dimer | 4/6 | 494 | 12 |
| Broyden Cartesian | 4/6 | 363 | 12 |
| Broyden native block form | 4/6 | 382 | 12 |

Total: 1340 direction +48 independent certificate =1388 requests. No backend
failure occurred. `completed` means the bounded routine returned, not that it
converged. Cartesian Broyden failed the certificate on both Cu31 anchors;
native-form Broyden and plane dimer failed on both bicyclobutane anchors.
Ritz used fewer requests in each of the six matched comparisons.

Cartesian Broyden reached higher-curvature stationary directions for Cu13
seed11 (3.084 versus Ritz1.581) and bicyclobutane seed11 (3.970 versus1.550).
These observations confirm different root selection, not an SSW advantage.
The screen does not justify changing the default or adding new safeguards.
A bounded full-escape comparison is needed to determine whether the different
modes discover useful minima, preserving failure costs and common Gaussians,
quench optimizer and stopping tolerances. This screen used provisional FACT1;
ongoing parser provenance work is separate from performance-based tuning.

## Complete fixed-cell escape comparison (FACT0.05)

The source-backed native initial FACT0.05 was recovered before launching this
campaign; this was not selected by optimizing the FACT1 screen results.
Evidence: `broyden-ssw-multicase-fact005-20260912`, audited by
`audit_broyden_ssw_multicase.py`. Eighteen runs use three cases, seeds11/29 and
Ritz/Cartesian-Broyden/native-metric-Broyden. All run two outer proposals with
3000-request/90-second caps; none hits either cap. The shared plane-dimer
presweep remains at5, with total rotation bound100 including both centers.
Only its main solver callback changes. That callback receives the actual
rotation surface including deposited biases and frame projection.

Other controls: width.1, maxGaussians25, T150, innerfmax.1, outerfmax.01,
localmax400, Safe-total history10. No height, quench or MC rule changes.

| Main direction | Search requests | Fresh requests | Noninitial landings | Rotation failures |
|---|---:|---:|---:|---:|
| Ritz | 6841 | 17 | 11 | 0 |
| Cartesian Broyden | 6879 | 17 | 11 | 0 |
| Native-metric Broyden | 8250 | 18 | 12 | 0 |

Total21970 search+52 fresh=22022 requests. All52 saved minimum records satisfy
freshfmax<=.01, energy agreement1e-8 and exactly unchanged cell. These counts
include repeated structures and initial states; they are not distinct-basin
counts. All eighteen runs finish their requested two proposals. No backend
failure or request denial occurred. Native first-rotation factor updates,
trace metric, paid counters and both-stage budgets were audited.

Cu13 finds no lower structure in any arm. Cu31 changes are at most1.4e-5eV,
which does not establish distinct structures or useful global improvement.
Bicyclobutane best energy changes (eV):

| Seed | Ritz | Cartesian Broyden | Native metric |
|---|---:|---:|---:|
| 11 | -.3304321 | -.3305151 | -.3300368 |
| 29 | -.4226010 | -.4227272 | 0 |

Cartesian Broyden has no established advantage: nearly equal aggregate cost
and small differences between relaxed energies do not certify improved
search. Native-metric Broyden pays more and misses the seed29 energy reduction.
Two seeds and two proposals are bounded integration/escape evidence, not a
production global-optimization benchmark.

Decision: keep Ritz default; preserve the recovered Broyden as an explicitly
independent root-solving reproduction option. Do not promote the degenerate
native metric to default or add compensating tuning. Assess a minimal public
opt-in Cartesian interface so SSW/LS/GA can use the same verified component;
its value is algorithm reproduction and controlled comparison, not a claimed
speedup. Further performance work must justify new budgets by a specific
hypothesis rather than repeating one failing case.

An offline molecular connectivity audit (`root-molecular-graph-audit.json`)
finds all saved C4H6 minima remain single connected components under the
existing H/C distance-graph rule. Best structures match the butadiene
connectivity class in five arms; native-metric seed29 retains bicyclobutane.
This supports the chemical interpretation of the observed energy reduction,
but a distance graph does not distinguish conformers or prove TS connectivity.
