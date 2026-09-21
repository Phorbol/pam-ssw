# TYPE4 fixed-support geometric operators

## Scope and scientific boundary

`pamssw/standalone/surface_ga.py` supplies independent Python topology,
rotating reload, disturbance/exchange, height-split reconstruction, all four
rebuild families and supported-cluster crossover.
No Java/binary calls, calculator calls, energy acceptance, or optimizer are
hidden in the physical interface. Cubic rebuilding does evaluate a separately
counted synthetic LJ/wall objective. These are experimental geometric operators;
a validated supported-cluster global optimizer is not established.

Source is the uploaded `sgn.jar`, decompiled with CFR 0.152 under
`/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/decompiled/sgn`:
`ga_Interface/TYPE4.java`, `ga_loaded_cluster/{CrossLoaded,MutateLoaded,
LoadedHandle,SuitLoad}.java`, `ga_cluster_cell/{Cut,Compete}.java`,
`other/{TemTools,CooHandle}.java`.

## Explicit geometry and API

- `surface_topology(atoms, substrate_indices, adsorbate_indices)` checks an
  exhaustive disjoint partition. PBC must explicitly be `(True,True,False)`;
  full right-handed cell remains fixed. Existing FixAtoms must match the
  explicitly frozen substrate. At least two adsorbate atoms are required.
- `reload_surface(..., rng, site_fractional=(u,v), min_distance=2.,
  accuracy=None, rotate=True)` rotates an intact Cartesian adsorbate and
  lowers it along the surface normal. Accuracy selects SuitLoad orientation.
- `disturb_surface(..., rng)` performs floor(.2*Nads) displacements sampled
  with replacement, width 2 A, followed by 10*Nads random species swaps.
- `cross_surface(parents, energies, ..., rng, n=..., site_fractional=...,
  max_cut_attempts=..., max_pair_attempts=...)` produces n candidates.
  All parents must share exactly the same support coordinates/species/cell.
  Source-default cut density is 10**(number_of_adsorbate_species+1) per slot;
  caller override is recorded. Compete requires >2 parents and a nonzero
  energy span. Parents/energies are caller metadata; no PES is evaluated.
- `surface_collision_free(atoms, bond_limits)` uses all relevant ASE periodic
  pairs, including in-plane self-images, excluding vacuum-direction images.
  Each species-pair minimum distance must be explicitly supplied.

Candidates contain ASE Atoms with FixAtoms, substrate and adsorbate index
sets, per-atom parent/source indices, operation and source/correction details.
Crossover/reload outputs are **unfiltered**; caller applies the explicit
collision filter then a real physical quench. No passed-geometry flag is a
minimum certificate. An intact unwrapped Cartesian adsorbate is required;
there is no implicit bond graph, image unwrap or rigid-body identification.
The current adapter freezes the whole designated substrate: support atoms
that should relax require a separate GA/static-support vs local-optimizer
constraint contract, not an inferred reinterpretation of segmentation.

## Recovered operators and deliberate corrections

CrossLoaded uses the same Cut and Compete as cluster GA. It cuts only the
adsorbate. Both parent-index columns are drawn, but column zero supplies cuts.
Within each offspring it draws sperm once, then resamples daughters until
composition matches. The released code appends the final pair even when its
matching budget expires; Python raises SamplingExhausted instead. For n
outputs, 2*floor(n/3) use SuitLoad accuracy 30, the rest use direct reload.
Selected support parent and all adsorbate source atoms remain in provenance.

SuitLoad tests the six signed Cartesian axes against `accuracy` spherical
Fibonacci directions, using minimal alignment rotations. It minimizes
`sum(z)-N*min(z)` with first-minimum tie ordering. This is geometric height
selection, not energy minimization. All operations use an explicit local
frame: e1 parallel cell a, e3 parallel a cross b, e2=e3 cross e1. This
retains the native z-surface convention while allowing a rigidly rotated
ASE cell. Bounding radii are computed in that same frame.

Native reload centers the standardized support, starts the adsorbate at
2*bounding-radius above its top and descends 0.125 A until a separation is
below min_distance+0.125 A. Python retains this quantized first-contact rule,
but computes first contact from full in-plane periodic distances. For pair
lateral distance rho and normal difference dz, the highest contact height
is `dz+sqrt(cutoff**2-rho**2)` when rho<cutoff. The maximum over pairs gives
the first contact; the nearest lateral image suffices because other images
have larger rho and no higher contact. This removes a potentially 10000-step
search without changing the step grid. Initial height is raised explicitly
if needed for initial clearance, and the source 10000-step bound remains.
A normal displacement <=0.125 A changes distance by at most that amount,
so the first crossing of min_distance+0.125 retains min_distance clearance.

Corrections are intentional and are not original trajectory parity:
1. The fixed support is never translated, wrapped, cut or rescaled. Native
   whole-structure standardization/final x/y/z translation is not applied.
2. Docking uses caller-supplied fractional lateral site; no guessed active
   site or unstated random placement rule is introduced.
3. Periodic contact/filter respects the two-dimensional physical domain.
   Native final filter is nonperiodic and can miss across-boundary clashes.
4. LoadedHandle.load erroneously appends loaded indices to the support list;
   explicit exhaustive topology and source-atom provenance correct this.
5. Native composition-mismatch fallthrough is rejected, not passed to PES.

The 2 A docking distance, .125 A step, .2 disturbance fraction, 2 A width,
10N swaps and orientation accuracies are empirical source compatibility
constants, not general physical optima. The supplied collision table and
lateral site remain explicit task inputs.

## Actual input and validation

Fixture `tests/standalone/fixtures/type4_tio2_au24o4.extxyz` is the first frame
of uploaded `GA-SSW_examples_run/global_exploration/input-templates/
TYPE4-TiO2@Au24O4/addition/add.arc`, with PBC explicitly changed from ARC's
3D convention to the independent slab domain `(True,True,False)`.
It contains 514 atoms, Au24O328Ti162. `segmentation.non` specifies support
1..486, adsorbates 487..514. Original lasp.in fixes 1..297 and separately
sets fixatommode 1..351; our test of all 486 frozen support atoms is an
explicit independent contract, not evidence of identical original constraints.

Tests check rigid-reload 28x28 internal distances, unchanged 486-atom support
and cell, exact species/source lineage in disturbance and crossover, true
periodic docking clearance, rigid frame covariance, vacuum-image exclusion,
in-plane self-image collisions and constraint/PBC fail-fast behavior.
These are zero-PES geometric checks on actual input. No physical relaxation,
SSW success, adsorption stability or global-search efficiency is established.

## Completed reconstruction and four rebuild families

`reconstruct_surface` implements the height>3 A split, preserving lower
adsorbate coordinates and their original atom indices. One output rotates
the upper fragment by pi about the normal; the second regenerates its
composition with TripleTangencyBallsPacking. If no upper fragment exists,
the source returns no structures, and Python likewise returns an empty tuple.
If no lower fragment exists, the source descends without contact up to its
10000-step cap; Python explicitly reports SamplingExhausted. The source
contact threshold here is 2.5 A with step .125 A (guaranteed floor 2.375 A).
Python uses bounded periodic first contact rather than source's unrelated
100/10 A starting heights, and restores atoms to their explicit partition
instead of retaining the inconsistent source lower/upper ordering.

`rebuild_surface` creates tangent packing, cubic (1,1,1), cubic (2,2,1),
and cubic (3,3,1), then loads each using SuitLoad accuracy 10. All atom groups
and species provenance remain explicit. Generated-coordinate provenance is
species origin, not inheritance of the old atom's spatial neighborhood.

Triple tangency uses a composition-average atomic radius. It places an
initial equilateral triangle, enumerates mutually adjacent triples with
pair distances in [1.9r,2.1r], then adds an available tetrahedral apex at
centroid +/- 2r sqrt(2/3) times the unit face normal. The first available
apex is preferred, and faces are randomly selected with a supplied budget.
The equivalent vector expression corrects the source's singular coordinate
divisions. No alternative packing rule is silently introduced.

Cubic generators are **not just uniform random coordinates**. Source code
creates 10 samples per aspect ratio, in a box whose volume is sum(8*r_i^3),
with initial placement distances >=1.5*average_radius. Each sample undergoes
up to 100 synthetic LJ/wall evaluations before the lowest auxiliary-energy
sample is selected. Pair sigma is r_i+r_j and epsilon=1 in source auxiliary
units; these are not physical energies of the material. The wall term is
10/(1+exp(10*x)) + 10/(1+exp(10*(L-x))).

`LJForDiff.getCubicPot` contains an energy/gradient inconsistency: it gives
both wall derivatives a minus sign, whereas the upper-wall derivative is
positive by the chain rule. Python retains that energy and corrects the
gradient. Stable sigmoid arithmetic avoids overflow. Independent SciPy
L-BFGS-B with history 5 replaces Java reverse communication, so convergence
norms and line-search trajectories are not claimed identical. Each of the
10 starts records auxiliary evaluations, terminal status, last evaluated
energy and gradient norm. The selected geometry is paired with its actual
evaluation; reaching the auxiliary budget is not a physical quench failure.

Atomic radii must be supplied explicitly from the intended source table.
The uploaded ElementPara.getAtomR returns tabulated diameter divided by 2:
O=1.269578/2 A, Au=2.574144/2 A in the tests. They must not be silently
substituted by an ASE covalent-radius table. Native HashMap traversal/random
generation of an unused Unlimited structure is not reproduced; explicit
ordered atomic input and injected RNG define the independent draw contract.

Validation after this extension: 7 surface tests passed in 1.50 s, including
both reconstruction outputs and all four rebuilt candidates on the actual
514-atom input, fixed support, composition and source indices. Synthetic
wall/LJ directional derivatives were checked against finite differences.
Each cubic family retains all ten auxiliary-run records with <=100 calls
each. These auxiliary calls are real proposal computation cost; physical ASE
energy/force/stress calls remain zero. No original Java trajectory oracle,
physical relaxation or supported-cluster global-search efficacy is implied.

## Full bounded proposal batch and three-stage controller

`propose_type4` now implements the entire family-count schedule in TYPE4.getGA:
G//4 crossover candidates; G//4 rotating reloads; G//4 disturbances; G//8
reconstruction calls returning zero or two candidates each; G//4 rebuild
calls returning four candidates each. Mutations select uniformly from the
stably energy-sorted parent list; rebuilding uses its first/best parent.
The crossover pool follows original input order and Compete probabilities.
Whole passing batches are retained. Each batch stores all generated candidates,
pass/reject masks, failures and auxiliary cost, including rejected candidates.
Sampling exhaustion reports completed auxiliary-family ledgers instead of
hiding spent work. For G=8 and nonempty upper/lower fragments, the actual
batch contains 16 candidates, not 8. This count and 300 explicitly capped
auxiliary calls were checked on the uploaded 514-atom example.

`surface_ga_reference.run_surface_ga` executes quick constrained walks,
partition/proposal/offspring quick walks, then ranked fine constrained walks.
It calls `run_constrained_ssw`, with each input initialized exactly once;
there is no hidden additional quench or duplicate Metropolis selection.
Only active-atom forces qualify the fixed-support certificate. Full raw force
is separately retained and may be large because fixed support is constrained.
All certified landings, including MC-rejected ones, are observations. Identity
matching independently determines archive entries; routing never certifies a
minimum or decides identity. Archive-row-to-proposal-to-atom lineage is recorded.

The caller must provide `routing(atoms)->three finite features` and
`matcher(a,b)->bool` appropriate to a fixed-support 2D periodic geometry.
Neither the 3D periodic descriptor nor a zero-stress certificate is reused.
Legacy partition/ranking (including its empirical mixed E/E^2 score) remains
explicit. A default scientific surface descriptor and surface-symmetry identity
contract have not been established: caller callbacks are a declared requirement,
not claimed native descriptor parity. The controller reports all physical E/F
requests, reconciles them by walker, and separately reports auxiliary LJ calls.

## Bounded physical EMT workflow evidence

Artifact directory: `research/ga_ssw/evidence/type4-surface-controller-emt/`.
`run.py` constructs Cu(111), 2x2x3, a=3.6 A, 12 fixed support atoms and
three active Cu adsorbates. Three supplied seeds differ by lateral offset.
The cap was 1000 physical E/F requests / 60 seconds; no GPU or HPC was used.
Main `result.json` uses the exact uploaded Cu radius 2.259876/2 A:

- completed: 3 initial quick walks, 7 offspring quick walks, 1 fine walk;
- quick and offspring walks initialize/quench with zero escape steps; fine
  performs one complete SSW escape, true constrained quench and MC selection;
- 188 physical E/F requests, 1083 synthetic LJ/wall evaluations, 0.4073 s;
- 12 qualified observations, 10 approximate archive entries, no failed walks;
- maximum active fmax 0.048768 eV/A (threshold .05), maximum full raw fmax
  0.216764 eV/A; frozen support and cell remain exactly equal to the input;
- final fine landing is accepted with delta E=+0.00037989 eV. This small
  response is near the loose force tolerance and is not a new-basin claim.

Routing in this demonstration uses the three Cartesian adsorbate-center
coordinates (height relative to support); matching uses species-homogeneous
minimum-image Hungarian assignment with max matched distance <1e-5 A.
It intentionally does not quotient support symmetries or infer basin identity.
Thus archive count is interface evidence, not distinct-phase/minimum coverage.
The best-energy change is ~0.00316 eV, also not an efficiency result.

An initial run used radius 1.180418 A before checking the precise source
entry. Its complete result remains `result-initial-radius-1.180418.json`
(185 physical requests, 1146 auxiliary calls, 0.4019 s). After correcting the
radius the main result was rerun. Combined spent budget is therefore 373
physical requests and 2229 auxiliary calls; the initial run is not silently
dropped or pooled into a success rate. Both are short integration checks,
not a tuned benchmark, production validation or scientific efficacy claim.
