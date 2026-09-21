# TYPE1 expansion, TYPE3 auxiliary refinement and TYPE0 packing inventory

Primary source is the uploaded sgn.jar/CFR tree,
`app_ga/GaInitialStructure.java`, `lj/{LJBase,LJ_Monomer}.java`,
`ga_cluster_cell/MutateForCell.java` and `snowFlake/*.java`.
Implementation remains independent Python. This work performs no physical PES
calls and makes no new scientific-efficacy or minimum-certification claim.

## TYPE1 seed expansion now implemented

`initialize_type1` executes forced MutateForCell.ForDoping even on single-
element seeds. For request n its source batch counts are n//2 exchanges,
two n//8 disturbance groups and two n//4 disturbance groups. n=30 therefore
produces 35 candidates. Coordinates are Cartesian; cell is copied unchanged.
Sorting uses explicit seed metadata, and newly generated structures inherit
that metadata only for subsequent sorting; these are not evaluated energies.
Output origins flag this distinction explicitly.

The source generate1 for-loop accumulates the passing-batch count in its
condition, checks whether count<=30, and appends only in the loop body.
A first passing 35-member batch is thus silently omitted. Independent code
corrects this by appending the complete passing batch before testing the
count target. Defaults 30 new structures, mutation request30 and100batches
come from source and are explicit arguments. Full-image periodic filtering
is the already documented independent correction to native 2x2x2 filtering.
All candidates/pass masks and parent population indices remain in the ledger.

## TYPE3 optional LJmonoOpt now implemented with consistent gradients

`molecular_auxiliary.MolecularLJChart` represents each supplied physical
monomer by its center plus three Euler angles. For row coordinates,

    R_i = (R_i0 - center_i0) Rx(ax_i) Ry(ay_i) Rz(az_i) + center_i.

Energy contains only atom pairs belonging to different monomer groups:

    Eaux = sum_intergroup 4 [(sigma_ab/r_ab)^12 - (sigma_ab/r_ab)^6].

Sigma comes from the caller's explicit table, epsilon=1 in source auxiliary
units. Intramolecular geometry remains rigid, and internal LJ contributions
are absent. Native LJ_Monomer uses forward finite differences dx=.001 A and
da=.00314159265 rad, then assigns the negative of monomer i's angular
response to monomer j. Unlike paired Cartesian translation derivatives,
rotations about two different molecular centers do not have opposite
coordinate derivatives in general. That source gradient is inconsistent.
The independent implementation retains the source energy/Euler order and
uses each monomer's own analytic chain rule. This is an intentional numerical
correction, not native trajectory parity.

`optimize_molecular_lj` uses independent L-BFGS-B, memory5, explicit tolerance
(.1 source default) and at most1000 auxiliary E/G calls (source default).
Every call, including a failed one, is counted. The returned structure and
energy come from the same actual evaluation, including budget exits. Solver
line search and convergence norms differ from Java reverse communication.
Output explicitly says certified_physical_minimum=False. No physical ASE
calculator is involved.

`initialize_type3(lj_monomer_optimization=True, lj_pair_sigma=...)` now refines
all filtered/shuffled structures before sorting by auxiliary energy and
applying the original output cap. All starts, including discarded candidates,
remain in the auxiliary ledger. Initial failed evaluations carry their spent
call count and are excluded from energy sorting; raw generation, auxiliary
state and physical certification remain separate. The shipped example LJ
file contains C/O/V pairs, so water tests use explicitly labeled numerical-
check sigma values; these are not claimed water force-field parameters.

## Complete TYPE0 source-family inventory

| Family | Source construction and formula | Independent state |
|---|---|---|
| Unlimited | add a sphere tangent to a randomly selected existing sphere at r_i+r_j; reject overlap with all spheres; nonuniform elevation/azimuth draws | implemented with explicit per-insertion cap |
| TripleTangency | average-radius tangent triangle/tetrahedral apex; neighbor window [1.9r,2.1r]; unoccupied apex >=1.9r from existing atoms | implemented, explicit radii and bounded face draws |
| SimpleCubicPacking | enumerate integer boxes (i,j,ceil(N/(ij))), deduplicate sorted triples, remove excessively elongated 1-layer boxes; grow along six axes by2r from exposed sites | implemented with explicit box; asymmetric negative-y boundary corrected |
| IrregularBall | concentric shells spaced2r, shell occupancy from surface area/(1.3*pi*r^2), golden-angle points, random layer rotation, jitter +/-0.1*(2r) | implemented; final single-point shell singularity corrected explicitly |
| IrregularBallOri | grow inside sphere radius r*(N/occupancy)^(1/3), initial occupancy .5, reduce by.01 after1000*current_size failed draws | implemented with explicit density and partial-output failure accounting |
| IrregularCage | shell thickness2.2r; fit occupancy polynomial in log10N forN<=100, else.446; solve shell-volume quadratic, decrease occupancy after failed draws | implemented with explicit density; empirical polynomial is not a physical density law |
| RegularRing | ring radius r/sin(pi/k), spacing sqrt(3)r between layers, alternating pi/k twist, ceil(N/k) layers truncated toN | implemented with explicit ring sizes and multiplicity |
| RegularCage | original spherical-Fibonacci points scaled so mean distance between successive listed points equals2r | implemented; successive-index distances are not nearest-neighbor distances |
| CustomStructure | load addition ARC/GJF; optionally load N-atom PSD xyz templates, rescale by new/old mean element radius and reassign species | caller-template conversion implemented; no hidden template corpus synthesized |

The exact IrregularCage occupancy for b=log10N is
-.283 b^5 +1.966 b^4 -5.232 b^3 +6.5352 b^2 -3.659 b +1.0799 +.05,
truncated to three decimals forN<=100. Its cubic-shell equation is
3w R^2 -3w^2 R +w^3 -3V/(4pi)=0, w=2.2r, V=N*(4pi/3)r^3/occupancy.
These are source heuristics with explicit coefficients, not newly invented
physical defaults or claims of universal initialization quality.

`initialize_type0_regular` exposes only the three completed packing families
through explicit counts. It does not pretend to implement missing families
or silently choose a mixture. Nonperiodic ASE represents the original
bookkeeping50A vacuum. All radii and ring sizes are caller inputs; candidates
remain uncertified geometry. All remaining family primitives are now implemented in packing.py; see
type0-packing-implementation.md. Family allocation/templates remain explicit,
and physically certified initial population yield is unvalidated.

## Verification

Eight tests passed in5.06s across initializers and molecular auxiliary module.
Actual XXXII172 input verifies TYPE1 produces and keeps35 passing candidates
plus its supplied seed, with unchanged cell/composition. Actual(H2O)15 checks
all90 center/Euler gradient components at nonzero rotations against central
finite differences, retained intramolecular geometry during bounded auxiliary
optimization, and the TYPE3 refine/sort/cap path with7 starts x3 calls=21
auxiliary calls. Au24O4 composition from the uploaded514-atom example checks
ring/cage length formulas and explicit packing counts. Physical PES calls=0.
These checks establish implementation/math consistency, not relaxed initial
population yield or global search efficiency.
