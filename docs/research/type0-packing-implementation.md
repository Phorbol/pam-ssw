# Completed remaining TYPE0 packing primitives

`packing.py` implements the remaining geometric families from uploaded
`snowFlake/{Unlimited,SimpleCubicPacking,IrregularBall,IrregularBallOri,
IrregularCage,CustomStructure}.java`, with a caller-supplied atomic radius
table and injected RNG. RegularRing, RegularCage and TripleTangency already
exist in `initialize_type0_regular`. No family mixture is silently selected,
no physical PES is evaluated, and generated or partial structures are never
labelled physical minima.

## Interface and bounded failure

`pack_type0(numbers, family, rng, atomic_radii=..., max_attempts=..., ...)`
returns PackingResult(atoms,status,source_atom_indices,details). Unknown
families or missing physical parameters fail before generation. Random
insertion and available-direction draws have a per-insertion cap. Exhaustion
returns the actual partial structure and failed atom index; it never pads
coordinates to the required atom count. Details retain successful and final
failed attempts, their total, density changes and intentional corrections.
`physical_requests` is explicitly zero; geometry work is not called free.

The two occupancy-dependent families require explicit occupancy. Callers may
choose the original source values: .5 for IrregularBallOri, or
`source_cage_occupancy(N)` for IrregularCage. No proposed optimal density or
new element-radius table is supplied. SimpleCubic requires an explicit
integer box, and `source_cubic_boxes(N)` reproduces the native enumeration
and elongated-box pruning. The caller therefore retains control of how many
shapes are requested and how much geometric work is allowed.

## Recovered formulas and corrections

- Unlimited adds each atom tangent to a randomly selected existing atom,
  at radius r_i+r_j, rejecting overlaps against every existing atom. Its
  nonuniform elevation/azimuth draw distribution is preserved. Exactly
  tangent comparisons include a stated 32-machine-epsilon distance tolerance
  to avoid rejecting mathematically tangent positions through roundoff.
- SimpleCubic grows integer lattice sites with spacing2*mean_radius, starting
  at (r,r,r). Select an occupied site with the least positive count of open
  directions, then rejection-sample the six directions. Integer occupancy
  avoids floating point duplicate sites. The source's asymmetric negative-y
  inequality is corrected to the same admissible-box condition as other faces.
- IrregularBall fills concentric golden-angle shells, spacing2r, using source
  packing factor1.3, nonuniform Euler shell rotations, and +/-0.1*(2r)
  Cartesian jitter. A final one-point shell made native y=1-2i/(n-1)
  undefined; Python places that single point on the equator before the
  original random shell rotation. This deterministic convention is an
  explicit correction to undefined source behavior.
- IrregularBallOri and IrregularCage draw around an existing atom at radius
  2r*(1+.1*(U-.5)), enforce minimum distance1.75r, and impose the source ball
  or shell boundaries. After1000*current_atom_count unsuccessful draws they
  lower occupancy by.01, subject to the explicit overall attempt cap. A
  nonpositive density or missing positive shell radius is reported rather
  than producing invalid coordinates.
- Custom template conversion requires caller-provided N-atom ASE geometry
  and the template-species radius table. It applies source scale
  mean(target_radius)/mean(template_radius), then assigns the shuffled target
  species. There is no inferred template corpus or silent external download.

Source formula details and original empirical polynomial are retained in
`initializer-final-source-closure.md`. Newly implemented functions supersede
its earlier “not ported” statuses; statistical initialization effectiveness
and native exact random trajectories remain unqualified.

## Actual-composition geometry checks and failures

Seven tests passed in2.50s. Tests use Au24O4 composition taken from the
uploaded514-atom supported-cluster example and the172-atom XXXII composition.
They check composition/source-index preservation, finite coordinates,
Unlimited pair exclusions, explicit template scaling, cubic boxes,
one-point-shell correction and bounded partial outputs. The172-atom shell
check is geometry generation, not a chemically plausible molecular assembly.

The original-density seed7 cases **did not complete** within10000 attempts
per inserted atom: IrregularBallOri(.5) and IrregularCage(.490). This evidence
is retained, not converted into success by retuning the default. A separate
explicit sparse occupancy.2 verifies the code can complete the same28-atom
composition in a different supplied domain; .2 is not adopted as a default
or represented as an efficiency improvement. The native density-reduction
trigger can exceed a user's smaller budget, so truncation is a legitimate
observable outcome. Full successful and partial coordinates, parameters,
lineage and attempt counts are saved in
`research/ga_ssw/evidence/type0-packing-geometry/result.json`.

All requested source packing families now have independent geometric
primitives. A production initializer must still explicitly select family
counts/templates, retain generation failures, run physical quenches and
assess post-quench yield/diversity. Geometry completion is not that validation.
