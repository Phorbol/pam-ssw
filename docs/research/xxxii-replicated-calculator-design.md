# XXXII replicated periodic representation

`research/ga_ssw/xxxii_replicated_calculator.py` is a research-only ASE
calculator for the audited 172-atom XXXII model.  Its public API accepts the
original ordered 172 atoms with a finite positive full-periodic cell and a
required positive integer tuple `repetitions=(nx, ny, nz)`.  It explicitly
constructs the internal repeated representation for each evaluation; callers
do not pass pre-replicated atoms.

The internal LAMMPS engine contains `172 * nx * ny * nz` atoms.  Type, charge,
and atom ordering are checked for every image.  Total energy is divided by
the replica count; corresponding force blocks are reshaped to
`(replicas,172,3)` and averaged.  Stress is converted through the existing
Prism tensor path and retains the per-cell scale.  `api_calls` counts public
attempts, while `engine_calls`, `requests`, and `atoms_evaluated` count only
actual internal `run 0` evaluations; the latter counts expanded atoms.  The
last corresponding-image force spread is recorded in
`last_force_image_max_difference`.

Before any engine evaluation, explicit original Bonds are expanded as graph
pairs at bond distance 1--3.  Each corresponding pair in the repeated
restricted Prism frame must satisfy the strict component condition
`abs(delta_component) < replicated_restricted_box_component / 2`.  A failure
raises `ReplicatedDomainError` before PES evaluation.  This preserves the
caller's lifted molecular coordinates and does not apply an ad hoc minimum
image repair.

The adapter reuses the audited converter/adapter source hashes and constants,
including `ENERGY_TO_EV`, `REAL_NKTV2P`, `TYPE_NUMBERS`, table setting 0,
Ewald accuracy `1e-12`, and `gewald=0.47570069`.  It is an exact numerical
representation diagnostic, not a new force field, solver, or SSW heuristic.

The existing 33-EF qualification is archived at
`research/ga_ssw/evidence/xxxii-replicated-qualification/`.  It compared
the primitive and explicit replicated representations at three saved points
using 1x1x2, 1x1x3, and 2x2x2.  The largest reported cross-representation
derivative discrepancy was on the order of `1.6e-6` in the tested derivative
table; this is a numerical floor for that qualification, not an exact
conservativity claim.  The evidence does not establish a whole-trajectory
SSW result or production equivalence.
