# Approved active-coordinate direction integration

User approved docs/research/2026-09-26-constrained-direction-geometry-proposal.md.
Base1ec428e. Scope nonperiodic constrained driver, true atom identity for selection,
active projection everywhere including stage history, explicit direction checkpoint.
Legacy no-option behavior and file compatibility remain. No pool/VC/periodic axes.

Root owns driver/reduced lifecycle and checks; Luna owns direction geometry and its
focused tests, no overlapping files. Tests before code via CPU Slurm <=5min each;
no login PES. New selection is finite active-conditioned native group score,
not native refresh parity; active first atom, second may be fixed reference,
group support active only, no retry/replacement when native outer band is empty.
None path retains legacy draws. Exact details documented in implementation.

Expected checks: forbidden fixed directions/coordinates zero, nonperiodic guard
before RNG/PES, required direction state/active identity on restore, continuous vs
split trajectory/LS/cost identity. Existing constrained, standalone and RC tests
must retain behavior. Real Cu cluster fixed subset and a point-restrained molecule
use bounded qualification, not global performance tests. Resource plan for those
will be frozen before submission; stop on contradictory state/physics evidence.

Real qualification protocol: Cu13/EMT fixes atoms0,1,2; same two-outer-step trajectory
continuous and split1+1, <=4000 search+6 independent minima checks/5CPU minutes.
C4H6 uses existing qualified MH-1/omol butadiene, one explicit Hookean point .2A
from atom0 with rt=.1A,k=1eV/A^2 (interface stress only, not confinement parameters).
Same continuous/split protocol, <=4000+6/one V10020min. No LS in these two physical
panels; earlier LS qualifications remain scoped, no combined-LS benefit claim.
Direction settings from previous qualification; constrained fmax.03, biased
coordinate gradient norm.1, max3Gaussians. Require full completed attempts,
fixed coordinates exact, all saved minima independent force/energy qualification,
exact state/RNG/request identity. Failure does not authorize loosening thresholds.
All actual source changes preserved in source.patch plus new source files before
execution; outputs refuse overwrite. No production search or efficiency inference.

Molecular1498463: both trajectories complete269 requests each, same RNG and counts;
strict comparison fails, with initial-quench position differences2.16e-12A already
before any resume. Do not tune tolerance or attribute this to checkpoint logic.
One discriminating followup records one real MH-1 oracle stream and replays those
exact E/F responses through split1+1; require every queried coordinate and final
state exactly equal. Recompute only the live trajectory's three minima; replay
identical minima share that qualification. <=3462 new requests including replay,
<=3 fresh; cumulative stays within4000+6 after original538+3. Same20min ceiling,
new run directory/source, no production-code change. Original failed output retained.

Also qualify Cu13 native LS+active-direction recovery with the existing native
Cu fixture (3eV,2.8A,defaultscale5), same continuous/split protocol, <=4000+6/5CPUmin.
This checks interaction/state ordering, not LS efficacy. Current source remains
frozen in source.patch and constrained_direction.py; original runner archived as
qualify-original.py before adding explicit replay/native-LS options.

Completed1498598:269 live +269 replay +3fresh; exact query/state identity.
Pending job moved4V100->8V100V0 with same20min/oneGPU bounds.
Final copy-only optimization skips nonterminal in-memory history snapshots;
explicit disk snapshots unchanged. RED1498611, GREEN1498659(133passed).
