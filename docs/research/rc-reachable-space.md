# Does native lambda add molecular configurations missing from our forest?

2026-09-10. **The source-defined rigid-tree configuration space is already
represented by the current exact-Jacobian pose/torsion forest. The recovered
native lambda operation is consistent with a choice of inherited-twist gauge;
there is no demonstrated additional reachable degree of freedom requiring a
new public parameter.** Full release forward/force equivalence remains unproven.

This qualifies the earlier description of lambda as a missing coupled policy:
an omitted numerical policy does not by itself mean an omitted physical space.

## Geometry argument, independent of force redistribution

Take two nondegenerate rigid bodies sharing two distinct axis atoms, with the
parent pose fixed. Any proper rigid transform of the child preserving both
shared points differs from another such transform by a rotation about that
axis. Thus its remaining configuration fiber is exactly SO(2), or S1.
The current `RigidChainChart` permits an unrestricted child torsion theta and
therefore covers this entire fiber. Repeating this construction along an acyclic
body tree gives root SE(3) times one S1 per edge (up to discrete geometric
symmetries). Forests add independent root poses. This statement assumes precisely
the implemented tree constraints; it excludes closed mechanisms, extra native
joint types, topology changes and additional external constraints.

If a native inheritance convention adds an axis twist chi(parent coordinates,k)
to a child's self-angle, its relation to our torsion is

```
theta_ours = theta_native + chi(parent coordinates,k).
```

For any fixed parent and k, this map is a translation of S1: onto and invertible.
Locally its coordinate Jacobian is triangular with a unit torsion diagonal;
its determinant is one, so Cartesian tangent rank is unchanged. An arbitrary
value k=.7 does not change this argument. For a tree, ancestor-dependent twist
changes give a triangular reparameterization recursively. This does **not**
assert that every release branch satisfies the premises; complete native
matrix composition and topology handling would be needed for that assertion.

A reparameterization can nevertheless change proposals. Isotropic draws in its
coordinates are generally not isotropic in ours; the induced Cartesian step
covariance is J A A.T J.T instead of J J.T. Search trajectories, step limiting,
Gaussian distance geometry and numerical conditioning can therefore differ
without any new physically reachable configuration. This is a coordinate/metric
effect, not evidence of greater physical representational capacity.

## New native instruction evidence

1. The body-type-2 self-angle branch **0x81f1b7 → 0x81d7ae** was executed using
   original ELF instructions. For two supplied axis endpoints it normalizes
   their difference and writes **rotation_vector = theta * axis**, with the
   stored signed angle exactly theta. No krot is read in this slice. Five
   signed angles −2, −.1, 0, .7, 2.8 were checked first on ASE trans-butane and
   then on a rotated, oblique copy; maximum vector discrepancy is 2.22e-16.
   This confirms an unsuppressed local torsional input at this stage, not the
   entire downstream transform.
2. `r2vec_` takes RINH and the AXIS2 vector supplied by `rigid_x_r2c_`. Its
   terminal instructions **0x81fa7b–0x81fa9b** multiply the input axis components
   by a signed/unwrapped scalar angle and store both. They do not return an
   arbitrary newly chosen rotation axis. The subsequent **abs(1-krot)** scaling
   in the caller therefore scales an **axis-aligned inherited twist**. Earlier
   acos/fmod branches select the angle and its winding; their complete numerical
   branch behavior was read but not executed by this new slice oracle.
3. The native lambda force-buffer oracle in `rc-native-transmit-contract.md`
   remains a separate result. Its radial/tangential split does not establish
   the derivative of the complete map. No force-conservativity conclusion is
   inferred from a single internal buffer.

Together these observations support the axial-gauge interpretation much more
specifically than merely seeing a krot symbol in a forward routine. They still
do not prove whether the complete inherited twist depends only on ancestor
coordinates, or whether all later matrix products preserve the declared shared
axis. Those are the remaining finite-angle release checks; they are not a reason
to add krot to our public chart now.

## Explicit local rank/work check, with its provenance boundary

`research/ga_ssw/probe_rc_reachable_fiber.py` saves two distinct evidence layers:

- Original-instruction self-angle slice, with synthetic registers/body fields;
  no native entry, allocations, MPI, main or protection execution. At most 500
  instructions and one second per case. It is not full native initialization.
- An **independently reconstructed axial-inheritance gauge example** on the
  same actual trans-butane two-body geometry. A smooth parent twist psi is
  extracted by swing/twist decomposition and theta_ours=theta−k*psi. This is a
  representative member of the proven fiber equivalence, not an asserted
  reconstruction of every native matrix product.

On the oblique copy, all k=0,.7,1 cases have Cartesian Jacobian rank **7** and
coordinate-change determinant **1**. At k=.7, the difference between Cartesian
column-space projectors is 9.4e-11; the smallest nonzero singular value changes
from 1.212 at k=0 to 1.668 at k=.7. Thus conditioning changes while the local
reachable space does not. These values belong to this one geometry/coordinate
point and are not an optimization-efficiency result.

For the reconstructed map, exact J.T F was checked against central differences
of a linear Cartesian scalar objective at h=1e-6: maximum force discrepancy
across all three k values is 9.6e-10. This checks the mathematical
reparameterization, **not native `rigid_f_c2r_`**. The latter's whole-map force
finite-difference closure is explicitly unfinished.

Artifacts:
`research/ga_ssw/evidence/rc-reachable-fiber-oblique/{probe.py,result.json}`;
the earlier axis-aligned evidence remains separately at `rc-reachable-fiber/`.
Static excerpts are `rc-transmit-evidence/{type2-self-angle.asm,r2vec.asm}`.
No PES calls or public RC changes occurred.

## Development decision

Keep the current exact work-conjugate forest as the baseline. Do not introduce
lambda merely to reproduce a nonunique gauge or a parameter list. A future
native gauge implementation needs a falsifiable benefit—better conditioning or
more useful equal-cost proposals on real molecular systems—and consistent
energy derivatives. It is not needed to fill a currently demonstrated missing
rigid-tree degree of freedom.

The narrowly outstanding native task is complete finite-angle two-body matrix
composition plus its final force outputs, including all root/child channels.
A resulting difference may be gauge, constraint policy or derivative error;
only that same-map comparison can distinguish them. Periodic Kabsch/cell
coupling and loop constraints remain separate coverage questions and are not
resolved by this fixed-cell two-body result.
