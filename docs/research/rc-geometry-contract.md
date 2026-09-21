# First RC geometry core: articulated tree and exact finite-angle pullback

2026-09-10. Implemented `pamssw/standalone/rc_geometry.py`, tested by `tests/standalone/test_rc_geometry.py`: **3 tests passed**. These are coordinate/derivative checks on an actual ASE trans-butane geometry, not PES optimization or RC-SSW efficacy. No calculator, MACE, native main program, expiry path or job was executed.

## Sources and native reverse boundary

The archived 2025 RC-SSW paper (`225.txt`, DOI 10.1021/acs.jctc.5c00350) §2.2 defines one center rigid body with translation T and angle-axis rotation p, and one torsion theta per parent-child bond. SI §1 illustrates reference-bond alignment followed by child self-rotation. SI §7 explicitly requires rigidbody membership to include atoms connected by the rotatable bond: neighboring groups share axis endpoints and are not disjoint partitions. Local primary sources are under `/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/literature/`, including `ct5c00350_si_001.pdf`/`.txt`.

Only two native symbols were disassembled for this task:

- `rigid_x_r2c_` (0x81c0f0): uses global `module_rigidperi` arrays/dimensions and has direct calls to `kabsch_rotate_` (0x81cc1d, 0x81d716), `vec2r_` (including 0x81cc87, 0x81d7c2), `r2vec_` (0x81db97), acos and crossproduct. This supports finite-rotation/alignment machinery, not its complete recovered argument contract.
- `rigid_f_c2r_` (0x8132a0): consumes globals including rbs, srb, s_sub, refcellb, rigid_rotfact and rigid counts, contains sincos and MPI_Allreduce calls. A safe isolated oracle needs initialized topology/global descriptor semantics and exact argument layout; those are not available from the inspected instructions alone. No opaque whole-routine emulation or fabricated global state was attempted.

Full bounded disassemblies are in `docs/research/rc-native-evidence/`. The routines have no useful source-line mapping in this ELF excerpt, unlike the SSW class routines. This is a **native evidence gap**, not a reason to guess a primitive and call it parity.

## Supported mathematical topology

`RigidChainChart(atoms, bodies, parents=..., joints=...)` supports one connected rooted **tree** of rigid bodies on an isolated, unconstrained reference. Parents must precede children; each child shares exactly its two oriented bond-axis atoms with its parent. Every atom's body-membership set must itself be connected in the body tree, preventing incompatible shared atoms in separate branches. All atoms must be covered. Cyclic mechanisms, closure constraints, multiple independent molecules, periodic wrapping and cell motion are not implemented. PBC and ASE constraints are explicitly rejected.

This is an articulated chain/tree, not independent disjoint rigid molecules: torsions propagate into every descendant transform. The three-body test has two real C–C axes in trans-butane and overlapping groups; the geometric degrees of freedom are 6+2=8. Body definitions are explicit test inputs, not inferred from approximate covalent radii.

## Finite coordinate map and coordinate-gauge distinction

q consists of root translation t (Å), root rotation vector p (radians), and child torsions theta (radians). Root rotation is `R=exp(skew(p))` about the reference root-body geometric center c. Its homogeneous transform is

```
T0 = [ R, t+c-Rc; 0, 1 ].
```

For child j with parent i, let a be the reference first endpoint and u the normalized reference oriented bond axis. Define

```
Gj = [ skew(u), -skew(u)*a; 0, 0 ],
Tj = Ti exp(theta_j Gj).
```

Each atom is evaluated through its earliest owning body. Shared endpoints give identical positions through either adjacent body because the local rotation fixes both points of its axis. Connected membership extends this consistency to atoms shared across multiple levels. Every body's internal distances remain rigid at arbitrary finite angles.

This is the **transported-parent torsion gauge**: child orientation includes the complete parent transform. The paper's stated shortest bond-alignment plus self-rotation construction can distribute twist differently between its alignment and theta. Consequently our q is a valid independently defined articulated coordinate map, but **its numerical theta values are not claimed identical to native/paper coordinates**. No inverse native-coordinate conversion is supplied. Equivalence of shape families does not establish coordinate-stream parity.

Root angle-axis is a local redundant chart globally: at full-turn rotation vectors the Jacobian can lose rank. The implementation evaluates finite derivatives there without division by angle, but does not claim a global nonsingular inverse or silently reset references during a trajectory.

## Exact derivative and force contract

Root derivatives use the matrix-exponential Frechet derivative, including finite p. Child derivatives follow the exact product rule and `d exp(theta G)/dtheta = G exp(theta G)`. The returned Jacobian has shape (N,3,6+n_bonds). No infinitesimal `delta_r = delta_theta × r` approximation is passed off as a finite angle-axis derivative.

`pullback(q, F)` returns the exact work-conjugate generalized force `J(q).T F`. Translational components have eV/Å units and angular components eV/radian. They cannot be concatenated into an SSW norm with an implicit metric; selecting explicit length/angular scaling is future work. Dense Jacobian construction is appropriate for this bounded geometry core but has cost growing with atoms × generalized DOFs; no large-system performance claim is made.

The paper §2.2 introduces adjustable lambda-dependent force/torque transmission. Its conservation of resultant force does **not automatically make it the exact Jacobian pullback of this fixed coordinate map**. This module does not implement lambda projection, native rigid_rotfact scaling, or claim that those operations equal J.T F. Any subsequent reproduction of those policies must name them separately and establish their energy/gradient or search-preconditioning semantics.

## Verification and remaining implementation

Three tests check zero-map identity; preservation of all body pair distances at finite root rotations and two finite torsions; full Jacobian finite differences; independent scalar-work finite differences; finite derivatives at zero, pi and 2pi; original input preservation; and rejection of incompatible topology/PBC. All pass. ASE emits an existing NumPy shape deprecation warning; it is not a failed numerical check.

This code closes a useful RC **geometry and conservative pullback subset**. It does not supply topology inference, the LASP rigidbody/blist parser, native global-state initialization, inverse coordinate recovery, lambda force transmission, variable-cell rigid-preserving Kabsch coupling, RC direction softening/bias/optimization, or the required final unconstrained atomistic quench and MC. Those remain explicit tasks; none can be claimed completed from the geometry tests. The next integration must keep its exact coordinate/gradient pair and metric consistent and eventually validate the full pipeline on supplied molecular inputs.
