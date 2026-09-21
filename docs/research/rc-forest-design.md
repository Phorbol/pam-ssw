# Isolated RC forest: explicit geometry design before implementation

2026-09-10. This is a bounded next-component design, **not implemented and not PES validated**. The current `rc_geometry.py` and `rc_reference.py` support one articulated tree. Extending them by removing all six root coordinates of every molecule would eliminate the intermolecular relative motions that the extension is supposed to explore.

## Source and scientific scope

The locally archived original RC-SSW paper, *Global Optimization of Large Molecular Systems Using Rigid-Body Chain Stochastic Surface Walking*, 2025, DOI [10.1021/acs.jctc.5c00350](https://doi.org/10.1021/acs.jctc.5c00350), §2.1 defines a rigid body's geometric-center translation and finite angle-axis rotation. Section 2.2 assigns a central body pose and child bond torsions to a connected chain. The SI §7 supplies explicit body and bond lists, with joint endpoint membership shared between connected bodies. Sources: `literature/225.txt` and `literature/ct5c00350_si_001.txt` under the external research directory. The exact finite map and inspected native symbols are recorded in `rc-geometry-contract.md`.

Those sources justify pose-plus-torsion degrees of freedom; they do not establish the forest API or the isolated-cluster gauge chosen below. These are **independent mathematical constructions**. Native `rigid_x_r2c_`/`rigid_f_c2r_` references to rigid counts and global arrays do not by themselves close native multi-molecule indexing, force transmission or lambda semantics. No such parity is claimed.

The next component targets several disjoint, nonperiodic, unconstrained molecular chains/rigid bodies interacting on a common translation- and rotation-invariant atomistic PES. Relative translations and rotations can change intermolecular energy. Global translation and rotation cannot. This assumption excludes external fields, fixed substrates, ASE constraints and periodic cells; a force-capable calculator alone does not establish invariance. There is no cell coupling, automatic chemical topology inference, closed-loop constraint solver or adaptive fragmentation in this change.

## Topology and exact finite map

An explicit forest consists of K connected trees. Within each tree retain the existing two-shared-endpoint joint convention and connected atom-membership rule. Across trees, atom index sets must be disjoint and together cover all atoms. A tree may contain one rigid body and no torsions; that is a supported molecular rigid body, not a fake articulated chain. Parent ordering and axis orientation remain explicit.

For tree k use root center c_k, translation t_k, finite rotation vector p_k and internal torsions theta_k. Apply the existing root transform

```
T_k = [exp(skew(p_k)), t_k + c_k - exp(skew(p_k))*c_k; 0, 1]
```

and transported-parent products along its edges. Combine each tree's output through its disjoint global atom indices. Exact Cartesian Jacobian blocks follow the same Frechet/root and product/child derivatives already tested in `RigidChainChart`; derivatives with respect to coordinates of other trees are zero. This does not use a small-angle update as a finite rotation. Shared endpoints inside a chain remain exactly coincident by construction.

## Remove global rigid freedom once

Use one explicitly selected anchor tree whose root body has at least three noncollinear points. Fix that root body's t and p at zero in the current reference; retain all its internal torsions. Every other tree retains its root translation and rotation and all torsions. With nondegenerate root bodies the reduced dimension is

```
D = 6*(K-1) + sum_k(number of internal joints in tree k).
```

This fixes one representative of each global Euclidean-motion class while retaining relative poses. It does not fix every molecule in space. Any configuration in the rigid forest can be globally transformed to restore the anchor root pose without changing its invariant PES energy or internal shape. Fixing the first body is consequently a valid local gauge for this specified domain, not an Eckart projection and not a root-independent metric.

Initially require **every root body noncollinear**, including the anchor. A linear body has only two physical orientation freedoms and a single atom has none; assigning either three free angle-axis directions would introduce null search directions. Supporting those objects needs an explicit lower-dimensional orientation chart and is deferred, rather than pretending that subtracting six always gives the correct shape dimension. Torsion axes also must move at least one non-axis atom in their descendant branch; rank checks in numerical geometry validation must expose redundant explicit topologies.

The anchor identity must be explicit and logged. Changing it preserves the accessible rigid shape family but generally changes the Euclidean coordinate metric and therefore an SSW trajectory. No anchor-choice performance claim or adaptive anchor heuristic is justified here.

## Metric, gradient and lifecycle

Use translations in Å, scaled root rotation coordinates x_rot=L_rot*p and scaled torsions x_tor=L_tor*theta, with separately explicit positive lengths in Å/radian. Neither physical masses nor a guessed molecular radius silently determine these values. They are search-metric inputs with units, not universal paper defaults. Their sensitivity belongs in later matched-cost experiments.

If S is the diagonal coordinate scaling and q_free=S^{-1}x, then

```
g_x = S^{-T} J_free(q).T g_cart.
```

All dimers, finite differences, Gaussian projections and Safe-total optimization must operate on this same x and total scalar objective. No separate per-body torque subtraction is added. Root gauge columns are removed once; relative root columns remain. The frozen chart must stay unchanged throughout one biased proposal. Unwrapped angles and finite exponential derivatives preserve the declared objective locally; angle-axis has chart singularities near full turns, so rank/domain failure must be reported rather than resetting a live bias history or claiming a globally nonsingular parameterization.

After a proposal, remove all rigid constraints and perform the existing unrestricted true Cartesian quench and full-force certificate. Rebuild all tree references only from the selected next landing. Count all E/F calls, preserve failed and MC-rejected candidates, and do not treat intermolecular dissociation as a successful new basin merely because forces are small. Native lambda-dependent force transmission remains an unresolved separate policy; exact J.T F is the conservative independent contract.

## Minimal implementation boundary and falsification checks

A subsequent implementation can add `RigidForestChart` beside `RigidChainChart` and a scaled forest surface, then generalize the driver without changing the single-chain baseline. It should first verify zero-map identity, finite-angle full-Jacobian/work finite differences, rigid distances and shared endpoints, disjoint ownership, retained relative root motions, and reduction to the existing one-chain map. A two-body reference must have six relative pose DOFs, not zero. Nonlinear single-body isolated input should have no search DOF; degenerate roots and loops must fail explicitly.

These checks establish geometry only. A bounded genuine molecular-dimer or multi-molecule ASE-calculator run must subsequently demonstrate changed relative pose, full true-quenched force checks and intact molecular identity; a single low-gradient dissociated candidate would not validate intermolecular exploration. Broader RC search efficiency still requires a matched Cartesian baseline and repeated seeds. This task starts no new PES calculations and makes no full-RC or native-parity claim.

Implementation status update: `rc-forest-implementation.md` records the now executable forest subset, tests, failed manual-input probe and separate S22 reference probe. The design above is preserved as the preimplementation contract. Explicit finite-rotation chart-rank failure handling remains open and is not silently claimed implemented.
