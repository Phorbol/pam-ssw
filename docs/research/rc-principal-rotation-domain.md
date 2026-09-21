# RC optimization root-rotation domain

The finite angle-axis map is mathematically correct at a full turn, but its Jacobian loses two root rotation directions there. Consequently a small pulled-back gradient at 2π does not certify stationarity even on the rigid manifold. This is a coordinate failure, not a reason to change the physical potential or add a force floor.

The optimization layer now restricts each root rotation vector, after removing its explicit Å/rad metric, to the **open principal ball ||ω|| < π**. The bound is the principal SO(3) logarithm cut, not a fitted search radius. The exponential differential is nonsingular throughout this ball (its transverse singular values are 2 sin(θ/2)/θ). The π boundary itself is not a differential singularity: it is excluded to maintain a unique principal root representation. This statement concerns each root rotation, not global injectivity of the entire torsion/chain configuration map.

Reference: Ethan Eade, *Lie Groups for 2D and 3D Transformations*, [author notes](https://ethaneade.com/lie_groups.pdf), SO(3) exponential/logarithm and derivative sections. The domain choice is our independent mathematical implementation, not demonstrated LASP parity.

## Implementation and conservative objective

- `rc_optimization_domain.py`: `RootRotationChartError`, principal-ball check, optimization-only `PrincipalRigidForestCellChart`.
- `ForestSurface.coordinates`: checks all free relative roots after metric unscaling. The isolated anchor's root pose is already absent.
- `run_rc_vc_ssw`: constructs the optimization-only chart. Every root rotation is checked, including the periodic anchor, whose orientation relative to the cell remains physical.
- Raw `RigidChainChart`, `RigidForestChart`, and `RigidForestCellChart` retain arbitrary finite angle maps and their exact derivatives. Internal torsions remain unwrapped.

The check occurs before any physical energy/force/stress request, for both direction finite differences and optimizer evaluations. Current Safe-total reports an invalid trial as `evaluation_failed`; it does **not** contract domain-invalid trials. This is the explicitly supported chart-failure option, and can reduce feasibility near the chart boundary. It is never reported as convergence. The last accepted iterate/current minimum and charged physical calls remain preserved. Domain-rejected numerical callback attempts are not physical oracle calls.

No angle wrapping, chart rebase, Gaussian relocation, history change, or gradient approximation was introduced. Within the domain the original scalar objective and its derivative are unchanged. Final unrestricted true quenches and their full physical certificates are unchanged. We do not use a bare physical-force certificate for a biased stationary point: the bias may legitimately balance the physical force. Other topology-dependent chain degeneracies still require separate analysis; this fixes the specific root full-turn singularity.

## Verification

`tests/standalone/test_rc_optimization_domain.py` uses actual ASE S22 Water_dimer geometry. At a relative root full turn the raw six-coordinate Jacobian has rank 4, versus 6 at zero, while geometry returns to its reference. A constructed nonzero rotational Cartesian force orthogonal to the surviving root direction is invisible to this deficient Jacobian. This is a geometric counterexample, not a claim about a physical PES at that structure. The optimization domain rejects it before the oracle, including through Safe-total. Tests cover exact π boundary, metric unscaling, periodic anchor, and both complete drivers' failure accounting/current preservation.

Together with the original geometry and full-path tests: **25 passed** (ASE/NumPy deprecation warnings only).

The unchanged bounded real S22/GFN2-xTB experiment was rerun separately using `research/ga_ssw/probe_rc_s22_water_dimer_principal_gfn2.py`, same seed 3, one outer step, two Gaussians and all prior numerical parameters. Evidence: `research/ga_ssw/evidence/rc-s22-water-dimer-principal-gfn2/`. It completed with **183 total E/F requests** (181 search + 2 independent endpoint checks), **0.411 s**; initial/landing O–O distances 2.835014/2.836939 Å, biased endpoint 2.993035 Å. This reproduces the prior feasible trajectory without reaching the new boundary; it is regression evidence, not evidence of improved search efficiency or a distinct minimum. The older evidence remains intact. The attempted isolated-env launcher was unavailable; the successful command used the existing system Python with `PYTHONPATH=/tmp/pam-ssw-tblite-20260909:.` and all three BLAS/OpenMP thread variables set to one.
