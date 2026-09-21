# Independent single-chain RC-SSW driver and first real-molecule wiring

2026-09-10. `pamssw/standalone/rc_reference.py` now connects the exact articulated geometry to a complete **isolated, single-chain proposal → unrestricted quench → selection** path. It is an independent RC-SSW subset, not native lambda-transmission, coordinate-stream or full RC release parity. Seven combined geometry/driver tests pass; no exports or other walkers were modified.

## Algorithm and consistent metric

`RCSSWConfig` requires `torsion_length` (Å/radian), Gaussian width (Å in scaled coordinates), and rotation bias (eV/Å²). Internally `x=L*theta` and `g_x=J_theta.T*g_cart/L`. Angular metric is explicit rather than mixing radians with Cartesian Å. The root translation and rotation coordinates are fixed during a proposal: for an isolated molecule without external orientation/position fields they are gauge freedoms, not useful escape directions. This restriction is an assumption about the physical PES and cannot be inferred merely from an ASE calculator supporting forces.

The driver performs:

1. Unrestricted true-PES Cartesian quench using the existing Safe-total backend and full per-atom force certificate.
2. Rebuild `RigidChainChart` from the **selected** true-quenched structure at each outer step. Explicit user body membership and parent/joint topology persist; reference body geometry is reset only here.
3. Generate an internal torsion direction, soften it with `generalized_dimer` and its explicit rotation-bias objective. An unconverged direction is recorded as a failed proposal, not silently certified.
4. Continue with projected Gaussian biases on the unwrapped scaled torsion coordinates. Set a positive height from the same independent forward-force formula used by the joint reference walker, then Safe-total relax the full physical-plus-Gaussian objective. Biased-quench residuals refer to that reduced modified objective.
5. At the Gaussian budget or a lower true energy, remove all bias and all rigid constraints, run the **full Cartesian** true quench, and apply one outer Metropolis decision. Preserve force-certified rejected landings in `minima`; `current` changes only on acceptance.

A proposal chart remains fixed while its bias history is live. There is no torsion wrapping or reference reset during climbing: such a reset would change the linear projected-Gaussian function. The selected next minimum may have changed bond lengths/angles under the unrestricted quench; the next chart then freezes that new geometry, not the original input forever.

Inputs require one nonperiodic rooted body tree with at least one internal torsion and no ASE constraints. A single free rigid body has no useful reduced escape DOF and is rejected before an oracle call. Multiple molecules, periodic rigid-body translations, joint cell modes, loop closure and native force transmission remain outside this driver.

## API and cost/failure semantics

Use `run_rc_ssw(atoms, surface, bodies=..., parents=..., joints=..., steps=..., config=RCSSWConfig(...), rng=...)`. `surface.evaluate(atoms)` provides counted true E/F; it can be `ASESurface` around an appropriate ASE calculator. No native binary or PAM walker is invoked.

`TorsionSurface` exposes the exact reduced E/gradient for independent derivative checks. The result includes initial/current/best, every valid landing, and each outer event with all climbing modes, Gaussian heights, optimizer results, status, spent requests and final true quench. Initial failure terminates explicitly; proposal failures preserve current. Stage request totals are measured from the supplied physical oracle, including HVP, line-search, failed evaluator and certificate costs. A caller needing an absolute cost/wall ceiling must use a capped counted surface, as in the probe; `relax_steps` is not an oracle cap.

The final quench certificate checks full physical Cartesian forces, **not only torsional torque**. It is still a stationarity certificate, not a Hessian or chemical validation. Native lambda-dependent force transmission remains separate from the exact pullback; it has not been implemented by silently rescaling the force.

## Numerical tests

`tests/standalone/test_rc_reference.py` checks metric-chain-rule finite differences on actual ASE butane coordinates, exclusion of root DOFs, failure cost/current preservation, rejection of zero-internal-DOF input, and the full dimer/Gaussian/Safe-total pathway preserving an MC-rejected candidate. The latter uses a clearly artificial flat surface to isolate stage wiring and is not a physical result. Together with `test_rc_geometry.py`: **7 passed**, with existing ASE/NumPy deprecation warnings only.

## Real butane/GFN2 feasibility result

`research/ga_ssw/probe_rc_butane_gfn2.py` ran one neutral-singlet trans-butane proposal using tblite GFN2-xTB (accuracy 0.001), CPU with OMP/OpenBLAS/MKL threads one. The predeclared ceilings were 400 E/F and 30 seconds; observed cost was **64 search + 2 independent-calculator fresh checks = 66 E/F requests, 0.229 seconds** in this environment. Every request has an E/F/geometry JSONL record; all 66 rows reconcile with the counter.

Development inputs were L=2 Å/radian, width=0.4 Å, two Gaussians, seed 3. These are explicitly selected feasibility settings, not fitted optimal values or established RC paper defaults. The width corresponds to 0.2 rad for a unit scaled direction; the full relaxed trajectory need not remain within that angular displacement.

| Observable | Initial true quench | Final unrestricted true quench |
|---|---:|---:|
| Fresh energy (eV) | -371.847059150793 | -371.819974715317 |
| Fresh maximum force (eV/Å) | 0.00416044 | 0.00992875 |
| Central C–C–C–C dihedral | 180.000° | 70.627° |

Both biased stages converged. The landing is +0.0270844 eV above the initial structure and the outer MC rejected it; it remains recorded as a valid observed candidate, while current stays at the initial structure. Fresh energy differences from stored values are below 4.6e-13 eV. The dihedral change supports actual internal conformational motion rather than global rotation/translation. No Hessian, high-level quantum comparison, repeated-seed efficiency comparison or global-conformer enumeration was done. This is real-molecule pipeline feasibility on a molecular approximate PES, not a general search-effectiveness claim.

Complete evidence, config, code snapshots and request trace: `research/ga_ssw/evidence/rc-butane-gfn2/`. Full native topology/force policy, variable-cell rigid-preserving coupling, scalable Jacobian implementation, rigidbody/blist parsing and RC scientific benchmarking remain subsequent work.

## Fixed anchor correction and unchanged bounded rerun

Parent review identified an unsupported difference from the shared atomic/joint-VC baseline: the first RC implementation replaced the dimer anchor by the returned softened direction after every Gaussian. No inspected RC paper/SI or native instruction evidence establishes that as an RC-specific policy. The live driver now retains the initial sampled anchor throughout an outer proposal. Gaussian centers and softened deposition directions still belong to their individual stages; fixing the rotation-bias anchor does not freeze those directions.

The original `evidence/rc-butane-gfn2/` code snapshot and result remain intact. `probe_rc_butane_gfn2_fixed_anchor.py` reruns exactly the same settings into `evidence/rc-butane-gfn2-fixed-anchor/`: **66 E/F, 0.238829 seconds**, identical final Cartesian positions and energy to the original result. The same rejected conformer is retained. This one result cannot establish that the policies are interchangeable on other trajectories. The driver regression deliberately changes the returned softened direction between two stages and asserts that both received the same original anchor; the combined geometry/driver tests remain **7 passed**. This is alignment of an independent baseline policy, not a new native-parity claim.
