# Independent ASE implementation — development status

Current concise entry-point and evidence guide: [固定胞使用说明](../../docs/research/FIXED_CELL_USER_GUIDE.md).

This package implements search components in Python using ASE calculators for
the physical potential. It does not launch LASP or Java. It does not call the
existing PAM walker. Uploaded programs are external test oracles only.

**Executable independent drivers exist:** `run_ssw`, `run_ls_ssw`,
`run_ga_ssw`, and experimental `run_vc_ssw`. GA supports the fixed-internal-unit
TYPE3, atomic TYPE0 and periodic TYPE1/TYPE2 research branches. LS, RC
and joint VC workflows are implemented with explicit independent contracts. This is not full uploaded-release parity or all-system support.
The runtime search is independent; ASE calculators may themselves use compiled
MLIP, tight-binding, or DFT libraries/programs as physical energy oracles.

Current fixed-cell implementation and evidence are tracked in
[`MAINLINE.md`](../../docs/research/MAINLINE.md) and
[`fixed-cell progress`](../../docs/research/2026-09-12-fixed-cell-progress.md).
Experimental staged rotation and PAM Gaussian policies are opt-in;
current cross-system results do not justify a universal default change.

Implemented interfaces:

- `surface.ASESurface(calculator, force_consistent=False).evaluate(atoms)`:
  energy/forces; no ownership change to the input Atoms. Choose
  `force_consistent=True` if the calculator's forces differentiate its
  free_energy. Missing backend capabilities propagate as errors. Backend
  energy/force consistency is not inferred from the ASE interface.
- `surface.quench(atoms, surface, fmax=..., steps=...)`: fixed-cell true
  quench; inspect `converged`, `max_force`, and `surface` on the result.
  Supplying `terms` instead returns a modified-surface certificate.
- `gaussian.ProjectedGaussian` and `GaussianSum`: additive projected bias;
  `adjust_native_weight`: static finite-domain 87-degree height update.
- `direction.reference_soft_mode`: bounded finite-difference Ritz solver.
  This is explicitly NOT the native biased Broyden dimer rotation.
- `softening.FrozenBondSoftening`: paper exponential pair potential; explicit
  element-pair energy/neighbor tables, frozen neighbors and reference lengths.
  `LSResponseState` updates the next step from true pre-quench energy response.
- `ls_cycle.prepare_ls_step` / `finish_ls_step`: soft-only preparation and
  true-surface finishing, with explicit failures on unconverged stages.
- `legacy_descriptor`: independently implemented archived NNA/selection
  behaviors. Its known atom-order dependence is preserved and documented;
  it is not a general structural-equivalence certificate.
- `population.partition` / `rank_regions`: Java-derived projection grouping,
  parent capping and empirical scores. Regions are not kinetic funnels.
- `ga_operators`: molecular operations and geometric docking; see
  `research/ga_ssw/GA_OPERATORS.md` for supported TYPE3 branches and provenance.

Minimal currently runnable physical-surface example:

```python
from ase import Atoms
from ase.calculators.emt import EMT
from pamssw.standalone.surface import ASESurface, quench

atoms = Atoms('Cu2', positions=[[0, 0, 0], [2.7, 0, 0]])
surface = ASESurface(EMT())
result = quench(atoms, surface, fmax=1e-3, steps=100)
assert result.converged  # force stationarity only, not a Hessian certificate
```

Complete ordinary SSW example using the tested Cu13/EMT configuration:

```python
import numpy as np
from ase.cluster.icosahedron import Icosahedron
from ase.calculators.emt import EMT
from pamssw.standalone import ASESurface, SSWConfig, run_ssw

config = SSWConfig(width=.2, rotation_bias=10., max_gaussians=1,
    temperature_K=300., fmax=.001, relax_steps=100,
    fd_step=.001, rotation_hvp=40, rotation_tol=.02)
result = run_ssw(Icosahedron('Cu', 2), ASESurface(EMT()),
    steps=1, config=config, rng=np.random.default_rng(9))
print(result.status, [step.status for step in result.records])
```

For the PAM Safe-total numerical backend, explicitly select
`quench_optimizer="safe-lbfgs-total"` on `SSWConfig`. Its optional
`lbfgs_memory=400` retains at most 400 curvature pairs; omitting it preserves
10. The same option is available on `surface.quench` with
`optimizer="safe-lbfgs-total"`. Other fixed-cell backends reject this option.
History storage scales as O(memory × number of optimization coordinates).
This is a numerical capacity setting, not a change to the physical potential
or an adaptive search policy. 400 is an experimental choice motivated by the
native implementation and controlled comparisons, not a universal optimum.
See `docs/research/explicit-lbfgs-memory-design.md` for evidence and limitations.

These are short integration settings, not recommended optimal search defaults.
Replace Atoms and Calculator with the intended physical system/backend.
`run_ls_ssw` additionally requires `ls=LSSettings(...)`; the C60 script shows
explicit paper-derived parameters and true energy-response recording.
`run_ga_ssw` additionally takes initial structures, groups, frozen descriptor
references and `PaperGAConfig`; the water script provides a complete tested
invocation. Install the package from this checkout with `python -m pip install .`.

EMT is an example backend, not a requirement. Each concurrent walker must
own a calculator and, for file calculators, its own calculation directory.
The bare `ASESurface` rejects constrained input. Use `run_constrained_ssw` for
FixAtoms/Hookean so the persistent restraint objective is assembled exactly once.
Gaussian positions must
remain unwrapped through an escape; LS uses MIC on its frozen fixed cell.
No joint variable-cell search or stress interface is claimed by these modules.

`paper_reference` implements SSW2013 steps 1-8, BP-CBD2012 forward-force
height and the LS lifecycle. It explicitly substitutes one-sided Ritz rotation
for native Broyden and uses ASE LBFGS. Paper direction sampling combines a
Maxwell global direction with a >3 Angstrom atom-pair move; an explicit global
variant supports small systems lacking that pair, without a silent fallback.
`paper_ga` connects quick exploration, TYPE3 generation/short searches, and
ranked-region fine searches. Budget choices are explicit; uploaded-release
hidden multipliers/carry rules are not silently reproduced.

Remaining algorithm work: native Broyden/stopping/MC execution parity,
exact native TYPE4 routing/scheduling,
closed-loop RC constraints and full native periodic/VC behavioral parity. Independent
TYPE0 and joint-VC implementations are documented below. Recovered release bugs and paper-level
algorithm behavior must remain distinguishable.
The Cu13/EMT probe in `research/ga_ssw/evidence/independent-cu13-surface/`
checks surface/direction integration only. It is not an SSW search benchmark.

The optional `SSWConfig(..., rotation_solver='dimer')` selects the independent
plane-minimizing dimer rotation. The default remains `'ritz'`. Both use the same
explicit rank-one anchor bias and forward finite-difference convention; dimer
retains only the current two-dimensional rotation plane, while Ritz grows a
Krylov subspace. Neither is the original CBD history update. The selected solver
is recorded per climbing segment and also applies to `run_ls_ssw` and the SSW
walks inside `run_ga_ssw`. See `docs/research/standard-dimer-direction.md`.

Experimental isolated-cluster geometry: `SSWConfig(..., cluster_frame='eckart')`
uses a fixed equal-weight Eckart-type section during each Gaussian escape.
This requires a non-linear isolated cluster whose calculator energy is invariant
under overall translation and rotation; PBC, external-field symmetry breaking,
and atom constraints are outside this option's scope. The default remains
`'cartesian'`. Both direction solvers and the full modified quench use the same
section; the final true quench is unrestricted and checks complete atomic forces.
Section loss is a failed step, not a silently repaired geometry. This is an
independent formulation, not exact LASP parity, RC-SSW or variable-cell support.
See `docs/research/cluster-eckart-design-review.md` and the Cu13 diagnostic reports.

`SSWConfig(..., cluster_frame='direction_only')` projects the anchor and rotation
operator at each current Gaussian center but retains unrestricted Cartesian
biased quenching. It requires the same isolated-cluster symmetry assumptions as
`eckart`. This reflects the recovered native rigid tangent span, not full native
CBD parity. The Cu13 ablation finds new structures with either option, with
substantial and different failure rates; neither is a universally validated
default. See `docs/research/2026-09-10-direction-only-ablation.md`.

`SSWConfig(..., quench_optimizer='safe-lbfgs-total')` selects the existing PAM
numerical `Relaxer` for initial, LS pre-quench, Gaussian climbing and final true
quenches. It optimizes the full physical + Gaussian + frozen LS objective using
its existing Safe-total settings; it does not invoke the PAM walker. The default
remains `'ase-lbfgs'`. Direct `quench(..., optimizer='safe-lbfgs-total')` is also
supported; existing ASE optimizer-class arguments remain compatible.
`cluster_frame='eckart'` with this backend is explicitly unsupported;
`'cartesian'` and `'direction_only'` are supported. Physical calculator metadata,
cell and PBC are preserved while numerical coordinates remain unwrapped.
Iteration limits do not bound line-search E/F calls: `evaluation_requests`
includes actual surface requests through final force certification. Modified
certificates use the full modified force; true landings remove both biases and
check the complete physical force. This backend option is experimental and does
not establish native optimizer parity or general search superiority. Interface
and LS lifecycle checks are in `tests/standalone/test_safe_quench.py`.

### Atomic/alloy TYPE0 GA

Use `PaperGAConfig(..., proposal_type=0)` with `run_ga_ssw(..., groups=None)`.
The controller accepts equal composition with different atom order, dispatches
atomic crossover and pure/alloy mutation by selected region, and preserves
per-atom parent provenance. `SSWConfig(quench_optimizer='safe-lbfgs-total', ...)`
also selects Safe-total for initial and offspring quenches. Physical E/F always
comes from the supplied ASE calculator.

Full mutation requires more than 10 atoms for pure elements or more than 5 for
alloys. Reinsertion deliberately accepts non-colliding trials and reports budget
exhaustion; it does not reproduce the uploaded JAR's reversed collision loop or
origin fallback. This difference and the original empirical geometric constants
are explicit in `atomic_ga.py`. TYPE3 remains the default.

Cu13/EMT short full-loop results and limits:
[TYPE0 report](../../docs/research/2026-09-10-type0-ga-full-loop.md).
This is an experimental fixed-cell implementation, not a demonstrated universal
optimizer or native trajectory reproduction.

## Experimental fixed-cell PBC and joint VC (2026-09-10)

`run_ssw` supports fully periodic, unconstrained fixed cells with explicit
`cluster_frame='translation_only', direction_sampling='global'`. It uses
continuous unwrapped Cartesian climbing coordinates, removes translation from
rotation directions, and retains the complete biased force. Periodic LS uses a frozen periodic-image bond graph. The potential must be invariant under global atomic translation.

### Fixed-cell SSW/LS outer checkpoint

`run_ssw`, `run_ls_ssw`, and `run_native_ls_ssw` accept `checkpoint_path=` as
an output path. The file is written after each completed outer attempt;
`steps` means additional outer attempts. Resume is explicit: load the trusted
local file with `load_ssw_checkpoint(path)` and pass it as `checkpoint=` with
a new `ASESurface`, the same SSW/LS settings, and a generator with the same
bit-generator class. The checkpoint restores the current minimum, archive,
frozen paper/native LS response state, policies, RNG state, and accumulated
request count. An existing path is never an implicit input.

Full `recovered_direction` uses schema 4 to retain the selected atom pair,
group mask, group marker, selection diagnostics and existing RNG/MC state.
On resume its settings may be omitted (inferred from the checkpoint), or
provided identically. No initial quench or direction initialization is repeated.
Schemas 1–3 remain supported; an old checkpoint without full direction state
cannot be converted to that mode by supplying new settings. Pool selection
(`starter_selector`) remains incompatible with checkpointing. Full directions
retain their free, nonperiodic cluster restriction.


The format is a trusted local pickle. Calculator objects are not stored; the
caller supplies a fresh calculator/surface. A hard kill during an inner
Gaussian or optimizer stage has no completed outer boundary and cannot be
resumed by this interface; an external calculator ledger is needed to account
for work already paid. Terminal error checkpoints are retained for diagnosis
and rejected on resume. Existing Gaussian/stage checkpoint interfaces retain
their separate contracts.

An optional `structure_matcher=callable` adds an `identity_view` while keeping
raw `minima` and `records` unchanged. Its representative and mapping indices
refer to raw observation indices; the caller matcher defines translation,
rotation, permutation and PBC semantics. No default identity metric or
threshold is applied, and matcher failures are recorded without dropping the
observation. The matcher is not a dynamical-basin certificate.

The offline Cu13 demonstration in
`research/ga_ssw/identity_cu13_demo.py` selects qualified source structures
from `17-dimer.json`, then presents four raw observations: an identity and a
rigidly transformed copy of the first structure, followed by a permuted and
rigidly transformed copy of the second structure. With its explicit caller
comparator (sorted all pair distances, `atol=1e-6`), the saved result is
`representative_indices=[0,2]` and
`observation_to_representative=[0,0,2,2]`, with four matcher calls and no
failures. This comparator is an illustrative geometric approximation; it does
not establish general structural identity, chirality equivalence, or basin
connectivity. The full standalone suite currently reports 501 passed and 2
skipped; this is implementation evidence, not a search-quality claim.

`run_vc_ssw` is a separate joint atomic/strain driver. It requires energy,
forces AND stress, and explicit `strain_length`, width and rotation bias:

```python
import numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT
from pamssw.standalone import ASEStressSurface, VCSSWConfig, run_vc_ssw

result = run_vc_ssw(
    bulk('Cu', 'fcc', a=3.6, cubic=True), ASEStressSurface(EMT()), steps=1,
    config=VCSSWConfig(strain_length=3.6, width=.2, rotation_bias=100),
    rng=np.random.default_rng(3))
# Explicit research settings, not transferable optimal parameter values.
print(result.status, result.requests, [r['status'] for r in result.records])
```

Coordinates are `(X, L*s6)` with `R=X exp(S)` and `H=H0 exp(S)`, symmetric
log-strain S and orthonormal six-component basis. L in Angstrom sets the relative
atomic/cell search metric; changing it changes the proposal distribution. The
chart remains fixed during each complete proposal and is rebased only after
its biases are discarded; cell rotations and automatic wrapping are excluded, and
positive determinant follows from the exponential. The objective is `E+pV`;
pressure/stress use eV/Angstrom^3, forces eV/Angstrom. Both atomic and strain
components participate in random directions, dimer rotation, Gaussian bias,
modified relaxation and true quench. See `docs/research/vc-logstrain-chart.md`.

True quenching stops on physical Cartesian force and stress thresholds at the
accepted state, independently of the generalized metric; a fresh physical
certificate follows. Modified-surface quenching uses the generalized norm. Failed moves preserve the
current state and costs. `status='completed'` means the requested loop ended;
inspect each record and certificate for failed proposals. Stored minima are
certified stationary candidates, not Hessian-certified minima or distinct phases.
This is an independent mathematical VC extension, not recovered native VC
schedule parity. This Cartesian VC entry point has no periodic constraints or external position
fields or automatically chosen optimal atomic/cell metric are claimed.

## Experimental sequential cell/atomic VC

`BlockSSWConfig` and `run_block_ssw` expose the independent block implementation.
They use the same `ASEStressSurface(calculator)` contract as joint VC, and require
full PBC, no atom constraints, and a translation-invariant energy model. Supply
an `SSWConfig` with `cluster_frame='translation_only'`,
`direction_sampling='global'`, `quench_optimizer='safe-lbfgs-total'`, and explicit
atomic search settings, then `BlockSSWConfig(atomic=atomic, quench_length=...)`.

Each proposal performs cell soft-mode displacement and limited fixed-cell atomic
relaxation, optionally atomic Gaussian climbing, then a common full true quench
and enthalpy MC. `cell_cycles` is the actual count; one-based outer steps divisible
by `atomic_period` include the atomic climb. Intermediate atomic `maxiter` is
allowed; final physical force/stress certification is required. Rejected valid
landings are retained. This block driver also uses cell degrees of freedom during
escape; it is distinct from adding cell relaxation only to the final quench.

The cell block uses nine physical row-cell entries, fixed fractional positions,
`d(E+pV)/dL = V L^{-T}(stress+pI)`, and projects three infinitesimal rotations.
Its displacement magnitude is `cell_step_fraction * ||L||_F`. Partial relaxation
is numerical Safe-total; native CBD/Broyden history is not reproduced. Currently
a fresh random cell anchor is drawn each cycle; native direction persistence
remains unresolved and this is explicitly recorded in every cycle. Paper example
settings are research starting values, not universal or optimized defaults.

`atomic_climb` is an internal reusable stage without initial/final quench or MC.
Its checkpoint resumes at a completed Gaussian boundary, preserving the anchor,
bias terms and prior cost. It redoes any incomplete Gaussian, without restoring
optimizer history. Stage replay alone does not restore an outer MC trajectory.

AlOH26 and brookite48 MACE development artifacts are under
`research/ga_ssw/evidence/block-*`. Distinct low-force/stress candidates and
budget-censored proposals are recorded separately. These limited runs do not
establish efficiency, Hessian stability, DFT accuracy or production readiness.

## Periodic LS and joint VC-LS

`run_ls_ssw` now supports full fixed-cell PBC using `LSSettings` and the
translation-only/global configuration. Every periodic bond `(i,j,S)` is counted
once with its reverse identified, including multiple images and self-image
bonds in small cells. Images and reference lengths remain frozen throughout a
proposal on continuous unwrapped coordinates; there is no dynamic MIC switching.
This is an independent extensive periodic extension, not native MIC parity.

`run_vc_ssw(..., ls=LSSettings(...))` composes this LS potential with the joint
log-strain kernel. The soft-only prequench is explicitly fixed-cell atomic
relaxation; the climbing potential includes exact LS atomic forces and cell
stress, including self-image virials. The final true quench and MC remove both
LS and Gaussian biases. Response updates use true prequench energy per atom and
rebuild bonds on the selected next structure. No new numerical search setting
is introduced. A failed deterministic LS preparation ends the run with its
state and cost preserved. Mathematical and EMT integration checks pass;
covalent/material efficiency remains unvalidated.

## Articulated-chain RC-SSW

`run_rc_ssw` accepts explicit overlapping bodies, topologically ordered parents
and two shared joint atoms for each child. `RCSSWConfig` requires an explicit
angular length metric, width and rotation bias. The current single isolated
chain driver searches internal torsions with exact finite rotations and Jacobian
force pullback; the free global pose is excluded. Final quenching is full
Cartesian ASE optimization, and new charts use the selected relaxed geometry.
It is not a permanently rigid constrained-minimum calculation.

See `docs/research/rc-ssw-driver.md` for the actual GFN2-butane result, accounting
and limits. Closed-loop constraints and native
lambda force distribution are separate outstanding capabilities.


## Multi-molecule, variable-cell and substrate entry points

`run_rc_forest_ssw` retains relative translations/rotations and chain torsions
across disjoint molecules; `run_rc_vc_ssw` additionally searches symmetric cell
strain without affine stretching of rigid interiors. Both release rigidity in
the final true-PES quench. Explicit rotation/torsion/strain metrics are required.
`read_rigid_topology` reads the supplied rigidbody/blist formats, while
`unwrap_rigid_molecules` reconstructs finite molecules from explicit periodic
bonds and rejects ambiguous images or winding networks.

`run_constrained_ssw` handles `FixAtoms` (or explicit fixed indices) and ASE
`Hookean` pair/point/plane restraints, including two-dimensional PBC. It also
allows no fixed atoms, retaining the full Cartesian active space. Other
constraints are explicitly rejected. Hookean contributes to energy and forces
throughout SSW and either LS variant, including true quench and MC; only the
SSW/LS biases are removed for true quench. Returned Atoms retain the constraints.
This does not make arbitrary constraints available to GA atom-reordering moves. Optional `direction_fixed_indices` excludes additional atoms only from
initial directions and mode refinement. Those atoms remain free in biased and
true quenches unless also physically fixed. Rotation residuals are then measured
in the selected subspace; this is not a full-space lowest-mode certificate.
Its certificate tests active forces and separately
reports unprojected objective forces; fixed atoms and cell remain fixed throughout.
For Hookean, certificates additionally store bare physical and restraint E/F.
A restrained minimum need not be stationary on the unrestrained physical PES.

`run_periodic_ga` supplies three-stage TYPE1 exploration with a periodic image
routing descriptor and an independent caller-supplied identity matcher.
`run_molecular_periodic_ga(..., molecules=...)` runs the same lifecycle with TYPE2
whole-molecule proposals. Molecular groups are disjoint, unlike overlapping RC
bodies. Initial molecules should be unwrapped; no fixed image table is reused
against a changing parent archive. The default local walker is full joint VC;
a compatible walker callback can replace it. Lineage is per atom for TYPE1 and
per molecule for TYPE2, with the original proposal retained in the report.

These are independent implementations. Recovered native collision, cell-transfer,
shared-mutation and energy/gradient inconsistencies are documented rather than
silently reproduced. Physical force/stress convergence, distinct stable minima,
and search efficiency are separate validation claims.

`cluster_reconnection.reconnect_clusters` is an optional, calculator-free
geometry component for finite isolated unconstrained clusters. It returns a
copy of the ASE `Atoms`, the medoid and initial component, every component
translation, and the last attachment separation; it rejects PBC and
constraints and leaves composition unchanged. Its `0.7` displacement fraction
is the recovered empirical native value. A distance such as `1.7` A came from
the carbon SI comparison and is not a universal default. The implementation
uses O(N^2) distance storage and O(N^3) worst-case sequential repair time. It
does not provide balanced sampling, physical dynamics, boundary-overlap
handling, or a closed-loop guarantee, and is not connected to a walker.

`run_ssw(..., reconnect_distance=d)` and `run_ls_ssw(...,
reconnect_distance=d)` expose this component only at the completed-climb
boundary, immediately before the unbiased true landing quench. The returned
reconnection trace is stored on `SSWStep.cluster_reconnection`; the original
climb work and `last_atoms` remain unchanged. `None` preserves the existing
lifecycle exactly. This restores only the Allopt-entry geometry operation; it
does not implement an in-quench vapor stop, `globalcompress`, or `Ratio_Local`,
and is not a complete native-policy reconstruction.


`run_surface_ga` now supplies full TYPE4 proposal families and three-stage
fixed-support SSW. It requires explicit support/adsorbate indices, two-dimensional
routing features and a separate identity matcher; physical E/F and auxiliary
synthetic-potential evaluations have distinct ledgers. `SurfaceGAConfig` records
all finite proposal and auxiliary-optimization budgets.

`run_ssw(..., height_policy=ConservativeNativeHeightPolicy(...))` enables the
experimental native-derived height/history policy while retaining conservative
single-count Gaussian energy/forces and stage-frozen weights. All six settings
are explicit, and the old forward-force policy remains the default. Native
history reweighting is part of the returned preparation event. This profile is
currently unsupported with the Eckart-section quench.

`run_ssw(..., gaussian_policy=PAMCurvatureGaussian(mode="height_width"))`
enables the existing experimental PAM curvature-based width/height policy.
Import it with `from pamssw.standalone.pam_gaussian import
PAMCurvatureGaussian`. `gaussian_policy` and `height_policy` are mutually
exclusive; the optional argument is also forwarded by `run_ls_ssw`,
`run_native_ls_ssw`, and `run_ga_ssw`. The returned PAM weight is used verbatim,
and `None` preserves the existing fixed-width/default stopping behavior. This
is an experimental fixed-cell Gaussian policy, not full PAM or native LASP
parity.


TYPE3 now accepts `run_ga_ssw(..., change_types=...)`: 0 preserves a whole unit;
1 reconstructs its atom-level candidate library before assembly. Mixed-source
units retain per-atom provenance and have no fictitious single parent. The
independent complete-library mode repairs source quota underflow using the
smallest sufficient native family request; strict native quota remains explicit.
`initialize_type2/3/4` expand supplied physical seeds using recovered proposal
families. Their results are unrelaxed candidates, with no automatic claim of
stationarity or descriptor-reference qualification.

`MinimalAngleHeightPolicy` is the separate analytic positive-height solution of
the 87-degree condition. It removes native initialization/growth overshoot and
keeps old history unchanged. Undefined or nonpositive-height cases are explicit;
it does not silently invent a floor or an alternate transition rule.


Fixed-substrate LS is available via `run_constrained_ssw(..., ls=LSSettings(...))`.
It also accepts `NativeLSSettings` and reuses the same native-derived table and
cycle runtime as the unconstrained driver. Fixed endpoints remain in the bond
set and normalization; only the existing active-coordinate chart projects
forces. No automatic zero atom filter is assigned to fixed atoms. Native state
is included in the existing attempt-boundary checkpoint.

The supplied pair tables and target retain their explicit provenance; `ls=None`
preserves ordinary SSW. The soft prequench, biased walk and physical final quench
use the same fixed-atom manifold. Direction exclusions only restrict modes.
Periodic fixed-cell LS includes image pairs on the declared1D/2D/3D PBC axes;
nonperiodic image components are zero. Variable-cell LS still requires full3D
PBC. Cu28/EMT contract checks support the composition, not a general LS gain.

Constrained SSW also accepts `checkpoint_path=` for explicit attempt-boundary
save and `checkpoint=` loaded from `load_constrained_checkpoint(path)` for
resume. `steps` counts additional attempted outer iterations. The returned
state is available as `result.current.atoms`, with cumulative `result.requests`;
raw failed climbs remain in `result.records`. Initial/LS initialization,
LS-prequench, and LS-update failures produce terminal diagnostic checkpoints
that can be loaded for inspection but cannot be resumed. Ordinary rotation,
biased-quench, and true-quench failures retain the recoverable
`completed_with_failures` flow. Resume validates composition, PBC/cell/masses,
all fixed coordinates and masks before any PES request; `steps=0` preserves a
valid boundary. The file is a trusted local pickle written atomically. ASE
calculators and physical surfaces are never serialized; callers provide a
fresh surface with the same settings. Current implementation checks use
Harmonic and paper-LS interfaces. The Cu/Al EMT continuation diagnostic
completed with equal request totals and qualified constrained landings. Cu has
exact paid-ledger replay; Al has small trajectory differences, so exact replay
is not claimed. See `docs/research/2026-09-12-constrained-resume-audit.md`.

`LSSettings(..., energy_filter={(26,26): 0.0})` sets the Fe–Fe LS energy
contribution to zero while retaining its geometric pairs. Omitted factors are1;
the immutable filter survives every response rebuild. Explicit positive energy
and cutoff tables are still required. This independent paper response uses
filtered bond-energy weights, not the native `Nb`-normalized table update.
Fe7C3/MACE controls exercise this interface but have not shown an LS search gain.

`run_vc_ssw(..., height_policy=MinimalAngleHeightPolicy())` applies the existing
analytic87-degree force-angle height in the configured scaled joint atomic/cell
metric. It reuses the height background evaluation and preserves conservative
Gaussian energy/gradient pairing. Default `None` preserves fixed-forward-force
height; a nonpositive required height stops the proposal rather than silently
releasing a biased point. This extension is experimental, not native VC parity.

### Experimental PAM Gaussian and inner-force controls

For sequential VC block experiments, `BlockSSWConfig.atomic_gaussian_policy`
accepts `PAMCurvatureGaussian` from `pamssw.standalone.pam_gaussian`.
`mode='height_only'` preserves the atomic config width and uses PAM curvature
height; `mode='height_width'` also applies PAM's curvature-adaptive width.
The default `None` preserves the existing forward-force height. This is the
Gaussian core only: it does not reproduce PAM's trust feedback, archive
controller, direction selection or default per-atom-RMS coordinate scale.

`SSWConfig.bias_fmax=None` preserves the former shared tolerance; a positive
value changes only Gaussian biased relaxation, leaving initial/final true
quench force thresholds at `fmax`. `bias_stage_steps` separately opts into
bounded-stage completion and remains disabled by default. The mere use of a
looser inner tolerance does not enable budget-based release or loosen final
force/stress certificates. Experiments and limitations are documented in
`docs/research/pam-gaussian-controlled-plan.md`.

`BlockSSWConfig.partial_atom_fmax=None` keeps the cell-interleave partial
atomic relaxation tied to `atomic.fmax`. A positive override changes only that
partial relaxation's force stopping tolerance (eV/Angstrom). This permits
controlled inner/outer tolerance comparisons while holding cell-interleave
stopping fixed; initial/final all-DOF quenches still use `atomic.fmax` and
`stress_tol`. The override is experimental, not a new recommended default.

Fixed-cell public interfaces (2026-09-12): `run_ls_ssw` and
`run_native_ls_ssw` accept the same explicit `height_policy` and
`height_update_budget` as `run_ssw`. This forwards an existing conservative
height rule; it does not change LS anchor sampling, response updates, or defaults.

`run_ga_ssw(..., structure_matcher=callable)` separates archive identity from
legacy descriptor projections used for routing. The callable receives owned ASE
copies; matching structures retain their lower-energy representative. A matcher
failure rejects that incoming observation and is recorded. `None` retains legacy
projection deduplication for reproducibility; `result.identity_mode` identifies
which contract was used. Descriptor collisions are not structural identity,
and a supplied approximate matcher still needs domain-specific validation.
Nonfinite-energy observations cannot enter either archive mode.

The fixed-cell controller also accepts `max_evaluations=None` (legacy unlimited)
or an explicit nonnegative total request allowance. The allowance counts the
underlying surface's `requests` increment from entry, including initialization,
offspring quenching and all SSW stages. A blocked request is not sent to the
calculator. The returned `budget_limit`/`budget_exhausted` distinguish truncation
from natural completion; this is an oracle API budget, not an SCF-iteration cap.

`PaperGAConfig.cycles=1` retains the original single generation/fine schedule.
For `cycles>1`, fine minima stay in the archive and are available to the next
cycle's GA parent selection. Initial supplied seeds and initial quick walks are
processed once. All cycles share the random generator and total request budget;
`GAStage.cycle` records the zero-based cycle. This implements the paper's
fine-to-GA feedback at framework level, without Java's hidden schedule factors.


TYPE0 quota boundary: multi-element inputs with GA generations require
`ga_candidates >= 4`; smaller recovered integer quotas generate zero operators
and are rejected before PES evaluation. `4` enables crossover only, `8` also
enables exchange, and `16` enables all recovered mixed-element mutation kinds
in region zero. These are count-contract thresholds, not recommended search
budgets. See `docs/research/ga-type0-quota-domain.md`.


`NativeLSSettings.bond_geometry='native-mic'` preserves the default single-MIC
pair behavior. Explicit `'periodic-images'` uses complete undirected image
records, including nonzero self images, for both the frozen potential and Nb
in initialization/response updates. Frozen shifts use continuous unwrapped
positions at fixed cell. This is an independent periodic extension, not native
image enumeration parity or a variable-cell/stress implementation. Nonperiodic
inputs keep ordinary pairs. Cu/Al bulk/vacancy EMT results and limits are in
[`periodic LS evidence`](../../docs/research/2026-09-12-native-ls-periodic-results.md).

Explicit H/C/O release lookup data are available as
`native_ls.HCO_BOND_ENERGIES` and `native_ls.HCO_BOND_LENGTHS`. Pass these to
`NativeLSSettings` (or explicitly choose the paper LS cutoff convention); no
binary is needed at runtime and no existing default table changes. The data
match all18 ordered H/C/O leaf returns, and H2O/CH3OH native initialization
prefixes at scales5/2.5 match the Python matrices and counts. These are raw
reference parameters before filtering/normalization, not potential-specific
GFN2 or MLIP bond strengths. The HCO real-system development runs are under
`research/ga_ssw/evidence/hco-fixed-cell-20260912/`; they establish bounded
lifecycle evidence, not general LS efficiency or physical binding accuracy.

GA continuation is opt-in at completed quick, generation, or cycle boundaries:

```python
from pamssw.standalone import GACheckpoint, run_ga_ssw

paused = run_ga_ssw(..., checkpoint_callback=lambda state: state.phase == 'generation_complete')
paused.checkpoint.save('ga.chk')
resumed = run_ga_ssw(..., checkpoint=GACheckpoint.load('ga.chk'))
```

The caller recreates the surface/calculator and validator or matcher with the
same contract. The saved request count is cumulative, so the total budget is
not reset on resume. Budget exhaustion and crashes do not create a new
boundary; the last saved boundary is the only replay point.

`run_ssw(..., bias_quench_adapter=callable)` is an experimental fixed-cell
stage interface; `None` preserves the existing biased quench. The callable
returns `BiasStageQuenchOutcome`: a separately evaluated modified-surface
`QuenchResult`, `stage_stopped`, `release_all`, and diagnostics. A stage stop
allows the next Gaussian; release ends climbing and invokes the ordinary
true-surface quench and Metropolis decision. Neither flag certifies a true
minimum. This interface currently rejects checkpoint use.

The research `StatefulNativeStageAdapter` uses recovered scalar stopping
conditions with explicit independent energy references and evaluation budgets.
It supports all-mobile Cartesian biased relaxation, including the existing
`translation_only` direction projector, but not an Eckart quench frame.
It pays for its center-energy reference and reports that cost. Its stopping
point is an evaluated optimizer trial, not necessarily an accepted iterate;
native optimizer lifecycle parity and scientific efficiency remain separate
validation questions. See the frozen
[`stage-control protocol`](../../docs/research/2026-09-12-stage-control-e2e-protocol.md).

`run_periodic_ga(..., fixed_cell=True, walker_config=SSWConfig(...))` selects
the existing fixed-cell SSW walker for TYPE1. Supply fully periodic parents
with exactly the same cell and a `translation_only` SSW configuration using
global or isotropic directions. The original cell orientation is retained;
TYPE1's internal canonical crossover coordinates are mapped back before the
physical calculator is called. Different cells are rejected rather than
strained into the input cell.

Fixed-cell observations carry a true-energy/max-force certificate;
`objective` is an explicit alias for energy, and `forces`/`stress` in the
controller observation are `None`, not fabricated arrays or hidden extra
calculations. Independent final E/F checks remain separately budgeted.
The default `fixed_cell=False` retains the existing VC walker and stress
certificate. TYPE2 fixed-cell mode is rejected before PES calls because its
current molecular proposal constructs a new cell from molecular extents;
supporting it would require a different proposal contract.

### PAM Gaussian with ASE constraints

The explicit constrained entry accepts the same existing
`PAMCurvatureGaussian` as the ordinary fixed-cell driver:

```python
from pamssw.standalone import ASESurface, ConstrainedSSWConfig, run_constrained_ssw
from pamssw.standalone.pam_gaussian import PAMCurvatureGaussian

# atoms carries the desired FixAtoms / Hookean constraints; calculator and rng
# belong to this walker. These numerical values are diagnostic, not universal.
result = run_constrained_ssw(
    atoms, ASESurface(calculator), steps=1, rng=rng,
    config=ConstrainedSSWConfig(width=0.1, rotation_bias=100.,
        gradient_tol=0.1, fmax=0.03, rotation_solver='dimer'),
    gaussian_policy=PAMCurvatureGaussian(mode='height_width'),
)
```

The policy operates in active Cartesian displacements. Each historical Gaussian
retains its own width. `gradient_tol` is the full active-gradient L2 norm in the
biased optimizer, whereas `fmax` is the maximum active-atom force norm for true
quench; equal numerical values do not represent the same stopping test. LS
prequench inherits `fmax` unless `ls.prequench` is explicitly set. Optional `ls=LSSettings(...)` or
`ls=NativeLSSettings(...)` composes the corresponding existing LS route.

`gaussian_policy=None` preserves the reference forward-force branch. Checkpoint
resume requires the same persistent Hookean specifications and Gaussian policy
parameters; it does not silently change the searched objective or strategy.
This is an explicit option, not a new default or a claim of full PAM parity.

### Recovered CBD with ASE constraints

`ConstrainedSSWConfig` also accepts `recovered_rotation=RecoveredRotationSettings(...)`
and `rotation_exit_policy='force'` (the default) or `'force_or_budget'`.
This reuses the existing CBD controller in active Cartesian coordinates;
`FixAtoms` stays exact and persistent `Hookean` energy and force enter the same
rotation objective. `direction_fixed_indices` restricts only the rotation,
not the subsequent relaxation. No free-cluster rigid-body projection is applied
to a fixed substrate.

When recovered rotation is selected, its explicit settings control rotation;
legacy `rotation_solver`, `rotation_hvp`, `rotation_tol` and `rotation_bias`
do not set the recovered controller's limits or bias. `fd_step` remains shared.
Recovered force tolerances apply to `10 * fd_step * HVP_residual`, not directly
to the legacy HVP residual. Existing legacy fields still require valid values.
Budget release preserves `mode.converged=False` and the recorded stop reason;
it does not certify rotation convergence or release invalid directions.

The default `recovered_rotation=None` preserves the old solver. Schema-1
checkpoints missing the new fields restore those defaults; new checkpoints
require the same requested rotation settings. Explicit presweep and PAM Gaussian
composition with recovered rotation are rejected before evaluation in this
minimal implementation. It does not add native MC, pool persistence, or periodic
pair/group direction selection. This is an experimental interface, not a
validated universal performance improvement.

### Separate fixed-cell LS prequench limits

```python
from pamssw.standalone import LSSettings, LSPrequenchSettings
ls = LSSettings(bond_energies=bond_energies, bond_lengths=bond_lengths,
                target_per_atom=target_per_atom,
                prequench=LSPrequenchSettings(fmax=0.1, steps=50))
```

The same optional `prequench` field is available on `NativeLSSettings`. These
explicit soft-surface limits apply to the fixed-cell ordinary/native LS and
constrained Cartesian LS drivers. They do not relax the outer true-start or
final force certificate. `None` preserves the previous inherited limits and
strict convergence behavior. An explicit `exit_policy="force_or_step_limit"`
option permits finite normal iteration exhaustion on the unconstrained fixed-cell
Safe-total path; raw `converged=False` is preserved. Line-search and backend
failures remain failures. Constrained paths reject this optional exit policy. This field is not wired into the experimental
joint/VC preparation paths; the VC driver rejects explicit overrides before evaluation. The values above illustrate an explicit setting,
not a validated universal choice; current Cu EMT tests check interface and
state flow only.

### Experimental recovered direction controller

`RecoveredDirectionController` is an experimental outer-state controller for
recovered fixed-cell CBD directions. It requires free, nonperiodic ASE
`Atoms` and an explicit outer `direction_only` contract. The following settings
are diagnostic values for a reproducible probe, not recommended optimum values:

```python
import numpy as np
from dataclasses import replace
from pamssw.standalone import ASESurface, run_ssw
from pamssw.standalone.recovered_direction import RecoveredDirectionSettings

settings = RecoveredDirectionSettings(
    ratio_local=50,
    local_probability=.5,
    group_threshold=.5,
    pre_rotmax=2,
    rotmax=8,
    pre_ftol=.01,
    ftol=.01,
    metric='euclidean',
    max_force_calls=40,
)
# `atoms`, `calculator`, and `config` are caller-owned existing inputs.
result = run_ssw(
    atoms, ASESurface(calculator), steps=2,
    config=replace(config, cluster_frame='direction_only',
                   pre_rotation_hvp=None),
    rng=np.random.default_rng(17), recovered_direction=settings,
)
```

Use this shared ASE path with `direction_only` and without
`pre_rotation_hvp`. Full-direction outer checkpointing uses schema 4 as
described above. `SSWConfig.rotation_solver`, `rotation_bias`,
`rotation_hvp`, and `rotation_tol` do not control this CBD controller;
`fd_step` and `rotation_exit_policy` still apply to the surrounding SSW/dimer
machinery. `calculator` may be any ASE-compatible backend; the subsequent
scientific validation target is MACE-OMAT-0-small. Q/compression recovery is
incomplete, and startup is an explicit Python contract. This `run_ssw` entry
reuses the shared quench and MC routes; it is not complete LASP execution
parity. An existing LS configuration may still be
passed through `run_ssw(ls=...)` and uses the shared surface, but no new-material
LS acceptance has been established.

### Optional recovered MC acceptance

`run_ssw`, `run_ls_ssw`, and `run_native_ls_ssw` accept
`mc=NativeMCSettings(energy_tol=..., maxtrap=...)`. Import the settings from
`pamssw.standalone`. Omitting `mc` retains ordinary Metropolis acceptance.
Both settings are explicit: the tolerance is in eV and `maxtrap` controls the
recovered repeated-energy counter. These are reference-behavior parameters,
not recommended general-purpose defaults.

This mode retains the uploaded binary's fixed energy-difference divisor 20,
stateful repeated-energy counter, and one uniform draw per eligible landing,
including downhill moves. Therefore its input temperature is not equivalent
to the same temperature in ordinary ASE-unit Metropolis. It does not implement
LASP's outer forced-acceptance overrides or its random-number generator.
A numerical-domain failure terminates with `mc_failed`, retains the landing
and cost, and skips subsequent LS strength updates.

Native-mode checkpoints use schema 2 and save MC settings/state together with
the walker RNG. Resume requires the same explicit settings. Old schema-1
checkpoints remain usable in ordinary mode; they cannot reconstruct missing
native counter state. Existing restrictions on checkpointing other optional
controllers still apply. See
[the recovered contract](../../docs/research/native-mc-contract.md) and
[integration evidence](../../docs/research/2026-09-18-native-mc-integration.md).

### Experimental fixed-cell starter selection

`run_ssw` accepts `starter_selector(snapshot, selector_rng)` and a separate
NumPy `Generator`. The snapshot contains copied, true-force-qualified
observations, including MC-rejected landings. Return an observation index to
choose the next starter, or `None` to keep the ordinary MC result:

```python
def uniform_observation(snapshot, selector_rng):
    return int(selector_rng.integers(len(snapshot.observations)))

result = run_ssw(
    atoms, surface, steps=steps, config=config, rng=kernel_rng,
    recovered_direction=direction_settings,
    starter_selector=uniform_observation,
    selector_rng=np.random.default_rng(31),
)
```

A different observation index explicitly restarts direction selection from
that stored geometry, without another true quench. Otherwise the recovered
controller retains its existing cross-step state, including information from
MC-rejected landings. `record.accepted` remains the MC decision;
`record.starter_selection` records the actual subsequent selection and restart.
The selector runs after a qualified landing and successful state update;
failed landings do not trigger this first-version hook. All failed work remains
charged to the surface counter.

Observation indices are **not deduplicated basin IDs**. Uniform selection here
weights repeated observations repeatedly; it is an interface example, not a
validated search policy. Snapshot mutation cannot modify stored geometries.
Selectors cannot inject unqualified coordinates, and invalid selections raise
rather than silently falling back. The hook currently excludes LS and
checkpoint recovery. Omitting both new arguments preserves the existing API,
MC behavior and kernel random stream. MACE descriptors are not required.
