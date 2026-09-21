# Joint periodic LS preparation (research-only)

`pamssw.standalone.joint_ls.prepare_joint_ls_step` is an explicit alternative
to the existing fixed-cell `prepare_ls_step`. It optimizes the full
`3N+6` symmetric-log-strain objective

`E_physical + E_frozen_periodic_LS + pV`

with the existing VC `gradient_tol`, `fmax`, `stress_tol`, `max_step`,
`relax_steps` and `lbfgs_memory`. No new numeric parameter is introduced. The
frozen pair/image labels and reference distances must describe the input within
the same 32-ULP allowance as fixed-cell preparation. Physical E/F/stress and
LS E/F/stress are combined in one chart evaluation; the cached initial value is
reused by Safe-total without an extra convergence request.

The Safe-total callback returns the chart's translation-projected joint
gradient. After convergence, the accepted geometry receives a fresh physical
E/F/stress request and a fresh analytic LS F/stress check; the returned
softened force/stress certificate is from that final check. The returned record
keeps physical energy and enthalpy before/after, LS energy, volume, joint
gradient, force/stress certificates, optimizer state and charged requests as
separate fields. At nonzero pressure, response consumers must use
`true_enthalpy_after - true_enthalpy_before`; the physical E difference remains
available separately. For a fixed-cell comparison the volume term cancels, but
this joint helper must retain it because the cell is an optimized coordinate.
Backend or optimizer failure raises `LSCycleError` with
the last accepted q/atoms and request count. The VC walker exposes this helper with explicit `ls_prequench="joint"`;
it does not claim native LASP parity.

The accompanying EMT tests cover cell-gradient relaxation, nonzero-pressure
H accounting, frozen-reference validation and failure-state cost retention.

## Actual motivating observation and matched control

The zero-new-PES audit `research/ga_ssw/audit_ls_prequench_cell_gradient.py`
reconstructs all8 saved preparations of Fe7C3-80 from their exact recorded
geometries, physical E/F/stress and frozen LS. Atomic modified-surface forces
passed, while the cell-block gradient norms were15.543–27.969 eV/A against
the existing joint gradient tolerance0.001. This is a measured mismatch in
preparation coordinates, not evidence about native LS cell scheduling or a
proof that changing preparation will improve global exploration.

`run_vc_ssw(..., ls_prequench='joint')` is the explicit alternative; default
`fixed_cell` preserves the previous baseline. The joint response uses true
enthalpy, with physical energy and volume separately recorded. Fixed-cell
response already satisfies ΔH=ΔE even under nonzero pressure because ΔV=0.
No extra numerical tolerance, fallback, or adaptive parameter is introduced.

The planned Fe7C3 comparison retains both all-pair and Fe–Fe-filtered LS,
seeds7/101, the same2000 total EFS/480s per arm, original height rule/history10,
and two requested attempts. Only the prequench coordinate boundary changes.
Old fixed-cell preparation results remain the complete comparison baseline.
Include failed preparation costs and last accepted geometry, and independently
recheck every physical landing; joint softened stationarity alone is not search
success. No budget extension or retuning is authorized by this plan.
