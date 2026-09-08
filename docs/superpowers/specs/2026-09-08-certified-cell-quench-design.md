# Certified minima and cell-relaxed quenching

Approved direction: the 2026-09-08 repository audit and the user's subsequent
"ok continue". Work starts from fb02469 in fix/certified-cell-quench.

## Scope

Implement the first three connected improvements: certified search landings,
geometry-first basin identity, and opt-in variable-cell true quenching after a
fixed-cell SSW/LS-SSW action. Do not import the old generalized VC walker.
Do not claim joint atomic/cell escape, production MACE validation, canonical
sampling, or a universal basin equivalence matcher.

## Terminal contract

The low-level relaxer returns incomplete RelaxResult values for diagnostics.
A failed initial search quench raises QuenchConvergenceError with the relaxation
and exact evaluation counts. A failed landing is recorded as uncertified and
excluded from archive insertion, best-energy updates and productive credit;
search continues while budget remains. The posterior worker maps an uncertified
starter/landing to INVALID, never WORKER_ERROR or false budget exhaustion.
Certificates require finite energy and nonnegative finite active force norm within
fmax. Cell quenches additionally require the allowed stress residual within
quench_stress_tol (eV/Angstrom^3). Optimizer success is not a certificate.

## Basin identity

Geometrically matching endpoints do not become new basins solely because their
energies differ. Preserve the first representative and record energy mismatch
counts/max discrepancy; do not silently update archive node identity or fabricate
new discoveries. Existing species-order-aware cluster/MIC matching remains an
approximation, explicitly documented. Different PBC or incompatible periodic
cells never merge; use a separate relative cell tolerance for cell relaxation.
No species-aware descriptor redesign or general permutation search in this change.

## Cell quench

SSWConfig exposes quench_cell_mode=fixed|volume_only|shape|slab_xy,
quench_stress_tol=1e-3, external_pressure_gpa=0, dedup_cell_tol=1e-3.
Default fixed mode retains existing atomic proposal and optimizer settings.
Opt-in cell quenching requires ase-lbfgs or ase-fire for primary/fallback.
CellRelaxer uses the actual cell in every evaluation and returns it unchanged
from the optimized Atoms. ASE FrechetCellFilter supplies the generalized
coordinates/gradients, with independent physical force/stress stopping checks.
Return energy=E+pV, potential_energy=E, volume=V, stress_norm=residual. Preserve
these values in state metadata for structure output. Missing/nonfinite stress,
invalid cells or unsupported PBC/mask combinations fail explicitly.
Fixed atoms follow affine cell deformation (fixed fractional coordinates).
slab_xy fixes the vacuum vector and only allows xx, yy and xy strain for
axis-aligned XY slabs; nonzero hydrostatic external pressure is bulk-only.
The posterior runner rejects cell mode before spending budget until its action
contract has explicit enthalpy/stress metadata. Standalone run_ssw/run_ls_ssw
are the supported search routes for this experimental feature.

## Verification

Use regression tests for uncertified bootstrap/landing, worker mapping,
energy-only splits, PBC/cell mismatch, and valid same-basin matching. Validate
cell quenching on analytic volume and atom/strain objectives, nonzero pressure,
nonzero deformation, slab/fixed-atom constraints, unavailable stress, force
budget interruption, and a small periodic LJ search. Re-run affected unit and
integration tests; record any baseline failures separately. No GPU/Slurm jobs.
