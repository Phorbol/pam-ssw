# Fe7C3 block cell/atomic energy diagnosis

This is a zero-PES readout of the saved `evaluations.jsonl` and block records.
For each completed cell cycle, the charged search evaluation at the exact
ledger endpoint (the cumulative request count after that cycle) is reported in
`fe7c3-block-baseline/cell-energy-diagnosis.json`. The report includes E,
force max, stress max, volume, cell singular values and condition number,
minimum MIC atom distance, cell-mode convergence/residual/curvature/force
calls, displacement length, and partial atom status.

The request ledger advances over the complete preceding outer record, so the
cell+atomic outer index 1 endpoints are not confused with the cell-only outer
index 0 prefix. The first atomic center is checked against the exact entry
evaluation; each completed true-energy boundary is checked against the next
event center, and the final boundary against checkpoint atoms.

The cell-cycle endpoint is the state after a cell direction move followed by a
fixed-cell partial atomic relaxation (`partial_steps=25`). It is therefore a
diagnostic endpoint, not a full stress-quench certificate. The atomic-climb
entry geometry and each completed Gaussian boundary are also matched from the
saved energy evaluations.

## Findings

The initial structure is E=`-682.1259014 eV`, volume `686.1044 Å³`, cell
condition number `3.81735`, and minimum MIC distance `1.9000 Å` for both seeds.

- **Cell-only outer index 0:** all five cell modes are reported
  `converged=False`, with `partial_status=maxiter` after 25 atom steps. Seed7
  cell-cycle endpoint energies move from `-627.58` to `+132.48 eV`, volume
  `578.17` to `333.04 Å³`, and stress max reaches `9.40 eV/Å³`. Seed101
  energies range `-648.89` to `-595.49 eV`, volumes `549.85–678.67 Å³`, and
  stress max reaches `0.97 eV/Å³`. These are unstable partial trajectories,
  not evidence that a full cell quench would have the same endpoint.
- **Cell+atomic outer index 1, before atomic climb:** seed7's last cell
  endpoint is E=`-649.9752 eV`, volume `646.4889 Å³`, stress max
  `0.2045 eV/Å³`, condition number `4.0959`, minimum distance `1.4126 Å`.
  Seed101's endpoint is E=`-152.9542 eV`, volume `371.9020 Å³`, stress max
  `5.59 eV/Å³`, condition number `7.8492`, minimum distance `1.3253 Å`.
  Thus seed101's high-energy atomic input is already produced by the cell
  preparation trajectory; the subsequent fixed-cell atomic climb does not
  change the cell.
- **Completed atomic boundaries:** seed7 keeps volume `646.4889 Å³` and
  condition number `4.0959`; its boundary fmax is `0.596–1.609 eV/Å` and
  stress max `0.182–0.229 eV/Å³`. Seed101 keeps volume `371.9020 Å³` and
  condition number `7.8492`; its boundary fmax is `1.177–3.166 eV/Å` and
  stress max `5.590–5.744 eV/Å³`. The boundary energies remain in the
  previously reported `+20–26 eV` and `+508–517 eV` ranges relative to the
  common initial reference.

The `cell_mode.converged` flag and the `partial_status=maxiter` flag are
separate. A cell direction can be marked converged by its direction solver
while the subsequent 25-step fixed-cell atom partial relaxation reaches
maxiter. Conversely, a nonconverged direction residual can still produce a
saved partial endpoint. This distinction is why the high seed101 energy is
diagnosed as a trajectory/input condition with large stress and distorted cell
metrics, rather than automatically labeled physical failure. A true stress
relaxation was not performed in this block preparation, and fixed-cell atomic
relaxation cannot remove the cell stress.

No basin identity or material-physics conclusion follows from these metrics.
The records retain the budget cutoff and all partial failures; no PES calls or
retries were added.
