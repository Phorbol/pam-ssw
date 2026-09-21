# XXXII converted-GAFF RC-VC one-step plan

This is an executable bounded experimental plan for the 172-atom XXXII structure using the existing `run_rc_vc_ssw` interface and the audited stock CHARMM conversion. It is an experimental converted-GAFF PES check; it does not claim native LASP whole-engine parity or production benchmark validity.

The frozen inputs are `tests/standalone/fixtures/type2_xxxii.extxyz`, the original `rigidbody`/`blist`, and the converted `lmp.data`/`in.simple`/manifest from `xxxii-lammps-qualification-table0-ewald12`. The calculator must issue `pair_modify table 0`, `kspace_style ewald 1e-12`, and preserve the first logged `G=0.47570069` as a fixed screening parameter (not a guarantee of a fixed reciprocal vector set). RC-VC uses rotation/torsion/strain lengths `1/1/5`; `width=.6` and `rotation_bias=100` are current material-baseline values explicitly provisional, with `max_gaussians=12`, seed3 and baseline Safe-LBFGS memory10 (`None`).

The proposed ceiling is 1500 EFS/120 s, with two independent fresh checks included within the total1500 cap; the initial and candidate are checked when available. Every request must preserve coordinates, energy, forces and stress, including failed evaluations and budget exceptions. The fixed-G derivative review completed52EFS: dominant cell residual matches the known erfc force approximation, with RC1.7e-7 remaining; original tolerances are retained; the run must remain separate from the existing qualification artifacts and cannot establish native parity or a general RC-VC benefit.

Preparation entry point: `research/ga_ssw/probe_xxxii_rc_vc_lammps.py`. The default prepares inputs without PES; `--execute` runs the reviewed implementation using the qualified existing LAMMPS launcher. A full source/input snapshot and per-request API/engine counts are saved; interrupted or failed work is retained. No general efficiency claim follows from this one input/seed.

## Execution and predeclared follow-up

The memory10 whole step consumed385API/382engine calls, initial certificate
passed, first biased quench maxiter300 with norm0.331. No landing was obtained.
An identical frozen first objective was tested with only memory400:291API calls,
283iterations, norm0.00383591, converged. Initial energy mismatch3.46e-14 and
gradient mismatch3.55e-15 verify the comparison. The saved topology bond lengths
remain unchanged by the rigid chart; this is an optimizer failure, not evidence
of molecular dissociation. The next whole-step control changes only memory400,
retaining1500API/120s, seed3 and all other configuration. No automatic default
change or continuation is implied. All failed baseline and diagnostic costs
remain part of the campaign.

## Separate the nested iteration limit from total cost

Both whole-step arms stopped below their shared1500API ceiling because of a
per-quench300-iteration guard: memory10 at385API, memory400 at741API (second
biased norm0.02494). These results do not distinguish exhausted useful search
budget from an earlier implementation guard. Next, two matched controls retain
the same1500API/120s total, input, tolerances, seed and memory10/400, setting
per-quench maxiter equal to1498 (the shared search API ceiling). Since each
iteration requires at least one call, this guard cannot bind before the declared
total budget. This is a budget-accounting diagnostic, not a fitted new search
parameter. All earlier failures remain reported; success means a true landing
with independent force/stress and geometry checks, not just an extra stage.
No further automatic budget enlargement is part of this experiment.
