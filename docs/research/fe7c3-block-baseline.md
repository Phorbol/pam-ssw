# Fe7C3-80 sequential cell/atom block baseline

2026-09-11. Intermediate 25-step atomic relaxation does not need a full physical
certificate: both cell-only block proposals yield independently certified
endpoints. Neither lowers energy. The combined block attempts are censored by
the fixed request budget, not evidence of impossible biased relaxation.

Artifacts: `research/ga_ssw/fe7c3-block-baseline/` (frozen source, runner, input,
plan, ledger, per-stage results, fresh certificates, audit and identity outputs).
The existing independent `block_ssw.py` implementation was exercised without
algorithm changes. Same qualified Fe7C3-80 input, MACE-OMAT-small float64,
pressure0, temperature300K, fmax0.001eV/A, stress0.0001eV/A^3, memory10,
maxiter300, two outer attempts, seeds7/101 and2000EFS per arm inclusive of up
to3 fresh checks as the joint comparison. Five cell cycles, fraction0.15 of
lattice Frobenius norm,6cell-mode EFS at most,25fixed-cell atomic steps per
cycle, atomic SSW on the second outer attempt. Parameters are explicit existing
example-inspired choices, not universal optimum values. The runner selects the
block arm and only Safe-total. It is not a one-variable optimizer ablation.

Job1261467 completed exit0 on one V100/4v100n13 in1m57s. Both search results
are correctly marked censored despite scheduler completion. Total3998EFS
=3994search+4fresh. Initial structures are not counted as new candidates.

| Seed | Cell-only proposal EFS | Final joint quench EFS within proposal | Energy above initial | Cell+atomic attempt EFS | Atomic climbing EFS within attempt |
| --- | ---: | ---: | ---: | ---: | ---: |
| 7 | 222 | 61 | 15.1149496eV | 1773 | 1620 |
| 101 | 221 | 61 | 18.8031856eV | 1774 | 1613 |

Every intermediate atomic relaxation in all20 completed cell cycles stopped
at25steps with `maxiter`. Both first proposals then passed physical joint
quench and independent E/F/stress checks. They match neither initial nor one
another under strict/default/loose pymatgen profiles, but are MC rejected
because of high energy. No Hessian or new stable-phase claim is made.

The combined second attempts finish their cell blocks and complete9(seed7)
and8(seed101) atomic Gaussian relaxations below0.001eV/A before the request
cap interrupts further work. They are not a sequence of failed inner quenches.
Their last errors explicitly name `BudgetExhausted`; no final true quench is
performed after the cap. Do not describe these as SCF failure, physical failure
or proof that the kernel cannot escape. Nor can a terminal quench be appended
outside the budget and advertised as a successful same-budget run.

## Consequences for the main line

- Full stress convergence at every partial block is not necessary. This now
  has direct complex-material end-to-end evidence under the chosen MLIP.
- Splitting cell moves and atomic response makes the cell-only proposal
  feasible here, but these two samples do not demonstrate lower-energy search
  or a fair global superiority claim against joint SSW.
- Next isolate the atomic climbing cost/stopping lifecycle and native cell/atom
  coordinate constraints. Preserve checkpointed Gaussian boundaries so a
  censored trajectory can be analyzed without confusing a trial with an
  accepted point. Any continued run must record its additional cost.
- Fresh cell direction each cycle remains an explicitly unverified native
  substitution. The paper's description alone does not unambiguously settle
  the exact direction-state lifetime; avoid promoting a guessed continuation
  rule as native parity.

Registered Fe7C3 EFS now60940 =56942+3998. CPU Cu/EMT checks remain separate.

## Following observability change

After this frozen experiment, `surface.QuenchResult.optimizer_telemetry` now
passes through the original Safe-total `RelaxTelemetry` (defaultNone for
backends without it). Atomic climbing records that telemetry and its exact
termination reason. This fixes information loss between the existing optimizer
and kernel diagnostics; it changes neither stopping rules nor trajectories.
The frozen job outputs above predate this field. The last completed Gaussian
checkpoint remains distinct from an interrupted pending optimizer state.

Final integration check: 51 targeted tests passed in isolated mace_env,
including the joint release policy, SciPy stop classification, ledger collector,
real Cu/EMT quench telemetry, and fixed-cell extraction geometry/cost parity.
The native force-mask probe separately passed both flag branches. These checks
are implementation evidence; the scientific scope remains the finite material
experiments above.
