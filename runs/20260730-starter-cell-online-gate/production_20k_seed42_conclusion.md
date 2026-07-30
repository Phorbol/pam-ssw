# 20k production-kernel starter-cell gate: seed 42

## Question

Does replacing uniform sampling over all retained minima with a full-support,
MACE-FPS cell-first distribution improve a frozen production LS-SSW search
kernel at the same force-evaluation budget?

This gate changes only the outer starter distribution. Direction generation,
local softening, serial bias uphill propagation, proposal relaxation, true
quench, structure matching, and archive insertion remain frozen.

Execution commit:
`15f6d44caaed298c2474714f991e71807ca7e57d`.

Each campaign uses:

- master seed 42;
- total budget 20,000 force evaluations;
- per-action cap 1,000 force evaluations;
- batch size and worker count 1;
- full archive support and no archive deletion;
- 16 cells, derived before execution from a 50-trial resolution reference and
  three desired observations per cell.

## Fixed-budget results

| System | Policy | Actions | Archive | Duplicate rate | Best energy (eV) | Energy drop (eV) | Gain AUC (eV) | Total FE | Wall (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C60 | uniform | 43 | 42 | 0.0455 | -488.786011 | 14.116394 | 13.024437 | 19,062 | 351.20 |
| C60 | fps_cell_uniform | 45 | 44 | 0.0435 | -488.369476 | 13.699707 | 12.362872 | 19,683 | 349.27 |
| PdO | uniform | 72 | 65 | 0.1096 | -573.070923 | 4.674072 | 3.601366 | 19,232 | 461.28 |
| PdO | fps_cell_uniform | 70 | 62 | 0.1268 | -574.357056 | 5.959961 | 4.117418 | 19,112 | 458.35 |

`Gain AUC` is the integral of best-so-far energy improvement over the complete
20,000-FE axis, divided by 20,000. The last best value is carried through any
unused tail of the fixed budget.

The paired effect changes sign by system:

- C60 cell-first minus uniform:
  - final improvement: -0.416687 eV;
  - gain AUC: -0.661565 eV.
- PdO cell-first minus uniform:
  - final improvement: +1.285889 eV;
  - gain AUC: +0.516052 eV.

Cell-first therefore changes the search trajectory materially, but this single
seed does not support a system-general positive claim. It also does not
consistently reduce duplicates: C60 is effectively tied, while PdO is worse.

## Mechanism and accounting audit

All four campaigns satisfy:

- benchmark eligibility is true;
- zero failed attempts;
- exact per-action costs;
- purpose counts close exactly to total evaluations;
- used plus unused evaluations close exactly to 20,000;
- `unattributed == 0`;
- every action probability equals its immutable policy snapshot.

Both cell campaigns additionally satisfy:

- sidecar partitions exactly match event-log snapshots;
- every retained starter has non-zero probability;
- every starter is in exactly one cell at every decision;
- no archive entry is deleted.

Final pre-action partition sizes were:

- C60: `[3, 3, 2, 4, 2, 4, 4, 6, 1, 3, 1, 2, 2, 1, 1, 4]`;
- PdO: `[5, 2, 7, 1, 3, 3, 6, 2, 5, 4, 2, 10, 7, 1, 2, 1]`.

Descriptor overhead remains negligible relative to the action kernel:

- C60: 43 new-entry descriptor calls, 0.782 s (0.22% of campaign wall time);
- PdO: 61 new-entry descriptor calls, 1.305 s (0.28% of campaign wall time).

The force-evaluation bottleneck remains physical search work. Across the four
campaigns, biased proposal relaxation consumes roughly 59--71% of total FE,
landing true quench roughly 15--28%, and direction-oracle work roughly 10--12%.
Repeated starter true-quench is below 1.3% and is not the present critical path.

## Decision

The cell-first implementation passes the accounting, full-support, and online
integration gates. It does not pass a production-promotion gate.

The effect is large enough, and changes sign strongly enough, that the next
scientifically useful step is to finish the prespecified paired variance check
with seeds 43 and 44 at the same 20,000-FE budget. No UCB, Thompson sampling,
PCA threshold, hard top-k pruning, or new reward is added before that result.

