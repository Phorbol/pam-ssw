# Production-kernel starter-cell smoke conclusion

## Scope

This is a mechanism and integration smoke, not a selector ranking experiment.
It freezes the evidence-backed system-specific LS-SSW action kernels and changes
only the outer starter policy:

- `uniform`: uniform over every retained archive entry;
- `fps_cell_uniform`: deterministic MACE-feature FPS cells, uniform cell mass,
  then uniform sampling within the selected cell.

Both policies retain full archive support. No archive node is deleted. The run
uses one seed (`42`), a 6,000-force-evaluation campaign budget, a 1,000-force-
evaluation action cap, and four cells derived from a 12-trial resolution
reference. These settings are intentionally too small for performance claims.

Execution commit:
`20296d2816c58b721cc5bf08fb46b26214ce1bf9`.

## Results

| System | Policy | Actions | Archive | Energy drop (eV) | Total FE | Unused FE | Wall (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| C60 | uniform | 12 | 12 | 10.359924 | 5,239 | 761 | 94.76 |
| C60 | fps_cell_uniform | 11 | 11 | 11.795227 | 5,175 | 825 | 97.12 |
| PdO | uniform | 19 | 19 | 2.923340 | 5,089 | 911 | 126.58 |
| PdO | fps_cell_uniform | 20 | 19 | 2.656982 | 5,156 | 844 | 124.09 |

The opposite ordering on C60 and PdO is not evidence for or against the cell
policy. It is the expected ambiguity of one short stochastic seed.

## Accounting and mechanism checks

All four campaigns satisfy:

- benchmark eligibility is true;
- zero failed attempts;
- purpose counts sum exactly to total force evaluations;
- total force evaluations plus unused budget equal 6,000;
- every action has exact cost;
- `unattributed == 0`;
- every logged action probability equals the probability in its immutable
  policy snapshot.

For both cell campaigns:

- the sidecar partitions exactly match the event-log snapshots;
- every archive entry belongs to exactly one cell;
- all archive entries retain non-zero probability;
- the final C60 cell sizes are `[3, 1, 4, 2]`;
- the final PdO cell sizes are `[9, 1, 5, 3]`.

MACE representation overhead is small in this smoke:

- C60: 10 descriptor forward calls, 0.231 s;
- PdO: 18 descriptor forward calls, 0.396 s.

The repeated starter true-quench in the isolated one-action adapter is also not
the present force-evaluation bottleneck:

- C60 cell campaign: 11 of 5,126 action FE (0.21%);
- PdO cell campaign: 40 of 5,118 action FE (0.78%).

The dominant costs remain biased proposal relaxation and landing true quench.
This observation does not prove that the repeated starter quench should remain;
it only removes it from the current critical path.

## Decision

The production-kernel integration and the full-support probability mechanism
pass the smoke gate. The cell policy is not promoted to production and no
Bayesian/UCB layer is added yet.

The next bounded scientific gate is a 20,000-force-evaluation, one-seed paired
comparison on C60 and PdO. It keeps the same action kernels and uses 16 cells
derived from a 50-trial resolution reference. Its purpose is to determine
whether the representation-level abstraction produces a measurable change
before spending the three-seed budget.

