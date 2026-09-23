# Periodic-return mechanism diagnostic protocol

## Question and scope

Use only the four completed seed-41 runs archived under `../periodic-rotation-priority-20260923/` to test whether a completed SSW attempt can move its final biased-quench geometry away from its accepted input and then return toward the same geometry during the true-surface landing quench. Separately audit whether the stop label `lower_true_energy` agrees with the last successful event's physical true-energy comparison against the pre-attempt accepted-state energy.

This is a zero-PES reanalysis of one seed on two previously used periodic inputs. It is diagnostic evidence, not an independent search, basin census, phase assignment, or general performance comparison.

## Frozen population and selection

Population: all records in the four archived arms (`aloh3` and `brookite48`; `ritz` and `recovered`; seed 41). Preserve every record's `evaluation_requests` in the reconstructed cost ledger, including failed and terminal request-censored records. Reconstruct the current accepted state from `result.initial` and each record in order: replace current geometry and energy only when `record.accepted is true` and a converged landing is present.

Geometry sample: for each arm, select the first three records in original order whose outer status is `gaussian_limit` and which are not terminal request-censored. Selection is based only on record order and status, never on energy or geometry response. This yields at most 12 sampled records.

For each sampled record compare (1) reconstructed current input vs `record.last_atoms`, (2) current input vs `record.landing.atoms`, and (3) `record.last_atoms` vs landing atoms. For each pair use `StructureMatcher` with `ElementComparator`, `primitive_cell=False`, `scale=False`, `attempt_supercell=False`, at both frozen tolerances: tight `(ltol=0.05, stol=0.10, angle_tol=2°)` and broad `(0.20, 0.30, 5°)`. Maximum 72 fits. A match is approximate geometric equivalence only, not strict basin or phase identity.

Also report ordered-atom Cartesian displacement after ASE `find_mic` using the saved fixed cell/PBC: RMS `sqrt(mean_i(|d_i|^2))` and maximum `max_i(|d_i|)`. This is not permutation invariant and is reported separately from StructureMatcher.

## Stop-sign audit and energy fields

For every complete record, identify the last climb event containing finite `true_energy`. Compare that physical energy with the accepted-state `current_energy_before` reconstructed before the outer record. A `lower_true_energy` record is sign-consistent only if the last successful event has `true_energy < current_energy_before`; a `gaussian_limit` record is consistent only if it does not. Preserve the signed difference. Other statuses are retained but not forced into this binary check. Terminal request-cap records are excluded from complete-record sign counts; their requests and failure classification remain in the cost ledger.

For the 12 sampled records, report the last successful event's `true_energy` and `biased_energy` separately, along with reconstructed current energy, final `landing.energy`, status, and request costs. `biased_energy` is the biased objective; `true_energy` is physical PES energy at the final biased-relaxed work geometry; landing energy is after the separate true-surface quench. Do not rank tiny energy differences as scientifically meaningful.

## Execution and outputs

`analyze.py` reads the frozen plan and four existing `result.json` files plus each arm's summary/request ledger for provenance and cost accounting. It makes no calculator/PES calls and writes `analysis.json` and `report.md` incrementally after each arm so a five-minute timeout preserves completed work. `cpu.sbatch` requests one task for at most five minutes on CPU-MISC/rush-cpu under `sjtu-caoxiaoming`; submission is deliberately left to the parent.
