# Independent three-stage Python/ASE GA-SSW controller

`pamssw/standalone/paper_ga.py` connects the independent `paper_reference.run_ssw`, true-surface ASE quench, TYPE3 proposal implementation, projection archive, partition and region ranking. It calls no Java, LASP, subprocess, input-file generator or alternative potential.

## Executed method and scope

1. Every explicitly supplied initial Atoms is quenched on the true surface. Each eligible result seeds an explicitly budgeted quick walk.
2. For each requested GA generation, partition the current archive, flatten selected parents in region order, generate real TYPE3 candidates, quench every returned offspring on the true surface, update the archive, then launch the explicitly budgeted short walks from region minima.
3. Rank current regions and start at most the explicitly requested number of fine walks with the explicitly requested fine step count.

The supplied SSWConfig is unchanged across these stages. There are no hidden temperature or step multipliers, no forced extra fine iteration, no division by nine and no inherited terminal appended to a sorted minimum list. This is a paper-level three-stage reference architecture, **not execution parity with the uploaded Java schedule**. SSW numerical differences are documented in `paper_reference.py`.

The available descriptors, KMeans and TYPE3 operations still follow separately recovered release semantics: this controller does not imply that the legacy descriptor is exactly the paper's permutation-invariant DCCD, or that KMeans equals the paper's grid-based partition. The configured energy window is applied globally after archive merging. Full observation history is retained separately, including observations outside that window.

Supported structures: nonperiodic, unconstrained, complete explicit monomer partitions with fixed internal reconstruction flags (all changeType0). Controller inputs must already have contiguous monomer-group atom order matching the frozen references; this avoids silently reordering atoms behind the legacy descriptor. General TYPE0, periodic crystals, overlapping rigid-chain groups and changeType1 internal atom GA are outside scope. This is an executable controller for this supported subset, not universal validated GA-SSW.

## Required configuration and API

`run_ga_ssw(initial, surface, *, groups, references, descriptor_bonds, descriptor_weights, neighbor_range, proposal_bond_limits, config, ssw_config, rng, ls=None, structure_validator=None)`

`PaperGAConfig` requires all these fields:

- Stage budgets: `quick_steps`, `generations`, `generation_steps`, `fine_steps`, `ga_candidates`, `regions`, `fine_regions`.
- True-quench contract: `quench_fmax` (eV/Å, max per-atom norm), `quench_steps`.
- Operational proposal/partition budgets: `proposal_max_batches`, `proposal_max_cut_attempts`, `proposal_max_pair_attempts`, `partition_max_draws`.
- Archive policy: `projection_tolerance`, `energy_window` (eV).

At least three frozen reference descriptors are currently required because the partition implementation uses the first three projection coordinates. The NNA bond-length table is separate from the proposal BLLimit table; neither receives invented physical defaults. `initial` geometries are supplied by the caller; the controller never constructs an unsolicited initial atomic model.

Initial/offspring explicit quenching and the walker's own initial quenching are both executed and counted. This repeats a small stationarity check at walk entry, rather than pretending it was free. The walker may use a different configured tolerance, but archive admission additionally requires `max_force <= PaperGAConfig.quench_fmax` for every landing, so a loose walk certificate cannot silently relax the archive requirement.

## Returned evidence and failure semantics

`PaperGAResult.archive` contains retained row dictionaries with stable observation ID, independent Atoms, energy, projections and observation ID. `best` returns the lowest-energy retained structure. Both are **force-stationary results**, not positive-Hessian or chemically validated minima.

`observations` retains every returned walk minimum, any additional failed/modified landing in SSW records, failed initial-quench exception results, and each explicit initial/offspring quench. Every available geometry is projected. Admission additionally requires true surface, force convergence and common archive force tolerance; the optional caller `structure_validator` may reject it. `physical_validation_performed` only records whether this optional validator was actually invoked, not whether every scientific stability criterion was proved.

Each observation stores phase, generation, seed ID, full QuenchResult, projection, eligibility, per-monomer parent observation IDs, GA operator and details. Failed candidates are not replaced with an old structure and are not called converged. `walks` retains complete underlying SSWResults; `stages` preserves requested budgets, event records, and per-stage evaluation requests. `failures` records the cause and available local request count. Failure counts can describe multiple causes in one stage; do not sum them as a cost ledger.

The authoritative total `evaluation_requests` is the final minus initial surface request counter, including failed calls. Sum of stage request counts tracks the same serial execution. SSW internal SCF iterations and wall time are not inferred from this counter. Proposal failures from degenerate parents or exhausted sampling are explicit; further GA generations stop and independent fine exploration may still use the existing archive. There is no silent fallback parent population or calculator.

## Verification boundaries

`tests/standalone/test_paper_ga.py` currently has seven tests: all returned minima are considered, failed geometry is retained/projected but excluded, every offspring receives a true quench, lineage survives, backend cost is retained, SSW record and initial-quench failures retain structures, a looser walk certificate cannot bypass the archive threshold, and incompatible monomer order is rejected before calculation. Tests use mocks specifically to isolate scheduling/accounting, not as evidence of scientific efficacy.

One test uses the actual independent walker and actual ASESurface at zero walk steps on a harmonic numerical fixture anchored to an existing 45-atom water archive. It verifies three actual evaluations (explicit initial quench plus quick/fine entry quenches) with no binaries. The harmonic function is not asserted to be a physical water model, and this test does not establish a successful real-PES GA search.

Real-potential multi-generation behavior, search efficiency, force-energy consistency of each scientific backend, chemical validation and comparison to the stable baseline remain separate required evidence.
