# GA full recovered-direction integration (2026-09-23)

Purpose: implementation and numerical qualification, not a search-efficiency claim.
User approved independent state ownership per GA walk: preserve pair/group/displacement
memory within a walk, initialize fresh state at every new walk. No parent-inherited
state and no interrupted-inner-walk recovery are introduced.

## Implementation

`run_ga_ssw(recovered_direction=...)` forwards the existing settings to quick,
offspring_ssw, generation_short and fine. None retains the old call and checkpoint
contract. Enabled settings enter the existing contract; incompatible restore or
unsupported combinations fail before PES. GA checkpoint version remains 1.
Only unconstrained nonperiodic clusters with direction_only are supported by this
full direction mode; mutually exclusive recovered_rotation and pre_rotation_hvp
are rejected. This does not narrow the default SSW API.

Core commits: b609726, 092a997; fixture correction: 0d27801.
Independent read-only review found no production defect. Actual execution found
and corrected two test-fixture errors (duplicate config keyword and omitted
zero-step fine initialization). CPU1453519's 27 pass/2 fail log is retained.

## Fixed protocol and execution

[protocol.json](protocol.json) reuses the three archived Cu13/EMT minima and
Cu13 GA configuration; both arms explicitly use full_fingerprint and one offspring
SSW step to exercise the fourth phase. The direction settings come from the
existing Cu13/Cu55/Au13 smoke protocol. They are bounded diagnostic settings,
not optimized or promoted defaults. Full protocol/input source paths are recorded.
Search cap: 8000 requests per full trajectory; total cap 24000 across baseline,
full direction and split-full; CPU wall cap 10 minutes. No GPU/model changes.

CPU1453528:

- `python -m pytest -q tests/standalone/test_ga_recovered_direction.py tests/standalone/test_ga_walker_options.py tests/standalone/test_ga_checkpoint.py tests/standalone/test_ga_cycles.py tests/standalone/test_recovered_direction_driver.py`: **29 passed**.
- [run.py](run.py): baseline and full direction complete every GA walk phase.
  Baseline 7296 requests; full direction 5498. Each produced 39 observed structures.
- Full direction created 18 fresh controllers for 18 independent one-step walks.
  The split run created 3 before the quick boundary and 15 after resume.
- Split costs: 903 + 4595 = 5498. Saved observations (IDs, phase, eligibility,
  energy, positions), stage states/costs, total cost and terminal status agree
  exactly with the continuous run on this EMT backend.
- Total **18292 search + 126 fresh E/F requests**. All 126 saved-observation
  checks pass the configured force threshold, composition, cell/PBC and fresh
  energy agreement. These include repeated endpoints and inherited observations;
  they are not 126 independent structures. No positive-Hessian qualification.

The initial reporting tail incorrectly assumed completed results had checkpoints;
CPU1453528 and offline CPU1453539 preserve these readout failures. All searches
and raw results had already been saved. No PES was rerun. Terminal RNG was not
exported in those results, so exact terminal-RNG equality is **not** claimed.
[analyze.py](analyze.py), CPU1453545, read the saved data with zero PES calls;
[summary.json](summary.json) reports `errors=[]` and exact observed-trajectory
resume equality. This is a correction of readout, not an algorithm retry.

## Artifacts and conclusion

`baseline/full/first/resumed.pkl` retain trusted-local raw results in this directory
outside Git; corresponding JSON summaries and extxyz observations are committed.
The runner and protocol are versioned; the protocol records the core code HEAD.
[verify.sbatch](verify.sbatch) and [analyze.sbatch](analyze.sbatch) give exact commands.

Retain the optional integration; no default change. GA now exposes the same complete
recovered-direction mechanism as standalone fixed-cell SSW under the approved
per-walk state ownership. Single-system short-run costs do not establish an
algorithm ranking, generalization, C60 cage discovery, or exact LASP parity.
