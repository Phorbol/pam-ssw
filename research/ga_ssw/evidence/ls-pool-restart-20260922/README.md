# LS with explicit pool restarts — 2026-09-22

User-approved design: continuous walking keeps LS; choosing another qualified
observation reinitializes the frozen potential and response history from the
existing settings. Completed outer-step response is recorded before the jump.
Local native LS steps restart at0; outer count, MC and paid costs continue.
Preparing the new LS and direction state precedes committing the current state.
Failure is terminal with restarted=false and the previous current/LS retained.
No new reward, physical parameter, public framework or checkpoint format.
The existing `run_ssw(ls=..., starter_selector=..., selector_rng=...)` is used;
LS wrapper signatures and all pool-checkpoint restrictions remain unchanged.

## Implementation and regression

Core commit ad1269b, reviewed isolated implementation3f658ee. Root corrected a
test's fixed index0 to actual snapshot.current_index and checks main RNG equality.
This changed no production behavior. Existing all-mobile fixed-cell boundaries
remain; no constrained/VC/RC integration or pool persistence is claimed.

CPU1448359: initial helper tests reproduce missing initializer (2 fail), existing
50 tests pass. Initial tests retained as red-tests.py; no numerical test was run
on the login node. CPU1448391:55 pass,1 fails because the test falsely equated
current observation with index0 after MC. CPU1448392:56 pass after that test
correction. Command and environment in cpu-tests.sbatch. Coverage includes native
LS local steps[1,1] across a jump, transparent and actual-current selections,
failed preparation, fresh paper LS response controller, legacy selector/RNG
checks, nonpool LS, periodic LS, and ordinary/recovered-direction checkpoints.
CPU1448441:66 pass after adding real Cu2/EMT paper-LS restart history and
reconciling failed LS restart requests in the existing PAM adapter. The adapter
final report retains requested chosen_index but derives actual_index from the
committed core event. These checks establish implementation contracts, not search efficacy.

## Real-system qualification protocol

GPU1448395:1V100, sjtu-caoxiaoming,4V100/rush-1o2gpu,30min cap. Frozen package in
prepared-v1/source, copied inputs/config sources, complete effective plan and
runner retained under prepared-v1/. Core production source is ad1269b; later
runner/test/doc edits are separately recoverable in this folder/final commit.

Two cases times two selector arms: saved C60_17093/MH-1 omol and anatase/OMAT-small
omat_pbe; unchanged archived LS/SSW parameters and C60 native MC. Transparent
selector returns None; deterministic diagnostic selects another observation.
Four outer attempts,2990 search + at most10 independent E/F per arm; total<=12000.
This is an interface qualification, not a proposed new pool scoring policy.

Raw result pickle precedes scalar summaries. Requested selections and successful
committed restarts are separate. analyze.py verifies next LS update step1 where
available; no subsequent update means continuation remains unverified. It also
reports fresh force, composition, cell and PBC checks and preserves failures or
budget truncations. A successful job alone does not pass these checks. Source
manifest, plan, logs and result files are authoritative; current analysis is
interim unless its complete flag is true.

GPU1448395 preserved both C60 arms (4766 total E/F). The collector then failed
before any TiO2 PES call because config_source named an old plan without the
config field. GPU1448487 completes ONLY the unrun TiO2 arms using the already
archived execution.json; no C60 rerun. prepared-v2 explicitly records its parent
run, previous cost and6000 remaining cap (combined maximum10766). A preparation
check now rejects missing config/LS before GPU submission. Original script,
error and results remain in prepared-v1. Final combined analysis has all4 arms, total9146 E/F, within12000. Do not infer search benefits,
long-budget stability or production qualification from the short protocol.


## Final real-system result

| Case / selector | Search E/F | Independent E/F | Status | Committed LS restarts | Next local update1 |
|---|---:|---:|---|---:|---:|
| C60 / transparent |2990|4|request-cap truncation|0|not applicable|
| C60 / other index |1768|4|completed|4|3/3 observable continuations|
| anatase / transparent |2431|4|completed|0|not applicable|
| anatase / other index |1941|4|completed|4|3/3 observable continuations|

Total9130 search+16 independent=9146. Every independent check passes finite
energy/force, fmax<=0.03eV/Angstrom, exact composition/cell/PBC preservation.
Only the first4 observations per arm were independently force-checked, per
frozen protocol. Last-step restarts have no next step and are not counted as
continued-use evidence. Observation indices are not certified distinct basins.
The transparent C60 arm did not complete4 outer attempts within its cap;
therefore these totals cannot support an efficiency ranking. The other-index
selector is a diagnostic, not a new PAM scoring rule or a default recommendation.

Combined readout command (JSON metadata only):
`python analyze.py prepared-v2 prepared-v1` from this directory.
`prepared-v2/analysis.json` names both source summaries. `export_results.py`,
run by `export.sbatch` on CPU, exports every saved observation and current
structure without PES calls and checks the complete search ledger and topology.
Pickles and full frozen source remain local; Git carries results, trajectories,
configuration sources, source manifests and executable runners.

Decision: retain the approved optional LS+pool interface and corrected actual
restart reporting. Keep default MC and all physical/search parameters unchanged.
No general LS/pool search benefit or production-scale validation is claimed.
Next public-design boundary is the separately proposed pool checkpoint state
contract; its current guard remains until user approval.
