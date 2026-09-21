# Walker failure and paid-state review

The confirmed issue was missing failure-state evidence in the RC and joint-VC drivers, not a demonstrated incorrect E/F sign or loss of their total request counter. A second-stage dimer/evaluator exception could leave a result containing only the selected initial minimum, even after a paid first Gaussian stage had reached a different accepted biased geometry. Gaussian centers were not retained explicitly in these records, so the attempted scalar objective was incomplete for replay.

## Minimal repair

`rc_reference.py` (shared by single-chain, forest and constrained fixed-cell drivers), `rc_vc_reference.py`, and `vc_reference.py` now record:

- the proposal's frozen chart-reference geometry;
- each added Gaussian's copied center, direction, height and width, before calling its optimizer;
- `last_work` at the latest available accepted optimizer state, updated before any following physical-energy/direction request can fail.

The joint VC record also retains its frozen LS term object from before the response update, separately from the next proposal's LS state. No charts are rebased, no Gaussian scalars are changed, and no new E/F/EFS calls are made. These fields describe attempted biased optimization states; they do not certify those structures as true minima. The selected current/best and the independent true-landing certification rules remain unchanged.

Three regressions (isolated forest, periodic RC-VC and joint VC) inject failure during the second direction solve after charging an extra physical callback. All require current-minimum preservation, exact total request reconciliation, first-stage frozen Gaussian retention and a displaced `last_work`. They failed before the repair and pass afterward. **31 related tests passed**, including real short Cu EMT VC wiring at zero/nonzero pressure and VC-LS/block/RC lifecycle tests. Existing ASE/NumPy deprecation warnings remain. No long PES experiment was launched.

## Other failure paths inspected

- Atomic `run_ssw` stores `height_prepared` before optimizer calls, including the full replacement history for optional height policies, and preserves last working geometry in its step result. Initial quench uses an exception contract (`InitialQuenchError` for an uncertified result), unlike the VC result-status contract; callers must account via the attached result/surface counter when initial setup fails.
- `atomic_climb` used by block SSW now has its own checkpoint with completed Gaussian history, boundary geometry and a pending stage. Its pending height/biased-quench data is retained on failure; no new change was needed here.
- `block_ssw` retains outer `last_work`, per-cycle costs and the atomic subresult; shared `cell_quench` returns uncertified failures with the optimizer's last accepted coordinate and paid request count. If the fresh certificate itself exhausts the oracle, its evaluation is absent rather than falsely certified.
- RC and joint VC charge requests by the underlying physical surface counter at every completed outer event. A domain-rejected chart trial is a numerical callback attempt, not an extra physical E/F request. The optimizer's callback count and physical counter therefore need not be equal.
- Run-level status conventions still differ: joint VC and block may return `completed` despite individual failed proposals, whereas RC returns `completed_with_failures`. This is not success evidence; consumers must inspect every proposal and certificate. Unifying statuses would be an API change and was not folded into this evidence repair.

Remaining boundary: a calculator that raises midway through one physical evaluation may have backend work not expressible by this wrapper's completed E/F-request counter. No wrapper can infer unreported SCF iterations or partial backend time. Existing request accounting remains an explicit interface measure, not an estimate of electronic work.
