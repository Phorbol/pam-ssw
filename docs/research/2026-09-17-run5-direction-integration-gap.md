# Run_type 5 fixed-cell direction integration gap

## Current update: experimental shared-driver integration completed

The gap table below is retained as the earlier audit, not the current status.
`native_direction_control.py` now closes the restricted c1/c4/c6/c9 generator,
including separate bond groups, connected-pair fallback and final-zero release.
`LocalDirectionState` owns the saved Gaussian center and selected local inputs;
`recovered_direction.py` owns the outer selection lifecycle. `run_ssw` accepts
explicit `RecoveredDirectionSettings` via `recovered_direction`, keeping the
existing default intact. No binary is called at runtime.

The new route shares Gaussian construction, biased and true quenches, cost
accounting and conventional MC. Recovered CBD now accepts the existing fixed
center rigid projector after angular capping (native call at 0x6e780e), while
keeping the physical endpoint forces separate from direction cleanup. Actual
post-PreRot anchor and stage-dependent curvature shift feed height policies.

Selection precedes MC: a qualified landing updates pair/group even when MC
rejects its coordinates. This order is supported by the Allopt/make_decision
static audit, not a complete dynamic oracle of rejected native trajectories.
Startup selection uses the input-to-initial-quench displacement as an explicit
Python contract. Q, compression, periodic native routing and checkpoint recovery
are excluded from this experimental entry. It is not full LASP parity.

On a post-landing selection failure, the driver retains the qualified landing,
best-so-far and paid cost, records `direction_selection_failed`, skips MC and
stops. It does not fabricate another pair or lose the completed landing.
Direction-generation failure inside an escape likewise remains a recorded
failure. Startup domain errors are explicit exceptions.

Verification: 55 targeted tests cover shared-driver defaults, checkpoint
regression, native LS defaults, recovered CBD and controller, including a
regression for preserving a landing when later selection fails. The bounded
EMT integration set (Cu13/Cu55/Au13, seed19, paper/recovered) used 1573 search
and 17 independent qualification requests. All returned structures satisfy
fmax <=0.03 eV/A with fixed cell. These 17 include six initial structures and
are not deduplicated or physically validated. One paper Cu13 attempt ended
with `nonpositive_height`; that failure is retained. No efficiency ranking
follows from this small check or its different rotation termination rules.
The raw summary's `gaussians` field counts any event containing a weight,
including this rejected nonpositive weight; use the event records to count
actual deposited Gaussians. `derived-deposition-audit.json` now records the
correction: 47 deposits, including seven rather than eight for paper Cu13.
Another historical metadata limitation is that the EMT snapshot's recovered
`initial_direction` stores the last proposal; per-Gaussian direction records
remain available. The later MACE snapshot fixes that field and records each
generator proposal, route and effective (initial or update) coefficients.

Evidence directory: `../../research/ga_ssw/evidence/recovered-direction-shared-smoke-20260917/`.
MACE-OMAT-0-small follow-up is complete in
`../../research/ga_ssw/evidence/recovered-direction-mace-e2e-v2-20260917/`.
Job1369266 used one V100, finished in45seconds with exit0, passed30 frozen
preflight tests, and completed6/6 arms. Cost:1534 search+18 independent E/F
requests. All18 frames satisfy fmax<=0.03 eV/A (maximum0.02977091) and fixed cell.
These short two-attempt trajectories establish backend integration; energy
changes relative to initial minima are at most~0.0012 eV and do not establish
new basin discovery. Distinct rotation stopping rules prohibit an efficiency
ranking from raw call counts. Larger-budget water15/Cu55 LS checks are the next
step; the Gaussian limit in this interface probe is not a search recommendation.

Scope: existing code and evidence only, for a nonperiodic, unconstrained,
fixed-cell Run_type 5 path with Q mode disabled. This is an integration audit;
it does not turn Q-off into the ordinary paper default and does not add a
second walker.

## What is already available

| Native stage/behavior | Existing evidence or helper | Boundary |
|---|---|---|
| Initial coefficient selection (`get_random_mode0`) | `docs/research/2026-09-17-mode-default-coefficients.md`; native disassembly and coefficient probe | Q-off disables the `c5` Q branch, but `c4`/`c6` remain conditional; this is not the final direction |
| Initial generator entry (`gen_randommode`, `0x5d5c50`) | `docs/research/native-global-random-workspace-audit.md`, `native-mode-mixing.md`, native generator assembly | Several coefficient/mixing/normalization branches are documented, but no complete Python final-output function is present |
| Later update (`update_mode0`, `0x5d55f0`) | `docs/research/2026-09-17-fixed-direction-state-decision.md`, `native-mode-mixing.md`; `direction-update.json`, 4/4 | `n0` is rebuilt from `current - selected_record`; `c9=1.2*(c4+c5+c6)` was verified, but the probe stops at the generator entry |
| Local axis/group geometry | `pamssw/standalone/native_local_group.py`; `2026-09-17-native-axis-group-selection.md`; 10 geometry tests and 9 combined helper cases | Pair/group caller lifecycle and final consumption are not part of this helper |
| Rotation arithmetic | `native_rotation.py`, `native_rotation_control.py`, `recovered_cbd.py` | These are caller-supplied-anchor primitives/stage composition; they do not implement native `gen_randommode` or `update_mode0` |
| Paper reference path | `paper_reference.py` and `docs/research/2026-09-17-direction-lifecycle.md` | Intentionally samples an outer anchor and independently solves each Gaussian; it is a separate rule |

## Run_type 5 Q-off dependency boundary

With `Lmode_Q` false, the documented `get_random_mode0` path skips the Q-only
`c5` activation branch (`0x5c0350–0x5c0398`) but still reaches the conditional
local `c4` or `c6` writes. The existing Q-mode contract concerns the separate
PTSD/`BrickPTSD` path reached when the generator takes its Q branch; it should
not be imported into this Q-off scope. Conversely, Q-off does not imply the
paper random-anchor rule, and no evidence supports silently substituting that
rule for the native generator.

`update_mode0` remains dependent on a selected trajectory record even with Q
off: `0x5d5bf4–0x5d5c06` forms and normalizes the current-minus-record vector,
then `0x5d5c15` re-enters `gen_randommode`. The selected-record index is now resolved as the current Gaussian record (see
2026-09-17-native-displacement-boundary.md); the complete final generator
output and driver state integration remain required for native parity.

## Minimum remaining work before a shared ASE driver

The two-rule shared driver boundary is already approved. The smallest native
addition needed before wiring it is:

1. Implement one instance-owned Run_type 5 direction state that carries the
   outer reference snapshot and selected-record index, while preserving the
   existing outer-step checkpoint boundary.
2. Close the Q-off `gen_randommode` final vector for the reachable `c4/c6`
   branches: component accumulation, final normalization, and the all-zero
   result/Allopt transition. Existing probes only close prefixes and helper
   arithmetic.
3. Feed the actual pair/group result from the fixed-cell caller into the
   local-group geometry helper, including the conditional `get_atompair_`
   overwrite of the pair scalar. Do not treat the standalone helper's pair as
   final until this caller mapping is wired.
4. Add one bounded state-transition check covering
   `NewStart -> copy_str/cart_copy -> update_mode0 -> gen_randommode`, including
   zero displacement and the status/re-entry branch. This is an interface
   check, not a PES experiment.

No new checkpoint fields for stage-internal rotation counters or Broyden arrays
are required if the driver retains the approved outer-step recovery boundary.
The existing `recovered_cbd.py` can remain an independent experimental helper;
it should not be presented as the native Run_type 5 generator.

## Subsequent closure, 2026-09-17

`native_pair_selection.py` now independently reproduces Run5 pair refresh.
Canonical 24-case and 24-edge comparisons match pair, draw count, acceptance
and rejection counters. `native_random.py` independently reproduces VMB2/RAN3
(46 archived outputs, maximum error 1.39e-17). These are Python algorithms,
not executable calls. The selected-reference timing is closed separately in
`2026-09-17-native-displacement-boundary.md`.

The full generator, Q branch, compression and common-driver integration remain
open. Reference geometry has a native coordinate-chart limitation documented
in `2026-09-17-native-axis-group-selection.md`; existing ASE defaults are not
changed to imitate that limitation.
