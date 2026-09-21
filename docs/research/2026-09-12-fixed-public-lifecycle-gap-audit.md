# Fixed-cell public lifecycle gap audit

Bounded read-only review of current `pamssw/standalone` code, tests and
`docs/research/2026-09-12-fixed-kernel-contract.md`. No PES run or production
change was made.

## Actionable findings

### 1. GA failure records do not carry per-record request deltas

In `paper_ga.py:407–416`, `walk()` records failed/nonpositive-height
`SSWStep`s with `fail(..., cost=0)`. The enclosing `SSWResult` and its
`GAStage` retain aggregate `surface.requests - before` at
`paper_ga.py:426–427`, but each corresponding `GAFailure.cost` is zero.

Trigger: any GA quick, generation-short, offspring, or fine walk that returns
a failed climb, failed landing, LS failure, or nonpositive height while the
walk made E/F requests. This does **not** lose total search or stage cost:
`GAStage.evaluation_requests` retains the aggregate delta, and `ingest()` may
already record the failed quench cost as an observation failure. It does mean
`GAFailure.evaluation_requests` is a zero-cost overlapping diagnostic entry,
so consumers must not sum all `failures` as a total-cost ledger. Minimal fix,
if per-record attribution is required, is to retain each record's request
delta or label this entry explicitly as an aggregate/overlapping diagnostic;
aggregate stage cost should remain unchanged.

### 2. Resolved: GA height-policy validation before initial quenches

`run_ga_ssw` validates `gaussian_policy` before the initial phase at
`paper_ga.py:186–202`, but has no corresponding validation for
`height_policy`. Validation is delegated to `run_ssw` inside `walk()` at
`paper_reference.py:215–230`. GA first performs supplied initial true
quenches through `relax()` and the initial loop, so an invalid height-policy
type or an Eckart-incompatible height policy spends calculator requests before
being rejected inside a later walk.

Trigger: `run_ga_ssw(..., height_policy=bad_object)` or a valid height policy
with `ssw_config.cluster_frame == 'eckart'`. The standalone SSW entry already
has the required pre-PES checks; GA should apply the same type/frame checks to
both `ssw_config` and `offspring_ssw_config` before initial `relax()` calls.
This is lifecycle validation and cost containment, not a search-policy change.

Resolved in the current worktree: `_validate_height_policy_options` is shared
by SSW and GA, including offspring config, before initial PES requests.
Actual native-height nonempty-history paths have regression coverage. The
paragraphs above describe the pre-fix trigger, not the current behavior.

### 3. Native LS public wrapper omits the existing reconnect argument

`run_ls_ssw` exposes and forwards `reconnect_distance` at
`paper_reference.py:578–588`, and the README documents that entry at
`README.md:337–344`. `run_native_ls_ssw` accepts only `height_policy`,
`gaussian_policy`, and `height_update_budget` at
`ls_native_reference.py:66–79`; passing `reconnect_distance=` raises
`TypeError`, although its underlying `run_ssw` call supports the same
completed-climb geometry boundary.

Trigger: a caller selects `NativeLSSettings` and requests the documented
optional reconnection boundary. Minimal interface fix is adding
`reconnect_distance=None` to the native wrapper and forwarding it. It should
not alter the default (`None`) or native LS update rules.

## Checked boundaries

`ASESurface` accepts an arbitrary supplied ASE calculator and owns it for
serial use (`surface.py:24–51`); no separate calculator replacement defect was
established. The contract's missing full checkpoint and native CBD/Broyden
parity are intentionally excluded. Default SSW/LS trajectories,
optimizer choices, and height/Gaussian formulas are not implicated here.
