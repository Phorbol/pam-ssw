# Explicit fixed-cell Safe-total memory: design

Expose `lbfgs_memory: int | None = None` on Relaxer, fixed-cell quench and SSWConfig. None retains the existing10-pair path exactly. Positive integer values select a bounded history, with storage O(memory × Cartesian DOF). This is an explicit numerical option, not adaptive memory or a recommendation that400 is optimal.

Only `safe-lbfgs-total` supports this option; other backends reject it before physical evaluation. Explicit memory conflicts with legacy internal history0/1/10 or adaptive-no-history ablations; those existing controls retain their old behavior when no public memory is supplied. Forward the config through initial true quench, LS prequench, every Gaussian biased quench and final true quench. Native-derived LS already delegates to the same walker/config, so no separate numerical policy is introduced. Periodic/generalized/RC integration belongs to the other agent.

Validation: invalid/backend/conflict checks must charge0 evaluations; legacy tests remain valid. Reproduce the complete saved C60 stage8 memory10 and400 raw E/F trajectories with explicit API,0 new PES, unchanged global constant10. Confirm stored endpoints/requests and compare defaultNone to explicit10. Preserve frozen experiment artifacts.

## Implemented and verified

`Relaxer(..., optimizer='safe-lbfgs-total', lbfgs_memory=400)` and `quench(..., optimizer='safe-lbfgs-total', lbfgs_memory=400)` opt in directly. `SSWConfig(..., quench_optimizer='safe-lbfgs-total', lbfgs_memory=400)` propagates to initial, LS pre-, biased and final true quenches for ordinary, paper-LS and native-derived-LS drivers. DefaultNone remains unchanged10.

`verify_explicit_lbfgs_memory.py` used0 new PES calls: memoryNone and10 replay all423 original C60 stage8 coordinates exactly,400 replays all264 recorded coordinates exactly and reproduces248 accepted steps; global memory remains10 throughout. Evidence `research/ga_ssw/evidence/explicit-memory-api-replay/result.json`.

126 relevant tests pass, including legacy Relaxer, ordinary/paper/native-LS lifecycle and real Cu EMT memory propagation for all three driver variants. Invalid/backend/conflict tests verify failure before physical evaluation; defaultNone and explicit10 have identical real EMT trajectory/request count. The separately attempted legacy history-depth research suite has5 unavailable-fixture failures referencing `/tmp/SSW-worktrees/fixed-proposal-replay/runs/20260727-023234-fixed-proposal-replay-gpu/output/summary.json`; these do not arise from an optimizer assertion and were not worked around by fabricating input. Its remaining tests passed. Original experiment artifacts remain unchanged.

## Generalized and variable-cell integration

The same optional positive history size is now exposed by `safe_lbfgs`,
`cell_quench`, VCSSWConfig, RCSSWConfig, RCForestSSWConfig and RCVCSSWConfig.
Block-SSW takes the single setting from `config.atomic.lbfgs_memory` for its
partial atomic relaxation, atomic climbing and joint true quenches. It has no
second competing history field. All defaults retain10.

Regression snapshot after integration: `tests/standalone` 319 passed,2 skipped;
`tests/unit/test_relax.py` 101 passed. Logs are
`/tmp/pam-ssw-ga-standalone-regression-20260910.log` and
`/tmp/pam-ssw-relax-regression-20260910.log`. The real Cu EMT VC test exercises
initial, biased and final optimization; RC/RCVC structure-fixture tests use
flat oracles and mocked final quench certificates to isolate parameter wiring.
They are not real-potential RC/RCVC efficacy or stability evidence. No dedicated
VC-LS memory performance campaign has been run.

## Consumer audit: GA direct quench and fixed substrates

A later consumer audit found two remaining gaps: `paper_ga` selected Safe-total
but omitted the supplied SSW memory on direct initial/offspring quenches; the
fixed-substrate configuration had no public history option. Both are now fixed.
`ConstrainedSSWConfig(lbfgs_memory=...)` and `constrained_quench` validate the
option before oracle calls and forward it through initial, biased and final
quenches. TYPE4 uses this same constrained walker. Defaults remain unchanged.

The related 28-test subset passes. GA spies check all three initial and two
offspring direct quenches; the constrained lifecycle checks exactly three
initial/biased/final calls. Mock/Harmonic fixtures establish parameter wiring,
not real-surface search effectiveness. These fixes do not change the C60
fixed-cell source being used in the separately frozen recovery experiment.
