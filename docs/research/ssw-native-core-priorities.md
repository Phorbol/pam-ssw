# Native SSW-family core: remaining semantics and conservative height integration

2026-09-10. Bounded review of current standalone code and archived native/paper evidence. No public walker was modified in this task. A new independent height helper and an isolated original-instruction initializer oracle are delivered below. This is a prioritized reproduction audit, not an efficacy or full release-parity claim.

## Highest-priority conclusions

| Priority | Mechanism and evidence | Current state / action |
|---|---|---|
| P0: preserve objective consistency | `native-addgaussian-instruction-oracle.md` executes48 cases: energy contains each Gaussian once, but force contains older ones twice. `native-gaussian-consumers.md` shows the immediate optimizer caller does not remove that extra copy. | Current conservative Python Gaussians are mathematically consistent. Do not copy this native E/F mismatch into a line-search objective. Full runtime parity and intended native physics remain different questions. |
| P1: height policy is a genuine missing native search choice | `paper_reference.py`, `vc_reference.py`, `rc_reference.py`, `rc_vc_reference.py` use the BP-CBD forward-force formula. `gaussian.adjust_native_weight` recovers87° arithmetic, but was not connected as a stage policy. | New `ConservativeNativeHeightPolicy` can be tested explicitly against that baseline, with single-count force and frozen stage weights. No global/default replacement or performance conclusion yet. |
| P1: initializer mutates history, not only the latest weight | Newly executed `set_initial_gaussw`,27 normal-return cases, exact agreement. Levels1/2 rewrite earlier weight entries at each call. | New pure `initialize_history` reproduces these writes and reports changes. Caller must replace/rebuild the full frozen bias history at the stage boundary; simply appending a returned scalar is wrong. |
| P1: curvature provenance must be explicit | `direction.paper_biased_direction` and `generalized_dimer` return curvature after the negative rank-one rotation bias. `set_initial_gaussw` branches on `control.curv_real<0`. | Never feed the biased eigenvalue directly as native real curvature. Removing the analytic rank-one term recovers curvature of the declared rotation surface, which can still contain LS. Native LS inclusion in curv_real remains unclosed. |
| P2: width/move/termination state | `native-gaussian-caller.md` shows `moveds` stores a measured projection as saved width, with further retry paths incompletely recovered; stage completion can copy work1/work2 rather than an arbitrary last trial. | Independent kernels use an explicit requested width/displacement and declared termination. This is a search-policy difference, not automatically a gradient error. Exact release timing/width parity needs a bounded state-machine recovery; not prerequisite to testing an independently named height policy. |
| P2: numerical and variant policies | Broyden CBD, native LS eligibility, VC cell scheduling and RC lambda/Kabsch policies have distinct evidence boundaries below. | Keep each gap separate from missing end-to-end execution. Several workflows are already executable with conservative independent replacements; that does not make their streams native-identical. |

## What the87° condition actually controls

At a frozen evaluation point, write the single-count background force as `Fbg=f_parallel*n+Fperp` and the new Gaussian force as `c*W*n`, with unit n and `c=d1*d2>0`. Then

```
angle(Fbg+cWn,n) <= 87 degrees
```

is equivalent, for nonzero resultant force, to

```
f_parallel+cW >= cot(87 degrees)*||Fperp||.
```

Thus the native angle target ties required forward force to the transverse force at that point. The current BP-CBD policy sets the forward component to the explicit constant target0.1 eV/Å (its supplied-paper setting). Neither dominates for all landscapes. The angle rule can require less forward force when transverse force is small and more when it is large; its growth schedule and positive initial W may overshoot that analytic bound. This derivation explains a potentially meaningful comparison without inventing another controller or asserting that native is better.

The existing helper grows `W=min(W*scale,W+2)`, then multiplies scale by step. It checks `W>maxw` **after** growth; maxw is not a clip. The87° threshold and additive2 are recovered constants. Native energy-unit values are mapped explicitly into ASE eV in this independent profile; they are not universal optimized physical constants. No initializer/growth defaults were inferred from unexamined parser presets.

The native caller itself adjusts only when `substatus='climb_new'` and then switches to `climb_opt`. It does **not** adjust on every arbitrary force evaluation. The conservative profile retains once-per-stage preparation. Its distinguishing choice is single-count old Gaussian force and explicitly consistent energy, not an invented correction to native call frequency.

## New initializer original-instruction evidence

The routine at0x6e8700–0x6e8799 has no external calls. `probe_initial_gaussw_emulated.py` executes its original instructions under Unicorn, with no hooks replacing arithmetic, no native main or protection path. Twenty-seven cases span `ng=1,2,4`, `w_level=0,1,2` and `curv_real=-.2,0,.2`. Explicit controlled settings are w_initial=.6, w_neg=.07, maxw=10, growth_step=1.2 and growth_scale=1.5; these are fixture inputs, not release defaults.

The exact finite-domain contract is:

```
output maxw, step, scalefact := corresponding parameter fields
W[ng] := w_neg if control.curv_real < 0 else w_initial
if w_level == 1: W[1] := 5.6
if w_level == 2:
    W[1] := .5
    if ng > 1: W[2] := .5
```

Here the native array indices are one-based. All other preexisting array entries remain unchanged. Negative-curvature selection can therefore be overridden for the first or second height by the level rule. The strict zero boundary chooses w_initial. Even after a previous stage adjusted W1 upward, the next initializer can reset it to5.6 or.5. Reweighting between stages is compatible with a conservative *new* objective if both E and F are rebuilt; mutating weights during optimizer callbacks would violate the frozen objective.

`research/ga_ssw/evidence/native-initial-gaussw-emulated/result.json` saves every input/output and executed instruction address; all27 agree exactly with the recovered formula. Parameter/field evidence: `para+0x2dce0/maxw`, `+0x2dce8/w_initial`, `+0x2dcf0/w_step`, `+0x2dcf8/w_scalefact`, `+0x2dd00/w_neg`, `+0x2dcd8/w_level`; `control+0x40` is named curv_real in the archived DWARF member offsets. NaN, invalid array index and uninitialized-native-global cases are outside this oracle contract.

## Callable conservative helper and caller obligations

`pamssw/standalone/native_height_policy.py` adds:

- `ConservativeNativeHeightPolicy(initial_weight,negative_weight,level,max_weight,growth_step,growth_scale)` with explicit inputs;
- `initialize_history(weights, curvature=...)`, returning the full initialized sequence and historical changes;
- immutable `FrozenHeightGaussian` in flat scaled coordinates with exact bias E/force;
- `prepare(history, center=..., direction=..., width=..., point=..., background_force=..., curvature=..., curvature_scope=..., max_updates=...)`, returning the complete replacement history, final height, history changes, angle/update trace and angle-versus-maxw stop reason.

The supplied background force must include physical/LS force plus the **input history** once, at the same point. The helper recomputes the known force difference caused by initializer history writes, then adjusts only the new height. It returns `terms` as a **replacement of the entire history**, not a term to append to the unchanged old list. Every returned term is subsequently fixed, so ordinary energy and force callbacks remain conservative. It returns Gaussian energy separately from total preparation force because the caller did not supply the physical background energy.

No physical oracle calls are made. `max_updates` is an explicit numerical work budget, not a fallback/acceptance rule; exhaustion raises. Positive projection, finite nonzero resultant force, unit direction and growing positive weights define the supported domain; invalid/degenerate native operands are rejected rather than assigned guessed behavior.

For an existing normalized mode n and fixed anchor n0, the independently biased rotation Hessian is `Hrot=Hsurface-a*n0*n0.T`. Therefore the curvature appropriate to its underlying rotation surface is

```
k_surface = mode.curvature + a*(n dot n0)**2.
```

This removes only the analytically known rotation bias, without additional E/F and subject to the existing finite-dimer error. With LS enabled it remains **LS-modified** curvature; it is not the bare PES curvature. The helper requires a separate `curvature_scope` string and records it with the supplied value. The archived native control field name alone does not prove whether upstream LS was excluded. Do not call that conversion full native curv_real parity.

For VC/RC or constrained coordinates the angle must use force and direction in the same explicitly scaled active/generalized chart. Raw full Cartesian forces would mix units or include fixed-support reactions. Padding fictitious atoms or silently borrowing a native Cartesian metric is not a justified generalization. The flat helper supports arbitrary dimension, but acceptance of that coordinate metric is an independent policy choice. Eckart-chart integration needs its separate caller contract and is not supplied here.

`tests/standalone/test_native_height_policy.py`: **4 passed**. Tests cover all27 native initializer fixtures, comparison with the previously recovered87° helper in its matched operand domain, a nontrivial level-based historical reweighting with frozen-energy finite differences, post-update maxw behavior and numerical budget failure. These establish arithmetic and conservative objective consistency only. No new PES search was run by this task.

## Remaining SSW-family semantics, accurately scoped

- **CBD/Broyden direction:** current Ritz/plane-dimer methods are explicitly independent solvers of the biased low-curvature direction problem. Native BRZERO4 history, first-rotation adaptation, rotation caps and termination/rollback have not all been closed into a full native CBD stream (`native-rotation-followup.md`). Their absence is a numerical/search-policy substitution, not proof the independent Gaussian objective is wrong. A full CBD variant claim still needs those mechanics or an explicitly declared independent formulation.
- **LS:** native-derived initialization, amplitude/count normalization and cycle controller now have Python components and an executable adapter (`native-ls-python-component.md`). The caller updates after converged soft prequench and retains that certification policy; native iteration-limit response eligibility is not copied. Native MIC versus image-resolved periodic LS remains a separate issue. The old blanket statement 'LS lifecycle not implemented' is obsolete.
- **VC:** joint logstrain and block cell workflows are executable independent coordinate/schedule choices. Native coordinate metric, exact cell-direction lifetime and some block continuation semantics remain incomplete. Applying87° in scaled strain space would be a declared generalization, not recovered native cell parity. Cell displacement and quench gradient consistency have stronger evidence than native schedule identity.
- **RC:** isolated chain, forest, periodic rigid-center geometry and full VC-RC workflow now execute. Exact Jacobian force pullback is a conservative alternative; native lambda force/torque transmission and Kabsch coordinate coupling are still distinct unresolved mechanics. Whether lambda represents preconditioning or a different nonconservative update cannot be settled by conserving resultant force alone.
- **Outer MC/history:** native near-equal-energy trap counters and effective-temperature changes are separately reconstructed in `native-mc-contract.md`; the independent walkers still use their explicit plain MC. That is a missing release search policy, not a missing ability to select/minimize candidates, and must not be bundled into a height comparison.

The next clean experiment can compare current forward-force and explicitly configured conservative-native height profiles on identical input/PES/seed/cost, recording weight histories and failed attempts. It should preserve independent mode, quench, MC and LS settings so any effect is attributable to height policy. No recommendation to replace defaults follows before that measured comparison.

## Follow-up review of the parent-provided atomic integration

The parent subsequently added an explicit optional `height_policy` to `run_ssw` (default forward-force behavior retained; Eckart combination rejected before oracle work). Review of the actual loop confirms that physical/LS-plus-old-history force is evaluated once at the displaced point, the helper replaces the entire Gaussian history while preserving soft terms, the rank-one curvature contribution is removed, and the resulting objective stays fixed throughout quenching. No extra E/F call is introduced by the helper.

One bookkeeping defect was corrected in the public file with the parent's explicitly limited authorization: a biased-quench exception previously occurred before the prepared-height event was appended, losing which objective had been attempted. The loop now saves that event before quenching and updates the same event on success/error, retaining stage costs and preventing duplicate events after a later evaluation failure. A new injected-quench-error regression proves that the height history, final weight, error and cost survive. Height driver, paper driver and helper selection: **10 passed**.

The direction-only profile's angle uses the full Cartesian background force; any old-bias rigid-force component contributes to its transverse norm. That is the declared conservative Cartesian profile, not proof of native constraint-projected angle parity. Its energy/gradient pair remains consistent. The independently prepared actual-system height comparisons belong to the parent workflow and are not counted as results of this audit.
