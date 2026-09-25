# Compact SSW progress callback: approved minimal design

User approved this minimum contract on2026-09-25. Implementation commit d27252b; targeted compatibility tests passed. Independent Cu13/EMT trajectory and zero-PES copy qualification completed as CPU1488820; see [qualification report](../../research/ga_ssw/evidence/ssw-light-observer-20260925/report.md). Follow-up documentation clarifies that call-start events require successfully initialized resumable state.

## Goal and evidence

The user wants paper-scale, bounded global-search comparisons with complete cost/trajectory evidence. Today run_ssw exposes progress/early stopping through checkpoint_callback. Enabling it constructs a detached full checkpoint every outer step and then copies that full checkpoint again for the callback. The checkpoint contains all previous records and minima (`paper_reference.py`, checkpoint construction and `_checkpoint_copy`). With a fixed bounded per-step payload, repeated full-history copying has quadratic cumulative work in the number of outer steps. This is not a change in the mathematical SSW proposal or a force-evaluation cost.

CPU1488429 measured `_checkpoint_copy` on prefixes of a real400-step C4H6 SSW checkpoint (26.5MB). Median single-copy times:10 records0.057s;50 records0.271s;100 records0.528s;200 records1.454s;400 records3.047s. Three measurements per size, zero PES requests. These demonstrate history-dependent copy cost; they are NOT a measured decomposition of the active LJ jobs, nor an algorithm speedup prediction. Current LJ budgets are left unchanged and any wall-censored result remains censored.

## Recommended minimum contract

Add optional `progress_callback` to fixed-cell `run_ssw`, receiving a detached, bounded progress object: initial/outer-boundary kind; current completed SSWStep (or None for initialization); newly added minimum (if any); current atoms/energy; best minimum; cumulative requests and next outer index. No accumulated history, RNG, pair/group/LS adaptation state or restart payload is exposed through this object. Atoms copies omit calculators and cannot mutate live search state. Per-step payload size depends on the current bounded Gaussian path, not total trajectory length.

Return True requests a pause at that safe boundary, matching the present first-hit usage. Produce one full compatible checkpoint on final return/pause for callers that need restart. Full checkpoint construction must therefore be deferred in this mode rather than triggered by an incoming checkpoint alone. Initialization and resumed runs retain the existing RNG and initial-quench behavior. Failed initial quench still raises the existing error; terminal failure and its paid costs remain in the returned result. No extra PES requests, proposals, defaults, numerical thresholds or selection policy.

Keep existing checkpoint_callback, checkpoint_path and pickle formats unchanged. In the first implementation, reject simultaneous progress_callback and checkpoint_callback explicitly; checkpoint_path keeps its existing every-step persistence behavior and associated cost. Default calls remain unchanged. Existing result/history storage still grows linearly; disk streaming, history pruning and checkpoint-format redesign are outside this proposal. Scope fixed-cell driver first; no GA/VC/RC interface redesign.

## Alternatives

1. Keep using complete checkpoints to observe every step: zero API change, but repeated-copy cost remains; unsuitable as the long-term paper-scale observation mechanism.
2. Recommended compact observer with one final checkpoint: one small public contract, old persistence compatibility retained; removes the full-history copying induced by progress-only usage.
3. Redesign storage into append-only trajectory plus minimal search-state checkpoints: addresses both memory and persistence scaling, but larger format/migration scope; defer until actual linear-memory or disk-write limits require it.

## Verification and rollback

Before implementation inspect existing callback/pause/resume tests. Test payload detachment, initial and outer pauses, final error costs, callback conflicts and old checkpoint/default compatibility. Use matched Cu13/EMT short trajectories (real atomic system) with fixed seeds and old-vs-new observers: identical minima, acceptance, RNG continuation and E/F request counts; separately report observer runtime. Repeat the zero-PES prefix-copy measurement using bounded progress payloads to verify cost does not grow with accumulated history. These checks establish engineering equivalence, not new global-search effectiveness. Reuse completed LJ inputs for a subsequent fixed-budget run only if needed; do not erase or relabel existing runs.

Approval received under the user's supplied AGENTS.md §8. It covers this bounded contract and routine implementation/testing; it does not authorize a checkpoint-format or archive redesign.
