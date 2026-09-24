# GA boundary checkpoint (experimental)

`run_ga_ssw` can optionally call `checkpoint_callback` after a completed quick
phase, generation, or cycle.  If the callback returns true, the run returns a
`PaperGAResult` carrying a `GACheckpoint`; `GACheckpoint.save()` and `.load()`
provide trusted local persistence.  The checkpoint contains the archive,
observations, failures, stages, walks, validation flag, configuration objects,
NumPy generator state, cursor, and cumulative surface request count.
It also stores a serialized scientific input contract covering descriptor and
proposal data, initial topology, LS/height/Gaussian settings, and callback
presence; mismatches are rejected before a calculator request.

Resume passes the loaded object as `checkpoint=...` to `run_ga_ssw`.  The
calculator/surface and validator/matcher remain caller-owned and must be
recreated with matching settings.  Resume begins at the saved next
generation/cycle and therefore does not repeat initialization, partition, or
proposal work.  A budget exhaustion or stage error is a terminal result and
does not create a new resumable checkpoint.  A recoverable candidate failure
can still be followed by a completed generation boundary.  If a process fails
after the last saved boundary, the caller's external ledger must account for
the requests spent while replaying that unfinished part; the default phase-only mode has no pending-offspring snapshot.

Checkpointing is opt-in.  The default call path does not invoke the callback or
copy controller state, so its surface request ledger and trajectory remain
unchanged.  This is persistence and controller verification, not a claim of
calculator-state or scientific-search equivalence across environments.

## Active SSW walks (2026-09-25)

`checkpoint_walk_steps=True` also invokes the callback at completed SSW outer
steps in quick, offspring, generation-short and fine walks. Such snapshots have
`state.phase == 'active_walk'`; `state.active_walk` contains the selected queue,
cursor and nested SSW checkpoint. Save using the existing `state.save(path)`,
and return true only when a cooperative pause is wanted. Returning false saves
without stopping. This optional state copy/I/O does not evaluate the potential.

```python
def save_and_pause(state):
    if state.phase == 'active_walk':
        state.save('ga-checkpoint.pkl')
        return True
    return False

result = run_ga_ssw(
    initial, surface, **scientific_options,
    checkpoint_walk_steps=True, checkpoint_callback=save_and_pause,
)
# In another process, recreate the same calculator and scientific_options.
result = run_ga_ssw(
    initial, fresh_surface, **scientific_options,
    checkpoint=GACheckpoint.load('ga-checkpoint.pkl'),
)
```

The unchanged total `max_evaluations` belongs in `scientific_options`; resume
subtracts the already paid prefix and runs only the remaining outer steps.
It does not reselect parents/regions or import the same walk twice. Saved LS,
direction and MC state use the existing nested SSW format. New writes use GA
v2; v1 completed-phase checkpoints remain readable. Only load trusted local
pickle files. Callback save errors propagate to the caller.

This does not checkpoint inside a Gaussian or line search, serialize the
calculator, or guarantee bitwise CUDA trajectories across processes. Hard-kill
replay costs still belong in the caller's external experiment ledger.
Deterministic Cu13/EMT cross-process checks cover all four phases, full result
and RNG equality, cumulative cost and fresh force qualification; see
[qualification evidence](../../research/ga_ssw/evidence/ga-active-walk-qualification-20260925/README.md).
