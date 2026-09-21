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
the requests spent while replaying that unfinished part; there is no
pending-offspring state.

Checkpointing is opt-in.  The default call path does not invoke the callback or
copy controller state, so its surface request ledger and trajectory remain
unchanged.  This is persistence and controller verification, not a claim of
calculator-state or scientific-search equivalence across environments.
