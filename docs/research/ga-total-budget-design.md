# Fixed-cell GA global evaluation budget

`run_ga_ssw` accepts `max_evaluations=None` (the legacy default) or a
nonnegative integer counting requests made through the supplied surface,
relative to the request counter at entry. The cap covers initial quenches,
offspring quenches, and every E/F request made inside quick, generation, and
fine SSW walks. Proposal sampling does not consume surface evaluations.

When a cap is requested, a thin surface proxy checks immediately before each
physical `evaluate`. The public SSW walker performs its initial true quench even
when `steps=0`, so zero-step walks still require PES budget. If the next request would exceed the cap it raises
`BudgetExhausted` before calling the underlying surface. Already paid requests,
observations, and eligible archive representatives remain in the result. The
controller stops the current candidate/seed loop and all later phases after the
blocked request. `budget_exhausted=True` and status `budget_exhausted` mean a
scheduled request was blocked; reaching the cap exactly at natural completion
leaves that flag false. The result also returns `budget_limit` alongside the
existing request count.

This closes accounting and control flow only; it is not an algorithmic gain.
The default `None` path retains unlimited legacy behavior. The limit counts
surface API requests and does not represent backend-internal SCF work.
