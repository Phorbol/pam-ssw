# Direction-mode continuation repeat conclusion

Decision: `repeat_stable`.

The preregistered advantage condition, plateau/seed44, repeated with the same
classification: fixed-intent Ritz was non-meaningful and transported direction
was meaningful. There were zero advantage reversals.

The full 12-cell classification matrix was not identical. The control at
plateau/seed42 changed from non-meaningful to meaningful, so full-matrix
stability is `False`. This is retained as an explicit stochastic-robustness
limitation rather than hidden by the survivor gate.

| arm | meaningful | median mode cosine | direction FE | case FE |
|---|---:|---:|---:|---:|
| fixed_intent_ritz | 2 | 0.968889 | 768 | 2,429 |
| transported_direction | 2 | 0.999999 | 76 | 1,890 |

Transport again reduced direction cost by about 90% and case cost by 22.2%.
Including six shared initial modes, the repeat consumed 4,463 FE and produced
12 strict certificates.

PdO transfer was not run because the exact preregistered PdO intermediate and
plateau structure artifacts are absent from the available worktrees. Replacing
them with convenient PdO minima from another run would change the transfer
protocol. The next bounded action is to restore those two state-hash-matched
artifacts and run the unchanged two-arm matrix; no PdO tuning is justified.
