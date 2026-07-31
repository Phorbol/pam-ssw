# S-CR0 selector-support audit

Audit the recorded starter-ID traces from the seed-42, 20,000-FE C60, PdO
and CuO campaigns.  This stage performs no PES evaluation and changes no
algorithm code.

The only question is whether the existing node-level UCB-like selector uses a
large fraction of the growing archive as effective starter support, compared
with uniform archive sampling and a Metropolis chain.  Search quality remains
a separate output and cannot be inferred from selection entropy alone.
