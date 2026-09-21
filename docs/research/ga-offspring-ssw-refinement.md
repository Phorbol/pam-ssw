# GA offspring SSW refinement

`PaperGAConfig.offspring_steps` closes the Java-style offspring refinement
boundary without changing the default path. The default `0` keeps the existing
true-surface direct quench. A positive value sends each GA candidate through an
independent `run_ssw` call; its built-in initial quench is therefore the only
initial quench for that candidate. All landings and failures remain recorded,
while only the lowest eligible landing from that offspring walk is admitted to
the shared archive and subsequent parent selection. `offspring_ssw_config` is
an optional explicit `SSWConfig` instance, so callers can express a temperature
multiplier with `dataclasses.replace` without a controller hard-code.

The existing surface proxy, total budget, LS object, height policy and height
update budget are reused. This is a lifecycle compatibility feature; it does
not establish a search-efficiency or scientific advantage claim.
