# GA fine feedback cycles

The fixed-cell GA controller exposes `PaperGAConfig.cycles`, a positive integer
with default `1`. Initial true quenches and quick walks run once. Each cycle
then runs the configured GA generations followed by ranked fine walks. Fine
walk landings are ingested into the shared archive; the next cycle partitions
that updated archive and uses its representatives as parents.

Cycles share the same archive, observations, random generator, surface request
counter, and optional global evaluation cap. No Java schedule multipliers or
new selection policy are introduced. A proposal failure retains the existing
behavior within that cycle: the cycle proceeds to its fine walks, then the next
cycle may continue from whatever archive remains. Budget exhaustion terminates
the whole run without resetting its accounting.

`GAStage.cycle` records the zero-based cycle for proposal, generation-short, and
fine stages; initialization and quick stages remain cycle zero. The default
`cycles=1` path is intended to preserve the previous trajectory and cost.
The cycle mechanism tests controller lifecycle only; fine feedback improving a
real PES search remains a separate scientific question.

The GA entry point also forwards the caller-selected `height_policy` and
`height_update_budget` to every SSW walk, with the SSW defaults
(`None` and `1000`) preserved. GA does not select or alter a height strategy;
the policy remains an explicit caller input, including when LS is enabled.

Source: user-provided GA-SSW PDF, section2.4.3, extracted GA-SSW-user.txt lines236–252 (DOI10.1021/acs.jctc.6c01078). The paper calls for minima from fine exploration to seed subsequent GA-SSW operations. This independent controller retains its existing population rules and explicit operation counts; it does not reproduce Java execution scheduling.
