# Atomic contribution in a bounded TiO2 block search

Runner e68a7ed, job1756999, four arms completed or reached their predeclared
per-arm cap. Campaign elapsed258.91s. Raw directory: `run-1756999/`.
`readout-1756999-v2.json` is the current compact analysis; v1 predates explicit
unassigned-stage and valid-atomic-landing fields. Raw evidence is unchanged.

| Seed | Treatment | Paid search / actual calculations | Valid landings | First cold-confirmed anatase cost |
|---|---|---:|---:|---:|
|26101241|cell only|3709 / 3680|20|1175|
|26101241|cell + atomic|4000 / 3900|6|not found within cap|
|26101242|cell only|3910 / 3881|20|712|
|26101242|cell + atomic|4000 / 3898|7|2569|

All three target-positive arms have a cold-qualified target frame. In the last
arm the first target and best endpoint are identical, so one fresh check serves
both roles. Search ledgers independently sum to15619 paid /15359 actual calls;
11 arm fresh +1 reference =12 additional calls. No paid backend failures, two
request-cap denials. The23 zero-request tail records are unstarted, not failed
physical proposals. Raw core `completed` is not the research completion gate.

The atomic arms completed59 Gaussians, all using the existing finite-budget
rotation release. Five atomic-stage landings were numerically valid, all
MC-rejected and none anatase. Two further atomic attempts stopped on
nonpositive computed height; the current documented contract discards those
proposals without a final true quench. This is not an unimplemented approved
release route and does not justify adding a fallback from two observations.
The atomic treatment's sole anatase target was reached on a cell-only outer
step (zero-based index4), not an atomic-stage landing. Earlier atomic steps
may still affect later RNG consumption; no unique trajectory causation follows.

Contrary evidence: seed41 cell+atomic best is -428.27803eV, below cell-only
-428.13512eV, but is not the target geometry. Seed42 has the reverse ordering:
cell-only best is a lower-energy non-target structure. Thus target arrival and
lowest-energy discovery give different comparisons. Do not call cell-only a
universally better global optimizer or treat anatase as this model's GM.

Paired-input qualification passed, but trajectories are not exact counterfactuals.
Before the first atomic stage, maximum saved cell-entry differences are about
4.2e-6A and0.058856A for the two seeds. They are reported as diagnostics, not
used to relax physical target gates. No precision-tuning campaign follows.
The results support a bounded treatment comparison from one input and two
seeds, not an isolated deterministic estimate of one atomic move's effect.

Stage subtotals leave175/71 paid calls unassigned in the atomic arms; those
costs are explicit in v2 and remain included in totals. Do not use the existing
stage helper to claim an exhaustive allocation or wall-time bottleneck.

Decision: no default change, no continuation or tuning of this input. The
atomic module demonstrably executes; lack of activation no longer explains its
missing target benefit here. Prioritize independent input transfer or a clearly
identified mechanism over another optimizer/backend or threshold scan.

Verification: six targeted contract tests and zero-PES input/config/import
preflight passed. Static review corrected two startup defects before submission.
Root re-counted ledgers and checked cold gates; an independent reviewer obtained
the same totals and highlighted the lower-energy non-target counterexample.
The first analysis call found the stage-key name differed from the draft; it
was corrected against actual output schema without altering raw data or rerunning
PES. No core/default/API changes in this experiment.
