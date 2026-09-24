# GA active-walk recovery implementation plan

Spec: ../research/2026-09-23-ga-active-walk-checkpoint-proposal.md. The user explicitly approved its minimum design. SSW completed-outer-step callback is already implemented; do not redo it. Resuming this previously deferred engineering item follows closure of the two bounded fixed-cell depth/tolerance probes; no supported change to the numerical default emerged. This work does not tune GA or initiate long production search.

## Constraints

Keep default phase-boundary checkpoint callbacks unchanged. Add one explicit opt-in for completed-SSW-step snapshots. Write checkpoint v2 with optional active walk, read v1 with old behavior. The active walk owns a nested existing SSW checkpoint, selected seed queue/cursor, current phase/cycle/generation, target/completed walk steps, offspring provenance, and cumulative costs. Preserve algorithm ordering, RNG, archive ingestion exactly once, current contracts and unchanged total budgets. Do not save optimizer/Gaussian internals, introduce adaptive search rules, or silently accept stateful callbacks without existing compatibility checks.

## Task 1: implementation and focused regression

Read ga_checkpoint.py, paper_ga.py, SSWCheckpoint and existing test_ga_checkpoint.py/test_ssw_boundary_pause.py. Extend existing checkpoint and shared walk/controller path for quick, generation_short, fine and offspring_ssw. Use the smallest explicit controller continuation cursor needed to resume selected work without repeating partition/parent selection or archive ingestion. Avoid duplication of whole phase loops. Prefer a small helper module if the continuation data handling has a distinct responsibility; no general workflow framework.

Add focused regression tests for cooperative pause within each walk phase, uninterrupted/resumed RNG and ordering, request conservation, no duplicate archive ingestion, old v1/default behavior and incompatible state rejection before PES. Validate through CPU Slurm, never numerical work on login. Baseline first: tests/standalone/test_ga_checkpoint.py and test_ssw_boundary_pause.py. Then extend only relevant tests plus test_paper_ga.py; no broad unrelated suite.

## Task 2: independent review and real-system qualification

Review the complete diff against the spec and verify actual test outputs. Deterministic EMT Cu13 continuous versus split process must preserve physical endpoints, observations, lineage and costs. Reuse existing real fixture/runner if present. Next, prepare a bounded MH-1 C60 cross-process smoke after deterministic qualification: no long search, independently check force and model identity, allow GPU roundoff without demanding bitwise trajectories. Record missing evidence rather than declare general search superiority.

## Delivery

Work in feature/ga-active-walk-recovery, based on a3ea940. Do not touch integration worktree experiment files. Commit scoped tested changes in this branch; parent reviews before integration/push. Keep mainline summary and existing proposal updated at phase boundaries. A genuine public-contract conflict beyond the approved spec must be presented before implementation; routine field naming and implementation choices do not require repeat approval.
