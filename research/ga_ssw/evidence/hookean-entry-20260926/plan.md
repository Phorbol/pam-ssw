# Approved pair Hookean automatic entry

User approved the referenced 2026-09-26 proposal. Scope: nonperiodic atom-pair
Hookean on run_ssw input, existing surface composition, explicit saved identity
(schema6 only when present), old no-restraint formats/behavior preserved.
No new confinement defaults, objective or generic constraint framework.

1. Root writes interface/identity tests before production changes. CPU1493564
   records 6 expected failures, 3 passes against 4c13652 (old rejection).
2. Delegate implements only driver/checkpoint + small constraint helper;
   root checks all construction/copy/error paths and version feature validation.
3. CPU targeted regressions: new identity tests, existing checkpoint/full-direction/
   pool/LS pool and ASE constraints; max5min, no GPU.
4. Real Cu13/EMT: explicit nonzero pair term, two outer steps continuous vs
   paused/resumed, ordinary and actual alternate-pool restart. Same physical
   configuration, RNG, costs and outputs required; ASE direct E/F verification
   on returned endpoints. 6000 total search requests + bounded fresh checks,
   single task max5min. No extrapolation to C60 efficacy, arbitrary constraints
   or other calculators. Persist failed runs; no threshold changes to get green.

Core code base is4c13652 plus committed implementation/runner in this series.
Artifacts contain actual scheduler IDs and return codes. Documentation/code
checks, interface qualification and scientific search effects remain separate.

## Execution-layer correction

CPU1493577 failed before any PES call while serializing Hookean in ASE extxyz.
Installed ase/io/extxyz.py move_mask branch converts FixAtoms/FixCartesian only;
Hookean remains a list and output_column_format accesses .dtype. Original
runner and partial input are preserved under runs/, log in cpu-1493577.out.
Correct only experiment serialization: write clean coordinates and canonical
Hookean specs separately in protocol.json. Retry has new runs-v2 directory;
same input, constraints, algorithm, random seeds and bounds. Core unchanged.

## Required-state guard correction

CPU1493580 completed all four real paths:1264 search+4 fresh, exact ordinary
and actual-pool continuation. This was the first schema6 implementation.
Root subsequently identified that presence-only feature validation loses old
schema4/schema5 mandatory-state guards when the field is absent. CPU1493683
reproduced two expected failures (missing direction and missing pool state).
Preserve original capability version as base_schema_version1..5 within schema6,
then apply unchanged legacy feature validation to that base. No algorithm or
new public option. First implementation source delta/helper are preserved in
v1-source/ relative to4c13652; final qualification must use the corrected format.
