# Cu13/EMT Hookean composition and restart protocol

## Question and scope

This bounded interface check asks whether the existing `normalize_constraints`
→ `ConstraintSet.clean_atoms` → `bind_hookean_surface` path composes with the
ordinary fixed-cell `run_ssw` checkpoint contract when the full recovered
direction controller is enabled. It reuses the saved Cu13/EMT input and
configuration from `../startup-order-20260926/qualify.py`. It is not a search
success, performance, or Hookean-parameter study.

The input carries one explicit atom-pair ASE `Hookean(0, 1, k=1.0, rt=0.9 * initial_pair_distance)`
constraint. Here `k` is in eV/Å² and `rt` in Å. The pair distance is measured
from the saved Cu13 geometry. Before any walk, the wrapper must report positive
Hookean energy and a nonzero Hookean force correction. This deliberately
ensures the wrapper is active. The values are an interface stress choice, not
a confinement recommendation. A pair spring depends only on the interatomic
distance, so its correction is invariant to global translation and rotation.

The script normalizes the constrained input, removes ASE constraints from the
walker atoms, and binds the serialized Hookean specification to a fresh EMT
surface. This is required because ordinary `run_ssw` expects unconstrained
atoms, while `RecoveredDirectionSettings` also requires a free, nonperiodic
cluster. The wrapper supplies the same augmented energy and forces throughout
initialization, direction search, climbing, and landing.

## Fixed comparison

Run two outer steps continuously, then compare against a second two-step run
paused after outer step 1, saved to disk, loaded, and resumed for one step. Both
arms use the same input bytes, EMT backend, one frozen Hookean specification,
`SSWConfig`, full recovered-direction settings, and seed. Resume receives a
different throwaway RNG seed so an exact match demonstrates restoration from
the checkpoint. Native LS and pool selection are omitted to keep the
composition test focused.

Acceptance requires exact equality of the completed records, minima/current/
best state, recovered-direction checkpoint state, main RNG state, and request
cost. It also requires the input positions to remain unchanged; continuous and
split arms to use the same Hookean pair specification; and the summed record
requests to equal the physical-surface request count. The preflight and final
endpoint checks require one physical calculator request per augmented wrapper
evaluation. At the final endpoint, independent raw EMT E/F must match the
wrapper’s physical component, while augmented E/F must equal physical plus one
Hookean correction. A separate ASE Atoms/EMT/Hookean evaluation must match the wrapper; the augmented endpoint force must meet 0.03 eV/Å. These checks detect accidental omission or double
addition of the constraint term.

The runner also performs a zero-step resume of the paused checkpoint with a
changed Hookean spring constant and a fresh wrapper. If accepted without a
runtime exception, it records acceptance and zero new PES requests. This is a
generic oracle-identity boundary of the current ordinary SSW checkpoint: the
checkpoint records SSW and direction state but not calculator/wrapper identity.
The altered-spec zero-step result is diagnostic only and must not be treated as
a valid changed-potential continuation or as a newly introduced code defect.

## Bounds and execution

The runner enforces at most 3,000 physical E/F API requests for each complete
two-step trajectory (the split and resumed segments share one counter), 6,000
requests across preflight, both trajectories, and endpoint checks, and a
270-second internal wall limit. A matching CPU allocation is one task on
`CPU-MISC`, `rush-cpu`, account `sjtu-caoxiaoming`, with a five-minute Slurm
limit. No GPU or model files are used. “Request” means one `ASESurface`
evaluation, not lower-level calculator work.

Reviewed launch: `sbatch --wait --parsable check.sbatch` from this directory
(or the corresponding repository-relative path from the checkout).
Job **1493555** was submitted after AST, shell syntax, diff and Slurm test-only
checks. Completion and numerical qualification are recorded separately.

The script refuses to overwrite an existing `runs/` directory. It writes the
input copy, protocol, initial decomposition, continuous and split results,
checkpoints, exact-continuation comparison, changed-spec identity diagnostic,
raw/augmented endpoint check, and final summary there. On failure it retains
completed checkpoint files and writes `failure.json` with the exception and
cost ledger; do not silently rerun over that evidence.

## Evidence limits

Passing this check would qualify API composition and same-oracle checkpoint
continuation for this Cu13/EMT fixture and these settings. It would not
establish Hookean confinement quality, GA/SSW effectiveness, native parity,
transfer to other calculators, or a public API that automatically accepts
`Atoms.constraints` in ordinary `run_ssw`.
