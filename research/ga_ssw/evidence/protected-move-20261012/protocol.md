# Protected-move paired paper-case protocol

Status: executed as GPU1754252; four completed arms, 1354 search requests and
four fresh evaluations. Frozen design below is unchanged. See [readout](readout.md).

The first CPU smoke, `smoke-1754155`, is retained as a failed plumbing attempt:
all four arms produced records but ended `evaluation_failed` at the recovered-CBD
native acos domain check (C60 stage 0, TiO2 stage 1). The isotropic quadratic smoke
calculator makes this direction calculation degenerate. This is not evidence
for or against the move guard, and those artifacts are not overwritten. The
revised smoke below uses an explicit identity-CBD stub solely to test runner
plumbing; it does not validate CBD numerics or provide scientific evidence.

## Question and decision

Does the optional fixed-cell scalar-guard move change first-outer-step landing
qualification or paid search cost in the two already qualified S2 paper cases?
The panel is a development comparison, not an efficiency or generality claim.
Protected move remains opt-in regardless of the outcome. A qualified-endpoint
improvement without a corresponding explanation in the raw move diagnostics
does not support keeping the mechanism; no guard activation means first audit
reproduction and configuration, without increasing the budget or tuning
thresholds.

## Frozen comparison

Cases are C60 isomer 2 with MACE-MH-1 (`omol`) and TiO2 phase 87 with OMAT-small
(`omat_pbe`). Each case uses S2 only. For each, run two independent invocations
from the same saved input and the same NumPy seed, one with `move_policy=None`
and one with `ScalarMovePolicy()`. Each invocation performs exactly one outer
step. The SSW configuration, recovered S2 direction settings, backend, initial
structure, and numerical precision are shared within the pair. No second outer
step, parameter sweep, or checkpoint-resume segment is included.

Frozen seed: `20261012`. Limits: four total arms, at most 3000 search requests
per arm, at most one independent fresh-calculator endpoint evaluation per arm,
one V100, and 30 minutes total. The seven-minute arm limit includes calculator
setup, search, and fresh evaluation. A request or wall limit is a retained
censored run; hard search or fresh-calculator failures stop later arms while
preserving all completed and failed artifacts. No automatic retry is allowed.

## Inputs, models, and provenance

The runner imports the existing `probe_q_paper_cases.py` source of truth for
input and model paths, frozen SHA-256 values, qualification checks, descriptor
preflight, SSW configuration, and geometry diagnostics. It records the actual
input/model hashes, qualification/manifest paths, checkout `HEAD`, runner and
imported helper paths, and hashes of those source files at execution time.
Model files are referenced in place and are not copied. Each arm writes its
starting structure, full result or partial result, final outer-step landing,
fresh endpoint structure, request ledgers, summary, errors, and calculator
call/request costs. The raw calculator is wrapped directly by the shared
`CountedSurface` ledger; this does not insert ASESurface into the search path.

Each available terminal candidate is checked at most once using a newly
constructed calculator. Prefer the actual outer-step landing. If there is no
landing, a saved last failed state may be checked as a diagnostic only; it
cannot pass endpoint qualification. If neither exists, record that the fresh
endpoint is unavailable and make no call on the initial input or current/best
fallback. Record fresh energy, maximum force, force-qualified flag, fixed-cell/PBC and
composition checks, and geometry diagnostics. These checks describe numerical
endpoint qualification and geometry; they do not establish phase identity or
first-principles accuracy. Guard accept/shrink/retry-exhaustion diagnostics
from the SSW record are retained without reclassification.

## CPU smoke and scientific limits

The opt-in CPU smoke uses a deterministic harmonic ASE calculator centered on
the case's original coordinates and the same runner, raw-calculator ledger, and
one-step public API path. It replaces only the CBD function with an identity
stub returning the supplied anchor as a converged direction. This stub is
installed only under `--smoke`; the scientific `--execute` path does not patch
CBD. Its request cap is 500 per arm and wall limit is 120 seconds per arm. It
reports success only after all four expected arms each return one completed
outer step without hard calculator failures, search failures, or request/wall
boundaries. Smoke artifacts, including any successful mocked-CBD smoke, only
validate runner wiring and carry no scientific evidence. The prepared GPU
command remains the only route to the frozen MACE panel; this protocol itself
submits no job.

One seed, two systems, and one step cannot establish a general benefit, a
change in global PES coverage, or phase stability. Report both arms including
failures, denials, failed calculator calls, guards, and fresh endpoint costs.
