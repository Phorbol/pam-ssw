# Native release condition still relevant to CuO/TiO2 no-landing cases

Date: 2026-09-11. This is a bounded, read-only audit of the archived
`GA-SSW_program/lasp` ELF and existing disassembly. No native executable,
PES, or new search was run. The fixed Gaussian width and the native
lower-energy stop have already been audited elsewhere; this note records one
different lifecycle condition.

## Direct evidence

The native local optimizer judge `ssw_fixlat_mp_allopt_judge_converg_` starts at
`0x5cf6f0` (complete disassembly:
`/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/analysis/judge-convergence.asm`).
It reduces absolute force components and compares the resulting `maxf` with
`para+0x2db28` (`ftol`) at `0x5cf976–0x5cf991`, setting the optimizer
convergence state. The same routine then reads `control+0x74` (`alloptstep`)
at `0x5cfaf1` and compares it with `para+0x2ddd0` (`maxoptstep`) in the
following branch through `0x5cfb0a`. The stop result is therefore an OR of
force convergence and the optimizer iteration cap. A normal return from this
judge does not imply `opt=true`.

There is a further explicit stop gate at `0x5cfd9e–0x5cfda7`, testing
`control+0x1b0` (`bfgs_must_stop`). These gates are downstream of the
Gaussian-biased local optimization and are distinct from the
`climb_convg_` energy-release comparison at `0x5cd44a–0x5cd469`.

## Relation to the independent no-landing records

The earlier CuO comparison records did hit their 300-iteration local budget,
but the later history500 control was censored much earlier by the shared
1997-request search cap: seed 7 stopped at 32 optimizer steps (35 attempted
requests), and seed 101 at 35 steps (38 attempted requests). In both controls
the outer report is `censored`, while the record is wrapped as
`biased_quench_failed`; the preceding Gaussian stages were converged. This
cost/state evidence supports budget censoring as the immediate cause of those
two latest no-landing records. It does not establish a native release
equivalence. The independent VC code requires `relaxed.converged` before
proceeding to a true landing quench (`pamssw/standalone/vc_reference.py`, the
biased-quench status branch around the `relaxed.converged` test).

The native outer path then restores the saved energy and enters the `Allopt`
status path at `0x5cb8dd–0x5cb928` after `lclimb_allstop` is set. That is a
consumer of the climb status, but the inspected slice does not prove that an
`allopt_judge_converg_` step cap sets this climb-release bit, nor that the
subsequent path performs a physical landing. The safe distinction is:
`allopt_judge_converg_` may stop the biased local optimizer on force, a step
cap, or `bfgs_must_stop`; this numeric stop can still return control to the
climb consumer and allow the bias-removal/`Allopt` path. It is not itself a
successful escape or a physical-minimum certificate. Conversely, the absence
of a native proof that the stop is a certified landing must not block an
independent algorithm test whose explicit contract performs a subsequent
unbiased true-PES quench and validates its result.

## What is known, unknown, and worth testing

Known: the native optimizer stop predicate has a force-independent `maxoptstep`
route and a separate BFGS-stop route, with exact addresses above. The two
original history10 CuO attempts hit their declared 300-iteration budget; the
history500 whole-search controls instead exhausted their total EFS budget.
A matching native cap value or release route has not been established. Unknown: the
native `maxoptstep` value and whether either gate was active in the CuO/TiO2
production inputs; unknown: which work buffer is released and whether the
subsequent `Allopt` path performs a fresh true-PES quench.

No lifecycle experiment is justified by this evidence alone. A future control
would first need a closed consumer trace and a stated landing certificate;
until then this remains a stop-boundary observation. It does not justify a new
threshold, changing the current acceptance contract, or claiming a search
improvement. No additional reverse engineering is warranted here.
