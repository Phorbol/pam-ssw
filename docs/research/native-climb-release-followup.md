# Native climb release: status boundary after the optimizer judge

Date: 2026-09-11. This is a bounded static audit of the archived
`GA-SSW_program/lasp` ELF and existing disassembly. No LASP process, PES call,
or production code was run. The purpose is to separate local-optimizer stop
from the subsequent climb-release status.

## What the one-level caller shows

Latest closure: the normal outer driver is now resolved as
`cal_pes_ -> move_ -> run_ssw_ -> return -> next cal_pes_` (addresses below).
Earlier statements in this chronological note that the first enclosing E/F
entry is unknown are superseded for that normal path. This does not close
all stop-flag conditions or establish a physical quench certificate.
Root also verified the actual `run_ssw_` table at `0x53ca680`: slots
`+0x100/+0x188/+0x1f8/+0x200` match `init_bfgs/set_status/allopt/allopt_judge_converg`.
Artifact: `evidence/native-moveds-retry/run-ssw-fixed-table.json`. The older
conditional use of candidate table `0x53cc8c0` is no longer the only table evidence.

Later table correction: the fixed-cell table0x53cc8c0 has slot+0x200 pointing
to allopt_judge_converg0x5cf6f0, verified in
`ssw-paper-native-release-boundary.md`. The original “unresolved slot” wording
below is historical; full-instance initialization and post-release E/F remain
separate from knowing this table entry.

The fixed-cell `ssw_fixlat_mp_allopt_` routine starts at `0x5d46e0`.
Its initial dispatch at `0x5d47d5` is an indirect slot (`object+0x200`), and
the fixed-cell table correction resolves that entry to
`ssw_fixlat_mp_allopt_judge_converg_` at `0x5cf6f0`. The same caller does
expose the post-optimizer boundary:

- `0x5d4c75` calls the local optimizer through slot `object+0x110`.
- On return, `0x5d4c7e–0x5d4c86` loads the optimizer's saved scalar
  `E_LAST` (`0x54c4020`) and the structure energy (`structure+0x230`).
  `0x5d4c8e–0x5d4c9b` compares them and conditionally clears
  `control+0x80`, which DWARF names `nfail`. This is an optimizer/result
  bookkeeping branch, not evidence of a true-PES landing.
- `0x5d52cb–0x5d52d3` resets `control+0x74`, named `alloptstep`, to zero and
  returns through the caller's continuation. Thus the optimizer iteration
  counter is lifecycle state for the next local optimization; it is not the
  climb release flag.

The judge itself remains the already established boundary at `0x5cf6f0`:
force convergence is compared with `para+0x2db28` (`ftol`) at
`0x5cf976–0x5cf991`, while the stop result also admits
`control+0x74 > para+0x2ddd0` (`maxoptstep`) at `0x5cfaf1–0x5cfb0a`.
`control+0x1b0` (`bfgs_must_stop`) adds another stop at
`0x5cfd9e–0x5cfda7`. None of these writes `lclimb_allstop`.

The return is consumed locally by `ssw_fixlat_mp_allopt_`: its indirect judge
call is at `0x5d47d5`, and the returned stop bit at `0x5d47e0` selects the
converged versus nonconverged bookkeeping paths. The later climb call at
`0x5caf9c` invokes the Allopt slot and does not test a returned physical
success value; it then calls `noncrystal_opt` at `0x5cb02b` and
`climb_convg` at `0x5cb03d`. Consequently a force/step/BFGS numeric stop is
eligible to proceed into the subsequent climb consumer and its possible
Allopt bias-removal path. That eligibility is separate from the climb
consumer's own `lclimb_allstop`/`lclimbstop` status and from a later unbiased
true-PES quench certificate.

## What the climb consumer actually tests

The direct climb status path is `ssw_fixlat_mp_climb_convg_` at `0x5cd130`,
called by `ssw_fixlat_mp_climb_` (`0x5ca8e0`; indirect call at `0x5cb03d`).
The following conditions are visible before the common status write:

- At `0x5cd44a–0x5cd469`, the routine computes
  `candidate_energy < reference_energy - 0.1`, using literal `0x4a45e08`,
  and masks that bit with `~control+0x1bc` (`multi_pes`). This is accumulated
  into the status mask at `0x5cd63a`.
- The Gaussian/index guards are separate bits. `r12d` is compared with the
  structure field `+0xf4` (`ng`) at `0x5cd5fe–0x5cd60c`; an integer local is
  compared with `para+0x2dd20` (`ngaus_relax`) at `0x5cd7bc–0x5cd7cc`, and
  with `para+0x2dd24` (`ngaus_relax_ini`) at `0x5cd9d5–0x5cd9eb`.
- A scalar is compared with `para+0x2dd30` (`climb_stopf`) at
  `0x5cd72c–0x5cd73b` and again at `0x5cd783–0x5cd79f`. The surrounding bit
  masks also depend on the `r12d == 1` branch. The assembly does not identify
  this scalar as a force angle or prove which status bit it represents.
- The accumulated mask is written to `control+0x78` (`lclimb_allstop`) at
  `0x5cd9b8`, and the derived status to `control+0x7c` (`lclimbstop`) at
  `0x5cd9b4`.

The all-stop consumer at `0x5cb377` enters the indexed work-buffer path and,
at `0x5cb8dd–0x5cb8e4`, restores `structure+0x230` from `tene0`. If soft-mode
cleanup is enabled, `0x5cb8ee–0x5cb904` calls `del_pot_bond_`; then
`0x5cb910–0x5cb928` invokes the `Allopt` status setter. The indirect target
after the setter can now be resolved one level further. At `0x5cb910`,
`r10 <- [r13+0x38]`, where `r13` is the same polymorphic descriptor passed as
the setter's first argument; the call at `0x5cb928` is therefore
`[r10+0x188]`. The statically identified fixed-cell table at `0x53cc8c0`
has `+0x188 = 0x5c0ac0` (`ssw_fixlat_mp_set_status_`), but this disassembly
does not itself prove that the live `[r13+0x38]` equals `0x53cc8c0`; that
requires descriptor initialization or a runtime-independent producer trace.
Inside the setter, `0x5c2860` reloads the same table pointer as
`r10 <- [r13+0x38]`, and `0x5c28e1` calls `[r10+0x100]`. Under the fixed-cell
table mapping, `0x53cc8c0+0x100 = 0x53cc9c0` contains `0x5ad9c0`,
`class_struc_mp_init_bfgs_`, so the candidate target is optimizer-state
initialization rather than an E/F calculation; the live descriptor identity
remains an explicit boundary.

`class_struc_mp_init_bfgs_` receives the freshly constructed record descriptor
from `0x5c287f–0x5c28e0`; it calls that record's table slot `+0` at
`0x5ada0f`. The record table written at `0x5c28b5–0x5c28b9` is
`0x53cd048`, whose `+0` entry is `0x5b1c30`,
`bfgs_class_mp_start_`. That routine initializes BFGS state and allocatable
storage; its inspected prefix contains no calculator, PES, force, or energy
evaluation. Thus, conditional on the fixed-cell table identity, the resolved
chain is `climb -> set_status(+0x188) -> init_bfgs(+0x100) ->
bfgs_start(+0)`, ending at optimizer-state initialization. It is direct
evidence for bias/state cleanup and an Allopt transition, but it does not
prove that the next action is a true physical quench. The first post-setter
E/F call remains outside this resolved chain and requires tracing the
continuation after `0x5c28e8`/the enclosing caller; resolving the descriptor
producer is also required before promoting the conditional target to a live
runtime claim.
The separate `lclimbstop` path beginning at `0x5cb9ae` copies/reallocates
buffers but has no `Allopt` setter in the inspected interval.

## One level after the climb returns

The fixed-cell `ssw_fixlat_mp_climb_` call site at `0x5be6e3` returns by
jumping to the common continuation `0x5be8a6`. That continuation first calls
descriptor slot `+0x38` at `0x5be8c7`, with `rdi=object descriptor`,
`rsi=object+0xe0` (cell), and `rdx=object+0x170` (coordinate descriptor).
For the statically mapped fixed-cell table `0x53cc8c0`, slot `+0x38` is
`0x59dbf0`, `class_struc_mp_update_str_`. Its body updates the object
cell/coordinate descriptors and may reallocate storage (`0x59dc9e`), but the
inspected prefix has no calculator, PES, force, or energy call. After its
return at `0x5be8cd`, the caller only computes status bits from existing
fields (`0x5be8d3–0x5be91c`) and returns; the all-stop debug branch at
`0x5cb92e–0x5cb9a9` likewise contains output calls only.

Within this one-level continuation the order is therefore
`Allopt setter -> init_bfgs -> bfgs_start -> climb return -> update_str`, with
no newly visible E/F evaluation before or after `update_str`. The first real
post-Allopt E/F entry is in the enclosing caller beyond this `ssw_move`
continuation, so the current static evidence cannot decide whether it is
before an Allopt first step or after each step. A minimal reproduction should
trace one fixed-cell transition at the backend boundary: record the last E/F
request before `0x5be6e3`, the `0x5be8c7` return, and the first E/F request
after `ssw_move` returns, together with Allopt status and structure
coordinates. No lifecycle or refresh rule is inferred from the absence of a
call in this slice.

## Enclosing `run_ssw_` boundary

The direct enclosing caller is `run_ssw_` in `analysis/run_ssw_.asm`. It
constructs the fixed-cell `SSW_A` descriptor immediately before the call:
`0x530732–0x530739` stores the object address, `0x53075d–0x530764` stores
`_DYNTYPE_PACK_16` (`0x534ef20`), and `0x53076b–0x530772` stores the type-bound
table `0x53ca680`. It passes the descriptor address in `rdi` at
`0x530818–0x53081f` and calls `ssw_fixlat_mp_ssw_move_` at `0x530822`.

The return continuation is concrete but contains no physical evaluation:
`0x530827–0x530836` tests `control+0x150` and may skip directly to `0x53095f`;
the enabled branch tests `para+0x2ddec` and `SSW_A+0x1b30`, optionally calls
`for_inquire` at `0x5308d7`, and then `0x5308f7–0x53095a` broadcasts
`control+0x14c` with `mpi_bcast_`. The subsequent path is allocatable cleanup
and status handling. A direct `cal_pes_` call is absent from this `run_ssw_`
disassembly. Therefore the static order is only: `ssw_move` returns, then
status/logging/MPI handling; it does not expose the first post-return E/F call.
The E/F entry is behind the next backend/dispatcher boundary, so this audit
cannot decide whether that evaluation is before the first Allopt step or after
each step. A trace that resolves the live `SSW_A` table and records calls at
`0x530822` return, the first subsequent backend dispatch, and the E/F oracle is
the smallest evidence that would close this lifecycle question.

## Reverse-communication caller order

The enclosing loop resolves the missing caller relation one level above
`run_ssw_`. In `main_lasp_loop_` (`0x4de0f0`), the branch at
`0x4debd4` first calls `module_str_mp_bcast_str_` (`0x4b26e0`), loads the
absolute address `0x4989e0` for `cal_pes_` at `0x4debe1–0x4debe8`, and calls
it at `0x4dec05`. The same loop then loads `move_` (`0x4ed3d0`) at
`0x4dec52` and calls it at `0x4dec5e`.

`move_` constructs the reverse-communication argument record and calls
`run_ssw_` directly: `0x4ed911–0x4ed918` loads `0x51cba0`, and
`0x4ed989–0x4ed997` passes the record and invokes it. On return, `move_`
copies status/coordinates and returns to the caller. Thus the resolved
outer order for a normal iteration is
`cal_pes_ -> move_ -> run_ssw_ -> return`, with the next loop back-edge
reaching the next `cal_pes_` call. The loop's branch at `0x4deb38` tests the
status bit and can jump back to the `run_ssw_` setup at `0x4de3b9`; this is a
control/retry path, not evidence of an additional E/F call inside
`run_ssw_`.

This establishes that the physical E/F oracle is called before the movement
dispatch in the resolved outer iteration, while the next oracle call occurs
only after `move_`/`run_ssw_` return and the loop repeats. It still does not
prove whether `cal_pes_` is reached after every accepted Allopt substep inside
`run_ssw_`; that inner schedule remains behind the reverse-communication
record and backend state. The strongest static claim is therefore the
outer-iteration order, not per-Allopt-step lifecycle semantics.

## Consequence for the CuO and Fe7C3 failures

The static chain now supports three distinct boundaries: optimizer stop
(`ftol`, `maxoptstep`, `bfgs_must_stop`), climb status accumulation (`ng`,
`ngaus_relax*`, `climb_stopf`, energy comparisons), and the later Allopt/state
cleanup path. It does not establish that a max-iteration optimizer stop is a
successful escape, nor that an NG or `climb_stopf` bit produces a certified
physical landing. Therefore the existing CuO/Fe7C3 no-landing records cannot
be reclassified as native-style released landings from this evidence.

The smallest useful future evidence would record, for one native climb, the
tuple `(ng, alloptstep, opt, lclimb_allstop, lclimbstop, structure energy,
force norm)` immediately before the `climb_convg` return and again after the
`Allopt` setter, followed by the first post-setter E/F evaluation. That would
close status polarity and whether a true quench follows without introducing a
new threshold or release rule. Until that trace exists, no lifecycle change or
real comparison experiment is justified.

Sources: `analysis/judge-convergence.asm`,
`analysis/allopt-convergence.asm`,
`analysis/kernel-ssw_fixlat_mp_climb_convg_.asm`,
`analysis/kernel-ssw_fixlat_mp_climb_.asm`, and
`analysis/kernel-dwarf-member-offsets.txt` in
`/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909`.
