# Native moved retry-exhaustion release into Allopt

Date: 2026-10-12. Scope: static caller-path closure for the fixed-cell LASP
`moveds` retry-exhaustion branch. This note uses the archived ELF disassembly
and existing repository audits; it is not a native execution, E/F dynamic
validation, or a claim that the resulting local optimization converges.

## Resolved caller path

The raw disassembly is archived at
`/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/analysis/kernel-ssw_fixlat_mp_moveds_.asm`
(function entry `0x5c4d60`). The coordinate restoration and status handoff
resolve as follows:

| Address | Static evidence | Interpretation |
|---|---|---|
| `0x5c75e6–0x5c7659` | Forms the source from the trajectory descriptor and copies `ng-1` coordinates when `ng>1`; destination is the current structure Cartesian array. | The released geometry is the saved prior trajectory point, not the final rejected trial. |
| `0x5ca223–0x5ca290` | The `ng=1` arm selects the first trajectory point and copies its coordinates to the current structure. | First-stage exhaustion restores `trajectory[1]`. |
| `0x5c782e–0x5c7847` | Joined failure arms load the literal `Allopt` (six bytes at ELF address `0x4a45ed4`) and call vtable slot `+0x188`. The fixed-cell table maps that slot to `ssw_fixlat_mp_set_status_` (`native-gaussian-caller.md`, table at lines 36–43). | Retry exhaustion requests the `Allopt` status transition. This is the relevant join; `0x5c6748` belongs to a separate failure path. |
| `0x5c7854–0x5c785e` | Writes `1` to `control+0x120`, then jumps to the `moveds` return at `0x5c976e`. | The failed proposal returns to its caller with the Allopt control flag. It bypasses the successful direction normalization/width construction beginning at `0x5c7863`. |
| `0x5d49b2–0x5d49ca` | In `ssw_fixlat_mp_allopt_` (`0x5d46e0`), calls status slot `+0x188` with `NewStart` before initializing and entering its optimizer. | This call occurs after the moveds handoff, in Allopt setup. |
| `0x5c0c19–0x5c0c56` | `ssw_fixlat_mp_set_status_` recognizes `NewStart` and stores zero to object offset `+0x1660`. The offset is mapped to active Gaussian/trajectory index `ng` in `native-gaussian-caller.md`, lines 108–115. | The active Gaussian index is reset before the Allopt optimizer; backing array storage is not shown to be deallocated or cleared. |
| `0x5d4c75` | `ssw_fixlat_mp_allopt_` calls type-bound slot `+0x110`; the fixed-cell table maps it to `class_struc_mp_noncrystal_opt_` at `0x5a88c0` (`native-gaussian-caller.md`, lines 36–43). The adjacent Allopt setup records `bfgs_opt` substatus (`allopt-convergence.asm`). | The restored current structure is passed into a local optimizer. This is a real optimization stage, not immediate termination. |
| `0x4debe1–0x4dec05`, `0x4dec52–0x4dec5e` | The enclosing `main_lasp_loop_` calls `cal_pes_`, then `move_`; `move_` calls `run_ssw_` (`native-climb-release-followup.md`, “Reverse-communication caller order”). On return, the outer loop back-edge reaches its next `cal_pes_` call. | The next outer E/F dispatch follows the `run_ssw_`/`move_` return. With `ng=0`, the active Gaussian index is released before that outer iteration. |

The broader restore arithmetic and the distinction between the failure branch
and the successful width path are also recorded in
`native-moveds-scale.md`, lines 123–158. The raw disassembly file is the
address-level source; the repository notes provide the descriptor/field
interpretation and vtable mapping. The enclosing outer-loop call order is
independently documented in `native-climb-release-followup.md` under
“Reverse-communication caller order”; its evidence resolves the ordinary
`cal_pes_ -> move_ -> run_ssw_ -> return -> next cal_pes_` order.

## Boundaries and coverage

Resolved statically: exhausted retries restore a saved trajectory geometry,
set `Allopt`, return to the caller, and reach an Allopt local-optimizer call.
Allopt invokes `NewStart` first, which resets active `ng` to zero. Thus active
Gaussian records are excluded by index before that optimizer call. The evidence
does not establish that Gaussian backing arrays are erased.

The outer-driver evidence closes the transition to a subsequent outer
`cal_pes_` dispatch after `run_ssw_` and `move_` return. Combined with the
`NewStart` reset, it resolves release of the active Gaussian index before that
outer call. It does not establish that the local Allopt optimization reached
force convergence rather than its step cap, which calculator energy/force
routine is used at every internal optimizer evaluation, or that the returned
geometry is a qualified minimum. The outer dispatch order also does not by
itself certify that every possible extra potential term is absent from the
actual E/F request. No E/F dynamic validation was run.

The existing Python callbacks documented in `pamssw/standalone/paper_reference.py`
wrap biased-quench evaluation or outer-boundary checkpoint/pause; they do not
receive a failed move outcome and restored native trajectory point. Therefore
this exact failure-to-Allopt lifecycle is not represented by those callbacks.
This note records the interface-coverage boundary only; it proposes no new
architecture or implementation.

## Direct-trial counter and Python index mapping (implementation preflight)

Root rechecked the direct branch: `0x5c6da9` sets esi=1, `0x5c6dd6` stores
it in int counter `[rbp-0x348]`, and `0x5c71d8` increments after constructing
the trial, before guards. After rejection, `0x5c749c` shrinks by0.95 and
`0x5c74be–0x5c74cb` allows another trial only when new ds>=0.1 and counter<=50.
Thus at most50 evaluated trials, not initial+50 retries. The first trial is
not gated by the0.1 lower bound. This remains the inspected direct branch.

For the independent driver, if current center is saved before trial at Python
stage k starting at0, native ng=k+1 recovery maps to saved_centers[max(0,k-1)].
This matches first-center/previous-center semantics; no rejected trial may be
inserted into the Gaussian history. Tests must check actual landing_input,
not just a diagnostic index. Python base-surface quench remains a documented
implementation choice; this static evidence is not full native E/F parity.
