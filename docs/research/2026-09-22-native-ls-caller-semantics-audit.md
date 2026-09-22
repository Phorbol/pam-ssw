# Native LS caller semantics audit (2026-09-22)

This bounded static audit checks the chain requested for the independent
`NativeLSRuntime`: soft prequench exit, direction/climb, MC decision, and the
strength/bond-count update. It uses the pinned LASP ELF
`/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp`,
the saved assembly under
`/home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity/docs/research/native-curvature-evidence/`,
and the LS-SSW paper, DOI [10.1021/acs.jctc.4c01081](https://doi.org/10.1021/acs.jctc.4c01081).

## Confirmed ordering and behavior

The native `ssw_fixlat_mp_ssw_move_` path saves the true starting energy when
`STEP_OPT_SOFTPES == 0` (`kernel-ssw_fixlat_mp_ssw_move_.asm`,
`0x5bd70a-0x5bd71e`) and writes
`1000*(E_after-E_before)/N` to `biasperatom_save` on either force or
`optsoftmax` exit (`0x5bf699-0x5bf6f4`). Thus an iteration-limit exit is an
eligible measured response in the native path, while an arbitrary backend
failure is not established as eligible.

The caller slice directly shows the prequench BFGS call at `0x5bdb72`, followed
on the inspected path by the indirect random-mode call at `0x5bfab1` and the
`CBD` status snapshot (`native-ls-prequench-caller.asm`). This confirms the
prequench-to-direction boundary only. The saved slice does not follow all
`iflag`/reverse-communication exits or alternate entrances.

The paper states that local softening precedes climbing and relaxation, and
that the relaxed trial is then subjected to Metropolis MC; if accepted, it
replaces the current minimum (paper Sec. 2.4, Steps 7–9). Eq. 15 is the
adaptive law in Sec. 2.3, rather than the workflow citation. The
Python driver follows that explicit paper-level state rule: it prepares the
softened start, samples/refines the direction, bare-quenches the landing,
performs MC, updates `current` only when accepted, then calls
`NativeLSRuntime.update` with that selected `current` and counts its bonds
(`paper_reference.py`, `0x`-independent source path around the MC/update
boundary; `ls_native_reference.py:60-76`).

The native MC routine itself is separately visible at
`ssw_commsub_mp_metropolismc_.asm`, `0x57e820`. It returns an acceptance flag
and updates only its `NSAME` trapping state. The inspected material does not
show a call from that routine to the LS table updater. Conversely, the static
`bond_info_init_` arithmetic proves save/zero/restore and the normal update
formula/order, but not which post-climb geometry the outer caller associates
with the update.

## Gap and decision

There is no evidence-supported local defect to patch in the current Python
caller. The unresolved question is whether the release updates `B` before MC
on the trial landing, after MC using the accepted current seed, or through a
caller path not captured by the saved slices. Existing Python behavior is an
explicit paper-reference convention, and its `caller_convention` field says
so; changing it to claim native parity would exceed the evidence.

The smallest safe code action is therefore no algorithm/default change. Keep
the current post-MC selected-current convention and its telemetry, while
retaining the distinction between measured response eligibility and convergence
qualification. A future code change should be considered only after the
following discriminator is obtained from the ELF: a complete outer caller
slice spanning the return from `ssw_move_`, the MC decision flag, the call (or
inline body) that writes `bond_ener_list`/`BONDNUM_SAVE`, and the geometry or
neighbor-count pointer supplied there on both accept and reject branches.

This note does not claim native caller parity or a scientific performance
result. No binary execution, PES run, scheduler job, or parameter change was
performed.

## Follow-up outer-caller slice

The local extracted source for the cited paper is
`/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/literature/215.txt`.
It is the 2024 LS-SSW article by Tong Guan, Cheng Shang, and Zhi-Pan Liu
(DOI 10.1021/acs.jctc.4c01081), not the 2013 SSW paper. The relevant excerpt is
the LS-SSW workflow at text lines 308–367 (two-column extraction), Sec. 2.4:
local softening/pre-relaxation, climbing and relaxation, then “Step 8: accept
or reject ... according to the Metropolis MC scheme”. Eq. 15 is separately in
Sec. 2.3 and defines the adaptive next-step penalty from the prior step and
the next neighbor counts. Eq. 15 does not itself specify MC caller order; the
MC-first/selected-current interpretation is an inference combining the Sec.
2.4 flowchart with the existing Python contract (the extracted PDF page has
the printed journal page header but no final pagination).

The outer static slice adds one useful ordering result. `run_ssw_.asm`
`0x530822` calls `ssw_fixlat_mp_ssw_move_` directly. In the inspected
`ssw_fixlat_mp_make_decision_` body, `0x5d195c` calls the native MC routine;
that routine writes the returned decision flag at its caller-provided result
slot (`0x57e8f4–0x57e908`) and has no LS-table call. The MC result is then
copied into the caller's decision state (`0x5d1961–0x5d1996`) before the
accept/reject-specific continuation. No direct `bond_info_init_` or
`bond_counter_` call appears immediately after the MC result handling in this
inspected body. Because the outer dispatch is indirect, this does not prove
geometry-pointer restoration or prove that no such call occurs in every
continuation path.

The next direct LS entry visible in the same lifecycle is the beginning of a
subsequent `ssw_fixlat_mp_ssw_move_`: its prequench setup calls
`pot_bond_mod_mp_bond_counter_` at `0x5bcf43`, which in turn enters
`bond_info_init_` (`ls-bond-counter.asm`, `0x6c4cda`). Therefore the available
outer evidence is only consistent with the existing Python convention that
the MC-selected current seed is the geometry used at the next LS update. The
accept and reject geometry-pointer restore branches are not proven by this
slice, so this remains a bounded caller-contract regression rather than a
claim of full binary parity.

For reproducibility, the minimal static slices used for this discriminator
are:

```sh
ELF=/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp
objdump -d -Mintel --start-address=0x57e820 --stop-address=0x57e932 "$ELF"
objdump -d -Mintel --start-address=0x5d18e0 --stop-address=0x5d19b0 "$ELF"
objdump -d -Mintel --start-address=0x5bcf20 --stop-address=0x5bcf60 "$ELF"
```

These reproduce the MC flag write, caller decision-state copy, and next
bond-counter entry. They do not resolve the indirect continuation or establish
the selected geometry pointer on either branch; that requires bounded vtable
target tracing or an equivalent dynamic-free call-graph recovery.
