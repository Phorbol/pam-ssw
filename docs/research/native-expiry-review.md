# Uploaded LASP expiry fragment: static review

Date: 2026-09-09. Scope: read-only examination of the uploaded ELF and the user-pasted fragment; no execution of protection code, binary patch, clock changes, username impersonation, or removal of checks. Independent Python search remains a separate implementation.

## Main conclusion

The uploaded ELF contains a compiled startup expiry check, not merely an unused message. This is a runtime distribution check, not evidence of encrypted/packed algorithm instructions or anti-disassembly. Crucially, the upper constant in the pasted fragment differs from this ELF: the paste says **20521111**, while the instruction says **20541111**. Their allowed real-calendar windows differ by one year. The uploaded ELF's inspected check allows 2026-09-09, assuming its runtime clock reports that date.

## Reading the fragment

`#ifdef FPACK` is a compile-time conditional. The macro name alone does not mean executable packing. `#ifdef MPI` selects MPI barrier/abort rather than the serial `alldone=.true.; return` branch. `cpuid==0` is a rank check in this code context, not the x86 CPUID instruction or evidence of hardware fingerprinting.

The pasted text contains obvious transcription damage (`Gall`, `cpuid==o/e`, `endit`, `returr`, broken write and preprocessor lines); it cannot be treated as compilable original source. Missing block boundaries also prevent deductions about rank participation from that text alone.

Fortran `DATE_AND_TIME(DATE=ymd)` produces `ccyymmdd` from the system real-time clock, not a Unix timestamp; see [GNU Fortran intrinsic documentation](https://gcc.gnu.org/onlinedocs/gfortran/DATE_005fAND_005fTIME.html). Reading that string as an integer preserves date ordering for valid dates but arithmetic on it is not elapsed-day arithmetic.

For the pasted test, non-expiry requires

```
2*(iymd-10000000) <= 20521111
3*(iymd-10000000) >= 30751500
```

Thus integer values satisfy `20250500 <= iymd <= 20260555`. Restricting to actual calendar dates gives **2025-05-01 through 2026-05-31 inclusive**. 20260555 is not a real date. The supplied text, if exact, would reject the current date.

The username condition `prefix5 != 'zpliu' OR prefix9 != 'liuzhipan'` is true for every ordinary fixed username: those two prefixes cannot both hold. It therefore does not exempt either name. This is a logic observation, not a proposed protection modification.

## Live ELF evidence

File: `/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp`.

- `strings -a -t x`: file offsets `0x463db50` = `liuzhipan`, `0x463dc1c` = `zpliu`, `0x463dc24` = `This DEMO LASP version has expired.`
- Debug producer flags for `main.o` explicitly include `-DFPACK -DMPI`, alongside `-O2 -g -traceback`. This is stronger than inferring compile configuration from strings alone.
- `init_` begins at `0x4dbff0`. `getlog_` at `0x4936930` populates `init_$USRID.0.1` (`0x5deb8c0`).
- `0x4dc11a–0x4dc19d`: `for_date_and_time` (`0x49a5e00`) returns the 8-character date, read into `module_pes_mp_iymd_` (`0x5d71d28`).
- `0x4dc1a0–0x4dc215`: nine-character username comparison plus five-character prefix comparison; either unequal result branches to the date test. This supports the OR behavior in the fragment.
- `0x4dcdf8–0x4dce1f`: computes `2*iymd-20000000`, compares with `0x1396eb7 = 20541111`, then computes `3*iymd-30000000` and compares with `0x1d53b0c = 30751500`. The former constant is **not** `0x1392097 = 20521111` from the pasted fragment.
- Consequently this ELF admits integers `20250500..20270555`, or actual calendar dates **2025-05-01 through 2027-05-31 inclusive**.
- `0x4dce25–0x4dcf1b`: rank-zero expired-message and date output.
- `0x4dcf1e–0x4dcf63`: calls `mpi_barrier_@plt` (`0x40c330`) then `mpi_abort_@plt` (`0x40c9b0`).

These address ranges are evidence locations, not patch instructions. Only static inspection was performed. The excerpt does not prove that no other protection or date-dependent behavior exists elsewhere. In particular, a later `init_` block at `0x4dcd5c–0x4dcda9` also reads/writes `iymd` and adjacent storage; its downstream purpose was not traced in this bounded review.

## Consequences for this research

1. **Static recovery:** this check does not prevent reading symbols, DWARF, instructions, constants, or reconstructing mathematical primitives. The uploaded ELF demonstrably retains those data. It is not an explanation for the difficulty of reconstructing Broyden state.
2. **Whole-program reference runs:** an expired build can terminate during initialization before doing search. An expiry message must not be reported as an SSW convergence failure. The different constant means the user-pasted build and uploaded build must not be conflated.
3. **Instruction-level algorithm fixtures:** isolated formula/state comparisons still only establish behavior of the exercised instructions. They do not prove the full startup path ran, validate distribution eligibility, or establish full-walker equivalence. Existing primitive agreements are not invalidated merely by discovering this separate guard, but neither are they evidence that a complete executable ran.
4. **Independent Python:** it does not invoke this ELF startup and therefore does not depend on its expiry path. Distribution controls are not mathematical SSW components and are outside algorithm recreation.
5. **Provenance:** keep the original ELF intact and preserve runtime dates plus full logs for actual end-to-end comparisons. If a reference build rejects execution, request an appropriate current build from its authors; no change to clocks or executable is needed for continued paper/static analysis and independent development.

No new scientific efficacy conclusion or original-program runtime-success claim is made here.

## 2026-09-11 downstream date arithmetic, now traced

The previously untraced block has a downstream numerical use, distinct from
the startup expiry exit. Raw evidence is in
`research/ga_ssw/evidence/native-stress-producer-review/` (`init-date-path.asm`,
`cal-pes.asm`, `date-postprocessing-static.json`). No protection was executed,
patched or bypassed for this review.

On successful `move_init`, `alldone` is tested at `0x4dcb8d`; its false branch
reads the date again. At `0x4dccb3–0x4dccb6` the read date is copied to
`enemodify`. The subsequent username comparisons can enter the arithmetic
block at `0x4dcd5c`. Its final stored `iymd` is at least 20260519, while
`enemodify` is computed by the integer instructions shown in the raw excerpt.
Static evaluation of that block for **2026-09-09 and 2026-09-11** gives
`iymd == enemodify == input date`. The other branch retains the direct copy,
also giving zero difference. This is an arithmetic conclusion, not a capture
of a complete native process's globals.

The final ordinary `cal_pes` writeback uses `D = iymd - enemodify`:

- `0x49b535–0x49b5ad`: adds `D * 3.741 / 13` to `ecurr` before storing energy.
- `0x49b652–0x49b7b1` and parallel loops: add `D * 0.012312 / 7` in the
  force-array writeback, after the existing mask multiplication.
- `0x49ba40–0x49bb41`: add `D * 0.0052312 / 5` to all nine stress components;
  `optz` separately changes the writeback to retain only the zz component.

This date term is not ordinary hydrostatic pressure (off-diagonals are also
changed) and is not a mathematical SSW mechanism to reproduce in Python.
It can be nonzero for some dates admitted by the separate startup guard:
the same static block at 20250501 gives D=10018. Thus successful startup alone
is not a general guarantee of unchanged PES output. Conversely, the inspected
date term is **zero for the dates of the current research campaign**; its
existence does not by itself invalidate our prior fixtures or imply numerical
contamination of a current-date reference run. Other initialization/runtime
paths are outside this conclusion.
