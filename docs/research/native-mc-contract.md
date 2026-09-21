# Uploaded ELF MC state contract (2026-09-09)

This is an independent Python translation of a bounded native acceptance domain,
not an equilibrium sampling method, a default policy change, or a complete SSW
reproduction. It is **not wired into the paper-reference driver**. It requires
explicit `energy_tol`, `maxtrap`, current state and one uniform variate per call.
There are no newly fitted search parameters.

## Evidence and exact recurrence

External source root:
`/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909`.

- `GA-SSW_program/lasp`: `ssw_commsub_mp_metropolismc_`, address
  `0x57e820`; assembly in `analysis/ssw_commsub_mp_metropolismc_.asm`.
- `analysis/selected-dwarf.txt`: subprogram DIE `aa248`, source line 1422;
  parameter order `energy1`, `energy2`, `temper_loc`, `laccept`.
  `NSAME` DIE `ab0ad`, static location `0x78e1588`, source line 1430.
- `analysis/kernel-dwarf-member-offsets.txt`: parameter offsets `0x12c` =
  `maxtrap`; `0x2de98` = `energy_tol`.
- Caller `make_decision` at `0x5d1940–0x5d195c` passes stored energy0
  (`+0x1ac0`) and current energy (`+0x230`) directly; no division there.
- `__powi4i4` at `0x4923ec0`, independently disassembled with objdump.

For delta = energy2 - energy1 and current integer state s:

1. Always call `rd_numb` (even downhill/equal moves), instruction `0x57e841`.
2. Define near = abs(delta) < energy_tol (strict inequality, `0x57e877`).
3. s_work = s + 1 if near, otherwise s (`0x57e87d–0x57e88d`).
4. increment = Fortran/int32 power(10, s_work - maxtrap)
   (`0x57e893–0x57e89e`), T_eff = input T + increment.
5. Native exponential weight is precisely

   `exp(((delta / 20.0) * 96485.0 / -8.314) / T_eff)`.

   Constants are read from ELF load segments at `0x4a43848`, `0x4a43850`,
   `0x4a43858`; operations at `0x57e8ac–0x57e8d2`.
   Energies at the ASE boundary are eV and T is K. This fixed divisor **20**
   is a native fact, not an inferred per-atom normalization; its scientific
   rationale is not established. No atom count enters this function.
   Do not replace the constants by ASE kB and claim native equivalence.
6. Accept if delta <= 0 or uniform <= weight; equality accepts
   (`0x57e8da–0x57e907`). Native Fortran true is -1, false is 0.
7. Reset s to zero only if accepted AND abs(delta) >= energy_tol;
   otherwise retain s_work (`0x57e908–0x57e919`). Near-equal rejected moves
   increment the counter too. Different-energy rejected moves do not reset it.

Input temperature is never mutated: heating is a per-decision effective value.
For maxtrap=2, repeated exact equal energies from s=0 give increments
`0, 1, 10, 100, ...`, not fractional values below the trap threshold.
`energy_tol` is an energy-difference criterion, not structural basin identity.

## Integer arithmetic and supported domain

The Intel runtime returns zero for negative exponents with base ten. Positive
powers use low-32-bit multiplication. A direct executable copy of the original
runtime function returned:

| exponent | integer result |
|---:|---:|
| -100, -2, -1 | 0 |
| 0 | 1 |
| 1 | 10 |
| 9 | 1000000000 |
| 10 | 1410065408 |
| 11 | 1215752192 |
| 12 | -727379968 |
| 32 | 0 |

`native_power10` reproduces these arithmetic semantics including wrap. The MC
wrapper supports finite energies, finite positive input/effective temperatures,
nonnegative finite tolerance and nonnegative signed-int32 state/maxtrap. It
rejects counter overflow and nonpositive effective temperatures rather than
quietly substituting a different heating rule. These are explicit domain
restrictions versus the original binary, whose arithmetic can enter such states.
The immutable per-walker state replaces process-global SAVE storage; callers
must persist it and consume one random draw on every eligible MC invocation.
The enclosing native `make_decision` acceptance overrides and geometry filters
are outside this module.

## Executable differential check

`tests/standalone/test_native_mc.py` contains an optional native instruction
oracle. It reads the uploaded ELF, verifies MC/runtime instruction slice identity,
copies the original code into an anonymous executable mapping and relocates
RIP references. This needs no debugger or ptrace and does not launch LASP jobs.
The original integer-power code is retained; `rd_numb` is replaced by an explicit
scalar callback and `exp` by host libm. Mutable parameters and NSAME live in the
isolated mapping. This checks original MC instructions, but does NOT establish
original RNG sequence or Intel-versus-system libm bitwise parity.

Run from the research worktree:

```bash
PAMSSW_NATIVE_MC_ELF=/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp python -m pytest tests/standalone/test_native_mc.py -q
```

Actual result: **11 passed**, including **224 native/Python comparisons** of
acceptance and post-call NSAME, covering equal, near, exact tolerance, downhill,
uphill, rejection, accepted reset, probability endpoints, heating threshold and
positive wrapped/zero wrapped integer-power regimes. All 224 calls consumed a
random draw, including downhill. The first native-grid attempt included a
negative-effective-temperature overflow state and correctly hit the Python domain
restriction; the final grid uses only the declared supported domain. Separate
unit coverage requires the rejected domain to remain explicit.
Without the environment variable the reference-free suite reports 10 passed,
1 skipped. The implementation never reads or executes an ELF at runtime.

These checks validate recovered arithmetic and control flow only. There is no
new real-system end-to-end result, no efficacy evidence for this MC policy and no
claim that the full native walker has been reproduced.
