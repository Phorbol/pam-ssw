# Fixed-cell native stage-control audit (authoritative summary)

Date: 2026-09-12. This is a static audit of the archived fixed-cell
`climb_convg_` disassembly and the current public paper reference. No native
main program, protection path, PES, or new search was run.

Address-level counter and RC details are maintained in
[`fixed-noncrystal-counter-trace.md`](2026-09-12-fixed-noncrystal-counter-trace.md)
and [`fixed-lbfgs-rc-lifecycle.md`](2026-09-12-fixed-lbfgs-rc-lifecycle.md).
Scalar parameter provenance and `maxe_height_gm` are maintained in
[`fixed-stage-parameter-provenance-audit.md`](2026-09-12-fixed-stage-parameter-provenance-audit.md).

## Closed stage-control facts

The scalar used by the native climb-stop comparison is the maximum absolute
component of the **current optimizer-side force array**, rather than a saved
bare-PES force or a force certificate produced after release. The complete
routine is `ssw_fixlat_mp_climb_convg_` at `0x5cd130`:

* `0x5cd147` loads the structure object from the incoming pointer. The loop
  loads its force descriptor fields at `structure+0x218`, `+0x228`, and the
  data/stride field at `+0x1d0`; `0x5cd1ca–0x5cd1f9` applies an absolute-value
  mask and `maxsd` to the entries.
* The resulting maximum is written to `[rbp-0x158]` at `0x5cd262` (and on the
  empty/alternate path at `0x5cd16c`/`0x5cd277` or `0x5cda61`). This is the
  scalar later reused by the status comparisons at `0x5cd521`, `0x5cd644`,
  `0x5cd72c`, and `0x5cd78b`.
* `0x5cd72c–0x5cd73b` compares that scalar strictly as
  `max_abs_current_force < [para+0x2dd30]`. The parameter object is proven by
  `0x5cd2a1` (`lea ... # 0x53ed7a0 <ssw_parameters_mp_para_>`), so
  `+0x2dd30` is a parameter field (`climb_stopf` in the existing DWARF
  mapping), not a structure field. The same comparison is repeated at
  `0x5cd783–0x5cd79f` on the alternate aggregation path.

This closes the data provenance of the force term: the native stage predicate
can be satisfied by the optimizer-side array that `climb_convg_` sees at call
time. Existing force-restoration evidence places restoration later in the
outer `climb` path; therefore this comparison cannot be called a final bare
PES certificate.

The final status writes are also now stated in their actual instruction order.
At `0x5cd9a8–0x5cd9bc`, `control` is loaded from `[rbp-0x38]`; bit 0 of the
accumulated `ebx` is tested at `0x5cd9b1`, `cmovne` sets the derived value to
`-1` at `0x5cd9b4`, and the stores are:

```
[control+0x7c] = derived lclimbstop       # 0x5cd9b8
[control+0x78] = ebx accumulated mask     # 0x5cd9bc, lclimb_allstop
```

Thus the stage/global distinction is a bitmask handoff, not a direct return
from the force comparison. In the `r12d == 1` branch, `0x5cd7a5–0x5cd7a9`
`jumps to 0x5cd9ce`; `0x5cd9ce–0x5cd9eb` compares the integer local
  `[rbp-0x1b0]` strictly greater than `[para+0x2dd24]` and ORs that result with
  the previously accumulated local status before the same final stores. Its
Its producer is now closed: `0x5cd434` loads `control+0x08`, and `0x5cd437`
  stores it into `[rbp-0x1b0]`; the control structure's DWARF name for `+0x08`
  is `climbstep`. The object value at `object+0x1660`, loaded into `r12d` at
`0x5cd32d` and `0x5cd42d`, is the nested trajectory's `ng` index. Thus the
The counter lifecycle is closed by the dedicated trace: `set_status` writes
`control+0x08 = 1` at `0x5c26a9`, and one `noncrystal_opt` dispatch increments
it once at `0x5a919e`, after the BFGS callback returns. `climb_convg` reads it
at `0x5cd434–0x5cd437`. The resulting rule is:

```text
budget = ngaus_relax_ini if ng == 1 else ngaus_relax
step_over = climbstep > budget
```

Both comparisons are signed strict-greater tests. `climbstep` counts enclosing
optimizer dispatches/RC consumptions; it is not a PES-request counter and is
not asserted to equal accepted iterates.

The remaining scalar aggregation is also explicit in the instructions. At
`0x5cd296–0x5cd2b4`, object fields `tene0` (`+0x1ac8`) and `energy0`
(`+0x1ac0`) form `tene0-energy0`, which is combined with `control+0x58` by
`maxsd`, giving `max_excursion`. The incoming/base energy may already include
LS or another upstream term. At `0x5cd518–0x5cd540`, the
three strict tests are `e_maxlimit < max_excursion`, `f_maxlimit < max_force`,
and `e_maxlimit_gm < control+0x60`. For `ng != 1`, their OR is retained; for
`ng == 1`, the initial branch masks these limit bits. The lower-energy bit is
`base_energy < initial_energy - 0.1` at `0x5cd44e–0x5cd469`, then
masked by `~control+0x1bc`. At `0x5cd840–0x5cd87f`, the later-stage comparison
is `base_energy < trajectory[ng-lower].energy - 1.0`; the `1.0` scalar
is the ELF value at `0x4a45e20`. Finally `ng == [para+0xf4]` contributes a
separate equality bit at `0x5cd5fe–0x5cd610`. These are status-mask inputs,
not independent certified-landing claims.

The returned low flag is consumed by an explicit bit-0 test at
`0x5cb043–0x5cb047`; status words must therefore be interpreted by their LSB.
The fixed caller has one BFGS/LBFGS dispatch per `noncrystal_opt` call
(`0x5a88c0` -> BFGS driver `0x5b2970`, LBFGS call `0x5b41cd`). LBFGS
can return an unevaluated `x_next` requesting a later E/G, while the evaluated
pre-dispatch input is `x_eval`; these are separate from the saved release
snapshot. Native release restores that pre-dispatch snapshot, so a returned
trial must not be called an accepted physical step.

`maxe_height_gm` is written by the run loop as candidate energy minus the
run-state `SSW_A/CSSW_A+0xb90` energy snapshot. It is that REFERENCE snapshot
which is updated through the strict lower-energy gate in `make_decision`
(`0x5d1d0a–0x5d1d1e`, store `0x5d3623–0x5d363e`), not the difference field
itself. The reference also has an `ssw_move` writer. Its complete lifecycle
must not be silently equated with a global archive minimum. Exact addresses
and provenance are linked above.

## Public reference gap

The public loop in
[`pamssw/standalone/paper_reference.py`](../../pamssw/standalone/paper_reference.py)
records a biased quench and, after an explicit true-surface evaluation, breaks
on `true_energy < current_energy` (lines 720–729 in this checkout). A failed
biased quench breaks at lines 721–723; otherwise the only exposed Gaussian
limit is the Python loop bound. It does not compute or expose the native
`max_abs_current_force < climb_stopf` stage bit, the `multi_pes` mask, the
  separate `lclimbstop`/`lclimb_allstop` outputs, or the native optimizer's
  reverse-communication trial boundary. This is a native-parity omission, not evidence that the Python
mathematical release rule is wrong.

The directly portable fact is the strict force comparison, provided the
caller defines the same optimizer-side force array and parameter units
(the parameter is a force threshold, conventionally eV/Å in this fixed-cell
contract). Porting the complete branch would additionally require the
  native `multi_pes`/status lifecycle and an explicit trial/accepted-point contract;
inventing those mappings would change the algorithm. No production change is
recommended by this static audit.

## Evidence boundary

The force-array provenance and the two final control offsets are instruction-
level facts from
`/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/analysis/kernel-ssw_fixlat_mp_climb_convg_.asm`.
The interpretation that the array is optimizer-side is independently recorded
in `docs/research/native-gaussian-consumers.md`, which notes that
`climb_convg_` reads `structure.fa` before later restoration. The audit does
not establish which later `Allopt` branch performs a true landing quench, nor
does it equate a native stage bit with a certified minimum.

## Isolated callee check

`research/ga_ssw/probe_native_climb_convg_scalar.py` executes only the callee
from `0x5cd130` to its return, with synthetic Fortran-like force and trajectory
descriptors and diagnostic printing disabled. The ELF SHA256 is recorded in
`research/ga_ssw/evidence/native-climb-convg-scalar-20260912/result.json`.
The initial result was invalid because its descriptor had not initialized
`object+0x200/+0x220`; it is retained as the invalid artifact. Version 2
initializes the verified `3x1` descriptor (`+0x200=3`, `+0x218=1`,
`+0x220=3`, `+0x228=1`) and writes saved energy at the `ng-1` trajectory
element. Sixteen cases cover `ng=1/2`, strict equality and adjacent
`nextafter` values for force/energy limits, the 0.1-eV comparison,
`multi_pes`, counter boundaries, final-ng equality, and the later 1.0-eV
saved-energy comparison. The probe records native
`control+0x7c/+0x78` beside the pure helper's scalar predicates. Native status
words are interpreted by their least significant bit, matching the native
`test ...,1` instructions. Version 2 asserts both stop and allstop LSBs against
the recovered helper. The earlier invalid
descriptor output must not be used to infer force or saved-energy behavior.

The root agent additionally executed 13 independent cases with force=0.2
above climb_stopf=0.15 to prevent force convergence from hiding other gates.
These cover first/later counter equality and +1, saved-energy ignored at ng=1
and active at ng=2, force-limit equality and nextafter neighbors, and multi_pes
suppression. All 13 native/helper assertions passed, saved in
`native-climb-convg-scalar-20260912-v2/root-unmasked-checks.json`.
No PES or LASP main calls were made. These establish the checked scalar gate
semantics, not full optimizer trajectory parity.
