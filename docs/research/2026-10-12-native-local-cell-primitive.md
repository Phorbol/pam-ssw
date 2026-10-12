# Native local-cell direction primitive and caller boundary

Read-only static audit of
`/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp`.
ELF SHA-256: `bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704`.
The two requested primitives are `newssw_basics_mp_get_random_crystalmode_`
at `0x6e4e50` and `newssw_basics_mp_generate_ncell_local_` at `0x6e4fe0`.
The audit answers whether the primitive and its data contract are sufficiently
closed for an independent Python implementation. It does not infer semantics
from the names, run LASP, or establish parity with a paper's cell-cycle
definition.

## Random crystal matrix

DWARF identifies the first routine's arguments at
`analysis/selected-dwarf.txt:188118-188149` as `cellx` and `fixlat`.
The disassembly at `0x6e4e7e-0x6e4eb7` calls `ssw_commsub_mp_rd_numb_` nine
times, in nested loops that write all nine contiguous `cellx` entries. Each
draw is transformed by subtracting the binary64 constant 0.5 at
`0x4a4cc70` and multiplying by 2.0 at `0x4a4ccb8`:

```text
x_i = 2 u_i - 1,        i = 0,...,8
x   = x / sqrt(sum_i x_i^2)
```

The sum-of-squares and square root are at `0x6e4ebc-0x6e4f70`; the inverse
norm is multiplied into every component at `0x6e4f75-0x6e4fc5`. The final
scale constant is 1.0 (`0x4a4cc98`). Thus the output has unit Euclidean norm
when flattened, equivalently unit Frobenius norm as a 3x3 array. It is a
normalized cube draw; it is not evidence of a uniform draw on the 9D sphere.
`fixlat` is not read in this routine's body. No Cartesian-to-fractional,
lattice-basis, or length rescaling occurs here.

## Local-cell generator

DWARF identifies the six by-reference formals at
`analysis/selected-dwarf.txt:188154-188184` as `latt`, `cell`, `ncell`,
`fixlat`, `flag`, and `flag2`. At entry (`0x6e4fff-0x6e5026`) the callee
zeros nine doubles at `ncell`. It then compares `*flag` with 1:

* If `*flag == 1`, control reaches `0x6e5381`. It draws
  `I = trunc(3 U_1)` (`0x6e5385-0x6e53a2`) and draws `Y = U_2`
  (`0x6e53a8`); a third draw at `0x6e549e` overwrites Y, so the
  effective sign variable is U_3, not U_2. It scans the three `fixlat` entries against `latt` to set
  static `MINLA`/`MAXLA` indices (`0x6e53ad-0x6e54a3` and
  `0x6e5439-0x6e5496`). If the selected length ratio exceeds 5, further
  ratio checks can select static mode `I=5` (`0x6e54b9-0x6e5538`). The visible
  code uses this compiled ratio, but the physical interpretation of the
  indices is not established here.
* If `*flag != 1`, execution skips those draws and bounds updates. It reads
  the existing static `I`, `Y`, `MINLA`, and `MAXLA` (`0x6e502c-0x6e5048`,
  and later branches). Therefore `flag==1` is the reset/re-initialize gate;
  the other branch reuses module state. These are process/module statics, not
  per-object saved fields. `flag2` is not read in the inspected body.

The initialized modes are visibly distinct. `I=0` starts from the `cell`
entries in the flat ordering `[0,1,2,6,7,8,3,4,5]` (`0x6e5330-0x6e5368`);
`I=1` uses `[6,7,8,3,4,5,0,1,2]` (`0x6e51ab-0x6e51e9`). Both paths
conditionally flip all signs when `Y > 0.2` (the threshold is the binary64
constant at `0x4a4cd28`; the tests are at `0x6e536e-0x6e537c` and
`0x6e51e9-0x6e5233`) and then normalize the nine outputs to unit flattened
norm (`0x6e5234-0x6e5321`). Mode `I=2` loads the original nine `cell` entries
at `0x6e5048` and stores them in row order `(b,a,c)` (first two rows swapped),
then follows its own sign/normalization path. It does not use `MINLA`/`MAXLA`
for placement. Mode `I=5` is the branch that handles those indices
(`0x6e50a2` onward), with further placement details intentionally left
unresolved here. Do not replace these branches with an inferred “local
strain” formula.

## Caller and stored coordinate

The containing caller is `ssw_crystal_basic_mp_gen_randommode_`; DWARF maps
the calls to `Class_ssw_crystal.F90:2219` and `:2222` (`addr2line` at
`0x607bda` and `0x6076ef`). The two call instructions are:

| Call | Caller address | Argument construction visible before call |
|---|---:|---|
| A | `0x6076ef` | `latt = this+0x358`; `cell = this+0xe0`; `ncell = scratch vector` data pointer plus computed cell-component offset; `fixlat = para+0x2dde0`; flag pointers are static literal-pack addresses `0x4a46a64`, `0x4a46a28` |
| B | `0x607bda` | same `latt`, `cell`, and `fixlat` bases; `ncell = scratch vector` (`object+0x1848`) data pointer plus computed cell-component offset; `flag = this+0x1660`; `flag2 = 0x4a46a28` |

Addresses and argument order are established by `0x60769a-0x6076ef` and
`0x607b89-0x607bda`; `addr2line` identifies the caller source lines. Direct
static inspection gives the literal-pack values `*(int*)0x4a46a64 = 1` and
`*(int*)0x4a46a28 = 0` for call A's `flag` and `flag2`. Call B's `flag` is
`this+0x1660`, whose runtime value is not established here. The callee writes
the nine local-cell components directly through `ncell`. At call B,
`0x607ba2` adds the scratch data pointer `[object+0x1848]`; its stride
comes from the same scratch descriptor. The earlier draft incorrectly
labeled this output as ssw_traj. A complete surrounding coordinate-mapping
contract is not established by this immediate pointer arithmetic.

The same caller invokes `get_random_crystalmode_` at `0x601ffa`, `0x602265`,
`0x60279f`, and `0x602a08`. The first visible call receives a temporary
output block and `para+0x2dde0`, then its three rows are copied into trajectory
workspace (`0x601fde-0x6020bc`). No lattice multiplication is visible in that
immediate handoff; the surrounding four branches are not asserted identical
beyond using the shared normalized-matrix helper.

The DWARF formals establish names and pointer ABI, not the physical convention
of `cell`/`latt` entries. The object offsets above are reported as addresses,
not assigned semantic fields beyond the `latt` member corroborated by DWARF.
The read-only literal-pack flags are passed by address; their packed storage
and the runtime value observed by the callee are not fully decoded here. The
second call's `this+0x1660` flag therefore cannot yet be equated with a
particular public option based only on its address.

## Evidence boundary and implementation decision

Established by this static evidence: the nine-draw normalized random matrix;
the local generator's explicit state-reset gate; distinct initialized-mode
branches; the two caller sites; and placement into the scratch cell-component block at call B. These results are consistent with the
separately observed dynamic local-cell/history participation, but are not an
independent dynamic validation of every branch.

This is not enough to implement a complete behavior-matched VC controller.
The saved-mode placement formulas have now been narrowed by root below;
whole-controller coordinate convention and branch timing remain incomplete. The key stateful
contract (static `I/Y/MINLA/MAXLA`, reset only when the integer flag equals 1)
also means a pure function of the current cell would not match the observed
lifecycle. A narrowly scoped implementation could be justified only after the
intended run's flag sequence and selected mode branch are fixed and the
remaining placement arithmetic is traced; do not add an empirical feedback
term to mask these unknowns. The current evidence does not close full Python
parity or authorize a new default.

## Source locations

* ELF disassembly: `lasp`, `0x6e4e50-0x6e4fdf`, `0x6e4fe0-0x6e5589`,
  `0x60769a-0x607910`, and `0x607b89-0x607afa`.
* DWARF extract: `analysis/selected-dwarf.txt:188118-188184` (helper formal
  names/signatures), with source-line mapping from `addr2line` for
  `0x6076ef` and `0x607bda`.
* Existing context: [VC scalar movement follow-up](native-vc-moveds-dscell-followup.md)
  and [cell-direction lifecycle follow-up](native-cell-direction-lifecycle-followup.md).

This audit ran only ELF/DWARF inspection commands. It made no binary, core,
MAINLINE, or existing evidence edits and submitted no computation.

## Root correction and narrowed state contract

Root separately decoded the constants with offline GDB: `0x4a4cd28=.2`,
`0x4a46a64` integer1 and `0x4a46a28` integer0. The initial draft's1.2
threshold and assignment of the extrema logic to I=2 were wrong and are
corrected above. No proposed Python formula is accepted solely from that draft.

Existing native-gaussian-caller.md identifies object+0x1660 as the Gaussian
index ng. Call B passes that integer as reset flag: the helper initializes
its static selector when ng==1 and otherwise reuses it, while recomputing the
vector from the current cell. This is state retention of a lattice-vector
permutation/shape mode, distinct from the separate normalized historical
trajectory displacement. Caller A explicitly forces reset with literal1.
The full caller branch selecting A versus B has not been dynamically logged.

From final register stores at0x6e52ec–0x6e531b, the ordinary modes in row-cell
notation are normalized `(a,c,b)`, `(c,b,a)`, `(b,a,c)` for I0/1/2, with a
global minus sign if Y>.2. I5 takes the longest selected row v, writes -v in
its old slot and +v in both remaining slots, then normalizes; it does not
use the ordinary sign branch. These are literal native heuristics, not a
universal definition of a soft lattice direction; CBD comes afterward.
Original-instruction check1755181 validates18 saved-state cases, maximum
absolute difference1.39e-17, with no mainloop or PES evaluation.
The corresponding native trace1755164 observed flag0/1/2 across the three
stages: zero-initialized I0/Y0 first, I2/Y=.0045422941381554035 drawn at
flag1, then retained atflag2. This closes this run's B-branch state timing,
not arbitrary multi-walker/global state lifecycle.
