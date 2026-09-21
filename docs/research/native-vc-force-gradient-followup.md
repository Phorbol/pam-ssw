# Native VC force/gradient and allowed-DOF boundary (static audit)

This is a zero-PES, read-only audit of the 20260909 `lasp` ELF.  It does not
claim that the native biased optimizer has been reproduced.

## Shortest confirmed BFGS data chain

DWARF identifies the main structure members used here as `cart` at `+0x170`,
`fa` at `+0x1d0`, `energy` at `+0x230`, and `bfgsopt` at `+0x1230`
(`analysis/kernel-dwarf-member-offsets.txt:517-522,370`).  In the fixed-cell
optimizer caller, `ssw_fixlat_mp_ssw_move_` constructs a local BFGS argument
record at `0x5bdaed` and calls `bfgs_class_mp_bfgsdriver_` at `0x5bdb72`:

* `0x5bdb20` sets record[0] to `object+0x1230` (`bfgsopt`);
* `0x5bdb36` forms `object+0x230`, then `0x5bdb53` loads from `+0x1d0`, so
  the fourth register argument is the `fa` array;
* `0x5bdb4c` loads `object+0x170`, the `cart` array, into the third register
  argument; the second register argument is the object pointer at `0x5bdaf8`.

At the callee entry (`bfgs_class_mp_bfgsdriver_`, `0x5b2970`), `r10=[rdi]`
therefore points at `bfgsopt`; its vector-array fields are consumed through
`+0x48`, `+0x90`, and `+0xd8` (`0x5b2a81-0x5b2ae0`).  The subsequent
`bfgs_basics_mp_lbfgs_` call at `0x5b2f50` receives those prepared arrays and
the parameter pointers at `object+0x180..0x1c0`.  This proves the
cart/force-array hand-off and the optimizer boundary.  It does not identify
the scientific sign convention of every BFGS vector or a VC cell-force map.

The extracted `ssw_crystal_basic_mp_ssw_move_` listing contains no direct
`bfgsdriver` call; its optimizer/allopt calls are indirect.  The fixed-cell
chain above is therefore a type/layout anchor, not evidence that the same
record is populated by the VC caller.

## What is and is not evidence for allowed DOF

The complete text disassembly was searched for the exact displacement
`+0x2db50` (`lnoatom_incell`), as well as `+0x2db58`, `+0x2db18`, and all
references in the extracted SSW/BFGS kernels.  `+0x2db50` has **no executable
consumer** in this ELF text.  The only direct parameter consumers in this
family are:

* `0x5aa010` reads `sswsteps` (`+0x2daf8`) while printing parameters;
* `0x5aa1e9` reads `ds_cell` (`+0x2db18`) while printing parameters;
* `0x5e8112` reads `ratio_atomcell` (`+0x2db58`) in `get_random_mode0_`;
* `0x5ece5f` reads `ds_cell` in `moveds_`.

Thus `lnoatom_incell` cannot currently be used as a native allowed-DOF mask.
The parameter parser writes it only as part of the input record (the parser
family begins at `0x6878xx`); no later load reaches a force/gradient or BFGS
array in the available ELF.  `force_factor` (`para+0x2ddb0`) and
`stress_factor` (`+0x2ddb8`) are likewise parameter fields, not proof that a
cell-force projection is applied in this caller.

The observed `bfgs_class_mp_bfgsdriver_` branch at `0x5b29b9` tests a bit at
`bfgsopt+0x1a8` and, if set, loads the six cell components into the BFGS
`CELL` common block (`0x5b29c7-0x5b2a48` and again at `0x5b3116-0x5b31b0`).
This is a conditional cell-data path, but the static chain does not connect
that bit to `lcellmove`, `lnoatom_incell`, or a force/stress projection.  It
therefore cannot justify claiming that biased VC relaxes all atomic and cell
DOFs, or that it freezes either subset.

## Boundary and next minimal audit

The real missing link is the producer of `bfgsopt+0x1a8` and the definitions of
its `+0x48/+0x90/+0xd8` vector fields in the VC (`ssw_crystal_basic`) caller.
An isolated follow-up should trace those fields from the VC `allopt` entry to
the first `bfgsdriver` call, with synthetic descriptors only, and assert
whether the six cell entries and the `fa` entries are populated or masked.
Until that link is recovered, production code should not infer VC allowed DOF
from `ratio_atomcell`, `lcellmove`, or the unused `lnoatom_incell` field.

## Cell-enable flag and scale provenance

The flag controlling the cell branch is now closed.  DWARF names structure
offset `+0x628` `lvariable_cell` (`kernel-dwarf-member-offsets.txt:430-435`).
`class_struc_mp_init_bfgs_` (`0x5ad9c0`) receives a structure descriptor in
`r12` and an optimizer descriptor in `r13`; at `0x5ada20` it loads
`eax=[r12->object+0x628]` and writes it to `bfgsopt+0x1a8` at `0x5ada3d`.
The same routine writes `para+0x2ddb0` and `para+0x2ddb8` to optimizer fields
`+0x1b0` and `+0x1b8` (`0x5ada28-0x5ada4b`).  DWARF names these parameters
`force_factor` and `stress_factor`.

The VC construction caller proves the descriptor identity: in
`ssw_crystal_basic_mp_construct_cssw_`, `0x5e1b34` forms
`r13+0x1230`, stores it as the second descriptor's data pointer at
`0x5e1b9b`, and calls `init_bfgs` at `0x5e1bca`; the first descriptor carries
the same object `r13` (`0x5e1ade`).  Therefore VC initialization sets
`bfgsopt+0x1a8 = object.lvariable_cell`, rather than deriving it from
`ratio_atomcell` or `lcellmove`.

In `bfgsdriver`, `test bfgsopt+0x1a8` at `0x5b2a89` selects the cell-enabled
branch.  That branch multiplies the ordinary vector entries by `+0x1b0`
(`0x5b2b21-0x5b2b64`) and the final three cell entries by `+0x1b8`
(`0x5b2c91-0x5b2ca5`).  This closes a real atomic-versus-cell scaling
difference and the enable predicate.  It still does not prove that the VC
caller fills those vectors from a complete Cartesian force/stress dual map,
nor that any atom subset is frozen: `lnoatom_incell` remains unused in the
executable text.

The `fixcell_climb` consumer is concrete.  In the true
branch at `0x5f25a2`, `rdi` is loaded as `natom` from `[r15]`, incremented at
`0x5f25b1`, and passed as the row/offset selector to `0x5f404e`.  The latter
loads `rdx=[r15+0x200]` (the `fa` vector length), `rcx=[r15+0x1d0]` (the
`fa` base), and initializes `xmm0=0` at `0x5f4058`.  It clears the selected
three row blocks with `memset` at `0x5f40d1`, advancing the row counter in
`r13` and `r14`; the loop terminates at `0x5f40e0` when `r13==3`.  The
remaining short-vector tail is explicitly written as zero at
`0x5f412c-0x5f4131` and `0x5f415d-0x5f4167`.  The completed branch restores
state and jumps back to `0x5f25b9` at `0x5f4195`; the alternate short-loop
exit sets `r14=0` at `0x5f41a6` and returns through `0x5f25b9` at `0x5f41bb`.

This is direct evidence that `fixcell_climb` zeros the three cell-related
force rows before the common `+0x108` optimizer callback.  It occurs during
climb setup, rather than only during a final quench.  It is still a force
input mask: the subsequent BFGS cell-enabled path reads/writes cell vector
entries and applies `+0x1b8` at `0x5b2c91-0x5b2ca5` and again in the later
cell update path (`0x5b3e03-0x5b3e28`).  No corresponding zeroing of the
LBFGS displacement output was found.  Therefore zero cell gradient does not
establish strict coordinate freezing; retained LBFGS history can still
produce a later cell displacement.  This `fixcell_climb` gate must not be
conflated with `lvariable_cell`/`bfgsopt+0x1a8`.

## Root cross-check: initialized ELF parameter bytes

Using the existing ELF PT_LOAD reader and little-endian unpacking at
`ssw_parameters_mp_para_=0x53ed7a0`, the uploaded file's initialized data has
`fixcell_climb=0` (+0x2dbe0, int32), `force_factor=0.05` (+0x2ddb0, float64),
`stress_factor=0.1` (+0x2ddb8, float64), and `ratio_atomcell=2` (+0x2db58,int32).
These are static initial bytes, not proof of effective runtime values after
input parsing or constructor overrides. In particular the optional cell-force
mask must not be described as always enabled in native VC.

## Bounded native mask probe

`research/ga_ssw/probe_native_fixcell_climb_mask.py` executes only the native
range `0x5f25a2 <= pc < 0x5f41bc` and stops at `0x5f25b9`; it does not enter the
BFGS callback, PES, protection logic, or the LASP main loop. The synthetic
Fortran-like array is five rows of three doubles: two atomic rows (indices
0--5) and three cell rows (indices 6--14). Its native indexing inputs are
`object+0x200=3`, byte stride `object+0x220=24`, and lower bound
`object+0x228=1`, so the selected rows begin at `FA+48`. The descriptor data
pointer at `object+0x1d0` is valid mapped storage; `r13` is initialized to a
descriptor containing the object pointer because the native epilogue restores
it before returning.

The evidence JSON records two guarded cases. With `fixcell_climb=0`, the
branch reaches `0x5f25b9` directly and all 15 entries remain unchanged. With
`fixcell_climb=1`, the short path reaches `0x5f404e`, `0x5f419a`, and
`0x5f4195`, executes the scalar stores at `0x5f415d`, leaves atomic entries
0--5 unchanged, and zeros cell entries 6--14. The `0x5f40d1` external memset
call is absent because the synthetic dimension is 3. This verifies the
force-input mask arithmetic and branch boundary only; it does not prove that
coordinates are strictly frozen, since later optimizer history/displacement
handling is outside the probe.
