# Native cluster controls: parser and selection evidence

2026-09-11. Static analysis of the same archived LASP ELF
SHA256 `bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704`.
No LASP main-program execution, PES calls or production algorithm changes.

## `globalcompress` is mapped; its downstream geometry is not yet closed

The saved parser at `0x687fec–0x68806f` passes the literal
`SSW.globalcompress` (rodata `0x4a49f54`) to `get_real`, with destination
`para+0x2db70`. DWARF calls this member `compress_mode`. Thus the earlier
statement that the input key could not be connected to this member is superseded.
This does not by itself connect the similarly named standalone function.

In `ssw_fixlat_mp_get_random_mode0_`, `para+0x100 == 5` dispatches to
`0x5c0a53`. If object `+0x1660` is zero, `0x5c0a72` obtains a random scalar
and `0x5c0a77–0x5c0aba` sets `control+0x68` true exactly when
`para.compress_mode > random`; otherwise false. Run-type 6 explicitly clears
that control at `0x5c017c`. The random helper `0x580640` calls
`for_random_number` and stores its returned double. Therefore, subject to the
usual uniform random-number contract, this is a probability selection, not
an energy penalty or an amount of coordinate compression.

At `0x5c0236–0x5c0255`, an enabled control modifies the mode construction's
local coefficient array: it zeros one slot and puts `max(1, localmode)` into
another. The complete direction generator is reached through an indirect
method; the full geometry action and ordinary/nonzero object+0x1660 branches
are not closed here. Do not infer that every step compresses coordinates by
0.0001, or that this control prevents all fragmentation.

## `Ratio_Local` is not a direct percentage test in this consumer

Parser `0x688154–0x6881bf` reads the literal `SSW.Ratio_local` into integer
member `para+0x2db54`. The mode-initialization branch at
`0x5c01d2–0x5c0203` obtains another random scalar u and computes

    localmode = 0.1 + 0.1 * Ratio_Local * u

before saving it to object `+0x1b28`. Constant `0x4a45e08` is double 0.1.
Thus the SI value 50 gives this intermediate coefficient in [0.1, 5.1)
under a [0,1) random input. The physical meaning and effect of the downstream
mode coefficients still require the direction generator; this is not proof of
50 percent localized moves, a radius in angstroms, or a universal parameter.

## Vapor enable guard

Constant `0x4a45e28` is double 100.0. Both the inspected Allopt setter and
judge test against it; the coordinate-repair call requires finite
`vapor_cri < 100`. The judge adds the local step >50 condition. See
[native-vapor-criterion.md](native-vapor-criterion.md) for the distinct logical
modes and unresolved complete component-translation/output contract.
The SI's carbon input value 1.7 is below this guard, but this does not establish
the configuration actually used by any archived native run.

## Evidence and next task

- `research/ga_ssw/evidence/native-cluster-control-parser.asm`
- `research/ga_ssw/evidence/native-cluster-control-selection.asm`
- Existing `analysis/kernel-dwarf-member-offsets.txt`: `ratiolocal`,
  `compress_mode`, `lcompress_mode`, `localmode`.
- LS-SSW SI, DOI 10.1021/acs.jctc.4c01081, sections 7.3–7.6: input values,
  not a mathematical specification of these three controls.

The next most direct correctness question remains the Allopt fragment action,
then the mode-generation consumer. No parameter or connectivity constraint is
added solely from these input names. The single-seed C60 comparison must not
be tuned retrospectively and reused as independent validation.


## Coefficient consumer closure, 2026-09-11

The pair-local branch of get_random_mode0 stores localmode in stack slot
rbp-0x60 at0x5c049b. Relative to the ten-double array starting at rbp-0x80,
this is zero-based slot4. The array is passed as rsi to the indirect generator
at0x5c0663–667. Run_type5 puts1.0 into slot1 at0x5c071b–725. This does not mean
every run selects the pair branch: additional controls and atom-group state
select slots5/6 instead, as the adjacent instructions show.

In gen_randommode, the pair-local call0x5d7d9e is followed by conditional
setconstraints and then unconditional n_normal at0x5d7dd7. Slot4 is loaded at
0x5d7dea and multiplies the normalized workspace before accumulation into the
aggregate; the aggregate is normalized at0x5d865a. Thus the consumed local
coefficient does not multiply a raw separation vector in this path. This differs
from the currently implemented2013 paper equation using the raw pair separation.
It is a concrete source distinction, not yet evidence of better search.

The isolated localatompair numerical oracle is still being corrected. Its output
is a raw3N double array; earlier attribution of zero output to an allocatable
output descriptor was incorrect. External neighbor-list helper arguments must
be supplied consistently before claiming any numerical parity or implementing
the complete native direction in the production kernel. The native neighborhood,
pair selection and constraint projection must not be replaced by guesses.

New evidence: native-cluster-control-generator/get_randommode_coefficients.asm
and gen_randommode.asm. No PES calls or default changes for this static closure.
