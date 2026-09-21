# VC `climb_convg` energy-consumer follow-up

2026-09-11. Bounded static review of
`ssw_crystal_basic_mp_climb_convg_` at `0x5f4360`. No LASP main, PES, or
runtime execution was used.

## Direct object-field data flow

The entry loads the VC object descriptor from `[rdi]` into `rbx` at
`0x5f4377`. The energy-related reads in the complete disassembled body are:

* `0x5f44a9`: `xmm2 = [rbx+0x1b20]`, the DWARF field `energy0`.
* `0x5f44b1`: `xmm1 = [rbx+0x1b28]`, the DWARF field `tene0`.
* `0x5f4996–0x5f49a9`: `xmm1 = [ [rbx+0x1668] + index*0x690 + 0x230 ]`,
  i.e. the indexed structure record's `+0x230` scalar, then the native
  constant at `0x4a46d30` is subtracted.

The first pair is combined at `0x5f44be–0x5f44c2` as
`xmm1 = tene0 - energy0`. The result is lower-bounded by
`control+0x58` at `0x5f44d0–0x5f44e0` and retained in the local at
`[rbp-0x178]`. A second use is direct: `energy0` is copied to
`[rbp-0x108]` at `0x5f44e9–0x5f44f0`, loaded at `0x5f463b`, reduced by the
constant at `0x4a46d48` (`0x5f466c`), and compared against `tene0` at
`0x5f4690`. The resulting mask is written into `r13d` at `0x5f469e–0x5f46a3`
and contributes to the all-stop mask. Thus `energy0` participates both in
the difference quantity and in a direct threshold comparison; it is not
merely bookkeeping.

The later energy branch loads `tene0` from `[rbp-0x38]` at `0x5f4996`,
loads the indexed `+0x230` scalar at `0x5f4996–0x5f49a9`, subtracts the
constant, and executes `cmpltsd xmm0,xmm1` at `0x5f49bd`. The resulting mask
is ANDed with the threshold mask at `[rbp-0x128]` and with the inverse of
the max-step mask at `[rbp-0x120]` (`0x5f49c2–0x5f49d7`), then ORed into the
`r12d` convergence mask at `0x5f49d7–0x5f49da`.

Therefore this later branch compares **`tene0` against an indexed candidate
record's `+0x230` scalar minus a constant**. It does not reread `energy0` in
this later branch, although the earlier direct `energy0` comparison controls
the accumulated stop mask. The native operand is not evidence for the Python
`current.objective` or `work` energy anchor.

## Branch control and returned status

The derived masks are combined with force/stress and step-count conditions.
For the energy branch, `r12d` bit 0 is tested at `0x5f4a6b`; when set, the
function additionally checks the current counter against `para+0xf4` and
`para+0xf8` at `0x5f4a78–0x5f4ab4`, with `object+0x2260` (`lcellmove`)
selecting the equality polarity. The final masks are written through the
control descriptor at `0x5f4b4e–0x5f4b66` (`+0x78` and `+0x7c`). The
disassembly establishes these boolean gates and writes, but the field names
of the two control outputs are not needed to identify the energy operands.

Correction: the earlier version missed the stack copy and incorrectly said
energy0 only affected the difference. There is one object load but multiple
uses. The direct all-stop comparison and the indexed stage-stop comparison
are distinct. The energy0 producer lifetime remains only partially recovered.

Sources: `native-cell-reference-evidence/climb_convg.asm` from the uploaded
ELF; `analysis/kernel-dwarf-member-offsets.txt:439-440,446`; and
`docs/research/native-vc-convergence-contract.md`.


## Root verification: indexed reference and constant

At0x5f4610–0x5f4620, the saved index is the signed integer at object+0x1660.
At0x5f4970–0x5f49a9, the code subtracts the descriptor lower bound at+0x16a8,
multiplies by0x690, and loads record.energy at+0x230 from base+0x1668.
Thus the record index is explicit, not an anonymous initial-energy scalar.
The read-only ELF bytes at0x4a46d30 are000000000000f03f (little endian double1.0).
This particular gated comparison is therefore `tene0 < record[index].energy-1.0`,
not the fixed-cell0.1 comparison. No new Python threshold follows from this:
the record producer and exact physical/bias energy scope remain to be resolved.
Also, the result AND at0x5f49c7–0x5f49d4 includes both a stored mask and a
conditional-zero predicate, so it must not be collapsed to an unconditional
energy-only stop. This is static instruction evidence, not runtime parity.


## Root isolated-instruction verification and actual stop levels

`probe_native_vc_stop_levels.py` executes two unmodified native instruction
slices. All48 cases pass, with zero PES requests. Evidence is
`research/ga_ssw/evidence/native-vc-stop-levels.json`.

The lower-energy slice0x5f462f–0x5f46cf confirms the contribution
`(tene0 < energy0-0.1) AND (externaltp < 1e-10) AND
NOT (min(fixlat[0:3]) == 0 AND lslab_search)`.
The0.1 and1e-10 literals are read from the uploaded ELF. They are native
units/conditions, not proposed portable defaults. Other all-stop causes,
such as maximum energy excursion, are outside this isolated predicate.

The tail0x5f4a6b–0x5f4b6a confirms:
- `r13d` writes control+0x78 (lclimb_allstop).
- `r12d` writes control+0x7c (lclimbstop); all-stop implies stage-stop.
- A stage-stop alone becomes all-stop at equality with the active mode's
  Gaussian cap (para+0xf4 for atom, +0xf8 for cell).
Thus the indexed energy-minus1.0 predicate does not by itself release to
Allopt at an intermediate Gaussian index.

The caller0x5f2663 tests all-stop first. Its set path restores the saved
energy `object.energy <- tene0` at0x5f2bf9–0x5f2c02 and calls the setter with
`Allopt` at0x5f2c10–0x5f2c1f. The stage-stop-only path at0x5f2ca6 instead
writes the next trajectory record: indexN+1 at0x5f323b–0x5f3245, with
record.energy=tene0 at0x5f34c3–0x5f34d1. See the producer follow-up.
The exact energy-callback scope and all coordinate-copy branches still need
separate verification; no arbitrary last evaluated trial is treated as a
released point.

Practical implication: the next parity question concerns stopping a biased
stage versus releasing the whole walk. The previous inference that native
VC omits an outer reference-energy stop is withdrawn. Python's use of an
outer reference is not disproved by this native consumer.
