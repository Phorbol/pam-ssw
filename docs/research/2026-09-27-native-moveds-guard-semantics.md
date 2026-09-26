# Native move guards: corrected operands and relevance to LJ diagnostics

2026-09-27. Bounded static/isolated-instruction audit of uploaded LASP ELF
SHA256 `bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704`.
No native main, protection modification, PES calls, or core algorithm changes.
This corrects `native-moveds-scale.md` and the retry-probe interpretation.

## Operand chain

- `0x5c71e3–0x5c741c`: sum Cartesian squared differences per atom, square root,
  atomwise maximum into `[rbp-0x40]`: `Dmax=max_i ||delta R_i||`.
- Calls at `0x5c67b3` and `0x5c7449` return before/after distance scalars into
  `[rbp-0xd8]` and `[rbp-0xd0]`, respectively.
- `0x5c7452–0x5c748e`: retry if short flag, `Dmax > disp_perstep`, or
  `d_before*(1-bonddisp_perstep) > d_after`. Scalar equality passes.
- Parser get_real calls `0x68d2b7/0x68d32b` use keys at `0x4a4a538/0x4a4a54c`,
  destination offsets `0x2e1f0/0x2e1f8`, defaults at `0x4a49950/0x4a49958`:
  **2.0 and 0.25**. `get_real` copies rcx's default before scanning input
  (`0x60f390–0x60f3a5`). Actual input may override them.

[Executable probe](../../research/ga_ssw/evidence/lj38-height-origin-20260927/check_native_guards.py)
checks literal defaults and five native branch cases: the saved extreme first
move, relative equality/rejection, maximum-displacement equality/rejection.
[Result](../../research/ga_ssw/evidence/lj38-height-origin-20260927/native-guard-instruction-check.json):
5/5 pass. Short flag is deliberately false; this is not full native move validation.

## Separate short-distance flag

The second callee argument is the `iza` array, structure field +0x8 in
`analysis/kernel-dwarf-member-offsets.txt`. At entry it is saved at -0xc8;
at `0x6da87d` it becomes r13. Integer tests against 1 and 20 use that pointer.
`$XACNA` supplies floating-point coordinates; calling it a class array was wrong.
The sorted values saved as BONDNAME are `iza` values, not pair indices.

Calls `0x6da89a/0x6da8ec` obtain two species radii. The observed per-image
predicate uses scaled distance d and the following strict lower bounds:

- unconditional 0.5 (`0x6da9f2` reloads xmm8; it no longer contains .7 times radii);
- 0.65 if either iza value equals 1;
- 0.9 if both iza values exceed 20;
- 0.8 if both are below 20 and neither equals 1;
- 0.7 times the radius sum if both exceed 20;
- 0.6 times the radius sum unconditionally.

Constants were read from ELF at `0x4a4c988`, `0x4a4ca08/10/18/20`, and
`0x4a4c9a8`; branches are `0x6dab7e–0x6dac3a`. The underlying radius table,
complete periodic coordinate preparation, MPI reduction and effective input
settings have not been executed together here. These bounds are **not** a
universal ASE guard or a length-scale-independent LJ criterion.

Raw disassembly source: `docs/research/native-moveds-evidence/present-tooshort-core.asm`
in the existing `ga-ssw-behavior-parity` worktree. Caller source:
`/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/analysis/kernel-ssw_fixlat_mp_moveds_.asm`.

## Decision-relevant negative result

[82 saved direct-move geometries](../../research/ga_ssw/evidence/lj38-height-origin-20260927/native-direct-guard-geometry.json):
maximum Dmax 0.582185; minimum same-pair distance ratio 0.751169; minimum
closest-distance ratio 0.839701. Thus neither default scalar check rejects
these displacements under the same Cartesian units/fixed pair set.
This excludes the separate species flag and differing periodic images.

The initial high-repulsion LJ38 move is accepted by the conditional native
branch probe. Therefore copying native scalar backtracking is not an evidenced
fix for that observed pathology. The height-only followup also leaves it
fragmented; coupled PAM height/width helps only that one tested escape.
Keep existing defaults. Close this panel without another threshold/width sweep.
