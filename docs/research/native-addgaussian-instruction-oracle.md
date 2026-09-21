# Whole-function Gaussian instruction oracle: old forces counted twice

Date: 2026-09-10. Scope: the uploaded fixed-cell `ssw_fixlat_mp_addgaussian_`, entry `0x5cda70`, including the original nested `set_thisgaussw` instructions when `climb_new` is true. This is instruction-emulation evidence, not a complete LASP execution or physical validation. No production search code was changed.

## Result and correction to the earlier static note

For the 48 explicitly constructed valid, contiguous array cases, the returned values satisfy

\[
 E_{out}=E_{in}+\sum_{j=1}^{ng} B_j,
 \qquad
 F_{out}=F_{in}+2\sum_{j<ng}F_j+F_{ng},
\]

where `B_j = W_j exp[-s_j^2/(2 sigma_j^2)]`, `F_j = B_j s_j n_j/sigma_j^2`, and the last weight is the **output** weight if the height controller was invoked. Energy agrees to `1.7763568394002505e-15`, and the force with the double-old formula agrees to `1.3322676295501878e-15`.

The earlier static note `native-gaussian-caller.md` incorrectly suggested the second subtraction lacked `s_j`. Instructions at `0x5ce437` and `0x5ce43f` explicitly load and multiply that projection before the second vector/scalar loop. The two subtractions are both the ordinary projected Gaussian force contribution. They are consecutive and execute in the oracle; the scratch trace after each pass shows the second identical increment.

This is **energy/force inconsistency**, not proof that the frozen force field itself is nonconservative: the latter is the negative gradient of a different potential, `E_in + 2 sum_old B_j + B_latest`, provided the input E/F are consistent. It is not the gradient of the energy returned by this function.

## Checks performed

- Atom counts `1, 2, 5, 15`; Gaussian counts `1, 2, 3`.
- `climb_new=False` and `True`; true branch executes the real weight-adjustment function as well.
- Both contiguous rank-two layouts `(3,N)` and `(N,3)` with correctly serialized values. The latter exercises SIMD loops and the `memset` path at N=15.
- Signed nonunit displacement projections, plus an exact `s_old=0` control (N=1, ng=3, first center equals the evaluation geometry). Its first old force is zero in both passes, contradicting the earlier missing-projection interpretation.
- 1,656 displaced-coordinate evaluations: central finite difference of returned Gaussian energy at `h=1e-5`, with the output weights frozen and `climb_new=False` for every displacement.
- For ng=1, maximum component error relative to finite differences is `2.4867041759080166e-10`.
- For ng>=2, errors range from `0.006095992808467385` to `0.22611608408673434` in the ELF's force units. They are explained by the additional old-Gaussian force, not finite-difference noise.

The input background force is a constant `0.03` per coordinate and the input energy is `-3`. The finite-difference check isolates the Gaussian part: it subtracts that known background force from the output, or equivalently adds `0.03` to the numerical negative derivative of the returned Gaussian energy. It does not falsely claim the deliberately fixed base-energy fixture has the supplied background force as its derivative.

## Executed instructions and explicit hooks

All Gaussian energies, scratch-array additions, forces, angle preparation, weight updates, energy bookkeeping, and branch decisions execute from the original ELF. No Gaussian function, force operation, or suspicious instruction is replaced.

Only these external runtime functions are hooked:

1. `exp`, `0x49207c0`: host `math.exp`, argument/result logged.
2. `acos`, `0x49206e0`: host `math.acos`, argument/result logged; no silent argument clamp.
3. `for_realloc_lhs`, `0x4970360`: a **no-op only after asserting** allocated LHS, rank two on both descriptors, identical extents and element size. All test arrays are preallocated, contiguous and conformable. Allocation/reallocation of other shapes is outside the oracle contract.
4. `_intel_fast_memset`, `0x4a10430`: byte-for-byte memset, with size logged.

Unicorn 2.1.4 is loaded only from `/tmp/pam-ssw-unicorn-probe`; it is not a package dependency. Unknown executed addresses fail the probe. Each call has a one-million-instruction and five-second ceiling and must return normally.

## Descriptor reconstruction

The Intel descriptor fields are set explicitly: data pointer `+0`, element bytes `+8`, rank `+0x20`, dimension extent/byte stride/lower bound at `+0x30/+0x38/+0x40`, then another 24 bytes per dimension. Object offsets follow the independently inspected caller/DWARF note:

- current coordinates `+0x170`, force `+0x1d0`, energy `+0x230`;
- Gaussian count `+0x1660`;
- center-record array `+0x1668`, record stride `0x690`, coordinate descriptor inside record `+0x170`;
- direction-record array `+0x16b0`, record stride `0x138`, vector descriptor at record `+0`;
- widths `+0x16f8`, weights `+0x1740`, scratch `+0x1a60`.

The object itself is passed by the original class-wrapper pointer indirection. Parameter globals set only height-control maxw=10, step=1.2, initial scale=1.5 and pi. These are controlled probe inputs, not proposed search defaults. Lower bounds are one. No assumptions about strided/reallocated arrays or additional class initialization are tested.

## Artifacts and reproduction

- Script: `research/ga_ssw/probe_addgaussian_emulated.py`.
- Full base inputs, outputs, weights, runtime-hook logs, scratch traces, and finite-difference coordinate/sign/step and numerical outputs: `research/ga_ssw/evidence/native-addgaussian-emulated/result.json`.
- Seed: 20260910. Uploaded ELF SHA256: `bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704`.

```bash
PYTHONPATH=/tmp/pam-ssw-unicorn-probe:. python research/ga_ssw/probe_addgaussian_emulated.py \
  --elf /home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp \
  --output research/ga_ssw/evidence/native-addgaussian-emulated/result.json
```

## Consequence for the project

Retain conservative paper Gaussians in the ASE implementation. If a native-behavior profile is needed, keep this output mismatch explicitly labeled and opt-in; do not silently encode it in a calculator claiming consistent energy and forces. The mismatch may affect an optimizer relying on energy/gradient consistency and the angle-based height controller after multiple Gaussians, but no actual convergence or search-effect claim follows from this bounded oracle.

Before calling it an end-to-end release bug, inspect whether callers apply a compensating operation (none is established here) and, if feasible, record E/F from a full native run at a frozen multi-Gaussian state. An internal subroutine result and the complete search trajectory remain different evidence levels. The important current conclusion is that exact release behavior and a conservative Python potential cannot simply be treated as interchangeable.
