# Native setconstraints: warm-cache instruction oracle

Date: 2026-09-10. This supersedes the unresolved orthogonalization suspicion in [the static rigid-motion audit](native-cluster-rigid-treatment.md).

## Result

For the three sampled non-principal-axis, nonlinear geometries (4, 7, 13 atoms), **the native function is numerically equivalent to the Euclidean orthogonal projector that removes the rigid tangent space**. Maximum matrix difference from an independent NumPy SVD projector is `4.997e-16`. Idempotence, symmetry, rigid-rotation removal, and rigid-coordinate covariance pass at floating-point precision in these cases.

The earlier static concern that the function merely normalizes Cartesian rotation modes independently was incomplete. Further address inspection locates inline Gram-Schmidt subtraction at original `simplefun.F90:1064` and `:1071`. No external QR call was needed, so absence of such a symbol was not evidence of absence of orthogonalization.

A two-atom, rank-deficient geometry behaves differently: the matrix is not idempotent or symmetric and does not equal the rank-aware projector. This is a separate degeneracy observation, not a failure of the three generic nonlinear cases.

## Scope and runtime boundary

Executable: `/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp`, SHA256 `bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704`.

The experiment executes original instructions from `ssw_commsub_mp_setconstraints_` entry `0x5790e0` until its normal return, using Unicorn. **It starts in a valid same-N warm-cache state:** six array descriptors are preallocated; NX/NY/NZ are initialized on the host to normalized Cartesian translations; FIRST is false. This corresponds to using translation caches from an earlier same-N initialization, but the cold allocation/initialization path was not itself executed or verified.

All coordinate-centroid calculations, rotation-mode construction, orthogonalization, normalization, dot products and final vector subtraction execute native instructions. No projection or geometric arithmetic is replaced by a host implementation. Only same-shape Fortran reallocation (checked no-op), memcpy and memset are hooked. No expiry logic was changed, entered, or bypassed by patching. This is isolated numerical-subroutine execution, not a complete native LASP run.

Each geometry is sampled from a seeded Gaussian distribution transformed by an anisotropic triangular matrix and translated away from the origin. These are numerical geometry probes, not physical-system efficacy tests. The independent reference uses the span of three translations and `e_alpha cross (R-mean(R))`, with rank determined by SVD and a relative `1e-12` singular-value threshold. The threshold is an explicit reference rank decision for well-separated ranks in these fixtures, not a proposed universal search parameter.

## Experiment and measurements

For each geometry, execute one random vector, all `3N` Cartesian basis vectors to reconstruct the entire native linear map, a randomly rotated coordinate/vector pair, a translated-coordinate case, and a pure rigid rotation. Total calls: `sum(3N+4)` for N=2,4,7,13 = **94 original-function calls**. Maximum-entry error is used for matrix/covariance statistics; Euclidean norm for rigid residuals.

| Atoms | Rigid rank | Max matrix difference from SVD | Idempotence error | Pure rotation residual norm | Rotation covariance error |
|---|---:|---:|---:|---:|---:|
| 2 | 5 | 7.823e-2 | 9.792e-2 | 5.549e-3 | 3.858e-2 |
| 4 | 6 | 4.719e-16 | 1.665e-16 | 7.303e-16 | 3.331e-16 |
| 7 | 6 | 4.996e-16 | 3.053e-16 | 1.064e-15 | 6.158e-16 |
| 13 | 6 | 2.220e-16 | 3.331e-16 | 2.720e-15 | 1.332e-15 |

For nonlinear cases, the returned rotation-mode Gram matrix is identity to numerical precision. Translation covariance errors are at most `9.993e-16`. This tests a finite set of asymmetric geometries and does not prove every nearly degenerate configuration is stable.

For the two-atom geometry, the returned third rotation mode has norm one although the rotational span has only rank two. It has dot products approximately `0.17628` and `-0.06865` with the first two modes, and also couples to translations. The native matrix symmetry error is `0.13463`, and translated-coordinate covariance error is `0.02608`. The evidence is consistent with normalization of floating-point leftovers after Gram-Schmidt; the static norm floor is extremely small (`1e-150`). This mechanistic explanation is an inference from the measured degeneracy and the norm branch, not a full conditioning proof.

For all cases, using the final returned translation/rotation modes, the actual map matches

\[
A=(I-RR^T)(I-TT^T)
\]

to at most `2.221e-16`. For generic nonlinear geometries these mode columns are orthonormal and mutually orthogonal, so this equals the correct rigid-complement projector. For the two-atom case they are not, which explains why `I-TT^T-RR^T` alone does not reproduce the observed map and why sequential treatment does not repair the rank problem.

## Inline orthogonalization addresses

The original source mapping in `objdump -dl -Mintel` identifies:

- `simplefun.F90:1064`, scalar multiply/subtract at `0x57c90f`, SIMD at `0x57c98d–0x57c9b1`, alternate scalar at `0x57caa0`: subtract a previously computed mode projection.
- `simplefun.F90:1071`, scalar two-component subtraction at `0x57d88d–0x57d891`, SIMD at `0x57d956–0x57d98f`, alternate scalar at `0x57daf5–0x57daf9`: subtract two preceding mode components.
- The subsequent normalized returned vectors are recorded in the evidence, establishing the intended Gram-Schmidt effect without relying only on source labels.

## Reproduction and artifacts

```sh
PYTHONPATH=/tmp/pam-ssw-unicorn-probe:. python research/ga_ssw/probe_setconstraints_emulated.py \
  --elf /home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp \
  --output research/ga_ssw/evidence/native-setconstraints-emulated/result.json
```

- Probe: `research/ga_ssw/probe_setconstraints_emulated.py`.
- Complete input geometries/vectors, rotation matrices, all reconstructed native/reference matrices, returned modes and their Gram matrices, pure-rotation inputs/outputs, covariance outputs, runtime hooks and errors: `research/ga_ssw/evidence/native-setconstraints-emulated/result.json`.
- Optional Unicorn dependency remains outside the package; no production module was modified.

## Consequence

A rank-aware Euclidean rigid-mode projector is now supported by actual native numerical behavior on the generic nonlinear fixtures, in addition to the independent geometry derivation. There is no evidence here that the project should adopt the native degeneracy threshold. The direction-only end-to-end walker can still fail for reasons involving the Gaussian potential and modified-surface optimization; agreement of this isolated projection does not establish that those later operations implement the same quotient geometry.

Remaining gaps: native cold-cache initialization, nearly linear/large-dynamic-range conditioning, all mask/periodic modes and callers, and a complete native trajectory. No global-search efficiency or physical-basin conclusion follows from this numerical oracle.
