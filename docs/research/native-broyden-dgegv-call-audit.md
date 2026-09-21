# BRZERO4 DGEGV call boundary (read-only audit)

Date: 2026-09-12. This note traces the call at `0x6fb198` in the archived
LASP ELF from the caller's registers and stack. It does not call LASP or let
the eigensolver execute, and does not modify production code.

## Actual argument mapping

The preceding source-951/952 block shifts the 50-column `SP` and `GP` work
matrices (`0x6fb0b8–0x6fb0ef`). At source line 985 the SysV/Fortran argument
setup is:

| DGEGV argument | location at `0x6fb198` | value/pointer |
|---|---|---|
| `JOBVL` | `rdi` | `0x4ba12ec`, one byte `'N'` |
| `JOBVR` | `rsi` | same `0x4ba12ec`, one byte `'N'` |
| `N` | `rdx` | `QWORD [rbp+0x20]` |
| `A` | `rcx` | `GP = 0x79236c0` |
| `LDA` | `r8` | `0x4a4cff0`, integer `50` |
| `B` | `r9` | `SP = 0x79288c0` |
| `LDB` | `[rsp+0x00]` | `0x4a4cff0`, integer `50` |
| `ALPHAR` | `[rsp+0x08]` | `AUXR = 0x792dac0` |
| `ALPHAI` | `[rsp+0x10]` | `AUXI = 0x792dc60` |
| `BETA` | `[rsp+0x18]` | `AUXBET = 0x792de00` |
| `VL` | `[rsp+0x20]` | `AUX = 0x792dfa0` |
| `LDVL` | `[rsp+0x28]` | integer `1` |
| `VR` | `[rsp+0x30]` | same `AUX` |
| `LDVR` | `[rsp+0x38]` | integer `1` |
| `WORK` | `[rsp+0x40]` | same `AUX` |
| `LWORK` | `[rsp+0x48]` | `0x4a4cff8`, integer `2500` |
| `INFO` | `[rsp+0x50]` | local `[rbp-0x2d4]` |

The two hidden character lengths are written at `[rsp+0x58]` and
`[rsp+0x60]`, both with integer value `1`. They are therefore ordinary
one-character arguments at the ABI boundary; the long literal text is only
the surrounding constant pool object.

The wrapper at `0x149d0c0` confirms the first six arguments by copying
`rdi/rsi/rdx/rcx/r8/r9` into its saved argument registers, then reads the
stack values as `LDB`, `ALPHAR`, `ALPHAI`, `BETA`, `VL`, and so on before
calling `mkl_lapack_dgegv`.

Consequently the mathematical problem sent to the LAPACK routine is the
generalized pair

```
GP * v = lambda * SP * v
```

with active order `N` and leading dimensions 50. `GP` and `SP` are the
post-shift work copies; the call does not pass `GMAT` or `SMAT` directly by
their construction descriptors. The eigenvalue outputs are the generalized
`alpha = (ALPHAR + i*ALPHAI)` and `beta` arrays. The immediate caller then
uses `AUXR` and `AUXI` in the source-987/988 loops beginning at `0x6fb1a1`,
which is the first visible spectral-consumption block.

## Eigenvector request boundary

The literal pool starts at `0x4ba12e0` with bytes `LITTLE_ENDIAN`, but the
caller passes `0x4ba12ec` (12 bytes into that object), whose byte is ASCII
`N` followed by the terminator. Thus both requests are standard
`JOBVL='N'` and `JOBVR='N'`; neither left nor right generalized eigenvectors
is requested. The call also aliases `VL`, `VR`, and `WORK` to the same `AUX`
allocation with unit leading dimensions, consistent with dummy vector
arguments for this no-vector request.

The standard mathematical interpretation of DGEGV is the pair `(A,B)` above;
it returns generalized eigenvalues as `alpha/beta` and optionally left/right
vectors. This audit identifies the actual A/B, dimensions, output arrays, and
the caller's explicit no-eigenvector request.

## Remaining evidence boundary

The post-call loop at `0x6fb1a1` reads `AUXR` and `AUXI`, but the subsequent
comparisons and transformations are still needed to identify the selected
eigenvalue sign/order and how that scalar spectral choice enters the rotation. The
existing first-matrix probe stopped before this call and only used one history
column; it therefore provides no evidence about off-diagonal matrix entries,
the generalized spectrum, or eigenvector selection. No eigensolver result was
hooked or fabricated here.
