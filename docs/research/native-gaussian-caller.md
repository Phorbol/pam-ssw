# Fixed-cell Gaussian caller: recovered state and remaining ambiguity

> Subsequent execution correction: see `native-addgaussian-instruction-oracle.md`.
> The second pass DOES contain the projection factor; the earlier missing-factor
> reading below is superseded. Whole-function instructions instead demonstrate
> two identical old-Gaussian force contributions but one energy contribution.
> The historical static reasoning below is retained for traceability, not as the
> current formula. Full-program compensation remains unverified.

Date: 2026-09-09. Scope: uploaded ELF `GA-SSW_program/lasp`, fixed-cell `ssw_fixlat` only. This document is a static reverse-engineering result, not a full-walker execution-parity or scientific validation claim. Repository AGENTS.md was read. No search implementation was modified.

## Main result

The 87 degree controller is invoked **once on entry to a new climbing segment**, selected by `substatus == "climb_new"`. It is not unconditionally invoked whenever the calculator returns forces. The actual arguments also differ from a generic "background energy/background force" wrapper: `fa0` is the current structure force array; `fa2` is a scratch array constructed inside `addgaussian`; `e2` contains only the sum of earlier Gaussian energies. True energy is added after the helper returns.

The scratch construction for earlier Gaussians contains two consecutive subtract passes in the ELF. The second lacks the displacement projection factor present in the first. This must be checked with a whole-`addgaussian` instruction oracle before replacing the scratch by the standard analytic Gaussian gradient. A scalar fallback path confirms the two arithmetic passes exist; this is not merely a duplicated SIMD/fallback reading. The scientific interpretation of the second term remains unresolved.

## Evidence and reproducible inspection

Original artifact: `/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp`.

Existing disassemblies under the adjacent `analysis/` directory:

- `kernel-ssw_fixlat_mp_addgaussian_.asm`, entry `0x5cda70`.
- `kernel-ssw_fixlat_mp_climb_.asm`, entry `0x5ca8e0`.
- `kernel-ssw_fixlat_mp_moveds_.asm`, entry `0x5c4d60`.
- `newssw_basics_mp_set_thisgaussw_.asm`, entry `0x6e1730`.
- `newssw_basics_mp_set_initial_gaussw_.asm`, entry `0x6e8700`.
- `kernel-dwarf-member-offsets.txt` and `selected-dwarf.txt`.

Additional functions inspected directly with `objdump -dl -Mintel --disassemble=FUNCTION lasp`: `ssw_fixlat_mp_set_status_` and `ssw_fixlat_mp_update_forcepara_`. `-l` provides original `Class_ssw.F90` line mappings. ELF PT_LOAD segments and the existing `lasp-symbols.txt` allow direct static reads of strings and type-bound function tables; no native process execution or ptrace is needed.

## Resolve the type-bound calls

A fixed-cell function table at virtual address `0x53cc8c0` gives these verified entries (64-bit little-endian pointers, matched to ELF symbols):

| Table offset | Function address | Function |
|---|---|---|
| `0x100` | `0x5ad9c0` | `class_struc_mp_init_bfgs_` |
| `0x110` | `0x5a88c0` | `class_struc_mp_noncrystal_opt_` |
| `0x130` | `0x59d970` | `class_struc_mp_set_atomfix_` |
| `0x188` | `0x5c0ac0` | `ssw_fixlat_mp_set_status_` |
| `0x190` | `0x5bc0f0` | `ssw_fixlat_mp_update_forcepara_` |
| `0x1d0` | `0x5ca8e0` | `ssw_fixlat_mp_climb_` |
| `0x1e0` | `0x5cda70` | `ssw_fixlat_mp_addgaussian_` |
| `0x1e8` | `0x5cd130` | `ssw_fixlat_mp_climb_convg_` |
| `0x1f0` | `0x5d55f0` | `ssw_fixlat_mp_update_mode0_` |

Multiple copies of this table exist. The mapping follows pointer contents, not the compiler-generated table names (which are nonunique in `nm`).

## Height-update gate and energy/force lifetime

At `climb:0x5caf49–0x5caf9c`:

1. Read current structure energy at object `+0x230` and save it to `tene0`, `+0x1ac8`.
2. Compare `substatus` at `+0x1b52` against the nine-byte string at `0x4a460a8`. Direct ELF read gives exactly `climb_new`.
3. Form the Fortran logical argument, true on equality, and call table slot `0x1e0` (`addgaussian`).
4. Call `noncrystal_opt` at `0x5cb02b`, then `climb_convg` at `0x5cb03d`.

At `addgaussian:0x5ceb93`, the second argument's low logical bit chooses the branch. The true branch calls `set_thisgaussw` at `0x5cec2c` and then writes `climb_opt` into the substatus (`0x5cec49–0x5ced00`). The false branch at `0x5ced0b` evaluates using the already stored weight; it does not call the height controller. Consequently an ASE port must store a stage-level initialized/frozen-weight state. It must not make Gaussian height a side effect of an arbitrary energy/force cache request.

`addgaussian` receives the current structure E/F on entry. These may already include LS or other enabled terms upstream; the function itself cannot establish that the input is the bare physical PES. Its returned structure force and energy are modified values. The base energy snapshot and force work arrays in `climb` allow later restoration; re-entering `addgaussian` on its own modified output would double-count terms.

## Actual helper arguments

The call at `0x5ceb9c–0x5cec2c` is:

```text
set_thisgaussw(
    na             = structure+0x0,
    d1             = exp[-s_latest**2/(2*width_latest**2)],
    d2             = s_latest/width_latest**2,
    fa0            = structure.fa,                # array at +0x1d0
    fa2            = structure.tf0,               # array at +0x1a60
    n              = saved latest direction,
    e2             = sum of earlier Gaussian energies,
    w              = saved latest weight,         # in-place array element
    e              = local output Gaussian energy,
    anglef_n       = local output angle,
    maxw, step, scalefact0 = para fields
)
```

The stack address `rbp-0x80` passed as `e2` is initialized to zero at `0x5cdbff` and incremented by the old-Gaussian energy at `0x5ce676–0x5ce689`. It never receives the physical energy. The returned Gaussian total at `rbp-0xa8` is only added to structure energy at `0x5cf681–0x5cf694`; `control.anglef_n` and `control.gauss` are written at `0x5cf69c/0x5cf6a1`.

`fa2` is not the force of the previous *evaluation*. The buffer is zeroed on every `addgaussian` entry at `0x5cda84–0x5cdbe6`, then rebuilt at the current geometry from old saved Gaussians. It excludes the latest Gaussian at helper entry. In the false branch the latest contribution is incorporated before the buffer is subtracted from structure.fa (`0x5cf159–0x5cf168` is a scalar example).

The helper then computes `fa0 - fa2 + d1*w*d2*n`. Hence the sign and exact contents of `tf0` matter. It is incorrect to pass an already accumulated background force as `fa0` while also subtracting the same old terms through `fa2`.

## Earlier-Gaussian scratch anomaly: confirmed arithmetic, unresolved interpretation

The loop runs over `1..ng-1`: load `ng` at `0x5cdbe6`, decrement at `0x5cdc03`, loop at `0x5ce695`. Define `s=(R-Rj) dot nj`, `Bj=Wj exp[-s*s/(2*sigmaj*sigmaj)]`.

The first pass, mapped to source line 1216, loads `Bj` from local `rbp-0x228`, multiplies by `s` from `rbp-0x1e8` (`0x5ce1af–0x5ce1ba`), then subtracts `Bj*s*nj/sigmaj**2` from tf0. Scalar fallback: `0x5ce308–0x5ce323`.

Control continues directly to a second pass, mapped to source line 1218, reloading `Wj*exp(...)` at `0x5ce38a–0x5ce39b` and storing it to `rbp-0x1e0`. It subtracts a second term using the same saved direction array and width squared. Scalar fallback: `0x5ce60f–0x5ce62b`. There is no intervening multiplication by `s` on this pass. Thus the current static reading is:

```text
tf0 -= Bj*s*nj/sigmaj**2
tf0 -= Bj*nj/sigmaj**2
```

These blocks are consecutive source statements, not alternate aligned/unaligned or vector/scalar implementations of one statement. This surprising extra term is **not yet promoted to a scientific formula or a proven original-program bug**. A full instruction oracle should initialize minimal Fortran descriptors, evaluate ng=1/2/3 with nonunit projections (including s=0), and compare raw tf0/fa/energy and both logical branches. It should also compare finite-difference energy gradients. Until then, the standalone paper Gaussian must remain conservative and should not silently adopt this suspected term.

## Center, direction, width, and weight lifetime

The persistent arrays used by addgaussian are:

| Object offset | Role from arithmetic |
|---|---|
| `+0x1660` | current Gaussian/trajectory index ng |
| `+0x1668` | trajectory structure array; each element size `0x690` |
| `+0x16b0` | direction record array; each element size `0x138` |
| `+0x16f8` | saved scalar width array |
| `+0x1740` | saved scalar weight array |
| `+0x1788` | current n0 direction record |

Fortran descriptor lower bounds must be honored when constructing addresses; offsets `+0x1738/+0x1780` are width/weight array lower bounds, not scalar algorithm parameters.

`set_status` increments ng at `0x5c1a3b–0x5c1a45` (source 640). For ng=1 it stores the current Cartesian coordinates into the trajectory center entry (source 651, initial copy at `0x5c1a6b` onward). It copies the current n0 into direction record ng (`0x5c1fbf–0x5c2236`); scalar mode metadata is copied at `0x5c2371–0x5c23ac`. It initializes the local BFGS history through slot `0x100` at `0x5c257c`, then invokes `set_initial_gaussw(ng, weight_array, ...)` at `0x5c25ae` (source 689). Therefore the initial weight is not newly reset on every force evaluation.

`moveds` stores a scalar accumulated from separately addressed weighted work arrays into width[ng] at `0x5c80c4`; the immediate accumulation is at `0x5c7d72–0x5c805e`. The selected displacement is normalized in-place earlier at `0x5c7c6a`, but this slice does not prove that width is a plain dot product with that normalized buffer. Additional move/retry paths also access/update this width; not all such branches were recovered here. It is therefore unsafe to equate native saved width unconditionally with the requested `ds_atom`.

On the climb completion path, the index `ng+1` is formed at `0x5cbf49–0x5cbf50`. Coordinates from `work2` (`+0xa28`) are copied into trajectory[ng+1]. The saved base energy `tene0` is stored in that next trajectory structure at `0x5cc209–0x5cc217`, and the force copy comes from `work1` (`+0x9c8`). These become the next segment's stored information. The latest center/direction/width are read during `addgaussian`; no center movement or width rescaling happens inside it. Only the latest weight is writable through the helper pointer in its true branch.

The exact choice of `work2/work1` versus last optimizer trial on every completion/failure branch remains to be traced before a full controller port. Merely copying the last ASE trial would be an assumption.

## Integration decision

Do not connect the current helper by guessing fa2 or call frequency. The recovered gate, energy bookkeeping, table mapping and persistent array roles are ready to use. The blocking evidence gaps are the old-Gaussian scratch anomaly, all moveds retry/width branches, and the optimizer completion geometry selection. A whole-function instruction oracle is the next bounded verification, followed by a same-PES single-segment real-atom comparison. No new heuristic or parameter is proposed here.
