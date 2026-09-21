# Native random-component workspace: bounded static audit

This is a static audit of the isolated ELF `/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp` (the same artifact used by the native control probes).  No LASP main program, protection path, or PES was executed.

## Coefficient workspace

`ssw_fixlat_mp_get_random_mode0_` zeroes the ten-double temporary at
`[rbp-0x80..-0x38]` (`0x5c018f–0x5c01a2`) and copies it to
`ssw_parameters_mp_control_+0x158..+0x1a0` in five 16-byte stores at
`0x5c04ec–0x5c0508`.  The ten slots are therefore, in order,

```
c[0..9] = rbp[-0x80, -0x78, -0x70, -0x68, -0x60,
              -0x58, -0x50, -0x48, -0x40, -0x38]
```

The ordinary path first computes
`localmode = 0.1 + 0.1 * int(para+0x2db54) * u` at
`0x5c01d7–0x5c0203`; `0.1` is the rodata value at `0x4a45e08`.
The value is also stored at `object+0x1b28` and stack `-0x130`.
The exact slot writes statically recovered are:

| condition | write | slot |
|---|---|---|
| `para+0x100 == 5` (`0x5c021d`) | `1.0` (`movabs 0x3ff0000000000000`) to `rbp-0x78` (`0x5c071b–0x5c0725`) | `c[1]` |
| `para+0x100 == 6` (`0x5c0223–0x5c0226`) | `1.0` to `rbp-0x80` | `c[0]` |
| `control+0x68` bit 0 (`0x5c0236`) | zero `rbp-0x78`; `rbp-0x70 = max(1.0, localmode)` (`0x5c023d–0x5c0255`) | `c[1]=0`, `c[2]` |
| later branch, after random/energy tests | `rbp-0x58 = localmode` (`0x5c0385–0x5c0398`) | `c[5]` |
| later branch, after random/energy tests | `rbp-0x60 = localmode` (`0x5c0493–0x5c049b`) | `c[4]` |
| later branch, after random/energy tests | `rbp-0x50 = localmode` (`0x5c04c4–0x5c04cc`) | `c[6]` |

All other slots remain zero in this routine for the inspected paths.  The
three `localmode` writes are conditional branches whose complete selection
state is not reconstructed here.  Thus run type 5 is proven to set `c[1]=1`
before the common path; normal SSW starts with all zero and can set `c[2]`
from the `control+0x68` branch and `c[4..6]` in later branches.  This is a
coefficient/workspace distinction, not yet a proof of which downstream mode
consumes each slot.

The run-type-5 control branch at `0x5c0a53–0x5c0abb` is separate: when
`object+0x1660 == 0`, it samples `rd_numb_`, compares the result with
`para+0x2db70`, and writes `control+0x68 = -1` or `0`; run type 6 clears that
field at `0x5c017c`.  The numeric comparison is established; the semantic
name of `para+0x2db70` remains a parameter-field question.

## `vmb2_` arithmetic and call ABI

`ssw_commsub_mp_vmb2_` is at `0x58c930`; its local scalar helper
`ssw_commsub_mp_velo_loc_` is at `0x58cb70`.  The rodata values are statically
decoded from the ELF:

```
0x4a43918 = 0.00172309
0x4a43920 = 2*pi
0x4a43928 = -2.0
```

For each enabled component, the helper loads `A = *(rsi)`. The outer
`vmb2_` `rdx` argument is the output buffer, not the denominator: `vmb2_`
initializes a local double to exactly `1.0` (`0x58c93e`, `0x58c9d2`) and
passes its address as `rdx` to `velo_loc_` (`0x58c9ec`, `0x58ca32`,
`0x58ca81`). Thus `B=1.0` for every component here (instructions
`0x58cb7d–0x58cb89`, `0x58cbb0–0x58cbc8`).

```
s = 0.00172309 * sqrt(A / B)
z = sqrt(-2 * log(u1)) * cos(2*pi*u2)
component = s * z
```

`u1,u2` come from two calls to `ran3_` at `0x58cb96` and `0x58cba3`.
The component flags are read at `flag[-0xc]`, `flag[-0x8]`, and `flag[-0x4]`
relative to the `rcx` argument (`0x58c9d7–0x58ca74`).  After all atoms,
`0x58caf1–0x58cb61` subtracts the arithmetic mean of each Cartesian component
from every output vector, so the final vector is zero arithmetic mean.

The two inspected callers (`gen_randommode` `0x5d66ef–0x5d6703` and
`0x5d7325–0x5d734b`) pass:

```
rdi = &N                 (rbp-0x4c)
rsi = 0x4a45ad0          (ELF double 300.0; helper input A)
rdx = object+0x1848      (outer output 3N buffer)
rcx = object+0x968
r8  = &scalar            (rbp-0x30; RNG seed input)
```

The available DWARF member map labels `object+0x1848` as `energy_copy`, but
the call ABI proves only that it is the destination buffer used by `vmb2_`;
its descriptor/data-pointer representation is not resolved here. It is not
the denominator in `velo_loc_`. Likewise the `r8` scalar is transformed at
`0x58c94a–0x58c984` as
`seed = -int((*(r8)+1.0)*10.0)` (`0x4a43830=1.0`, `0x4a438a0=10.0`), but its
producer/semantic role is unresolved.  Consequently this is a proven
Box–Muller-like arithmetic kernel with a fixed `300.0/1.0` scale in these
callers. It is isotope-independent at this stage, unlike a mass-scaled
`1/sqrt(m)` interpretation.

The recovered facts close the arithmetic and coefficient-slot questions
needed for a future isolated probe. They do not establish that this component
is an all-atom global direction: in `gen_randommode`, slot 1 calls
`atom_neighbor_radius_` immediately before `vmb2_` (`0x5d66ab–0x5d66fe`), and
the `rcx=object+0x968` flags are not independently identified as “all mobile”
by this audit. Remaining unknowns are the runtime
descriptor/data layout of the `object+0x1848` output field, the producer of the `r8` seed, and the complete
downstream interpretation of `c[4..6]`; no production integration follows
from this static result.

## Five-case instruction probe

`research/ga_ssw/probe_native_vmb2.py` independently maps the ELF text/data
segments, supplies the flat ABI above, and executes the original `vmb2_`,
`velo_loc_`, and `ran3_` instructions. It hooks only the host `cos` and `log`
calls; it does not execute the binary main program or any protection/PES path.
The reference uses the actual native random draws and the proven denominator
`1.0`, then applies the same componentwise mean subtraction.

The saved report is
`research/ga_ssw/evidence/native-vmb2-20260912.json`: all five cases pass with
maximum absolute errors in `[0, 1.74e-18]`:

| case | N | scalar input A | mask | seed input | native first seed | max error |
|---:|---:|---:|---|---:|---:|---:|
| 1 | 3 | 1.0 | all components | 0.17 | -11 | 4.34e-19 |
| 2 | 3 | 1.0 | diagonal masked, nonzero untouched initial entries | 0.17 | -11 | 0 |
| 3 | 3 | 300.0 | all components | 0.17 | -11 | 1.73e-18 |
| 4 | 3 | 300.0 | diagonal masked, nonzero untouched initial entries | 0.17 | -11 | 0 |
| 5 | 1 | 300.0 | all components | 0.83 | -18 | 0 |

The masked cases verify that disabled components are retained until the final
mean subtraction; the routine does not silently clear them. This closes the
unit-denominator and output-buffer behavior for the isolated kernel. It does
not close the full `gen_randommode` random-component distribution: the
pre-`vmb2_` neighbor-radius call can affect the atom count/mask, and the
post-return path calls `n_normal` unconditionally (`0x5d6724–0x5d6733` and
`0x5d7368–0x5d7376`) before coefficient-scaled accumulation. No mass or
per-atom scaling is proven in the inspected post-return arithmetic, but the
complete mask, normalization, coefficient mix, caller descriptor layout, and
downstream mode interpretation remain open.

## Slot-1 neighbor-radius filter

The durable disassembly is
`research/ga_ssw/evidence/native-cluster-control-generator/atom_neighbor_radius.asm`.
The slot-1 caller invokes `newssw_basics_mp_atom_neighbor_radius_` at
`0x6e3590` before `vmb2_` (`0x5d66ab–0x5d66fe`). Its directly established
argument mapping is:

```
rdi = &caller[-0x44]       (center/control scalar)
rsi = 0x4a45ac8            (rodata double 12.0)
rdx = &caller[-0x4c]       (N pointer)
rcx = object+0x170         (coordinate descriptor/pointer)
r8  = object+0xe0          (additional geometry argument)
r9  = object+0x968         (3N integer mask)
```

The callee reads `N=*(rdx)` at `0x6e35a5`, initializes the first `3N` mask
entries to one (`0x6e35b4–0x6e3621`), and does not write through the N pointer
in the inspected body. It then loops over the N-sized control range and calls
`get_dist_` (`0x6e366c–0x6e3696`). The returned distance is compared against
the literal `12.0` at `0x6e369a–0x6e36a4`; on the greater-distance branch it
writes three zero mask entries (`0x6e36a6–0x6e36c1`). Thus this is a radius
filter over a pre-existing geometry-dependent selection, with a 12.0 literal
passed to the callee.

The adjacent `get_dist_` ABI is now statically mapped in
`research/ga_ssw/evidence/native-cluster-control-generator/get_dist.asm`:
`r15=&caller[-0x44]` supplies the first index, `r12=&caller[-0x20]` supplies
the second index, `r14=&caller[-0x38]` receives the scalar distance,
`rbx=object+0x170` supplies the coordinate array, and `r13=object+0xe0`
supplies the cell/periodic 3x3 data. The function reads both
indices (`0x581278–0x581349`), applies reciprocal-lattice/PBC reduction
(`0x58125a–0x58126a`, `0x5813c2–0x581440`), and returns the minimum periodic
Cartesian distance at `0x581687–0x5816a3`.

The wrapper sets the candidate index at `caller[-0x20]` before each call and
increments its loop index (`0x6e3666–0x6e36cb`). However, on the
distance-greater-than-12 branch, the inspected stores use `3*(*(Nptr))-3`
as the mask displacement (`0x6e36a6–0x6e36c1`), i.e. the final 3-component
triplet for the supplied N, rather than visibly using the candidate temporary.
Whether this is an intentional Fortran-layout choice or a caller/data-layout
interaction remains unresolved; it must not be paraphrased as “clear the
candidate atom.” The evidence supports only the exact 12.0 PBC-distance test,
N read-only behavior, and observed mask write address. It still does not
support the stronger claim that slot 1 is all-mobile or all-atom sampling.

## Actual center and branch reachability

At `gen_randommode_` entry (`0x5d5c50`), `rdi` is a handle containing the
object pointer (`0x5d5c64–0x5d5c76`), and `object+0` is copied to `rbp-0x4c` as N
(`0x5d5c6f–0x5d5c84`). The normal path reads `object+0x1ad8` at
`0x5d5de6` into `rbp-0x44`; the DWARF member map names this field `atompair`.
That value is therefore the first index passed through the neighbor-radius
wrapper as its center pointer. The run type 5 and 6 branches at
`0x5d5dd4–0x5d5de0` do not write `object+0x1ad8` in their inspected bodies;
after their branch-specific work they jump back to `0x5d5de6`
(`0x5da96f–0x5da973` for type 6, `0x5daaee` for the type-5 zeroing path;
type 5 can also jump directly back at `0x5da984`).
Thus the evidence does not support a claim that run type 5/6 universally use
center zero; the same object `atompair` read remains reachable.

The mask-distance branch is reachable when a non-center candidate is farther
than 12.0; the normal center producer is identified below. No complete
production trajectory exercising the large-distance condition was executed.

One producer is closed in `ssw_fixlat_mp_init_string_` at
`0x5dea3f–0x5deaa7`: it initializes `object+0x1ad8=1` and `+0x1adc=2`.
For N>2 it draws each index as `int(N*u)+1` using `rd_numb_`, repeats until
the two indices differ, and writes the pair back (`0x5dea5c–0x5deaa7`).
Thus the normal initialization can select any 1-based center, including a
non-final atom; run type 5/6 branch work still returns to the common center
read. This establishes reachability of the observed final-triplet mask write,
while leaving later overwrite producers outside this bounded trace.

An isolated instruction probe, `research/ga_ssw/probe_native_atom_neighbor_radius.py`,
was run with the same ELF loader and a hook at `get_dist_` entry. The hook
records the real callee registers (`rdi=Nptr`, `rdx=centerptr`,
`rcx=candidateptr`), writes only a controlled distance to the callee's real
distance output, and returns without executing `get_dist_`; no main/PES path
is involved. The report
`research/ga_ssw/evidence/native-atom-neighbor-radius-20260912.json` contains
12 passing cases: N=1/3/5, multiple centers, exact 12.0 versus 13.0, and
non-final candidates assigned the out-of-radius distance. Exact 12.0 leaves
the mask unchanged (`jbe` at `0x6e36a4`), while 13.0 clears the final mask
triplet even when the out-of-radius candidate is not the final atom. This
confirms the observed final-triplet write as executed behavior under the
wrapper ABI; it does not establish that this is the intended physical atom
selection rule.

Root independently repeated all 12 controlled-distance cases in
`native-atom-neighbor-radius-root-verification-20260912.json`; all passed.

## `n_normal` and width scale

The shared `ssw_commsub_mp_n_normal_` at `0x578e20` reads `N=*(rdi)` and
sets its loop length to `3N` (`0x578e29–0x578e3f`). It accumulates every
component square (`0x578e8a–0x578f28`), takes one square root, and multiplies
every component by its reciprocal (`0x578f70–0x579070`). The zero norm path
is handled separately. This is complete 3N Euclidean L2 normalization, not
per-atom RMS, max-atom, or max-component normalization.

The native `gen_randommode` calls it unconditionally after `vmb2_` at
`0x5d6724–0x5d6733` and `0x5d7368–0x5d7376`; the previously audited `moveds`
width path calls the same function at `0x5c7c6a`. Consequently, when the
incoming direction is normalized and the move is the direct unclipped path,
native width is a total Cartesian displacement norm. A fixed-cell direction
with width `0.1` therefore has total norm `0.1` and typical per-component or
per-atom amplitudes that decrease with dimension (roughly `1/sqrt(N)` for
spread components). This is a normalization convention comparison only; it
does not explain the CuO64/TiO2-12 outcomes or establish a size-effect cause.
