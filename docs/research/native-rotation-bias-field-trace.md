# Native pre-rotation to biased-rotation transition

> 2026-09-17 correction: the earlier Allopt equality interpretation was reversed.
> `for_cpstr` operator 3 is inequality. The text below is corrected; see
> [verification and decision](2026-09-17-direction-lifecycle.md).

Evidence: uploaded ELF SHA256 bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704;
`analysis/kernel-ssw_fixlat_mp_unbiasedrot_.asm` and `kernel-ssw_fixlat_mp_biasedrot_.asm`.
Root independently executed the isolated original instructions with real `for_cpstr`:
`research/ga_ssw/evidence/native-rotation-weight-branch-20260912/root-verification.json`.
No main program, protection code, calculator or PES is entered.

## Recovered branch, with corrected constants and fields

`+0x18a8` is CBD rotation's curvature output. `+0x18b0` is the separately
produced curv_real field; those names must not be interchanged. In this
unbiased path, the former measures the rotation surface before the rank-one
bias; with LS this need not mean bare physical PES curvature.

At `0x5c44b7` the mode string is compared with `CBD_PreRot` using the actual
Fortran runtime. Constant address `0x4a45e38` contains bytes
`8dedb5a0f7c6b0be`, which decode to **-1e-6**, not zero. The branch is:

```
if mode == 'CBD_PreRot' and rotation_curvature > -1e-6:
    mode = 'CBD_biasedRot'
    control.rotstep = 1
    rotation_weight = rotation_curvature
```

Copy instructions: `0x5c45e9` reads object+0x18a8 and `0x5c45f7` writes
object+0x18b8. `0x5c45f0` writes control+4, which is **rotstep**, not an
unidentified independent flag. `0x5c4226` establishes the saved control pointer;
upstream `0x5c4232` clears that counter after LCONVERGE was checked at
`0x5c420f`. The isolated probe starts after that upstream gate and cannot
claim to reproduce all reasons why LCONVERGE became true.

The biased caller passes object+0x18b8 to add_rotation_bias at `0x5c4c45`.
The force primitive is `F += w * ((R-R0) dot n) * n`. Thus the native branch
supplies a curvature-derived rotation weight rather than a universal constant.
BP-CBD2012 (DOI10.1021/ct300250h, Eq8–9) motivates a=Ce, but the executable
also accepts zero and a tiny negative value above its -1e-6 boundary. Do not
silently rewrite its branch as strictly positive or max(Ce,0).

## Independent instruction verification and limitations

The root probe reads the comparison literal from the ELF, counts actual runtime
calls, checks output weight/mode/counter and exact stop address. Twelve cases
cover two modes and positive, zero, slightly negative, exact-boundary and lower
curvatures. All pass. The original agent result.json is retained but its
`passed` only meant reaching a stop address, and its rotstep field was an input
label rather than a measurement. The root-verification artifact supersedes
those assertions. Zero DOES assign weight0; a previous agent prose statement
that it skipped was inconsistent with its own raw output.

This verifies only the branch from 0x5c44b7 to0x5c45fe/0x5c4896. It stops before
the subsequent anchor-vector copy, and does not reproduce full pre-rotation,
Broyden history, all initialization paths or the complete walker. Current fixed
rotation_bias=100 remains an explicit independent development setting. The next
kernel task is to recover the coupled pre-rotation/counter/anchor lifecycle;
changing only that constant would not reproduce this source behavior.

## Pre-rotation versus biased-rotation call granularity

The fixed-cell `unbiasedrot` entry (`0x5c3fa0`) calls `cbd_rotation` once at
`0x5c40ca`, then rechecks the mode string and curvature. When the recovered
`CBD_PreRot` predicate passes, it writes an intermediate `CBD_UnbiasedRot`
mode around `0x5c411f–0x5c420f`. The separately verified transition to
`CBD_biasedRot`, `control+4=1`, and the rotation-weight copy is at
`0x5c44b7–0x5c45f7`; it must not be attributed to `0x5c411f–0x5c420f` (the
latter includes an upstream convergence gate). The fixed-cell `biasedrot` entry
(`0x5c4ae0`) performs `add_rotation_bias` at `0x5c4c45` and then one
`cbd_rotation` at `0x5c4cea`.

This proves the transition is per invocation of the rotation function: a
single `unbiasedrot` call makes one call into `cbd_rotation` before the mode
can select the biased path. The callee may itself iterate, so this is not a
claim that one call equals a complete rotation solve. `biasedrot` likewise has
one visible `cbd_rotation` call on its ordinary branch. These call counts do
not establish entry frequency per Gaussian.

The fixed-cell method table gives the relevant callback identities: slot
`+0x188` is `set_status` (`0x5c0ac0`), `+0x1e0` is `addgaussian`
(`0x5cda70`), and `+0x1e8` is `climb_convg` (`0x5cd130`), as read at
`0x53ca680` and recorded in `native-release-snapshot-followup.md`. `climb`
calls those latter two through the table at `0x5caf9c` and `0x5cb03d`, with
the optimizer callback between them (`0x5cb02b`). `climb_convg` itself is a
status/convergence consumer; its inspected body contains no direct call to
`set_status`, `unbiasedrot`, `biasedrot`, or `addgaussian`. Thus the static
path does not show a new Gaussian re-entering `set_status('CBD')` from
`climb_convg`; any such transition would have to occur in an unresolved
caller/state branch or in a subsequent outer iteration.

`set_status` does directly initialize the pre-rotation stage when its input
status compares equal to `CBD` at `0x5c0c30–0x5c0c7a`: it resets
`control.rotstep=0` at `0x5c0c94`, copies current coordinates/forces into the
stage snapshots at `0x5c0cae–0x5c1224`, writes the padded
`CBD_PreRot` string at `0x5c1249–0x5c1314`, sets `object+0x18a8` to `1.0` at
`0x5c131b–0x5c1325`, and dispatches the next type-bound procedure at
`0x5c1330`. The fixed-cell table resolves that descriptor slot `+0x1b8` to
`unbiasedrot` (`0x5c3fa0`) at table `0x53ca680` (the mapping is also recorded
in `native-curv-real-provenance.md`). Thus this is a real indirect
`set_status(CBD) -> unbiasedrot` edge, while the setter still does not make a
direct call to the symbol.

The paper has a stronger algorithm-level statement than this incomplete
native caller trace: `literature/74.txt:179–185` says each `Nni` is updated
from the initial random direction `N0i`, while the modified PES accumulates
Gaussians. Therefore the paper reference's per-Gaussian reuse of the outer
sampled anchor is supported by the cited algorithm text. The native ELF
evidence here does not prove that its `+0x1788` value is reset at the same
frequency. The anchor-copy audit shows `0x5c469f` copies the current
presweep-updated `object+0x1788` into `object+0x17e8`, after the
`cbd_rotation` writeback/rollback sequence (`0x5c40ae–0x5c40ca` and
`0x5c46ae–0x5c472c`); it is therefore not safe to call that source an
untouched random N0. Whether a later Gaussian reaches this copy remains
unknown from the resolved callbacks.

## Resolved post-convergence CBD re-entry

The fixed-cell `climb` body does contain one resolved indirect re-entry that
was absent from the earlier direct-call scan. After the ordinary
`addgaussian -> optimizer -> climb_convg` sequence, the `run_type == 5`
branch at `0x5cc4c3–0x5cc4cc` calls the mode-update slot `+0x1f0` at
`0x5ccb9e`. It then compares the current status (`object+0x1b34`) with the
six-character `Allopt` string at `0x5ccba8–0x5ccbc6`. If unequal, it calls the
fixed-cell `set_status` slot `+0x188` at `0x5ccbe7`, passing the three-byte
`CBD` literal at `0x4a42980`.

This closes a concrete conditional path
`climb_convg -> update_mode0 -> status != Allopt -> set_status(CBD) ->
unbiasedrot`. `set_status(CBD)` then performs the pre-rotation reset and
`CBD_PreRot` initialization documented above. It is evidence that a new CBD
stage can begin after convergence processing, and therefore that native
pre-rotation is not confined to the first stage. It does not prove that this
branch executes for every Gaussian: the path is gated by `run_type == 5` and
the post-update status being different from `Allopt`, and the surrounding trajectory-counter
meaning is not fully recovered. The native result is thus a conditional
stage re-entry, not evidence for unconditional pre-rotation before every
Gaussian.
