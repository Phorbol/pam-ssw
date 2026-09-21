# Native fixed-cell CBD re-entry guard

> 2026-09-17 correction: the earlier Allopt equality interpretation was reversed.
> `for_cpstr` operator 3 is inequality. The text below is corrected; see
> [verification and decision](2026-09-17-direction-lifecycle.md).

Bounded static audit of the uploaded fixed-cell ELF. No LASP main execution,
PES call, protection-path change, or production code change.

## Closed path

The fixed-cell method table at `0x53ca680` resolves `+0x188` to
`set_status_` (`0x5c0ac0`), `+0x1e0` to `addgaussian_` (`0x5cda70`), and
`+0x1e8` to `climb_convg_` (`0x5cd130`). In `climb_`:

| address | evidence |
|---|---|
| `0x5caf9c` | callback through `+0x1e0` (`addgaussian`) |
| `0x5cb02b` | optimizer callback |
| `0x5cb03d` | callback through `+0x1e8` (`climb_convg`) |
| `0x5cc4c3–0x5cc4cc` | branch guard `run_type == 5` |
| `0x5ccb9e` | callback through `+0x1f0` (mode update) |
| `0x5ccba8–0x5ccbc6` | compare `object+0x1b34` with six-byte `Allopt` at `0x4a45ed4` |
| `0x5ccbe7` | callback through `+0x188`, with three-byte `CBD` at `0x4a42980` |

The last callback is therefore a concrete conditional
`climb_convg -> mode update -> status != Allopt -> set_status(CBD)` edge. In
`set_status_`, the CBD branch resets `control+4` at `0x5c0c94`, writes the
`CBD_PreRot` status at `0x5c1249–0x5c1314`, and dispatches slot `+0x1b8` at
`0x5c1330`. The same table resolves `+0x1b8` to `unbiasedrot_` (`0x5c3fa0`),
so the re-entry reaches native pre-rotation.

## Frequency boundary

This proves that a new CBD/pre-rotation stage can begin after convergence
processing. It does **not** prove one such re-entry per Gaussian. The only
recovered guards are `run_type == 5` and the post-update status being unequal to `Allopt`; the
trajectory counter/`ng` meaning that determines how often this branch is
visited is not resolved in this slice. `climb_convg_` itself has no direct
call to `set_status`, `unbiasedrot`, `biasedrot`, or `addgaussian`; the edge is
in the enclosing `climb_` continuation.

The paper-level statement is separate: `literature/74.txt:179–185` says each
`Nni` is updated from initial random `N0i`. That supports the paper reference's
per-Gaussian anchor convention, but does not establish that the native
`object+0x1788` reset/copy occurs at the same frequency. The native anchor
copy at `0x5c469f` uses the current presweep-updated value.

Conclusion: native new-Gaussian entry is conditionally demonstrated after
`climb_convg`; unconditional per-Gaussian pre-rotation remains unknown.
