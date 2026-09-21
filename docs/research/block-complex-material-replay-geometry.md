# Complex-material atomic replay: offline endpoint geometry

2026-09-10. This is a separate followup to
[the original bounded runs](block-complex-material-initial-results.md).
The original 300-second censor is retained. Geometry script and results are
`research/ga_ssw/evidence/block-complex-replay-geometry/diagnose.py` and
`geometry.json`. No E/F/stress requests were made for this analysis.

## AlOH26 replay landing

Source: `research/ga_ssw/evidence/block-aloh26-atomic-replay/result.json`
and its landing geometry. The atomic replay reached `gaussian_limit` and
subsequent full relaxation produced a fresh-certified endpoint. This completes
that replayed atomic segment; it does not retroactively complete the original
censored experiment or establish relative search efficiency.

| Quantity | Original initial | Cell-only landing | Atomic replay landing |
|---|---:|---:|---:|
| Composition | Al8O14H4 | Al8O14H4 | Al8O14H4 |
| Volume, Å³ | 273.784997 | 309.618289 | 250.005734 |
| Density, g/cm³ | 2.692127 | 2.380557 | 2.948188 |
| ΔE from original initial, eV | 0 | +0.303659884 | +0.488095685 |

The replay endpoint is compressed by about 8.69% relative to the initial
cell, and is 0.184435801 eV higher than the cell-only endpoint. The latter is
an endpoint comparison, **not** a claim that the accepted chain transitioned
from the rejected cell-only endpoint to the replay endpoint. Replay artifacts
do not contain an MC decision; no acceptance decision is invented here.

The saved fresh certificate has energy error 0, fmax 0.009297803 eV/Å and
maximum absolute stress component 0.000125096 eV/Å³. No Hessian was calculated.
It is a numerically relaxed candidate on MACE's PES, not a confirmed stable
phase or an independently curvature-qualified minimum.

At Al–O cutoffs 2.2, 2.3 and 2.4 Å, the replay endpoint consistently has four
four-coordinate and four six-coordinate Al sites. Every H has exactly one O
neighbor across cutoffs 1.1, 1.2 and 1.3 Å. The zero-based H-site nearest-O
identities are:

| H index | Initial nearest O | Cell-only nearest O | Replay nearest O | Replay O–H distance, Å |
|---|---:|---:|---:|---:|
| 4 | 1 | 1 | 19 | 1.008535 |
| 8 | 2 | 2 | 0 | 1.012460 |
| 18 | 15 | 15 | 13 | 0.981151 |
| 20 | 16 | 11 | 2 | 1.064926 |

These are changes in endpoint coordination at fixed atom ordering; they do
not establish elementary proton-transfer pathways or barrier heights.
Periodic neighbor shifts, including large integer shifts of the lifted H8
coordinate, are preserved in the JSON. Large unwrapped coordinates alone do
not imply a detached atom; the periodic nearest O remains about 1.01 Å away.

Replay shortest species-pair distances, Å: Al–Al 2.696411; Al–H 2.217570;
Al–O 1.684545; H–H 1.844846; H–O 0.981151; O–O 2.491604. There is no
near-zero endpoint overlap. The reduced H–H separation should not itself be
called an H2 molecule: both H sites retain short O coordination, and no bond
order calculation was made.

Pymatgen matching uses `scale=False`, `primitive_cell=True`,
`attempt_supercell=True`, symmetric matching, and three settings
`(ltol,stol,angle_tol)=(0.1,0.15,2°),(0.2,0.3,5°),(0.3,0.5,10°)`.
The replay landing does not match either reference at any of these settings.
This and the coordination changes support geometric distinction, without
energy-only basin identity or an assertion of new stable phases.

## Cost and status accounting

Original AlOH run: 1113 search requests + 2 fresh checks, 300.010241 search
seconds, with its second proposal explicitly budget-censored. Replay adds
465 search requests + 1 fresh check and reports 135.189618 replay seconds.
Accumulated search requests are 1578; counting all fresh checks gives **1581**
requests across both artifacts. Recorded elapsed fields have their original
scope and do not include an inferred fresh-check time. The offline geometry
analysis adds zero PES requests.

## Brookite48 replay: terminal censor, no final landing

The replay has now returned `result.json`: its atomic status is
`evaluation_failed` with `RuntimeError: declared replay request/wall budget
exhausted`. The checkpoint records `next_index=9` and a pending
`biased_quench` stage. These are continuation state, not an endpoint minimum.
No final landing or fresh endpoint certificate was produced; no geometry of
this intermediate is analyzed or reported as a minimum here.

The replay adds 1092 search requests and reports 600.473007 seconds. Combined
with the original 609 search requests, this is 1701 search requests; retaining
the original 2 fresh checks gives **1703 total recorded requests**. Replay
fresh requests are zero. The original censor and replay censor remain separate
records. This updates the earlier pending-file status only and leaves the
completed AlOH endpoint diagnosis unchanged. The geometry JSON records this
terminal censor, checkpoint stage and full cost; additional PES requests remain
zero.
