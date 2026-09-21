# Separate direction support from physical constraints

2026-09-11 implementation plan, experimental until real-material checks.

The uploaded TYPE4 input fixes atoms1..297 and separately sets mode0 on1..351.
Initial quenching already correctly used217 movable atoms. Treating both
blocks as FixAtoms would incorrectly prevent54 additional atoms from relaxing.
Ignoring the second block in search also changes the declared direction space.

Static parser0x68bcd0 maps fixatommode to para.atommodefix(+0x2df78).
`class_struc.get_ffix` constructs integer `ffix=min(fixatom,atommodefix)` and
copies atommodefix to modefix. Three original-instruction fixtures, including
the514atom mask, match exactly (`fixatommode-oracle.json`):217 physically
movable atoms,163 mode-eligible atoms. The combined mask is passed into
fixed-cell biasedrot at0x5c4ccb and direction generators. The allopt references
inspected use it for least-moved atom/pair selection, not a replacement of
the physical oracle's force constraint. Further native rotation internals
are not claimed bitwise equivalent to our independent solver.

Minimal independent design: optional `direction_fixed_indices` in constrained
SSW, distinct from `fixed_indices`. If B selects eligible Cartesian components,
direction refinement solves the projected problem B^T H B using exact lifting
of each finite-difference trial and restricting its gradient with B^T.
The rank-one rotation bias uses the same projected anchor. Lift the resulting
direction into the full active chart; Gaussian energy and force use that
vector. Both biased relaxation and initial/final true quenching retain all
physically active coordinates. Thus no false force/energy projection is used
in a larger optimization space.

No fitted parameter or new force term. A selection is physical task input;
None keeps the previous path exactly. Validate index bounds and reject an
empty direction subspace before any PES call. Report projected rotation
residual and allowed indices so it cannot be mistaken for a full-space mode.

Verification: coupled-coordinate test must show zero excluded direction
components while those same atoms respond during biased relaxation; all
initial/final certificates use the original physical constraints. Compare a
real Cu/EMT supported-adatom run with the frozen default and masked variant,
retaining complete calls, fixed geometry, costs and failures. Then prepare
the514atom MACE case with its source-defined two masks. Do not infer improved
coverage from the formula test or tune these masks to a favorable result.

## Implementation and measured checks

Public `run_constrained_ssw(..., direction_fixed_indices=...)` now implements
the whole-atom selection above. It does not yet expose arbitrary per-component
native masks or fractional mask weights. The shared reduced lifecycle restricts
only its dimer problem; all biased and true gradients remain full active-chart
derivatives. Events retain selected coordinate indices and mark the rotation
residual as projected. None preserves the prior numerical path.

Six new checks failed before the interface existed; after implementation the
constrained/RC selection has17 passes. Complete standalone/rotation regression
has347 passes and1 skip in the reference-compatible user-site environment.
The separate real Cu28/EMT comparison uses isolated NumPy2.0.2/ASE3.26.0.

Same raw Cu(111)3x3x3 plus FCC adatom, seed29, physical bottom two layers fixed:

| Arm | Search E/F | Fresh E/F | Biased stages | Landing minus initial (eV) |
|---|---:|---:|---:|---:|
| Frozen old default |59|2|1|-0.00139905|
| New default |59|2|1|-0.00139905|
| Direction on adatom only |422|2|14|+0.00010676|

Old/new default positions, cell, energy and raw force are exactly identical on
every request. In the masked run every mode has zero top-layer support
components, yet that layer moves by up to0.082614 Angstrom during biased
relaxation, demonstrating the intended distinction from FixAtoms. All six
fresh checks preserve fixed atoms/cell exactly and meet active fmax0.01.
Both observed proposals are MC accepted, but the tiny energy differences at
this tolerance do not establish distinct minima. The masked run costs more;
this test supports the interface, not a preferred search policy. Total test
cost includes the duplicate verification arm:540 search+6 fresh=546 E/F.

Source and complete traces/results are in `evidence/constrained-direction-subspace`.
The broader514atom MACE two-arm test is frozen in
`evidence/type4-source-direction-mace-v100`, job1249131, one V100,
2000 total E/F per arm including up to2 fresh evaluations,720s search cap per
arm and30min job limit. Its source-defined mode mask is compared with physical
mask only. Both arms completed; strict endpoint Hessians and unbiased-descent controls are
reported in [TYPE4 results](type4-source-mask-results.md). The initial fails the
first strict curvature check, so raw descent is not claimed as a qualified
minimum-to-minimum escape.
