# Fixed-cell climb: evaluated-input snapshot and release restoration

2026-09-11, P0 targeted static audit, uploaded LASP ELF (same hash as prior
native reports). No main execution, PES calls or kernel change. This narrows
normal `lclimb_allstop` release; it does not establish every failure path.

The actual fixed-cell method table at0x53ca680 was read again from ELF data:
+0x110=0x5a88c0 noncrystal_opt, +0x1e0=0x5cda70 addgaussian,
+0x1e8=0x5cd130 climb_convg, +0x188=0x5c0ac0 set_status.

Inside `ssw_fixlat_mp_climb_`, before those callbacks:
- 0x5ca9ec..0x5cac43 sets work1(+0x9c8) from fa(+0x1d0), using array
  descriptors and contiguous/vector copies where applicable.
- 0x5caca0..0x5caf03 sets work2(+0xa28) from cart(+0x170).
- 0x5caf55..0x5caf71 saves energy(+0x230) in tene0(+0x1ac8).
- 0x5caf9c invokes addgaussian, then0x5cb02b invokes noncrystal_opt, then
  0x5cb03d invokes climb_convg. These are separate callback boundaries.

On the subsequent `lclimb_allstop` set branch (test0x5cb377):
- 0x5cb383..0x5cb629 restores fa from work1.
- 0x5cb647..0x5cb8d8 restores cart from work2.
- 0x5cb8dd..0x5cb8e4 restores energy from tene0.
- Optional LS cleanup0x5cb90b precedes Allopt setter0x5cb928.

Thus this normal release restores the entry snapshot that preceded the current
Gaussian/optimizer callbacks. It is not evidence for selecting an arbitrary
last unevaluated optimizer trial or searching all past minima for the best
release coordinate. The restored energy/force scope can include upstream LS;
cleanup and the next true-PES call must still be accounted for. The physical
force certificate is not provided by these array copies.

The optimizer callback itself has several branches; merely resolving its name
is not proof that every invocation uses LBFGS. Other stop/failure branches and
VC must be checked separately. The isolated Python pipeline must retain its
own accepted-point contract rather than mechanically reproducing a pre-callback
snapshot where its optimizer has already returned a valid accepted point.

Evidence: `native-release-snapshot-evidence/climb-snapshot-and-restore.asm`.
This closes the normal snapshot-copy ordering, not complete native end-to-end
SSW parity and not an algorithmic performance claim.
