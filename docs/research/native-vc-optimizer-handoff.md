# VC optimizer and saved-state handoff

2026-09-11. Bounded root audit, zero PES. Ordinary CSSW table0x53cacc0.

The actual VC optimizer call at0x5f2642 uses slot+0x108, resolving to
`class_struc_mp_crystal_opt_`0x5a7f90. Slot+0x110 does resolve to
`noncrystal_opt`, but is not called here. This corrects the prior MAINLINE
inference from an adjacent table slot. Raw caller: native-cell-reference-evidence/climb.asm.

Before addgaussian and the optimizer callback, the caller copies
sfa(+0x4d8) to work1(+0x9c8), at0x5f1e89–0x5f213c;
scart(+0x478) to work2(+0xa28), at0x5f2153–0x5f23fd;
and object.energy(+0x230) to tene0(+0x1b28), at0x5f241f–0x5f243b.
These are pre-callback snapshots of generalized coordinates/forces, not
necessarily an accepted L-BFGS iteration or Cartesian positions.

Call order: addgaussian+0x1e0 at0x5f2466; crystal_opt+0x108 at0x5f2642;
climb_convg+0x1e8 at0x5f2650; scart2cart+0xe8 at0x5f265d.
The convergence routine reads tene0, so its candidate scalar is the saved
pre-callback energy, not an unevaluated proposed point after crystal_opt.
The true/biased/LS scope of the incoming energy still depends on the upstream
energy-evaluation callback and is not inferred from the field name.

If allstop(+0x78) is set, the caller restores sfa from work1 (0x5f268a–0x5f291d)
and scart from work2 (0x5f2961–0x5f2bf9), restores energy=tene0
(0x5f2bf9–0x5f2c02), then selects Allopt (0x5f2c10–0x5f2c1f).
If only stagestop(+0x7c) is set, analogous work copies start at0x5f2ccd;
energy restoration is0x5f322d–0x5f3234, followed by the next indexed trajectory
record write. Thus both paths retain saved generalized state rather than
blindly retaining the optimizer's newest proposal. The later Cartesian/cell
resynchronization before the next PES evaluation has not been claimed here.

This distinguishes evaluated snapshots from uncomputed trial coordinates.
It does not imply an Armijo/Wolfe accepted-point certificate: that requires
examining crystal_opt/BFGS reverse-communication state. Any Python early-stop
implementation must make this distinction explicit and preserve the final
physical force/stress certificate regardless of inner-stage status.
