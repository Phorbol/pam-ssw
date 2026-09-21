# Actual XXXII molecular crystal: GFN2 RC-VC derivative preflight

**The actual 172-atom XXXII crystal executes with an independent periodic GFN2-xTB ASE backend, and its RC-VC directional derivative agrees with central energy differences at both tested step sizes.** This establishes a narrowly defined backend/coordinate qualification point. It does not reproduce the original GAFF model, qualify a local minimum, or validate an RC-VC search trajectory.

Input: `tests/standalone/fixtures/type2_xxxii.extxyz` (C84H68Cl8N4O8); original uploaded TYPE2-XXXII `mc/rigidbody` and `mc/blist`. The explicit bond image lift changes zero atom images. Four 43-atom molecular components form 32 bodies/28 joints. The optimization chart has **55 coordinates** (four root poses, minus one global translation, plus 28 torsions and six cell coordinates). All root rotations remain physical relative to the periodic cell.

The frozen setup uses TBLite GFN2-xTB, accuracy 0.001, neutral charge and singlet multiplicity; single CPU thread for OpenMP/OpenBLAS/MKL. Root rotation and torsion metrics are 1 Å/rad; strain length is 5 Å. These are explicit numerical chart definitions, not tuned physical parameters. A seed-17 Gaussian random vector is normalized once in the full chart; pose/cell subvector norms are 0.969744/0.244123. The same direction is used for base and ±h at h=1e-4 and 5e-5 Å of the scaled chart. Pressure is zero.

| h | Analytic g·n, eV/Å | Central energy derivative, eV/Å | Absolute discrepancy | Relative discrepancy |
|---:|---:|---:|---:|---:|
| 1e-4 | -0.5243965017 | -0.5243621581 | 3.43436e-5 | 6.54916e-5 |
| 5e-5 | -0.5243965017 | -0.5243877331 | 8.76858e-6 | 1.67213e-5 |

Halving h reduces the discrepancy by about 3.92, consistent with central-difference truncation over these two points. This is one mixed root/torsion/cell direction, not an exhaustive Jacobian test. It checks the actual calculator's E/F/stress plus the nonaffine rigid-body cell pullback together; numerical agreement alone does not validate GFN2's physical accuracy for this crystal.

All **5/5 attempted E/F/stress evaluations completed**, with all returned values finite. Total measured elapsed time: **76.227 s**, within the authorized 5-EFS/120-s cap. No quench, walk, parameter retry or additional physical evaluation was performed. The base energy is **-8039.036555383 eV**, maximum force **2.5254601 eV/Å**, and maximum stress component **0.01154388 eV/Å³**, at volume **1959.43951 Å³**. The sizeable residual force means the supplied structure is not a GFN2 stationary structure; no stability conclusion follows.

A setup-only keyword argument error occurred before constructing/calling the oracle (0 EFS). Its script and error were preserved, the keyword-only `natoms` call was corrected, and only then was the five-call physical preflight started. No failed physical call was retried.

Artifacts: `research/ga_ssw/evidence/xxxii-rc-vc-gfn2-preflight/` contains frozen input, lifted input, rigidbody/blist, geometry/interface source snapshots, plan with full normalized direction, chronological calls (including E/F/stress arrays), result and the setup failure record. Runner: `research/ga_ssw/probe_xxxii_rc_vc_gfn2.py`. It refuses to repeat once calls/results exist. Command used `PYTHONPATH=/tmp/pam-ssw-tblite-20260909:.` with the three thread variables set to one.

Next scientific qualification would require a separately budgeted true atomic/cell quench and a complete RC-VC proposal/landing on this explicitly alternative GFN2 surface. Those operations were not included here. At roughly 14–20 s per EFS in this preflight, such a trajectory should not be described as a cheap test without a concrete budget.
