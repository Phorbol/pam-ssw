# Rutile12 joint-VC: initial and landing curvature qualification

2026-09-10. **Both structures have positive sampled joint Hessians.** This bounded check does not support the suggested explanation that the observed 0.1219813 eV decrease merely starts from a high-symmetry saddle with a negative local mode. It does not by itself establish a new phase, two distinct exact stationary basins, or bulk thermodynamic stability.

Source: `research/ga_ssw/evidence/joint-vc-rutile12-l5/result.json`, initial and final recorded minimum. Each structure uses its **own** cell reference and the exact `SymmetricLogStrainChart`, L=5 Å, zero pressure. Coordinate dimension is 42 (36 atomic plus six strain); an orthonormal null-space basis removes the three uniform atomic translations, leaving 39 dimensions. No rotation is removed from a periodic crystal and all six symmetric strains remain.

At each structure, one fresh E/F/stress evaluation precedes 78 central finite-difference evaluations of the exact joint gradient, h=0.001 Å in q. Diagonalize the symmetric part of the 39×39 matrix. Re-evaluate the lowest eigenvector at h/2, h, 2h, recording both gradient-difference and energy-second-difference curvature (six more calls per structure). Total is **170 E/F/stress requests in 19.76 seconds**, under the 180-request/60-second limits. CPU float64 MACE OMAT-small, one thread, CUDA hidden, `PYTHONNOUSERSITE=1`; no relaxation or walking was performed.

| Quantity | Initial | Landing |
|---|---:|---:|
| Fresh energy (eV) | -106.63444154812946 | -106.75642285622841 |
| Maximum atomic force (eV/Å) | 0.00642518 | 0.00665719 |
| Stress Frobenius norm (eV/Å³) | 0.000174646 | 0.000163833 |
| Joint projected gradient norm (eV/Å) | 0.0168631 | 0.0166016 |
| Lowest Hessian eigenvalue (eV/Å² in L5 metric) | +0.13157245 | +1.30071271 |
| `||H-H.T||_F` (eV/Å²) | 0.000258985 | 0.000253078 |
| Lowest-mode gradient component (eV/Å) | 0.0000420316 | 0.00585998 |
| Lowest-mode atomic / strain norm | 0.978115 / 0.208067 | 0.802661 / 0.596435 |

Lowest-mode gradient curvatures at h/2, h, 2h are respectively:

- Initial: **0.13154830, 0.13157158, 0.13166471 eV/Å²**.
- Landing: **1.30069819, 1.30070057, 1.30071007 eV/Å²**.

Energy curvatures agree at these steps: initial 0.13154443–0.13160263 and landing 1.30069782–1.30070374 eV/Å². There are no negative eigenvalues in either central-difference matrix. The small antisymmetric residual and step consistency support a robust positive lowest local curvature for this chart and calculator.

Residual forces are not zero. In particular, the landing's minus-mode displacement lowers energy linearly even though its symmetric curvature is positive; this is explained by the measured nonzero gradient and must not be described as a negative mode. At nonstationary points, Hessian values depend on coordinate chart through gradient-dependent terms. This check intentionally uses each structure's own chart, records the residuals, and does not replace tighter stationary relaxation. Only the smallest mode is step-size repeated; the entire Hessian was not recomputed at three spacings.

The finite-cell calculation covers the supplied 12-atom cell's atomic modes and homogeneous strain. It does not test larger-cell phonons, finite-wavevector instabilities outside that cell, alternate electronic models, or thermodynamics. Distinct-basin/phase claims still need tightened endpoint relaxation and structure comparison. The defensible result is: **a lower-energy, force/stress-qualified landing was observed, and both saved geometries show positive local joint curvature at the tested resolution.**

Reproducer: `research/ga_ssw/qualify_rutile_joint_hessian.py`. Evidence directory: `research/ga_ssw/evidence/joint-vc-rutile12-l5-hessian/`, containing plan, frozen script, chart snapshot, source/model checksums, execution log, all 170 raw E/F/stress/geometry records, full Hessians/bases/eigenvalues and lowest-mode checks. All raw-call lines were reconciled with the reported request count. This is additional validation cost, not part of the original 173 search requests.
