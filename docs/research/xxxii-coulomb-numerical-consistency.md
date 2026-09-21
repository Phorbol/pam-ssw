# XXXII derivative failures: source-supported numerical hypotheses

The first actual172-atom qualification failed. This static review identifies two testable engine approximations; it does not attribute the measured errors to either without the independent numerical experiment. No PES was evaluated here.

## Coulomb table12 is not a conservative interpolation

Official [Pair constructor/table builder](https://github.com/lammps/lammps/blob/stable_22Jul2025_update4/src/pair.cpp) sets ncoultablebits=12 and tabinner=sqrt(2). It independently constructs etable and ftable from the analytic energy and force expressions, then separately differences each table. In [pair_lj_charmm_coul_long.cpp](https://github.com/lammps/lammps/blob/stable_22Jul2025_update4/src/KSPACE/pair_lj_charmm_coul_long.cpp), the real-space loop uses a float32 conversion of rsq for bitmap lookup **and for interpolation fraction**, then interpolates force from ftable/dftable. Energy is interpolated separately from etable/detable. The derivative of the interpolated energy is therefore not the independently interpolated force. Decreasing a finite-difference step need not converge to the returned force; float32 argument rounding further affects sufficiently small steps.

This is also native-default behavior: the uploaded Pair constructors at0x10cb510 and0x10cb760 load integer12 and write object+0x1a4 (0x10cb5ee→0x10cb6e2, second constructor0x10cb83e→0x10cb932). The custom pair single and compute consume that field to select their table branches. Thus changing to `pair_modify table 0` is an explicitly recorded numerical-evaluation change from the original default, not evidence that the reconstructed force field changed.

Minimal falsification: repeat the same failing fixed-cell atomic directional derivative and both original h values with **only** table0 changed. Keep structure, charges, model, Ewald settings and derivative convention identical. If the mismatch persists at the same scale, table interpolation alone does not explain it. Table0 still uses a finite erfc polynomial approximation; it is not an exact symbolic Ewald oracle. Do not silently tune model coefficients to absorb the difference.

## Ewald initialization can change the finite approximation with cell

Official [Ewald init/setup](https://github.com/lammps/lammps/blob/stable_22Jul2025_update4/src/KSPACE/ewald.cpp) does the following:

- init, unless gewaldflag is set, estimates g from accuracy, charges, atom count, cutoff and sqrt(volume), then transforms that estimate using log/sqrt. Reinitializing for a different cell can change g.
- setup, unless kewaldflag is set, increments integer kxmax/kymax/kzmax until RMS error crosses accuracy. Triclinic setup also transforms the chosen limits and converts them to integers.
- even with explicit kmax, reciprocal-vector inclusion uses `sqk <= gsqmx`; cell changes can modify the included set. Fixed limits alone do not prove a smooth fixed reciprocal sum.

The exact infinite Ewald sum does not depend on g. A truncated finite-accuracy implementation generally retains some g-dependence. A derivative of reported energy while the setup rules change is not automatically the fixed-setup virial. This is a possible contributor to cell FD discrepancies, not a diagnosis of the current adapter.

Minimal second experiment, only if table0 leaves material cell error: preserve the original point's **measured** G and integer reciprocal limits using `kspace_modify gewald G kmax/ewald KX KY KZ`, then repeat just the previously failing cell direction/h pair. The command and flag behavior are verified in official [kspace.cpp](https://github.com/lammps/lammps/blob/stable_22Jul2025_update4/src/kspace.cpp), lines487–492 and565–576. Do not guess G or K values. Record G, limits, kcount and whether init/setup reran at every perturbed point. If kcount changes, a reciprocal-shell boundary remains; if G/counts stay fixed and error remains, this hypothesis is insufficient.

A tighter-accuracy comparison is a later controlled numerical convergence test, with all extra cost counted, not an arbitrary tolerance relaxation. Translation invariance and fresh-engine agreement by themselves do not test E/F derivative consistency or cell-setup smoothness. Small global-rotation error may be compatible with finite reciprocal truncation but is not conclusive evidence for that explanation.
