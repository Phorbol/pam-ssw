# Published TiO2 cells: MACE-OMAT-small CPU stress preflight

2026-09-10. All **39 of 39** planned combined energy/force/stress calls completed
in **8.5872 seconds**, within the 120-second cap, on one CPU thread. No local
relaxation, SSW, cell optimization, GPU computation or model download occurred.

Inputs are the 12-atom rutile, anatase and TiO2-B coordinate tables extracted
from section 7 of the official SI to DOI **10.1039/c7sc01459g**. Exact source PDF,
its SHA256, coordinate manifest entries, copied input extxyz files, every trial
geometry, raw energy/forces/stress and componentwise comparisons are retained in
`research/ga_ssw/evidence/omat-small-tio2-stress/`. SI rounded VASP energies remain
source metadata; they are not numerical parity targets for this different PES.

| Input | Unrelaxed maximum force (eV/Å) | Maximum stress FD error (eV/Å³) |
|---|---:|---:|
| rutile | 0.04720619 | 2.2611e-10 |
| anatase | 0.07387179 | 3.2387e-10 |
| TiO2-B | 0.23979416 | 4.4335e-10 |

For each original cell, one initial E/F/stress evaluation plus symmetric
positive/negative strains for six components gives 13 calls. We use fixed
fractional coordinates, row cell Hnew=H(I+hB), h=1e-5, and offdiagonal B entries
of 1/2. Thus (Eplus-Eminus)/(2hV) directly equals the corresponding ASE Voigt
stress component. The predeclared absolute tolerance was 1e-5 eV/Å³.

The backend is the existing local `mace-omat-0-small.model`, CPU float64, cuEquivariance
and OpenEquivariance disabled. Model identity and package versions are in
`result.json`; both Ti and O must be in the recorded supported element table for
these successful evaluations. The exact launch used:

```sh
PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 /home/gengjianrui/.conda/envs/mace_env/bin/python -m research.ga_ssw.probe_omat_tio2_stress
```

PyTorch intra/inter-op threads were explicitly set to one. The script refuses
to overwrite its evidence directory and enforces 39-call and 120-second limits.
The environment variable avoids loading incompatible user-site packages into
`mace_env`; no environment installation or modification was needed.

These results establish backend loading and the E/stress sign, unit and strain
convention at three supplied periodic geometries. All three initial forces
exceed 0.01 eV/Å. Consequently none is a force-certified minimum under this
calculator. Their unrelaxed energy ordering does **not** establish stable-phase
ordering, and stress consistency does not validate MACE accuracy, joint VC
search efficiency, a new basin, or reproduction of the source VASP/NN PES.
