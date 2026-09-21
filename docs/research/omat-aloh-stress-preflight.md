# MACE-OMAT-0-small / uploaded AlOH CPU stress preflight

2026-09-10. **All 13 planned E/F/stress evaluations completed in 7.86 seconds on one CPU thread; six-component stress agrees with central energy differences to 2.41e-10 eV/Å³.** No relaxation or search was performed.

Input is first frame of the uploaded `TYPE1-AlOH/addition/add.arc`: 26 atoms, H4Al8O14, fully periodic, triclinic cell. The full original path, structure arrays, input checksum, model checksum, software versions and every evaluated E/F/stress/geometry are in `research/ga_ssw/evidence/omat-small-aloh-stress/result.json`. The native archive's energy is not used as a MACE parity target: the original AlOH.pot and OMAT-small are different PES models.

The actual local checkpoint `/home/gengjianrui/.cache/mace/mace-omat-0-small.model` loaded successfully with `MACECalculator`, CPU, float64, both optional accelerated backends disabled. Model elemental metadata includes H/C/O/Cu/Pd. Environment was `mace_env`: mace-torch 0.3.16, torch 2.8.0, ASE 3.26.0, NumPy 2.0.2. CUDA devices were hidden; OMP, OpenBLAS, MKL and PyTorch thread counts were one. No package installation or model download occurred.

Initial energy is -176.97909566649037 eV; maximum atomic force is **0.45023664 eV/Å**. This is a force-bearing input, not a qualified minimum. Initial stress in ASE Voigt order xx,yy,zz,yz,xz,xy is:

```text
[ 0.004034460386,  0.003228946712, -0.001016453290,
 -0.000995829631,  0.001425989230, -0.000390481444 ] eV/Å³
```

For each symmetric basis B, evaluate cells `A± = A(I ± hB)` with fixed fractional coordinates, h=1e-5. Diagonal B has one unit entry; each off-diagonal pair has two entries of 1/2, so `dE/dh / V = sigma_ij` without an extra factor two. Compare `(E+−E−)/(2hV)` with the initial analytic stress. One initial call plus twelve strained calls gives exactly **13** calculator requests. All calls requested energy, forces and stress together. The predeclared tolerance was 1e-5 eV/Å³; the maximum error was 2.406700897600311e-10 eV/Å³.

Evidence includes `plan.json`, a frozen script, complete `result.json` and `execution.log`. Reproducible script: `research/ga_ssw/probe_omat_aloh_stress.py` (its fixed evidence path refuses overwrite). SIGALRM limited total runtime to 180 seconds.

This establishes successful loading, interface availability and the tensile-positive stress sign/unit convention at this one real periodic geometry. It does **not** validate the joint VC coordinate map, Gaussian stress, cell search, endpoint stability or global-search efficacy. It supersedes the earlier inventory's “model load/E/F/stress not yet checked” status for this checkpoint and frame only.
