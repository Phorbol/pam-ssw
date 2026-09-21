# SSW/VC-SSW benchmark calculator readiness

2026-09-10. Read-only local inventory and official web verification. No model downloaded, no GPU calculation, no environment modification, and no atomistic benchmark executed.

## Backend matrix

| System / purpose | Backend candidate | Interface evidence | Current readiness and scientific limits |
|---|---|---|---|
| Pure Cu/Pd metal, cheap atomic/cell optimizer control | ASE EMT | E/F and periodic stress in installed ASE | Immediately importable; useful inexpensive metal-model control. Not a first-principles validation or assurance for every alloy configuration |
| Pd/Cu oxides, periodic inorganic crystals, strained cells | MACE-OMAT-0-small | MACE ASE calculator exposes E/F/stress | Checkpoint exists and MACECalculator imports in `mace_env`; actual model load and E/F/stress smoke still required. OMat materials training is relevant; unseen oxidation, surfaces, very small clusters and extreme strain require domain checks |
| C/H/O molecules, conformers, small covalent rearrangements | tblite GFN2-xTB | Official ASE E/F; method parameterized through Z=86 | Working tblite 0.7.0 research path imports; separate mace_env installation fails, see below. Semiempirical validation is distinct from DFT accuracy |
| Periodic C/H/O or Pd/Cu-containing systems | tblite GFN2-xTB, conditional control | Native tblite supports lattice/PBC and ASE stress | Element/periodic support does not establish quantitative reliability for metallic Pd/Cu or oxide energetics. Resolve import, then check SCF convergence and cell derivatives before VC tests |
| Pd/Cu/O or C/H/O chemical benchmark with EMT | Excluded as scientific backend | Parameters exist for H/C/N/O | Official documentation explicitly says those four elements are not seriously described; successful evaluation is not model validity |

The standard EMT metals are **Al, Cu, Ag, Au, Ni, Pd, Pt**. H/C/N/O entries are demonstration extras, so neither oxide stability nor covalent rearrangement should be claimed from EMT. This is confirmed both by local `ase/calculators/emt.py` and [official ASE EMT documentation](https://docs.ase-lib.org/ase/calculators/emt.html).

## MACE-OMAT-0-small

The official foundation-model table lists MACE-OMAT-0 as a materials model covering 89 elements, trained on OMAT with VASP54 PBE+U, in small/medium sizes. This metadata does not certify all geometries containing those elements. The intended role here is inorganic-material E/F/stress testing; molecule/cluster extrapolation must be evaluated separately. [Official model card](https://github.com/ACEsuit/mace-foundations).

Official release: [mace_omat_0](https://github.com/ACEsuit/mace-foundations/releases/tag/mace_omat_0). Official filename is `mace-omat-0-small.model`; source `foundations_models.py` maps `mace_mp(model="small-omat-0")` to the release asset. Its ASE entry also allows an explicit local model path, avoiding automatic network retrieval. E/F/stress handling is implemented in `MACECalculator`; stress uses ASE Voigt conversion. [Official calculator loader source](https://github.com/ACEsuit/mace/blob/develop/mace/calculators/foundations_models.py), [official calculator source](https://github.com/ACEsuit/mace/blob/develop/mace/calculators/mace.py).

Local file is already present:

```text
/home/gengjianrui/.cache/mace/mace-omat-0-small.model
67,630,500 bytes
SHA256 0abfde07862cf1e93b8b4d03cb702f29ce9c344ff2fc4de2ec0d7166d6c113a5
```

This is a locally hashed named checkpoint, not an upstream checksum authentication. Its actual elemental table and successful CPU/GPU model evaluation were not checked in this read-only inventory. Do not silently substitute local medium/MPA/OFF/OMOL models also found in the cache.

`/home/gengjianrui/.conda/envs/mace_env/bin/python` with `PYTHONNOUSERSITE=1` imported `MACECalculator`. Installed metadata: mace-torch 0.3.16, torch 2.8.0, ASE 3.26.0, tblite 0.5.0. System `python` instead has ASE but no mace/torch/tblite module specs. Run-path provenance is therefore essential. A subsequent explicit CPU float64 model-load and small periodic E/F/stress evaluation is needed before declaring the checkpoint calculator ready; no such evaluation was run here.

## tblite / xTB boundary

Official tblite ASE documentation lists `energy/free_energy`, forces and periodic stress, with PBC stress in eV/Å³. `TBLite(method="GFN2-xTB", ...)` accepts charge, multiplicity and electronic temperature. [tblite ASE interface](https://tblite.readthedocs.io/en/stable/users/ase.html).

The native Python `Structure` takes lattice and periodic flags; GFN2-xTB is parameterized for elements through Z=86. Therefore H(1), C(6), O(8), Cu(29), Pd(46) fall within the stated elemental range. This is interface/parameter coverage, not a benchmark of those chemistries. This finding concerns **tblite**, not an assertion that every `xtb` executable/version or every xTB variant has identical periodic support. [Official tblite Python API](https://tblite.readthedocs.io/en/latest/api/python.html).

Live import in the above environment failed:

```text
tblite/_libtblite.cpython-312-x86_64-linux-gnu.so:
undefined symbol: _gfortran_os_error_at
ImportError: tblite C extension unimportable, cannot use C-API
```

No environment was repaired. Root supplied an existing working research path, and this audit independently verified its import:

```sh
PYTHONPATH=/tmp/pam-ssw-tblite-20260909 python -c "from tblite.ase import TBLite"
```

It reports tblite **0.7.0**, module `/tmp/pam-ssw-tblite-20260909/tblite/ase.py`, and properties energy, energies, forces, charges, dipole, stress. Thus **tblite is available through this separate path**; repairing mace_env is not required. Root reports a completed C4H6 run on this path, but that trajectory was not independently reviewed here. An installed distribution record alone does not satisfy readiness. Once import works, E/F and finite-strain stress consistency should be checked under the chosen electronic-temperature and energy convention. SCF failures must remain explicit failed evaluations.

## What must be supplied versus already available

**No MACE model upload is needed for this named small checkpoint.** The working tblite 0.7.0 runtime is already available separately from the MACE environment; no runtime or scientific input upload is needed for that import.

For user-specific physical tests, the still necessary provenance is the actual starting structure(s) with element identities, cell/PBC, charge and multiplicity where relevant; for VC, allowed strain/cell degrees of freedom and external pressure; and any target polymorph/reference-energy data used to define success. Existing authorized structures may satisfy these needs—ask only for genuinely missing task-specific inputs. Public/built-in structures could support separately labeled controls, but must not be presented as the user's intended Pd/Cu oxide or molecular benchmark.

Model E/F/stress consistency and endpoint stationarity are implementation checks. Claims about discovery, stability or transferable SSW/VC efficacy still require complete real-system runs and appropriate physical comparison; neither successful imports nor the presence of a checkpoint substitutes for those results.

## Completed follow-up CPU preflight

The later authorized [AlOH preflight](omat-aloh-stress-preflight.md) loaded this exact OMAT-small checkpoint and completed 13 E/F/stress calls on the uploaded 26-atom periodic frame. Six-component energy finite differences match stress within 2.41e-10 eV/Å³; H/C/O/Cu/Pd are present in the loaded elemental table. This supersedes the untested-model-load status above for this explicit input and environment, without asserting relaxation or search validation.
