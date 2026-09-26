# VC mechanism screen: cost and geometry readout

Run `/home/gengjianrui/bin/pam-ssw-worktrees/vc-paper-panel-20260926/research/ga_ssw/evidence/vc-paper-panel-20260926/run-1499436/result.json` completed with status `completed` and scientific label `bounded_development_mechanism_screen_not_validation`. This analysis used the saved result and paid-request ledgers only; calculator/PES calls: **0**.

The reference gate used 6 requests and qualified the common start. Both arms used the same serialized start, seed 17, MACE-OMAT-small PES, and two outer steps; each arm used one request for its fresh start certificate plus two separate fresh endpoint checks. The arm cap was 6000 requests; no cap censoring or failed steps occurred.

| Arm | Cell rotation | Partial fixed-cell relax | Atomic climb | Full-cell true quench | Start certificate | Search total | Fresh endpoints | Total incl. gate | Wall (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cell_on | 20 | 265 | 1526 | 108 | 1 | 1920 | 2 | 1928 | 61.91 |
| cell_off | 0 | 0 | 1275 | 43 | 1 | 1319 | 2 | 1327 | 41.67 |

The gate is a shared pre-arm cost, so it is listed in each arm's inclusive total for comparability; it is counted once in the whole-run total.

| Arm | Step | Step requests | ΔH (eV) | MC accepted | Cell-cycle aggregate requests | Rotation force calls / cycle | Partial-relax steps / cycle |
|---|---:|---:|---:|---|---|---|---|
| cell_on | 0 | 905 | 0.659505 | False | [30, 29, 28, 28, 28] | [2, 2, 2, 2, 2] | [25, 25, 25, 25, 25] |
| cell_on | 1 | 1014 | 0.763256 | False | [30, 28, 28, 28, 28] | [2, 2, 2, 2, 2] | [25, 25, 25, 25, 25] |
| cell_off | 0 | 673 | 0.234709 | False | — | — | — |
| cell_off | 1 | 645 | 0.000978 | True | — | — | — |

The cell-on candidates were both valid landings but had positive ΔH (+0.659505 and +0.763256 eV), so MC rejected both. Cell-off produced +0.234709 eV (rejected) and +0.000978 eV (accepted). The cell-off endpoint on step 1 was accepted and became its final current state.

All four candidate endpoints passed their separate fresh E/F/stress qualification checks. Their measured force and maximum absolute stress values are retained in `analysis.json`.

Approximate structure matching uses the existing periodic-direction pilot's tight/broad tolerances and the same `StructureMatcher` options. Here the frames are the common initial structure and the four true-quenched candidate endpoints; cells may differ because this is variable-cell data.

| Tolerance | Approximate groups among initial + four candidates | Initial matches candidates | Cell-on/off candidate matches (2×2) |
|---|---:|---|---|
| tight | 4 | [False, False, False, True] | [False, False, False, False] |
| broad | 4 | [False, False, False, True] | [False, False, False, False] |

The pairwise matrices and greedy representative assignments are in `analysis.json`. These matches are geometry-based similarity only; they do not certify basin identity, Hessian stability, or phase identity. With one seed and two steps, the observed request and energy differences are descriptive, not an efficiency or generality result; they are insufficient grounds to remove or promote the VC mechanism.

Analysis ran as Slurm job 1499552 on CPU-MISC (rush-cpu); submit command: `sbatch --wait research/ga_ssw/evidence/vc-paper-panel-20260926/analyze.sbatch`. Elapsed analysis time 0.117 s (Python 3.12.11); no PES/calculator calls.
