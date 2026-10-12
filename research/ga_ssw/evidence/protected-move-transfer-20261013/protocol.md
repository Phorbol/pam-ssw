# Protected-move transfer panel (protocol created 2026-10-12; series ID 20261013)

Status: executed as GPU1754531 after CPU1754506; see [readout](readout.md). This is a bounded cross-initial-structure
transfer check for the fixed S2 protected-move comparison. It is not parameter
tuning, a success-rate estimate, or a generalization claim.

## Question and decision

Does the first-panel protected-move behavior transfer from C60 isomer 2 / TiO2
phase 87 to a separately qualified C60 isomer-1 minimum and the paper's
anatase-final (FS) endpoint? The paired arms differ only by `move_policy=None` versus
`ScalarMovePolicy()`. Results can support retaining the mechanism for further
testing only if the raw guard diagnostics explain a paired endpoint/cost
difference. No activation means stop and audit protocol/configuration; a mixed
or null result retains the opt-in status. This one-seed, one-step panel cannot
support global search efficiency or broad transfer claims.

## Frozen panel and budget

The cases are C60 isomer 1 with MACE-MH-1 (`omol`) and TiO2 anatase FS with
OMAT-small (`omat_pbe`). Each uses the existing S2-only setting from
`probe_protected_move_paper_cases.py`, same optimizer and direction parameters,
S2 descriptor construction, fixed-cell conditions, backend precision, and
guard policy as the 2026-10-12 panel. Four arms total: off/on for each case,
one outer step per arm, seed `20261013`. Limits are 3000 search requests and
one fresh endpoint request per arm, six Gaussians, one V100, 30 minutes total;
the shared runner retains its seven-minute per-arm wall cap. No retries or
automatic continuation. Hard search/fresh failures stop later arms; all
attempts and costs remain recorded. The fresh check uses the actual landing;
without a landing, a saved failed state may be checked for diagnosis only and
cannot qualify. No initial/fallback state is substituted.

The case structures and model hashes are fixed in the wrapper. C60 input
`qualification/isomer-1/final.extxyz` has a saved independent fresh force check
(0.004075 eV/A) and preserved connected 60-carbon degree-3 cage graph at
1.64/1.70/1.80 A. Its labeled source member is `c60/c60-iso-1_opt.xyz` from
The archived source dataset identifies the member as `c60/c60-iso-1_opt.xyz`
and records DOI `10.1038/s41524-024-01410-7`. The local endpoint is the
separately MH-1 force-qualified structure; this does not claim DFT force
qualification or vibrational stability. The saved cage already matches the
Ih graph with degree three at every carbon; this panel measures escape-from-a
qualified cage and endpoint validity/cost transfer, not the probability of
discovering Ih or a defect-repair success rate.

The TiO2 input is the 12-atom
`run-1663408/anatase_fs/endpoint.extxyz`, identified in metadata as the paper's
“FS (anatase)” endpoint and tied by the frozen manifest to SI section 7 of the
2017 paper DOI `10.1039/c7sc01459g`. The existing qualification record reports
optimizer convergence and independent fresh endpoint checks passing (fmax
0.01201 eV/A; stress max 0.0009538 eV/A^3) using the same OMAT-small model/head.
The repository's prior decision selects this paper FS target and excludes
rutile as a search input because its paper label and measured symmetry do not
agree. This transfer panel makes no new phase-stability or first-principles
accuracy claim.

Before execution, the CPU preflight must report finite nonzero S2 value and
gradient for both exact input files and validate all source/model/qualification
hashes. A zero or invalid S2 coordinate is a stop condition: do not perturb the
structure or tune its descriptor. The preflight makes no PES calls. The runner
and protocol are versioned in the resulting provenance; all SSW calculator
requests use the raw calculator wrapped by the shared `CountedSurface` ledger,
not `ASESurface`.

## Execution and evidence boundary

`probe_protected_move_transfer.py` is a thin wrapper over the current
checkout's protected-move runner and the existing Q-paper input/config/ledger
helpers. It does not import code from a data directory. The `--smoke` path uses
the inherited per-case harmonic calculator and identity CBD stub only to
exercise the four-arm runner plumbing; it is not scientific evidence. The
`--execute` path does not patch the CBD solver and loads the frozen models on
one V100.

Commands from the frozen code snapshot (with the validated evidence checkout
passed separately as the data root; each output directory must be new):

```bash
python research/ga_ssw/probe_protected_move_transfer.py \
  /path/to/data-root/research/ga_ssw/evidence/protected-move-transfer-20261013/preflight \
  --data-root /path/to/data-root
python research/ga_ssw/probe_protected_move_transfer.py \
  /path/to/data-root/research/ga_ssw/evidence/protected-move-transfer-20261013/smoke \
  --data-root /path/to/data-root --smoke
```

The prepared `cpu-preflight-smoke.sbatch` runs these commands on a CPU node;
`gpu.sbatch` contains the bounded one-V100 execution command. Both scripts were submitted after source freezing. Submit either script as `sbatch SCRIPT CODE_SNAPSHOT DATA_ROOT`.
The default data root is the validated worktree. Do not execute the GPU script until the CPU preflight shows
both S2 coordinates qualified and the complete source/hash gate passes.

Reuse `../protected-move-20261012/analyze.py` for paired readout; its optional
prior input supports transfer without copying analysis logic. Preserve raw
results, failures, boundary statuses, request ledgers, fresh endpoint records,
and calculate-call costs. Do not replace or overwrite first-panel artifacts.
