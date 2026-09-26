# C60 endpoint path diagnostic

This directory defines a bounded, endpoint-informed NEB diagnostic for the
local C60 cage-defect question. It is a diagnostic of one known endpoint pair;
it does not provide the endpoint or a path to SSW, and does not qualify an
SSW search strategy or a transition state.

## Question and competing explanations

The existing MH1 C60 defect-cage search often produces high-energy, damaged
structures. This diagnostic asks whether a physically connected path between
two independently specified endpoints has an energy scale and initial
tangent that can be related to the local soft-mode spectrum already measured
at the starting endpoint.

The main competing interpretations are:

1. A moderate-energy path exists and its initial tangent has appreciable
   overlap with one or more low-curvature local directions. This would make
   local tangent/curvature alignment a plausible explanatory variable, but
   not proof that SSW should find this path.
2. The path remains high in energy, or the initial tangent is poorly aligned
   with the soft subspace. This would weaken the claim that local softness
   alone explains the destructive SSW outcomes and leave finite-displacement,
   nonlinear, or basin-connectivity effects open.

The NEB uses both known endpoints, so its results are an independent path
diagnostic, not evidence that the endpoint was discoverable without target
knowledge. Do not pass either endpoint, the aligned path, or NEB information to
the SSW run being interpreted.

## Fixed protocol

- Inputs: two same-composition, same-order, isolated C60 extxyz structures.
  The separate spectrum study supplies/validates the atom mapping; this runner
  does not infer or repair it.
- Calculator: `/home/gengjianrui/.cache/mace/mace-mh-1.model`, MACE `omol`
  head, float64, CUDA, cueq and oeq disabled.
- Seven images including endpoints; ASE 3.26.0 `NEB(method='improvedtangent',
  k=0.1, allow_shared_calculator=True)`, serial images.
- IDPP interpolation; no endpoint relaxation.
- The linear seed and IDPP relaxation write their optimizer trajectory/log into
  the selected run output directory, not the repository working directory.
- FIRE phase 1: at most 50 steps, no climbing image. FIRE phase 2: climbing
  image enabled, at most 150 steps. Both use `fmax=0.05 eV/Å`, ASE's standard
  NEB projected-force stopping criterion.
- Hard cap: 3000 calls to the actual calculator `calculate` method, including
  final fresh energy/force evaluations. Program deadline: 18 minutes. The
  intended one-GPU Slurm allocation is 20 minutes; submission is managed
  outside this directory.
- Every optimizer iteration writes all seven geometries to `neb.traj`.
  The final path records fresh per-image energies and physical forces. The
  maximum physical force is reported separately from the NEB projected-force
  convergence criterion.

Parameters are fixed for this discriminating diagnostic, not proposed SSW
defaults. Results cannot establish a converged minimum-energy path, a
transition-state Hessian index, a qualified transition state, or SSW success
probability. In particular, a high point in this discretized path is not a
certified saddle.

## Run

Use the isolated MACE environment to avoid user-site ASE shadowing:

```bash
PYTHONNOUSERSITE=1 /home/gengjianrui/.conda/envs/mace_env/bin/python \
  research/ga_ssw/evidence/c60-path-diagnostic-20260927/run_neb.py \
  --initial research/ga_ssw/evidence/c60-path-diagnostic-20260927/inputs/initial.extxyz \
  --final research/ga_ssw/evidence/c60-path-diagnostic-20260927/inputs/final.extxyz \
  --out research/ga_ssw/evidence/c60-path-diagnostic-20260927/run-001
```

Do not run the MACE protocol on a login node. A zero-model EMT smoke can be
run on CPU with `--calculator emt --initial smoke-initial.extxyz
--final smoke-final.extxyz --out <new-empty-directory> --smoke`. The smoke
mode uses the same ASE NEB construction, counting, trajectory writer,
final-force serialization, and summary/error path, but only one step per
phase; it does not test the C60 model or physical conclusions.

The output directory is exclusive and must not already exist. On completion,
inspect `summary.json`, `neb.traj`, and `path.extxyz` together. If the process
hits its deadline or call cap, the partial trajectory and summary are retained
and the runner exits unsuccessfully. The summary distinguishes calculator
calls started from calls that returned successfully.
