# Held-out 72-atom SiO₂ joint-VC optimizer comparison

## Question and scope

This bounded development comparison asks whether Safe-total's lower joint-VC
cost observed on the AlOH26/TiO₂-phase87 short panel remains operational on an
alpha-quartz-derived 72-atom geometry. It tests optimizer robustness and cost on
the OMAT-small PES. It is not a global search validation, BKS reproduction,
phase-ordering calculation, or production optimizer ranking.

The sole arm variable is the existing optimizer adapter: Safe-total, ASE
LBFGSLineSearch, or SciPy L-BFGS-B. The input, OMAT-small calculator, joint
log-strain chart, pressure, VC config, seeds, outer-step count, request ceiling,
and wall ceiling are held fixed. Use the qualified final output of COD 1011097
job 1501259 as a 9-atom primitive-cell starting geometry, tiled 2×2×2 without
added perturbations to 72 atoms (Si₂₄O₄₈). This is larger than the 48-atom
TiO₂-phase87 example and has a different composition/network, while retaining
a traceable source and an already-qualified parent geometry. Replication gives
only one starting basin and does not make three outer steps a meaningful global
search test.

## Frozen settings and metrics

`build_plan.py` copies the configuration from the existing frozen
`plan.json` case `TiO2-phase87`: strain length 5 Å, width 0.6 Å, rotation bias
100, pressure 0, 300 K, forward force 0.1, 10 Gaussians, biased-gradient
tolerance 0.05, true force threshold 0.05 eV/Å, stress threshold
0.001 eV/Å³, max step 0.2 Å, 300 relaxation iterations, finite-difference
step 1e-4, 100 rotation HVPs, rotation tolerance 0.02, L-BFGS memory 500,
strict bias release. The AlOH26 case uses width 0.2 and 14 Gaussians; this is a
frozen TiO₂-derived transfer preset, not a universal parameter set. The 5 Å
strain length is held fixed operationally; it is not size-invariant, since at
fixed stress the log-strain cell gradient scales with the replicated volume.
The common fresh force/stress certificate remains essential.

The calculator is the existing local MACE-OMAT-0-small model, `omat_pbe`,
float64 CUDA, with cuEquivariance and OEQ disabled, and one Torch intra-op
thread. Each method/seed arm requests at most three outer proposals, 6000
charged search E/F/stress evaluations and four independent fresh endpoint
checks. Six arms give ceilings of 36,000 search and 24 fresh requests. Each arm
has a 27.5-minute cooperative search deadline and a 29-minute external process
limit inside a 30-minute one-V100 allocation. Six allocations at no more than
two concurrent GPUs cap the panel at 3 GPU-hours.

Primary operational metric: paid search E/F/stress requests per attempted outer
record, alongside fresh-valid landings per attempted record. Preserve the full
denominator of 18 planned outer slots and report actual attempts separately.
Slots never attempted after early termination or censoring are unobserved, not
optimizer failures. Failed quenches, MC rejections, missing arms and censoring
retain their status and any paid cost. Numerical endpoint criteria remain
`fmax ≤ 0.05 eV/Å` and maximum absolute pressure-residual stress `≤ 0.001
eV/Å³` at zero pressure.

The existing adapters retain their native stopping criteria and conversions:
SciPy's generalized norm conversion by √6, ASE's by √2, and their unequal
step-cap support. Therefore this is an operational optimizer comparison, not
an isolated line-search or identical-stopping cost experiment. Do not modify
the adapter or tune any settings on these outcomes.

## Execution and analysis path

Do not expand the structure on the login node. The 5-minute CPU-MISC
`prepare-plan.sbatch` uses ASE to repeat the already-qualified input and writes
a deterministic `plan.json` under `prepared-plan-<jobid>/`. That plan must be
reviewed before launching the GPU array. The six-arm GPU script writes one
arm per array task (`0–5`, concurrency at most two) and saves each worker's
result, fresh checks, task config and ledgers.

The existing `run_vc_e2e_optimizer_panel.py` and its existing `analyze.py`
remain unchanged. Because the six independent jobs are individual arms, the
zero-PES `merge_analysis_view.py` creates a symlink-only view compatible with
the existing analyzer's `run-0/plan.json`, six
`case0-{method}-seed{seed}` directories and `worker-exits.tsv` contract. It
does not copy raw ledgers or call a calculator. Preserve array outputs and
feed the resulting `run-0` to the existing analyzer in a scheduled CPU
allocation.

## Stop and interpretation

Run this only after review of the generated plan and the six-arm scripts. Stop
after the fixed panel or the existing per-arm request/time ceiling; do not
automatically extend a censored arm, increase Gaussian count, change the strain
length, or add noise. If physical qualification or biased-quench completion
degrades, retain those outcomes as evidence about transfer to this replicated
geometry. A consistent cost direction would justify at most a follow-up
decision; two seeds and three steps cannot establish stable optimizer benefit,
global-search efficiency, phase identity, α-quartz stability, or a production
ranking. No conclusion is made about the paper's BKS results.
