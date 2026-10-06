# TiO2 phase qualification before a VC search panel

Purpose: choose a phase-search test with a meaningful structural target on the
existing OMAT-small PES. This is a bounded development qualification, not a VC
global-search comparison, DFT reproduction, barrier computation, or stability proof.

## Source and task

The 2014 VC-SSW paper (10.1039/c4cp01485e, author full text in
`ga-ssw-behavior-parity/literature/benchmark-sources/vc2014`) tests BKS SiO2,
periodic LJ (hcp target), and a 12-atom MA TiO2 rutile/anatase network. Those
PES conventions are not interchangeable with OMAT. Available coordinates used
here actually come from the **2017 SSW-NN** SI, DOI
[10.1039/c7sc01459g](https://doi.org/10.1039/c7sc01459g), source SI MD5
e8e463594e595fa81b2cb2fec1b73997. That paper discovers porous phase-87 in a
48-atom PES and samples phase-87 to anatase, followed by VC-DESW TS validation.
We use its 12-atom pathway IS/FS plus rutile as a distinct phase control;
the larger phase-87 cell is an identity reference only. Rounded source NN
energies and reported DFT barriers are not OMAT reference energies.

## Competing explanations and resulting decisions

1. Distinct, force/stress-qualified phase-87 and anatase stationary structures,
with anatase lower on OMAT: a subsequent same-model target-discovery panel is
meaningful. It must still qualify actual escapes and structure identity; this
stage does not prove the proposed seeds are Hessian-stable minima.
2. Phase-87 relaxes into anatase without escape, or anatase is higher: this pair
does not support the intended lower-energy phase-discovery test. Preserve the
result, do not tune the quench to keep the source phase alive or relabel the task
as successful global optimization. Select a different qualified paper case.
3. Quenches fail numerically or identity tolerances merge known different phases:
resolve that qualification defect before search. More SSW steps cannot repair
an undefined target. Do not infer physical instability merely from quench failure.

## Geometry calibration (zero E/F/stress)

Existing `pymatgen_identity` uses scale=False, primitive_cell=True and
attempt_supercell=True. Existing tight (ltol .05, stol .10, angle 2 degrees)
and broad (.20, .30, 5 degrees) settings are tested without tuning. Positive
controls are an identical anatase FS after relabeling, lattice wrapping,
unimodular cell basis change and supercell repetition. Negative controls are
rutile, TiO2-II, brookite and phase-87 (two representations). Source IS/FS to
phase-reference matches are reported, not assumed from filenames. Space group
at symprec .01/.1 A is secondary diagnostic only. Failed checks remain evidence
and prevent using the corresponding matcher as a target criterion.

## Fixed model and numerical bounds

`plan.json` is the authority for exact source paths/hashes and settings:
MACE OMAT-0-small / omat_pbe / float64 CUDA, no cueq/oeq, p=0;
existing all-DOF Safe-total/history500, strain length 5 A, max_step .2 A,
maxiter 300, physical atomic fmax .05 eV/A and stress max component .001 eV/A^3.
These are explicit inherited development tolerances/metric, not claimed to be
2014 paper choices or universal optima. Stress tolerance is about .1602 GPa.
Cell and atom coordinates both change in this true quench. No Gaussian, LS,
MC, constraints, heating, perturbations or search is introduced.

Three sequential 12-atom cases, each at most 1000 paid E/F/stress requests and
180 seconds; at most 600 seconds combined search and 6 independent cold checks
(initial + final per case), one V100 allocation for at most 12 minutes. Empty
calculator cache per case; request cost and actual calculator invocations
reported separately. A cached request still counts; failures and cap denials
are retained. No continuation, resampling or automatic retry. This small budget
tests whether case/model/identity are suitable; it does not estimate GM rate.

## Evidence and smallest verification

Python compile, wrapper syntax and zero-PES input/API/ledger preflight precede
submission. Model hash, actual imports, core tree, code/plan/source snapshots,
scalar ledger, initial and endpoint extxyz, quench termination and independent
cold E/F/stress certificates are saved. Raw runs and derived geometric readout
use fresh directories. Force/stress qualification, execution completion and
phase identity are separate; independent fresh agreement is necessary before
an endpoint supplies the next experiment. No core change/default promotion.
