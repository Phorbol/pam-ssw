# Direction-component source panel (bounded integration pilot)

Question: with the same frozen first direction, configuration, input, and four
RNG seeds, does the local pair/group term or the previous-stage displacement
feedback change what one subsequent SSW outer attempt exposes and where it
lands? This is an implementation/exposure pilot, not a success-rate test.

## Frozen arms and resources

The matrix is LJ38 SLM, a saved non-GM intact LJ55 cluster, and the saved C60
source-3 defect cage. Each is run with full source (`6c70ef4`), local-only
(`a13ab976`), and feedback-only (`7e2a309`), using seeds 26101231–26101234.
Each arm requests one outer step, at most 2,000 paid search E/F requests
including startup, and at most one separate fresh E/F check. The ceiling is
36 arms, 72,000 search requests and 36 fresh requests. No retries, checkpoint
resume, budget transfer, redraw, or early success-based stopping is used.
The LJ cases use the existing untruncated nonperiodic FullPairLJ
(`epsilon=1 eV`, `sigma=2.7 A`) panel settings: width 0.6 A, 14 Gaussians,
temperature 464.2049555 K (ordinary `kBT` is about 0.04 eV; the existing
native Metropolis exponent includes a factor of 20, giving an effective
0.8 eV scale), true
`fmax=0.05 eV/A`, bias `fmax=0.1`, 1,000 quench steps, finite-difference step
0.001 A, rotation bias 1, HVP 39, tolerance 0.02, safe-total L-BFGS/history
500, and force-or-budget stopping. Existing direction settings are retained,
with the full direction's `c1_radius_policy=per_atom`; the height policy is
the existing ConservativeNativeHeightPolicy `(initial_weight=0.5, negative_weight=0.2, level=1, max_height=10, growth_step=1, growth_scale=2)` and
NativeMC `(0.1 eV, 99999)`. No setting is fitted in this panel.

C60 uses the existing source-3 MH-1 model (SHA256
`a522eb7f59c7879963d41586528f4980baf33e086c94aa92e3eafdeccad3be47`), omol
head, float64 CUDA, deterministic Torch seed 0/one thread, TF32 off, and
cuEq/oeq off. Its saved config is read from the source-3 plan: 12 Gaussians,
width 0.6 A, 150 K, true `fmax=0.03 eV/A`, bias `fmax=0.1`, 1,000 quench
steps, finite difference 0.001 A, HVP 39, rotation tolerance 0.02,
safe-total/history 500, and NativeMC `(0.1 eV, 99999)`. It has no LJ height
policy. No LS, pool, or new optimizer/configuration is introduced.

The LJ inputs are the existing LJ38 SLM endpoint and existing prepared intact
non-GM LJ55 geometry. C60 uses the existing `initial-fresh.traj` for source 3.
The runner requires the expected composition, no constraints, and `pbc=False`
for all three. Each method runs in a fresh process so the imported core must
resolve under its pinned worktree; the runner checks the exact revision and
the actual `native_direction_control.py` path/hash before executing.

Per arm, the runner saves the initial minimum, every outer landing and current
structure, climb diagnostics, request ledger, paid versus actual calculator
calls, cap denials, exception/failure, source provenance, and elapsed time.
Second-Gaussian exposure means `climb.gaussian_index >= 1` with recovered
direction diagnostics. A zero-release diagnostic at that index is still an
exposure: it is one of the behaviors this ablation is meant to observe. All
saved landings remain available even when Monte Carlo rejects one. The single
fresh calculation checks the lowest-energy force-qualified saved landing;
it explicitly resets the calculator before evaluating and records actual
calculator calls separately. It never substitutes `result.best/current` for
the saved landing set. If there is no force-qualified landing, fresh cost is
zero. LJ geometry uses the existing 1.3-sigma graph plus proper-rotation RMS
comparison to the published GM reference; C60 graph diagnostics use the
existing 1.64/1.70/1.80 A validator. For all cases, the runner’s qualification gate is only finite coordinates and
nonperiodic unconstrained input; the LJ/C60 graph and geometry checks are
diagnostics, not a C60 PES-qualification gate or basin labels. A short, one-outer endpoint cannot establish discovery rate,
basin population, or general algorithmic effectiveness.

The single-job caps are 10 minutes per LJ case (CPU), 20 CPU-minutes total for
both LJ cases, and 30 minutes total for the C60 case (one V100, with the three source workers run sequentially and
each loading the model once). A shared monotonic deadline allows 540 seconds
of search for each LJ job and 1,680 seconds for C60; the remaining scheduler
time is reserved for final serialization. The LJ two jobs may use at most two
CPU slots concurrently; their job-time ceilings therefore sum to the stated
20 CPU-minutes. Each process receives only the
remaining time to that same case-wide deadline, so a new method process does
not reset the budget. The scheduler time cap bounds process startup, model
loading, searches, and output together. If a process fails, `set -e`
stops that case and preserves the already written arm outputs; no arm is
retried. Untouched arms remain unrun, not failures. The request caps remain
independent of scheduler walltime.

## Commands

Zero-PES preflight (run inside the existing `mace_env`; it checks imports,
input hashes/composition/constraints/PBC, pinned source resolution, model
hash, frozen settings, and aggregate budgets):

```bash
cd /home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification
PYTHONNOUSERSITE=1 /home/gengjianrui/.conda/envs/mace_env/bin/python research/ga_ssw/evidence/direction-components-panel-20261012/run.py --preflight > research/ga_ssw/evidence/direction-components-panel-20261012/preflight-zero-pes.json
```

After review, the bounded commands are represented by
`run_lj_case.sbatch <lj38|lj55>` and `run_c60.sbatch`. They are not submitted
by the preflight. Each invocation creates a new job-ID-specific output root;
never point it at an existing root.

## Interpretation rule

Compare same-case/same-seed outcomes and costs, then retain both discordant
seeds and all non-target outcomes. A local-only or feedback-only difference
in index-1 exposure, landing energy, or structural diagnostics is evidence
about this one-step mechanism on these saved inputs. It does not identify a
transition-state path or prove a basin change. Mixed directions across seeds
or cases leave the mechanism context-dependent and do not justify a default
change. No PES outcomes exist until the explicit compute jobs are run.

## Corrected C60 execution after zero-PES configuration failure

Job1756507 stopped before its first PES call: the saved plan uses
`energy_tol_eV`, whereas NativeMCSettings accepts `energy_tol`. This is a
runner translation error, not a direction result. Its failed slot and logs
remain archived. Actual allocated runtime was 11 seconds. The correction
explicitly translates the field with the same 0.1 eV value and constructs all
settings under each source during zero-PES preflight. The corrected run has
a new job/output identity, 29-minute allocation and 27-minute search window;
together with the failed startup it remains below the 30-minute GPU-series
ceiling. No search was replayed, no seed changed, and no automatic retry is
installed. The 24 completed CPU arms are not rerun.
