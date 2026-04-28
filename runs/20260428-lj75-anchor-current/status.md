Current phase: latest anchor-direction LJ75 validation completed.

Code state:
- Commit under test: `a909043 add anchored ssw direction selection`.
- SSWConfig default `proposal_fmax`: 0.02.
- True quench `quench_fmax`: 0.001.

Command:
```bash
MPLCONFIGDIR=/tmp/mplconfig python3 benchmarks/lj_cluster_compare.py \
  --sizes 75 --seeds 0 --budget 60 \
  --ssw-steps-per-walk 8 --ssw-proposal-relax-steps 80 \
  --baseline-source ase \
  --output runs/20260428-lj75-anchor-current/lj75_seed0_budget60_metric.json \
  --trace-output runs/20260428-lj75-anchor-current/lj75_seed0_budget60_traces.json \
  --plot-output runs/20260428-lj75-anchor-current/lj75_seed0_budget60_curves.png \
  --minima-output-dir runs/20260428-lj75-anchor-current/minima_xyz
```

Results:
- SSW best energy: -377.85948824349464
- SSW gap: 19.63284275650534
- ASE-GA best energy: -377.42028098008603
- ASE-GA gap: 20.072050019913945
- ASE-BH best energy: -373.387373333605
- ASE-BH gap: 24.104957666394967

SSW diagnostics:
- force evaluations: 51299
- local relaxations: 60
- n_minima: 58
- duplicate_rate: 0.01694915254237288
- direction_selected_bond: 30
- direction_selected_random: 170
- direction_selected_soft: 243
- proposal_relax_count: 443
- proposal_relax_unconverged: 377
- true_quench_unconverged: 19
- walk_displacement_clips: 11
- fragment_rejections: 1

Geometry check:
- SSW best frame max nearest-neighbor distance: 1.1053127277571655
- SSW best frame median nearest-neighbor distance: 1.076825124930088
- SSW worst exported frame max nearest-neighbor distance: 3.291504331690055, corresponding to the rough initial structure.

Interpretation:
- Latest SSW still beats this short-budget ASE-GA and ASE-BH run on LJ75 seed0.
- Latest anchor direction mechanism is active; bond directions are selected in standard SSW.
- Duplicate rate is much lower than the previous LJ75 run.
- Best gap is worse than the previous pre-anchor SSW run, so anchor-direction defaults need retuning for LJ75.
- Proposal relaxation remains the main production bottleneck.
