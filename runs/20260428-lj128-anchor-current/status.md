Current phase: LJ128 short-budget comparison completed.

Reference:
- LJ128 Cambridge/Doye reference energy: -741.332100.

Command:
```bash
MPLCONFIGDIR=/tmp/mplconfig python3 benchmarks/lj_cluster_compare.py \
  --sizes 128 --seeds 0 --budget 60 \
  --ssw-steps-per-walk 8 --ssw-proposal-relax-steps 80 \
  --baseline-source ase \
  --output runs/20260428-lj128-anchor-current/lj128_seed0_budget60_metric.json \
  --trace-output runs/20260428-lj128-anchor-current/lj128_seed0_budget60_traces.json \
  --plot-output runs/20260428-lj128-anchor-current/lj128_seed0_budget60_curves.png \
  --minima-output-dir runs/20260428-lj128-anchor-current/minima_xyz
```

Results:
- SSW best energy: -698.6448762535367
- SSW gap: 42.687223746463246
- ASE-BH best energy: -689.3661122732149
- ASE-BH gap: 51.9659877267851
- ASE-GA best energy: -675.8481645987222
- ASE-GA gap: 65.48393540127779

SSW diagnostics:
- force evaluations: 50013
- local relaxations: 60
- n_minima: 55
- duplicate_rate: 0.0
- direction_selected_bond: 25
- proposal_relax_count: 427
- proposal_relax_unconverged: 348
- true_quench_unconverged: 23
- walk_displacement_clips: 21
- fragment_rejections: 5

Geometry check:
- SSW best frame max nearest-neighbor distance: 1.104601622599308.
- SSW best frame median nearest-neighbor distance: 1.0898827771337114.
- ASE-BH best frame max nearest-neighbor distance: 4.346118940747901, so this short-budget BH result contains a more weakly connected/sparse fragment than SSW best.

Interpretation:
- Under seed0 budget60, current SSW beats ASE-BH and ASE-GA on LJ128 gap.
- This is not a literature-grade claim; all three methods are still far from the CCD global minimum at this small budget.
- Proposal relaxation remains the main bottleneck and needs trust/step retuning before larger robust claims.
