# LS pool-routing panel: offline analysis

This report reads archived result, fresh-check, and request-ledger files only; it performs no PES evaluations.
Candidate counts are diagnostics. Scientific readout uses fresh-certified observations and paid search E/F cost.

## Arm summary

| Case | Mode | Status | Censored | Outer records | Landings conv / nonconv / no landing | Fresh certified minima / cert nonconv / failed / missing | Search / fresh paid E/F | Committed restarts | Issues |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C60-isomer2 | mc | evaluation_failed | True | 68 | 67/0/1 | 68 / 0 / 0 / 0 | 30017 / 68 | 0 | outer record count 68 differs from protocol steps 100 |
| C60-isomer2 | uniform | ls_prequench_failed | True | 73 | 72/0/1 | 73 / 0 / 0 / 0 | 27709 / 73 | 63 | outer record count 73 differs from protocol steps 100 |
| C60-isomer2 | pam | ls_prequench_failed | True | 68 | 67/0/1 | 68 / 0 / 0 / 0 | 26141 / 68 | 39 | outer record count 68 differs from protocol steps 100 |

## Common paid-search prefixes

An exact common endpoint is shown only when all three modes have closed search/result and fresh accounting with full fresh-check coverage. Ledger-only amounts are lower bounds and are never used as exact prefixes.

### C60-isomer2

Common search prefix: 26141 E/F requests.

- **mc:** {"best_fresh_energy_eV": -62214.17044504409, "c60_graphs_by_cutoff": {"1.64": {"cage_candidates": 12, "ih_graph_matches": 0, "source_defect_graph_matches": 12}, "1.7": {"cage_candidates": 12, "ih_graph_matches": 0, "source_defect_graph_matches": 12}, "1.8": {"cage_candidates": 12, "ih_graph_matches": 0, "source_defect_graph_matches": 12}}, "energy_window_eV": 0.01, "fresh_converged_certified_minima_in_prefix": 59, "ih_energy_window_observations": 0, "prefix_requests": 26141}
- **uniform:** {"best_fresh_energy_eV": -62214.170454110856, "c60_graphs_by_cutoff": {"1.64": {"cage_candidates": 8, "ih_graph_matches": 0, "source_defect_graph_matches": 8}, "1.7": {"cage_candidates": 8, "ih_graph_matches": 0, "source_defect_graph_matches": 8}, "1.8": {"cage_candidates": 8, "ih_graph_matches": 0, "source_defect_graph_matches": 8}}, "energy_window_eV": 0.01, "fresh_converged_certified_minima_in_prefix": 67, "ih_energy_window_observations": 0, "prefix_requests": 26141}
- **pam:** {"best_fresh_energy_eV": -62214.17056399603, "c60_graphs_by_cutoff": {"1.64": {"cage_candidates": 9, "ih_graph_matches": 0, "source_defect_graph_matches": 6}, "1.7": {"cage_candidates": 9, "ih_graph_matches": 0, "source_defect_graph_matches": 6}, "1.8": {"cage_candidates": 9, "ih_graph_matches": 0, "source_defect_graph_matches": 6}}, "energy_window_eV": 0.01, "fresh_converged_certified_minima_in_prefix": 68, "ih_energy_window_observations": 0, "prefix_requests": 26141}

## Interpretation limits

Committed restarts and their following landing are descriptive associations, not isolated causal effects.
Graph isomorphism classes are topology summaries, not geometrical minima, reaction pathways, barriers, or proof of chemical stability.
C60 cage/Ih graph matches and the IH energy window are separate criteria. This panel is a one-seed-per-case developer screen, not a population ranking.
