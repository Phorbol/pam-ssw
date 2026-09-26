# LS pool-routing panel: offline analysis

This report reads archived result, fresh-check, and request-ledger files only; it performs no PES evaluations.
Candidate counts are diagnostics. Scientific readout uses fresh-certified observations and paid search E/F cost.

## Arm summary

| Case | Mode | Status | Censored | Outer records | Landings conv / nonconv / no landing | Fresh certified minima / cert nonconv / failed / missing | Search / fresh paid E/F | Committed restarts | Issues |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C4H6 | mc | evaluation_failed | True | 51 | 50/0/1 | 51 / 0 / 0 / 0 | 33031 / 51 | 0 | outer record count 51 differs from protocol steps 100 |
| C4H6 | uniform | evaluation_failed | True | 71 | 70/0/1 | 71 / 0 / 0 / 0 | 35687 / 71 | 61 | outer record count 71 differs from protocol steps 100 |
| C4H6 | pam | evaluation_failed | True | 56 | 55/0/1 | 56 / 0 / 0 / 0 | 35536 / 56 | 36 | outer record count 56 differs from protocol steps 100 |
| C60-isomer2 | mc | recovered_from_raw_artifacts_postprocess_incomplete | True | 66 | 65/0/1 | 66 / 0 / 0 / 0 | 28338 / 66 | 0 | missing artifacts: offline-identity.json; outer record count 66 differs from protocol steps 100 |
| C60-isomer2 | uniform | ls_prequench_failed | True | 61 | 60/0/1 | 61 / 0 / 0 / 0 | 23877 / 61 | 58 | outer record count 61 differs from protocol steps 100 |
| C60-isomer2 | pam | evaluation_failed | True | 61 | 60/0/1 | 61 / 0 / 0 / 0 | 24009 / 61 | 39 | outer record count 61 differs from protocol steps 100 |

## Common paid-search prefixes

An exact common endpoint is shown only when all three modes have closed search/result and fresh accounting with full fresh-check coverage. Ledger-only amounts are lower bounds and are never used as exact prefixes.

### C4H6

Common search prefix: 33031 E/F requests.

- **mc:** {"best_fresh_energy_eV": -4243.921655520654, "c4h6_fragmented_new_graph_classes": 2, "c4h6_noninitial_connected_new_topology_classes": 3, "connected_certified_observations": 49, "fragmented_certified_observations": 2, "fresh_converged_certified_minima_in_prefix": 51, "prefix_requests": 33031}
- **uniform:** {"best_fresh_energy_eV": -4243.921610659256, "c4h6_fragmented_new_graph_classes": 1, "c4h6_noninitial_connected_new_topology_classes": 2, "connected_certified_observations": 60, "fragmented_certified_observations": 7, "fresh_converged_certified_minima_in_prefix": 67, "prefix_requests": 33031}
- **pam:** {"best_fresh_energy_eV": -4243.921683391849, "c4h6_fragmented_new_graph_classes": 2, "c4h6_noninitial_connected_new_topology_classes": 2, "connected_certified_observations": 40, "fragmented_certified_observations": 12, "fresh_converged_certified_minima_in_prefix": 52, "prefix_requests": 33031}

### C60-isomer2

Common search prefix: 23877 E/F requests.

- **mc:** {"best_fresh_energy_eV": -62214.17066180569, "c60_graphs_by_cutoff": {"1.64": {"cage_candidates": 12, "ih_graph_matches": 0, "source_defect_graph_matches": 12}, "1.7": {"cage_candidates": 12, "ih_graph_matches": 0, "source_defect_graph_matches": 12}, "1.8": {"cage_candidates": 12, "ih_graph_matches": 0, "source_defect_graph_matches": 12}}, "energy_window_eV": 0.01, "fresh_converged_certified_minima_in_prefix": 55, "ih_energy_window_observations": 0, "prefix_requests": 23877}
- **uniform:** {"best_fresh_energy_eV": -62214.17020995467, "c60_graphs_by_cutoff": {"1.64": {"cage_candidates": 3, "ih_graph_matches": 0, "source_defect_graph_matches": 2}, "1.7": {"cage_candidates": 3, "ih_graph_matches": 0, "source_defect_graph_matches": 2}, "1.8": {"cage_candidates": 3, "ih_graph_matches": 0, "source_defect_graph_matches": 2}}, "energy_window_eV": 0.01, "fresh_converged_certified_minima_in_prefix": 61, "ih_energy_window_observations": 0, "prefix_requests": 23877}
- **pam:** {"best_fresh_energy_eV": -62215.39289098097, "c60_graphs_by_cutoff": {"1.64": {"cage_candidates": 4, "ih_graph_matches": 1, "source_defect_graph_matches": 3}, "1.7": {"cage_candidates": 4, "ih_graph_matches": 1, "source_defect_graph_matches": 3}, "1.8": {"cage_candidates": 4, "ih_graph_matches": 1, "source_defect_graph_matches": 3}}, "energy_window_eV": 0.01, "fresh_converged_certified_minima_in_prefix": 61, "ih_energy_window_observations": 1, "prefix_requests": 23877}

## Interpretation limits

Committed restarts and their following landing are descriptive associations, not isolated causal effects.
Graph isomorphism classes are topology summaries, not geometrical minima, reaction pathways, barriers, or proof of chemical stability.
C60 cage/Ih graph matches and the IH energy window are separate criteria. This panel is a one-seed-per-case developer screen, not a population ranking.
