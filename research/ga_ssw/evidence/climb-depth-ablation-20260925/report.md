# Saved-path depth ablation readout

This readout asks whether a truncated true quench returns near the saved outer-step start, and how both compare with the original full landing. Start-related geometry comparisons are diagnostics because starts have no independent fresh force check.

Rows: 32 expected; 32 have result files; 0 are missing. Observed ledger calls: 1416; protocol cap: 32064.

Fresh force qualification uses independent checks. Structural and energy comparisons are scientific endpoint comparisons only when both endpoints pass fresh force qualification; otherwise measurements are archived-geometry diagnostics or unassessed. C4H6 graph equality does not imply the same conformational basin. C60 graph counts use 1.80 Å and 1.64 Å cutoffs.

| System | Rows | Missing results | Fresh-qualified truncated/full | Quench converged | Not converged | Quench/flag missing | Ledger issues |
|---|---:|---:|---:|---:|---:|---:|---:|
| C4H6 | 24 | 0 | 24/24 | 24 | 0 | 0 | 0 |
| C60 | 8 | 0 | 8/8 | 8 | 0 | 0 | 0 |

Fresh energy difference is truncated minus original full in eV. Cost difference is saved prefix plus actual new quench requests minus original full requests; fresh calls are accounted separately.

| System | Depth | N | Quench converged / not / missing | Fresh ΔE median [min, max] | Cost Δ median [min, max] requests |
|---|---:|---:|---:|---:|---:|
| C4H6 | 1 | 12 | 12 / 0 / 0 | -0.084 [-0.404, 0.000] (n=12) | -609 [-728, -509] (n=12) |
| C4H6 | 12 | 12 | 12 / 0 / 0 | -0.045 [-0.320, 0.084] (n=12) | -316 [-381, -243] (n=12) |
| C60 | 1 | 4 | 4 / 0 / 0 | -10.041 [-20.747, 0.776] (n=4) | -682 [-1368, -610] (n=4) |
| C60 | 6 | 4 | 4 / 0 / 0 | -4.549 [-20.747, 1.302] (n=4) | -406 [-826, -361] (n=4) |

Structural diagnostics by endpoint pair. Start-related rows use archived geometries and have no fresh-qualified start endpoint; they do not establish physical qualification.

| System | Depth | Pair | Geometry pairs / source rows | Fresh-qualified pairs | Same / different / unassessed graph | C4H6 RMS median [min,max] Å | C4H6 absolute Δtorsion median [min,max]° | C60 1.80Å same/diff/unassessed; both connected | C60 1.64Å same/diff/unassessed; both connected |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| C4H6 | 1 | truncated vs start | 12/12 | 0 | 12/0/0 | 0.023 [0.007, 0.074] (n=12) | 2.6 [0.4, 8.7] (n=12) | n/a | n/a |
| C4H6 | 1 | start vs original full | 12/12 | 0 | 10/2/0 | 1.268 [0.008, 1.415] (n=10) | 149.5 [0.3, 175.1] (n=10) | n/a | n/a |
| C4H6 | 1 | truncated vs original full | 12/12 | 12 | 10/2/0 | 1.285 [0.013, 1.407] (n=10) | 152.1 [1.6, 172.3] (n=10) | n/a | n/a |
| C4H6 | 12 | truncated vs start | 12/12 | 0 | 12/0/0 | 1.272 [0.005, 1.317] (n=12) | 150.0 [0.5, 157.0] (n=12) | n/a | n/a |
| C4H6 | 12 | start vs original full | 12/12 | 0 | 10/2/0 | 1.268 [0.008, 1.415] (n=10) | 149.5 [0.3, 175.1] (n=10) | n/a | n/a |
| C4H6 | 12 | truncated vs original full | 12/12 | 12 | 10/2/0 | 0.946 [0.028, 1.286] (n=10) | 110.1 [3.4, 152.7] (n=10) | n/a | n/a |
| C60 | 1 | truncated vs start | 4/4 | 0 | n/a | n/a | n/a | 4/0/0; conn=4 | 4/0/0; conn=4 |
| C60 | 1 | start vs original full | 4/4 | 0 | n/a | n/a | n/a | 0/4/0; conn=4 | 0/4/0; conn=4 |
| C60 | 1 | truncated vs original full | 4/4 | 4 | n/a | n/a | n/a | 0/4/0; conn=4 | 0/4/0; conn=4 |
| C60 | 6 | truncated vs start | 4/4 | 0 | n/a | n/a | n/a | 2/2/0; conn=4 | 2/2/0; conn=4 |
| C60 | 6 | start vs original full | 4/4 | 0 | n/a | n/a | n/a | 0/4/0; conn=4 | 0/4/0; conn=4 |
| C60 | 6 | truncated vs original full | 4/4 | 4 | n/a | n/a | n/a | 0/4/0; conn=4 | 0/4/0; conn=4 |

C4H6 torsion differences are shortest absolute circular differences of the detected CCCC dihedral; RMS is a continuous graph-compatible proper-Kabsch diagnostic with no similarity threshold. C60 geometric alignment is attempted only for connected graph-compatible pairs.

No basin count, universal ranking, or end-to-end stopping policy is inferred. These saved-state diagnostics can motivate a separate depth-policy comparison; starting-geometry physical qualification remains unmeasured.
