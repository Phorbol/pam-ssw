# C60 local-defect escape: fixed-protocol development result

Four trajectories (one source defect × seeds1101/1102 × SSW/native-LS), ten attempts each, completed on MH-1/omol. This is local-defect development evidence, not random-C60 global acceptance or a general LS ranking. Input qualification and sources: [report](report.md); frozen settings: [protocol](escape-plan.md).

## Execution and qualification

GPU1484387 completed in30m35, exit0; CPU1484415 analysis completed in1s, exit0. Actual26509 search requests +44 fresh checks;25003 search Calculator calls plus44 fresh calls. Every saved-record ledger closes, with zero requests outside saved records. The44 fresh checks represent four starting roles plus40 landing roles, not44 unique geometries.

| Method / seed | Search requests | Force-qualified landings | Changed graph observations | Intact cage landings | Fragmented at1.8Å | Ih recovery |
|---|---:|---:|---:|---:|---:|---:|
| SSW1101 |6320|10|9|1|2|0|
| LS1101 |6387|10|9|1|2|0|
| SSW1102 |6904|10|10|0|1|0|
| LS1102 |6898|10|10|0|0|0|

All40 landings meet fresh force≤0.03eV/Å, but only two retain cage topology; both retain the original defect graph. The38 graph-change observations are not38 unique basins or useful repairs. None reaches the Ih reference energy window. Common-search prefixes6320/6898 requests also contain no recovery (the slower arm has only nine completed landings).

Seed1101 accepts one near-identical defect landing per method; seed1102 accepts none. Best landing energies relative to Ih are1.2226075/1.2225490eV for SSW/LS1101 and9.9673265/9.9668587eV for1102. The retained best for1102 remains the INITIAL defect at+1.22316eV. Near-equal best energies do not imply matching trajectories: other landing energies differ substantially. No precision-level benefit is claimed.

## Mechanism readout and limits

[LS state audit](ls-state-audit.md) confirms active nonzero LS with strength updates in20/20 records. Positive prequench responses3.46–8.12meV/atom remain below the configured20meV/atom target: these ten-step runs sample adaptation startup, not settled-strength LS performance. This is not evidence that LS was disabled or that its strength should automatically be increased.

CPU1484825 completed in1m06, exit0. [Saved-path audit](saved-path-audit.json), produced by[audit_saved_path.py](audit_saved_path.py), uses480 biased endpoints and zero PES requests. Ih matches are0/480 at each cutoff1.64/1.7/1.8Å. Intact-cage counts are106/113/115; at1.8Å all40 first and40 second endpoints are cages, but all40 final endpoints are non-cages. Geometry under bias does not establish irreversible bond breaking or a true minimum.

Two explanations remain: early endpoints could quench to useful structures subsequently lost during further climbing, or the directions never reach a repair channel. Saved geometry alone cannot distinguish them. The next bounded diagnostic uses stages2/3/6 of the FIRST attempt of all four arms, chosen uniformly, rather than selecting favorable outcomes. No new full trajectory, parameter scan, strength adjustment or default change follows from this result. Prior C4H6/random-C60 depth evidence still argues against a general early-stop rule.

Raw results/checkpoints/request ledgers remain in `runs/` in this worktree. Derived readout:[escape-analysis.json](escape-analysis.json). Frozen runner f51ed24; runtime documentation HEAD f7b3496; execution snapshot `runs/execution.json`. Large raw artifacts remain outside Git and must not be deleted with the worktree.
