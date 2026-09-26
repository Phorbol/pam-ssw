# Public pool-adapter identity qualification

The opt-in ASE matcher passed the fixed self-pair adapter qualification on the prior 33-geometry panel. This is a narrow callback/checkpoint identity check, not evidence of search improvement or broad false-merge safety.

CPU-MISC job 1500597 (`sjtu-caoxiaoming`, `rush-cpu`) completed in 17 seconds with exit code 0 and zero PES evaluations. Against each of 33 references, it exercised an exact copy, three rigid transforms, three same-species permutations, and three combined transforms in both ordered-v1 and ASE-v1 modes: 330 cases per mode.

For ASE-v1, all 33 exact copies and all 297 transformed cases produced one archive entry with observation mapping `[0, 0]`. The first stored atomic numbers and coordinates remained exactly in the original order. Each duplicate outcome was marked duplicate and all 330 duplicate node-success counts were zero. Continuous callback execution and export/restore continuation yielded identical complete exported state for every case.

The explicit ordered-v1 control retained the previous behavior: all 33 exact and 99 rigid-only cases merged; all 99 permutation and 99 combined cases remained two entries. On exact-copy pairs, the default constructor's mapping and checkpoint contract equaled explicit ordered-v1. Ordered-v1 payload version 1 restored in ordered-v1 mode; ASE-v1 payload version 2 rejected both the old ordered-v1 payload and a v1-version payload.

Inputs and implementation provenance are embedded in [result.json](result.json), including the 33 geometry source hashes, main commit `f756b962ddf85cf17c34c9a8bbbd063395f86667`, and hashes of the three relevant main-worktree source files. The fixed protocol and run command are in [protocol.md](protocol.md) and [run.sbatch](run.sbatch).

Passing qualifies only the public adapter behavior for these geometries and transforms under the frozen 0.1 Å threshold. It does not establish matching precision on pairs of distinct structures, false-merge rates, pool contamination, physical basin identity, or improved restart/search performance. The ordered-v1 mode remains the default.
