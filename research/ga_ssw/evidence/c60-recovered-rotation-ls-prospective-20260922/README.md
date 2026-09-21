# C60 recovered-rotation + NativeLS prospective package

This directory is a frozen executable preparation package. It contains no
model copy and has not submitted or run a GPU search. `source` is a symlink to
the old long-budget source tree; the two `.traj` files are copied frozen
inputs because that is the runner's actual hash contract. The original ARC
paths remain provenance only.

The proposed comparison reuses the two old development inputs `17093` and
`17094` from `c60-random-native-development-20260917`. It deliberately does
not use the current held-out `17095/17096` inputs. Each proposed arm would add
the existing `NativeLSSettings` to the already prepared recovered-rotation
long-budget protocol: 60,000 search requests per arm, one V100, a 5,400 s
program wall cap inside a 6,000 s (100 min) allocation. Initial and best
structures receive separate fresh E/F checks; a
new first cage may receive one conditional extra check per case without
extending search.

The recovered-rotation settings, SSW configuration, NativeMC settings, model
and runtime must be inherited exactly from
`c60-recovered-rotation-long-20260921/plan.json`. The LS values are copied
field by field from `mh1-native-ls-equal-budget-20260920/plan.json`; they are
release-derived inputs, not fitted values. The planned factor is enabling LS
only. The existing 12,000-request MH1 NativeLS result used the ordinary
rotation path, so it cannot answer whether LS and recovered rotation compose;
this proposal targets that specific evidence gap without changing CBD or the
common SSW protocol.

The runner's actual frozen-input contract
uses the prepared `inputs/c60_17093.traj` and `inputs/c60_17094.traj` files;
their hashes match the old long-budget plan. The original `.arc` files are
listed separately as provenance only. Before execution, a zero-PES preflight must
prove that frozen source, input hashes, backend/runtime, seeds, recovered
rotation, NativeMC, SSW configuration and budget match the existing no-LS long
arms. If that comparison cannot be made exactly, these two arms must remain
unexecuted rather than becoming an unmatched comparison. `runner.py` retains
the held-out runner's failed-initial archive, conditional first-cage
validation, ledger accounting and checkpoint recovery. `preflight.py` performs
only hashes, geometry and effective-config checks; it does not initialize a
calculator or call the PES. The plan reaches `READY_NOT_SUBMITTED` only after
that preflight passes. The two sbatch files are submission descriptions only.

The eventual report must retain execution, numerical, physical graph, and
reference-energy qualification separately, with all failed, denied and
budget-censored attempts and their E/F costs. It must not infer fullerene
success from low energy or connectivity alone, and it must not use interim
results to tune, retry, extend the budget, or promote LS as a default. The
execution decision remains with the main agent after the current held-out
rotation result is available. No retry, extension or tuning is permitted; the
program cap is 5,400 s inside the 6,000 s allocation (100 min), one V100 per
case.


## 实际提交

预检1441604通过141项零PES检查；原失败1441550、误提交后取消1441563、元数据文本不一致1441565保留。原始ARC与冻结TRAJ的hash适用对象已区分，并非输入发生漂移。预检后补回GPU脚本的CUBLAS_WORKSPACE_CONFIG=:4096:8，与旧基线一致；核心/输入/数值配置未改，runner执行前再次检查全部hash。

GPU1441611/1441612已经提交，实际决定及预算见submissions.json。计划中的prepared状态保留为准备时快照，不代表未提交。最多120000搜索+6独立复核，无重试/预算追加。结果仍待运行与审计。
