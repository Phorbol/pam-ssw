# Fixed raw landing 严格 true-quench 消融结论

## 判据与数据边界

- 输入为冻结的 C60/PdO raw landing corpus，共 160 行；严格力证书为 `max_active_force <= 0.01 eV/Å`，不采信 optimizer_success 作为收敛判据。
- 总核算为 27128 次 `landing_true_quench` force eval；`unattributed=0`，逐行与汇总账本闭合。
- 表中 FE、P90、wall time 和降能统计均为 unconditional：包含成功与失败的全部 attempts。

## 精确结果

| system | arm | 证书成功 | termination reasons | 总 FE | 中位 FE | P90 FE | 最大 FE | 总 wall/s | 中位 wall/s | 总降能/eV | 中位降能/eV | 终态力中位/最大 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| c60 | scipy-lbfgsb | 0/16 | unconverged:16 | 1807 | 109.5 | 190.0 | 221 | 40.640 | 1.893 | 72.363 | 3.845 | 0.021546 / 0.052470 |
| c60 | safe-lbfgs-total | 6/16 | converged:6, line_search_failed:10 | 2137 | 132.5 | 218.5 | 259 | 35.430 | 2.201 | 72.369 | 3.846 | 0.021770 / 0.066235 |
| c60 | ase-fire | 7/16 | converged:7, maxiter:9 | 4904 | 401.0 | 401.0 | 401 | 82.296 | 6.648 | 70.405 | 3.801 | 0.031765 / 0.108391 |
| c60 | ase-fire2 | 6/16 | converged:6, maxiter:10 | 5349 | 420.5 | 422.0 | 422 | 89.688 | 7.042 | 69.941 | 3.797 | 0.050404 / 0.124450 |
| c60 | ase-lbfgs | 14/16 | converged:14, maxiter:2 | 2490 | 115.0 | 331.5 | 401 | 42.586 | 1.948 | 69.625 | 3.845 | 0.009188 / 1.502415 |
| pdo | scipy-lbfgsb | 7/16 | converged:7, unconverged:9 | 1494 | 66.5 | 190.5 | 243 | 31.496 | 1.414 | 57.923 | 3.264 | 0.010706 / 0.041664 |
| pdo | safe-lbfgs-total | 12/16 | converged:12, line_search_failed:4 | 1527 | 64.5 | 196.5 | 230 | 32.103 | 1.348 | 59.095 | 3.574 | 0.009178 / 0.020385 |
| pdo | ase-fire | 15/16 | converged:15, maxiter:1 | 2295 | 140.5 | 215.5 | 402 | 48.777 | 3.006 | 53.233 | 2.916 | 0.009438 / 0.337190 |
| pdo | ase-fire2 | 15/16 | converged:15, maxiter:1 | 2519 | 140.0 | 289.0 | 420 | 53.629 | 3.015 | 53.314 | 2.917 | 0.009593 / 0.018060 |
| pdo | ase-lbfgs | 14/16 | converged:14, maxiter:2 | 2606 | 107.0 | 380.0 | 402 | 56.261 | 2.317 | 56.714 | 3.123 | 0.009411 / 0.675479 |

## 最小结论

- 此次固定 corpus 中，没有单一优化器 arm 实现全覆盖。
- 按严格力证书的整体成功数，ase-lbfgs 为最强单一 arm（28/32）；这只是该固定 corpus 上的局部淬火结果，不是生产最终答案。
- C60 与 PdO 的 taskwise `any-arm` 覆盖分别为 16/16 和 16/16。
- 不同 arm 可能落入不同局部极小值，因此不能把终态能量差直接解释为公平的收敛速度比较。
- 本实验没有重跑 SSW proposal，也不能据此声称全局搜索性能或修改生产默认值。

## 唯一下一步

- 仅做基于严格力证书触发的 `primary + fallback` sequential rescue 小消融：先运行一个 primary，只有未通过证书时才从其终态调用 fallback；继续核算两阶段全部 force eval，并且不直接修改默认值。
