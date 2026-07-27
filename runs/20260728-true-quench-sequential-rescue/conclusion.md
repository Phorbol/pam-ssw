# Sequential strict-quench rescue：固定 corpus 结论

验证通过：输入是 ASE-LBFGS primary 严格证书失败的固定 4 个 terminal states；primary cohort 共 32 个任务，原本通过 28/32，成本为 5096 次 force evaluations。

- ASE-LBFGS restart：3/4 个 fallback 通过，完整 pipeline 为 31/32；离线 replay 总成本 5816 FE，连续实现投影 5812 FE。
- safe L-BFGS：1/4 个 fallback 通过，完整 pipeline 为 29/32；离线 replay 总成本 5370 FE，连续实现投影 5366 FE。
- FIRE：4/4 个 fallback 通过，完整 pipeline 为 32/32；离线 replay 总成本 5590 FE，连续实现投影 5586 FE。相对 primary，连续投影额外 490 FE，即 9.6153846%。

在这个固定 corpus 上，FIRE 是唯一覆盖全部 4 个 ASE-LBFGS 证书失败任务的 fallback。

## 解释边界

“连续实现投影”假设 primary 的 cached terminal evaluation 能直接交给 fallback，因此每个触发任务比当前离线 replay 少 1 FE；这是投影，不是当前实测。不同 fallback endpoint energy 不相同，因此这些 endpoint 的能量与 wall time 不作速度公平比较。本结果不声称 global-search 性能已经改善，也不声称生产默认优化器已经选定。

## 唯一下一步

最小化集成 `ASE-LBFGS primary + certificate-triggered FIRE fallback`，随后做固定预算 end-to-end 验证；不加入额外 fallback 层或参数。
