# Posterior starter-policy GPU smoke 结论

## 已验证事实

- 6/6 个 campaign 完成；共 76 次 action，失败 action 为 0。
- 每个 campaign 的 manifest、index、event/action 字段闭合、policy probability 与 posterior credit 重建、预算账本和 action optimizer diagnostics 均通过 fail-closed 校验。
- 有效配置固定为 proposal `safe-lbfgs-total`，true quench `ase-lbfgs`，失败时 `ase-fire` fallback；`fmax=0.01 eV/Å`、`maxiter=400`。

| system | policy | attempts | total FE | bootstrap FE | action FE | unused FE | wall s | archive | duplicate | best drop eV | fallback |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| c60 | uniform | 11 | 5660 | 49 | 5611 | 340 | 123.698411 | 12 | 0.000000 | 12.385955811 | 0/0 |
| c60 | posterior_proportional | 10 | 5147 | 49 | 5098 | 853 | 88.722294 | 10 | 0.090909 | 13.825988770 | 0/0 |
| c60 | minimal_ucb | 9 | 5074 | 49 | 5025 | 926 | 87.230282 | 9 | 0.100000 | 16.652435303 | 0/0 |
| pdo | uniform | 15 | 5011 | 84 | 4927 | 989 | 113.076200 | 15 | 0.062500 | 3.700683594 | 1/1 |
| pdo | posterior_proportional | 15 | 5159 | 84 | 5075 | 841 | 116.160798 | 15 | 0.062500 | 4.646850586 | 0/0 |
| pdo | minimal_ucb | 16 | 5074 | 84 | 4990 | 926 | 117.365245 | 14 | 0.176471 | 4.458007812 | 1/1 |

## 账本与收敛

- Bootstrap 固定成本：C60 为 49 FE，PdO 为 84 FE。当前 optimizer diagnostics 只记录带 action ID 的尝试，因此 bootstrap fallback 是否触发不可观测；不能从 action diagnostics 反推。
- Action 内 fallback 仅发生 2 次，其中 2 次收敛；最终 `true_quench_unconverged=0`。
- true-quench diagnostics 是 final-only：发生 fallback 时不用于总 FE 加和；总成本只以按 purpose 记账且闭合的 evaluator ledger 为准。
- C60 的主要成本是 proposal relaxation 与 direction oracle；PdO 的主要成本是 proposal relaxation 与 strict true quench。

## Policy 语义与结论边界

- `uniform` 和 `posterior_proportional` 在每次 policy snapshot 上都保留完整支持；后者对所有 eligible starter 的概率严格为正，但它不是 frequency-unbiased，因为抽样频率受 posterior mean 影响。
- `minimal_ucb` 是 one-hot 的确定性对照，不属于无偏或完整支持探索。
- 表中 policy 差异只能作为单个 seed 的描述性结果。已知 GPU 非确定性会被 PES 搜索放大，因此不能据此判定 policy 优劣，也没有比较 TS 与 UCB。
