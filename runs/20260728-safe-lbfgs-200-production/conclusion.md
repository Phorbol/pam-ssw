# Safe-LBFGS 200-step production conclusion

两套任务均完成 200 个 SSW macro trials，且 purpose ledger 求和等于总 force evaluations，`unattributed=0`。

| System | Initial eV | Best eV | Drop eV | Force evals | Wall s | Entries | Duplicate | Final best trial |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C60 | -474.669617 | -508.913483 | 34.243866 | 92265 | 1715.1 | 144 | 0.284 | 162 |
| PDO | -568.397034 | -579.845947 | 11.448914 | 49593 | 1120.2 | 175 | 0.129 | 191 |

## Optimizer and cost evidence

- C60: proposal relaxation 840/1034 converged, 194 failed; true quench 35/201 converged, 166 unconverged.
  Cost shares: proposal 51.2%, true-quench/validation 13.6%, direction oracle 30.5%, escape checks 4.7%.
  Best-energy improvement trials: 1, 2, 3, 4, 5, 7, 8, 26, 40, 83, 118, 127, 162.
- PDO: proposal relaxation 283/290 converged, 7 failed; true quench 201/201 converged, 0 unconverged.
  Cost shares: proposal 62.0%, true-quench/validation 22.9%, direction oracle 12.3%, escape checks 2.8%.
  Best-energy improvement trials: 1, 2, 5, 6, 10, 14, 16, 54, 76, 141, 145, 146, 147, 166, 172, 180, 183, 191.

C60 的 true quench 为 35/201 converged、166 unconverged，这是当前长任务最明确的瓶颈；PdO 为 201/201 converged。

## Historical FIRE boundary

- C60 FIRE default8: 58,000 force evaluations 后 完成 85 trials，best=-506.905304 eV，drop=32.235962 eV。
- PDO FIRE default8: 58,000 force evaluations 后 完成 121 trials，best=-573.166260 eV，drop=4.771362 eV。

这些 FIRE 结果是同名输入、seed-42、default8 的 58k-cap 部分轨迹，但完成 trial 数、总预算和 provenance 不完全匹配。因此它们只提供描述性边界，**不支持严格的 200-step superiority 结论**。
