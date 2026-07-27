# Five-trial GPU no-loss validation

trajectory_exact: false
exact_gpu_trajectory_claim_supported: false

数值差异按 JSON 中记录的 Python float 值逐项比较；没有引入容差。因此轨迹不一致是测量结果，而不是 analyzer failure。配置、provenance、purpose closure、unattributed=0 和 central-HVP 核算仍 fail closed。

| System | old vs current exact | current vs repeat exact | initial delta (old-current, eV) | initial delta (current-repeat, eV) | escape-only counterfactual FE saving |
| --- | --- | --- | ---: | ---: | ---: |
| C60 | false | false | 0.000152587890625 | -0.000152587890625 | 85 |
| PdO | false | not available | 6.103515625e-05 | not available | 31 |

结论：这份运行可以支持‘当前实现的 purpose 账本闭合，且按 trace 反事实重建只节省 `ESCAPE_TRUE_PES_CHECK`’；不能支持 exact GPU trajectory 的声明。
