# Direction candidate hard-cap: fixed-budget GPU evidence

纳入且仅纳入 6 个完成的 GPU run：C60 seeds 42/43 与 PdO seed 42 的 precap/hardcap 配对；`*.partial-failed` 明确排除。

六个 run 均耗尽精确的 6000 次 force evaluation；purpose 计数闭合、`unattributed=0`，且 `direction_oracle = 2 × candidate_count_sum`。arm-specific native contract 成立，实际 precap overflow 为正、hardcap overflow 为零；record/direction 统计闭合。配对配置除派生输出路径外一致，runtime/CUDA/calculator/input/model/frozen runner 运行环境身份一致。因此本实验支持的是 **hard-cap 预算语义正确**。

下表全部为 `hardcap - precap`：

| system | seed | trials | minima | duplicate rate | energy drop (eV) | wall (s) | candidate mean | direction FE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C60 | 42 | +1 | -1 | +0.147436 | +6.958374 | -2.948811 | -0.826087 | +6 |
| C60 | 43 | -1 | -2 | +0.090909 | -1.375122 | -0.033678 | -0.796610 | +290 |
| PDO | 42 | +5 | +4 | +0.030000 | -0.459656 | -0.134936 | -0.393939 | +118 |

性能结果不一致：C60 seed 42 的 energy drop 增加，seed 43 减少；PdO seed 42 也减少。候选池均值下降并不保证总 direction FE 下降，因为完成的 direction choices/trials 数会变化。

结论止于预算语义正确性；不宣称搜索质量或性能提升，也不为这一语义验证继续扩展 seeds。
