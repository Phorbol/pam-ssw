# Strict-quench 200-step production conclusion

严格组与 baseline 均来自同一 frozen base execution commit `32980ad9154ec6481b310477dd7b85597cefd49a`、同一输入/模型/runtime；每条 CUDA/MACE 轨迹均执行 200 个 macro trials 和 201 次 true quench。purpose ledger 均精确闭合到总 force evaluations，且 `unattributed=0`。严格证书指终止计数为 `converged=201`、`unconverged=0`，并满足 `max |F| <= 0.01 eV/Å`。

| System | Strict drop eV | Strict FE | Strict wall s | Baseline drop eV | Baseline FE | Baseline wall s | Strict-baseline drop eV | FE delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C60 | 33.408630 | 91490 | 1566.6 | 39.850586 | 92707 | 1592.7 | -6.441956 | -1217 |
| PDO | 6.767456 | 65531 | 1453.6 | 7.646057 | 54659 | 1215.1 | -0.878601 | +10872 |

## Cost structure and certificates

- C60: certificate 201/201; ASE-LBFGS→FIRE fallback 6/6. FE shares: proposal 51.59%, direction oracle 31.76%, landing true quench 14.98%.
- PDO: certificate 201/201; ASE-LBFGS→FIRE fallback 7/7. FE shares: proposal 46.53%, direction oracle 9.31%, landing true quench 42.97%.

## Decision boundary

- **C60:** `quench_fmax=0.01` 与 baseline 相同；严格配置仅替换 true-quench optimizer 为 ASE-LBFGS，并在它未给出证书时以 FIRE 补救。新运行把证书从 40/201 提升为 201/201（6 次 fallback，均收敛），总 FE 少 1,217、wall time 少 26.1 s；但 energy drop 少 6.441956 eV、archive 少 20 个 minima。故采用 **ASE-LBFGS+certificate FIRE** 作为 certified-data/validation candidate，而不是 global-search 默认。结合 `pamssw/walker.py` 的代码审计，当前框架允许未获证书的 landing 进入 archive，导致优化器终止语义与后续搜索状态直接耦合；该判断不是由本次 JSON 证据单独推出。严格组成本最大项仍是 proposal relaxation，其次为 direction oracle。
- **PdO:** baseline 在 `quench_fmax=0.03` 已经是 201/201 证书。严格运行同时改变 optimizer 和 `quench_fmax: 0.03→0.01`，FE 增加 10,872、energy drop 少 0.878601 eV；因此不能纯归因于 optimizer，拒绝把 0.01 设为 production default。干净策略是 **0.03 用于 search，只有需要严格数据的结果再 selective refine 到 0.01**；若要隔离阈值效应，再做同一 ASE-LBFGS+certificate FIRE 下 0.03 vs 0.01 的 paired threshold ablation。当前 PdO 的 proposal 与 landing true quench 是并列主要 FE 瓶颈。
- **P0 transition-dataset completeness:** C60 baseline 的 `stats.n_trials=200`，但只持久化了 199 条 walk records 和 200 个 energy-trace states；缺失的 trial 由 `fragment_rejections=1` 精确解释。这次分析保留 `recorded_trials=199`、`unlogged_trials=1`，没有伪造记录。对 posterior/credit 学习而言，失败 transition 也必须写入 action/outcome log。PdO 为 200/200，`fragment_rejections=0`。
- 两个系统在更换 true-quench 方案后观测到的能量与归档指标不同，后续 SSW 状态序列不能继续视为 paired trajectory comparison；且每个条件只有 seed 42。上述差分是严格核算下的描述性结果，不是 basin identity、optimizer 一般性或因果 superiority 声明。
