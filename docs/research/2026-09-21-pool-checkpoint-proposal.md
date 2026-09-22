# 已批准：池策略断点的责任边界（2026-09-23）

完整方向schema4已验收。用户已批准本文最小方案，正在实现；不改变默认MC，也不意味着PAM池策略已显示搜索收益。此前用户明确将池持久化留待单独讨论。

## 不能直接解除禁用的原因

`starter_selection.py`的观察序号是minima顺序，不是稳定盆地身份。SSW当前索引与last_landing_index只在函数中维护；池调度另持有selector RNG。`research/ga_ssw/pool_starter_adapter.py`还维护archive条目/原型/统计、observation→entry映射、代表索引、outcomes/decisions、source/executed及finalized标记。只恢复几何或只恢复主RNG会改变下一次起点、重复记账或把最后一次建议选择算成已执行工作。

## 最小可讨论方案

只对现有研究PoolStarterAdapter提供显式纯数据快照，不自动pickle任意callback。职责分为：

- 核心保存当前观察索引、最近落点索引、selector RNG，并在同一个完成外步边界获取策略快照。
- 策略明确导出/恢复带版本和策略标识的纯数据：archive配置、完整entries/prototypes及统计、mapping/representatives/outcomes/decisions/source/executed、selector/scorer配置。对象内部的任意calculator/callback不序列化。
- 在现有checkpoint原子写入中把核心和策略状态一起保存，避免两个文件的边界不一致。core不导入research模块；由调用者提供实现显式状态契约的同类策略，恢复前核对标识、版本及配置。第一版只验证现有adapter，其余callback继续拒绝checkpoint。
- finalize仅在整个实验结束后调用，不能在暂停边界先重算统计并终结adapter。

新增策略导出/恢复契约属于公开接口设计；用户已于2026-09-23批准由core统一保存、adapter负责显式状态解释的方案。

## 备选与取舍

1. 继续暂缓池checkpoint：保持现有连续池运行；将精力用于搜索效率证据。实现成本零，缺点是池长任务不能精确恢复。目前短程池对照没有显示优于MC，此项可合理暂缓。
2. 上述显式adapter状态契约：可恢复当前池研究，利于长任务；增加公共接口与版本兼容责任，但不引入新选择策略。
3. 调用者自行维护两个文件：改动较少，但不能保证策略和SSW保存于同一边界，不推荐作为完整恢复能力交付。

若决定实施2，验收为连续与分段回放逐项比较chosen/actual index、selector RNG、archive统计/映射、方向重启和成本；覆盖MC拒绝、池重启、失败外步及finalize；错误配置在PES之前拒绝。保留非池schema1–4兼容，不将恢复能力称为算法效果提升。


2026-09-23实现约定：池快照使用schema5，非池schema1–4保持兼容。策略通过checkpoint_contract()/export_state()/restore_state(payload)提供显式纯数据状态；核心保存当前观察索引、最近落点索引、策略契约/状态与selector RNG。复用核心已有LS、方向、MC、主RNG和成本字段。第一版不支持Gaussian/内层优化中间恢复；错误终止快照仅诊断，不恢复执行。暂停不调用finalize。

## 调用方式与边界

调用者仍创建相同配置的 `SSWConfig`、LS设置和Calculator；恢复不会从磁盘反序列化可执行的Calculator或策略对象。示意：

```python
from numpy.random import default_rng
from pamssw.standalone import run_ssw, load_ssw_checkpoint
from research.ga_ssw.pool_starter_adapter import PoolStarterAdapter

policy = PoolStarterAdapter(mode="pam", energy_tol=0.001, rmsd_tol=0.1)
first = run_ssw(atoms, surface, steps=8, config=config, ls=ls,
    rng=default_rng(19), starter_selector=policy, selector_rng=default_rng(23),
    checkpoint_path="search.pkl")
# 暂停时不调用 policy.finalize(first)。
checkpoint = load_ssw_checkpoint("search.pkl")
restored_policy = PoolStarterAdapter(mode="pam", energy_tol=0.001, rmsd_tol=0.1)
result = run_ssw(atoms, fresh_surface, steps=12, config=config, ls=ls,
    rng=default_rng(0), starter_selector=restored_policy, selector_rng=default_rng(1),
    checkpoint=checkpoint, checkpoint_path="search.pkl")
report = restored_policy.finalize(result)  # 整个实验结束后才结算。
```

此处8+12仅说明 `steps` 是本次新增外步数，不代表已完成20步模型验收；参数为已有研究适配器设置，非推荐通用最优值。若启用native MC、恢复方向或其他策略，两段须传相同设置。恢复RNG的种子可不同，但bit-generator类型必须相同；两套Generator仍须独立。

`PoolStarterAdapter`目前是研究模块：仍不支持约束输入，其近似结构身份与描述符限制未因持久化而消除。schema5没有引入GA三阶段恢复、VC或RC状态的新契约。旧无池checkpoint按原接口继续使用。
