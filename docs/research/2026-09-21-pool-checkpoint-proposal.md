# 待讨论：池策略断点的责任边界

完整方向schema4已验收。此提案尚未实现，不改变默认MC，也不意味着PAM池策略已显示搜索收益。此前用户明确将池持久化留待单独讨论。

## 不能直接解除禁用的原因

`starter_selection.py`的观察序号是minima顺序，不是稳定盆地身份。SSW当前索引与last_landing_index只在函数中维护；池调度另持有selector RNG。`research/ga_ssw/pool_starter_adapter.py`还维护archive条目/原型/统计、observation→entry映射、代表索引、outcomes/decisions、source/executed及finalized标记。只恢复几何或只恢复主RNG会改变下一次起点、重复记账或把最后一次建议选择算成已执行工作。

## 最小可讨论方案

只对现有研究PoolStarterAdapter提供显式纯数据快照，不自动pickle任意callback。职责分为：

- 核心保存当前观察索引、最近落点索引、selector RNG，并在同一个完成外步边界获取策略快照。
- 策略明确导出/恢复带版本和策略标识的纯数据：archive配置、完整entries/prototypes及统计、mapping/representatives/outcomes/decisions/source/executed、selector/scorer配置。对象内部的任意calculator/callback不序列化。
- 在现有checkpoint原子写入中把核心和策略状态一起保存，避免两个文件的边界不一致。core不导入research模块；由调用者提供实现显式状态契约的同类策略，恢复前核对标识、版本及配置。第一版只验证现有adapter，其余callback继续拒绝checkpoint。
- finalize仅在整个实验结束后调用，不能在暂停边界先重算统计并终结adapter。

新增策略导出/恢复契约属于公开接口设计，超出已批准的方向状态扩展。持久化职责是否留在caller侧、是否让core保存策略数据，需要用户确认后实施。

## 备选与取舍

1. 继续暂缓池checkpoint：保持现有连续池运行；将精力用于搜索效率证据。实现成本零，缺点是池长任务不能精确恢复。目前短程池对照没有显示优于MC，此项可合理暂缓。
2. 上述显式adapter状态契约：可恢复当前池研究，利于长任务；增加公共接口与版本兼容责任，但不引入新选择策略。
3. 调用者自行维护两个文件：改动较少，但不能保证策略和SSW保存于同一边界，不推荐作为完整恢复能力交付。

若决定实施2，验收为连续与分段回放逐项比较chosen/actual index、selector RNG、archive统计/映射、方向重启和成本；覆盖MC拒绝、池重启、失败外步及finalize；错误配置在PES之前拒绝。保留非池schema1–4兼容，不将恢复能力称为算法效果提升。
