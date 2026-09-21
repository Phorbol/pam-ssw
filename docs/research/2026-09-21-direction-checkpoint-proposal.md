# 已批准：完整方向控制的断点恢复

## 问题与当前边界

本轮批准的连续run状态和显式池重启已完成，但持久于一次run不等于可以跨进程恢复。提案时`run_ssw`明确拒绝recovered_direction+checkpoint及starter_selector+checkpoint。普通SSW checkpoint虽有坐标、历史极小值、RNG、MC/LS状态，却没有完整方向的pair/group/group_marker。因此不能仅删掉拒绝检查：恢复时initialize会重抽方向状态、消耗随机数，悄悄变成另一条搜索。

此缺口影响后续长预算实验的分段运行与恢复，不是新的搜索策略，也不提供效率收益。现有冻结实验不改写、不重跑。

## 推荐最小方案：先恢复完整方向，不同时持久化池策略

1. 保留现有外步边界checkpoint和受信任本地pickle存储方式；新schema增加有明确类型的方向状态和settings。旧schema1–3继续可读。
2. 方向快照至少含已选pair、group mask、group_marker；这些是下一外步需要的跨步状态。上一外步诊断已在records保存。Gaussian内部状态不作为本版本的中途恢复承诺，下一外步仍按原begin_escape构造。
3. 恢复时直接恢复已选状态及已有主RNG/MC，不再initialize或初始淬火；核对settings、元素顺序、PBC、cell、状态形状和现有配置契约。
4. 原run_ssw调用签名与无checkpoint结果不变。失败中的checkpoint仍按既有规则仅供诊断；不把中途Gaussian截断包装成可继续的外步边界。source/backend由调用者提供，不序列化Calculator。
5. starter_selector仍显式不支持checkpoint；LS现有checkpoint不被重设计，LS与pool仍暂缓。

变化范围：SSWCheckpoint schema/复制与校验、RecoveredDirectionController的状态导出/恢复、run_ssw恢复分支、针对性回归及文档。不是新通用session框架或archive格式。

## 备选：同时做池调度断点恢复

这还需要明确selector_rng、当前观察索引、caller-owned archive/entry映射、累计反馈与策略内部状态如何保存，以及任意callback如何声明可恢复。现有records只有选择结果，不包含任意策略的全部内部状态。此方案扩大公开策略契约和序列化职责，成本和兼容风险明显更高；不能通过自动pickle任意callback冒充通用安全/可恢复接口。

## 验证与验收

先用已有保存E/F账本做连续两外步 vs 一步保存、重新加载后一步的严格控制流回放：比较方向状态、Gaussian几何、落点/current/best、MC、RNG及累计请求，验证无额外初始淬火/选pair。覆盖接受和拒绝边界、设置不兼容、旧schema可读；旧无checkpoint路径保持现有目标回归。随后真实体系的分段运行只验证数值与结构资格，不要求GPU逐位一致，不将恢复能力称为搜索收益。

用户已批准第一种最小方案；[实现与验证已完成](2026-09-21-direction-checkpoint-results.md)，池策略仍不接入。讨论依据为AGENTS第8节：重要抽象与系统架构设计，包括状态/持久化格式重设计，需先与用户讨论。池状态契约另行界定。
