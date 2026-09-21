# 约束SSW/LS接入已有PAM Gaussian：实现与对照

2026-09-12。问题：FixAtoms/Hookean已支持，但约束SSW只有固定宽度的
forward-force Gaussian；用户此前要求保留PAM自适应高度/宽度作对照。
本改动复用 `PAMCurvatureGaussian`，没有重新设计算法或修改默认参数。

## 数学和实现边界

约束SSW使用活动原子Cartesian位移q，Hookean作为持久目标的一部分。
PAM规则在该活动坐标上计算曲率，排除方向求解器的rank-one rotation bias，
再加入历史Gaussian的解析方向二阶导数。LS若开启，已包含在方向求解的
背景目标里，不能又作为Gaussian history重复加入。诊断字段`k_true`在此
表示去掉rotation bias的背景曲率；开启LS时并非裸物理PES曲率。

每个Gaussian保存独立width/weight/center/direction。变宽后，旧项的能量、
梯度和Hessian仍使用旧width，不能全部套用当前width。分阶段旋转使用实际
presweep方向及实际bias_curvature；当只允许部分活动原子参与方向搜索时，
presweep方向需展平并提升到完整活动坐标后再调用policy。root审查修正了
此组合的shape错误。

公开入口：`run_constrained_ssw(..., gaussian_policy=PAMCurvatureGaussian())`，
支持plain/paper LS/native LS与已有FixAtoms/Hookean。默认None保留原公式、
调用顺序和随机数消费。checkpoint保存不可变策略参数；恢复时缺失/改变
参数会在首个PES请求前拒绝，不保留跨外层Gaussian偏置。旧checkpoint缺少
该字段视为None。不增加通用有状态policy框架，不宣称GA或RC/VC新增支持。

已有参数定义、单位与来源仍由`pam_gaussian.py`及其parameters记录，属于
PAM的经验操作点；这里不声称它们是从第一性原理唯一推导的通用最佳值。

## 验证

root standalone suite：**564 passed, 1 skipped，28.88s**，日志
`/tmp/pam-root-constrained-pam-suite-20260912.log`。
新增测试包括：

- 默认None和省略参数的逐次E/F输入输出、RNG、records一致；
- 活动坐标及staged+restricted旋转组合；
- driver实际使用两个不同width，逐项历史下的总梯度有限差分；
- PAM+Hookean与paper/native LS的continuous2和1+resume1，逐records、
  累计请求、RNG、LS状态及最终坐标一致；policy mismatch零调用拒绝。

真实固定12臂：水二聚体/GFN2-xTB、Cu111/EMT，各plain/paper/native LS，
reference vs PAM，两组相同seed11、初态、内外精度与预算，每臂一个外层
尝试。PAM用已有height_width默认；无结果驱动调参。此处是兼容性和短流程
对照，不是独立长期搜索效率benchmark。

| 体系/变体 | reference搜索 | PAM搜索 | 结果 |
|---|---:|---:|---|
| 水二聚体/SSW | 109 | 145 | 双方valid_landing |
| 水二聚体/paper LS | 119 | 178 | 双方valid_landing |
| 水二聚体/native LS | 120 | 157 | 双方valid_landing |
| Cu111/SSW | 49 | 89 | 双方valid_landing |
| Cu111/paper LS | 56 | 98 | 双方valid_landing |
| Cu111/native LS | 152 | 152 | 双方LS预淬火失败，尚未生成Gaussian |

reference605，PAM819，共1424搜索+22fresh=1446次E/F请求。22个已存储观察
全部fresh资格合格，包含重复初态，不能称22个不同盆地。PAM实际width为
水二聚体约0.293–0.349 Å、Cu约0.488–0.523 Å，历史项width确实不同。
参考六臂的整个搜索ledger与前一轮源码的保存产物逐项一致，包括失败臂，
不是只比较最终能量。新runner的结果成本字段已修正，无前轮汇总异常。

本组PAM没有展现更好的best-objective降低或更低调用成本；短步数、单seed、
宽度显著不同，不支持一般性的算法优劣结论。保留为显式可选项，**不改变
默认，不据此调整PAM参数**。Cu/native-LS失败先于Gaussian，故不能归因于
选用何种Gaussian；离线MIC审查另见相关文档，不添加自动fallback。

## 产物

- `research/ga_ssw/evidence/constrained-gaussian-reference-20260912/`
- `research/ga_ssw/evidence/constrained-gaussian-pam-20260912/`
- `research/ga_ssw/evidence/constrained-gaussian-comparison-20260912.json`
- `research/ga_ssw/audit_constrained_gaussian_comparison.py`，纯离线审计。
- 每组包含完整plan、源代码快照、逐调用ledger、result及fresh检查。

后续优先级：在已实现共同选项上进行更复杂跨体系验证，区分实际不同盆地
与重复淬火；普通/约束入口尚未统一的优化器选择与阶段停止接口保持显式
边界。VC度量问题已有独立方案，但不与本次Gaussian接口改动混合。
