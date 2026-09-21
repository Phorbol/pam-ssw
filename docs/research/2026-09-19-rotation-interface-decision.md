# 待讨论：将旋转求解与方向生成解耦

## 为什么现在需要决定

已完成的同状态MH-1实验显示恢复CBD减少局部成本，但尚无键重排收益；下一项验证必须进入完整多Gaussian链。
现有`run_ssw(recovered_direction=...)`同时切换pair/group方向生成、旋转阶段、旋转力投影及状态生命周期，
而且显式拒绝checkpoint。它可以比较完整机制包，但不能回答单独替换旋转部分的效果。
不以重复编写一个研究walker或monkeypatch解决这个问题。

## 备选

1. **保持接口，使用现有recovered_direction完整包。** 工程成本最低；可做端到端比较，
   但生成方向/几何/旋转同时变化，只能报告包级效果，且暂不支持这一路径恢复。
2. **推荐：在现有驱动内分离可选择的旋转器。** 用显式、不可变、可保存的旋转设置，
   使现有global/paper方向生成独立于CBD选择。沿用已有rotation_surface、Gaussian、quench、MC与统计流程。
   不增加population/controller框架，不修改默认，不复制外层walker。
3. **任意callable rotation_adapter。** 与bias_quench_adapter形式相似，但难以保证闭包状态、
   配置与恢复过程一致；仅为一个已知机制引入开放接口收益不足，暂不推荐。

## 推荐方案的具体契约（尚未实施）

- run_ssw新增可选的typed recovered_rotation设置；未指定时现有路径完全保持。
- 此设置仅拥有CBD阶段阈值、上限和度量；方向采样和几何仍由已有config决定。
  不把原生rotnum上限伪装成HVP次数，也不混用报告力与HVP残差单位。
- 返回真实的方向、curvature、residual、调用数、stop_reason以及实际bias_reference/rotation_weight。
  不为“看起来同条件”把动态旋转算子的a伪报成基线1；这会破坏曲率还原和诊断。
- 与完整recovered_direction同时提供时应拒绝歧义，或明确定义兼容规则；初版建议拒绝。
- 配置随checkpoint保存，恢复时核验；旧checkpoint缺省为None。不能依赖用户重新传入未验证的callable。
- Gaussian高度策略不改：默认前向力公式仍用返回方向处的实际力；两臂高度可因方向不同而不同。
- 默认路径回归、恢复等价、预算/停止状态、真实C60多Gaussian链为必要验证。
  前面的单Gaussian数值合格不能代替这一步。

## 范围与代价

涉及公共入口与checkpoint配置，按AGENTS.md需用户讨论后实施。
预计局部改动限于standalone旋转设置/调度、checkpoint兼容及针对性测试；不重构整个SSW类层次。
既有recovered_direction入口保留实验标签，不把其方向状态恢复问题夹带进本次范围。
用户本次批准的组账户单Gaussian实验已完成；后续计算预算需另行明确，当前未提交额外PES任务。

[本次真实证据](2026-09-19-mh1-matched-cbd-results.md)。

2026-09-20 用户已批准推荐方案，已[实施并验证](2026-09-20-recovered-rotation-integration.md)。上述为原始决策时的备选与理由，不改写历史。
