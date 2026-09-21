# 已实现：方向生成与恢复CBD旋转独立选择

## 能力变化

用户批准局部接口方案后，`run_ssw`新增`recovered_rotation`可选设置，
不再必须连带启用native pair/group方向生成。未指定时原有路径保持；
Gaussian、高度公式、淬火、MC及公共几何仍由既有驱动控制，无新walker或任意callable框架。

```python
from pamssw.standalone import RecoveredRotationSettings, run_ssw
rotation = RecoveredRotationSettings(
    pre_rotmax=5, rotmax=15, pre_ftol=.2, ftol=.02,
    metric='euclidean', max_force_calls=40,
)
result = run_ssw(atoms, surface, steps=steps, config=config, rng=rng,
                 recovered_rotation=rotation, checkpoint_path='ssw.pkl')
```

示例值只对应本次共同dr=.001 Å的C60机制对照；不是通用默认。
pre/main ftol是报告力阈值eV/Å，换算关系为10*dr*HVP残差；
阶段rotnum上限与总E/F预算分别记录。返回实际bias_reference/rotation_weight。

## 状态契约

- 新模式使用schema 3保存不可变typed旋转设置；旧schema 1/2无此字段时按None读取。
- 恢复时省略recovered_rotation会继承保存配置；显式不匹配在PES调用前拒绝。
  config仍须按原API显式传入且一致，未增加config=None语义。
- native MC配置/状态须成对保存；恢复时继续要求明确传入相同MC设置。
- recovered_direction和pre_rotation_hvp与新选项互斥；不扩展前者尚未支持的checkpoint功能。
- 保留rotation_exit_policy：预算结束不伪装成收敛，只有显式force_or_budget允许已求值方向进入后续阶段。
- 旧有约束/周期边界限制仍存在。本轮不新增FixAtoms、slab或VC能力。

## 主agent实际验证

均在CPU-MISC计算节点执行，无GPU模型推理：

| 作业 | 验证 | 结果 |
|---|---|---|
|1405489|初版接口测试|3通过、1失败；测试擅自假定config=None可用，纠正测试以保持原合同|
|1405523|既有checkpoint/native MC/CBD/原方向driver/Broyden回归|36通过|
|1405531|修正后的接口测试|6通过|
|1405543|补齐真实连续2步vs保存1步+恢复1步，两种MC|8通过|
|1405554|读取前序实际C60旧schema2 checkpoint|两份均可读取，保持60000请求和原evaluation_failed诊断状态|

8项与前面的6项是重复集合，不相加成14；最终相关测试为36+8=44项。
连续/分段运行的请求数、方向、接受决策、终态坐标匹配。
新路径在非共线Cu4/EMT上覆盖多Gaussian事件、强制预算退出与checkpoint生命周期；
这不证明C60或跨体系全局搜索优势。其他几何上的新CBD科学效果仍未评估。

`py_compile`、`bash -n`及`git diff --check`通过。
[测试日志与精确补丁](../../research/ga_ssw/evidence/recovered-rotation-integration-20260920/)。
补丁的前版本使用前次已冻结源码；中途采集的before目录含部分实现，不能称纯修改前快照。

## 下一项已经准备

[完整多Gaussian实验包](../../research/ga_ssw/evidence/mh1-multigaussian-rotation-20260920/)：
相同四保存结构，两种旋转，各一次完整外步、最多12个Gaussian；总上限24000 E/F。
公共入口直接执行，无monkeypatch；失败和checkpoint保留。计划1 V100最多30分钟。
用户已批准本次组账户预算，GPU作业1405922运行；结果完成后单独归档。
核心源码已实现但尚未提交Git/合并或发布；不将本次交付称为生产稳定版。
