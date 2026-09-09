# 独立 ASE SSW 家族实现计划

目标：Python 独立执行 SSW、LS-SSW、GA-SSW；用户通过 ASE Calculator 提供势面。运行搜索不得要求上传的 LASP/Java，也不得调用旧 PAM walker 冒充原版。

这是用户已明确授权的架构与开发方向。原程序保留在 research 中作为对照。新的代码在 pamssw/standalone；尚未完成完整搜索时不导出误导性的 SSW/GA_SSW 入口。

## 接口与实施顺序

1. `surface.py`：接收任意提供 energy/forces 的 Calculator；显式 energy/free_energy 选择，独立真实势与偏置势评估。ASE 局部优化后检查真实力，记录失败及所有评估请求。第一阶段固定胞、无约束，其他域明确拒绝。
2. `gaussian.py`、`direction.py`：投影 Gaussian 与静态恢复高度控制；有限差分软方向作为独立数值组件。原版 biased rotation、climb termination 和 MC trapping 状态尚需恢复，不以近似组件声称完整原版。
3. `softening.py`：调用者提供元素对键能和成键长度表；冻结邻居/r0，soft-only 预淬火、真实能量响应和跨步幅度更新分别保存。论文模式与发行版调度分开。
4. `ga_operators.py`：原 TYPE3 可确认的刚性分子操作；依次补 docking、父代竞争、描述符/网格及 quick/fine controller。不能用泛化随机交叉代替未恢复算子。
5. 完整 SSW 外层接通后，将 LS 接入相同 escape 生命周期，再接完整 GA controller。所有参数记录出处、单位与使用域；未知规则继续反编译。
6. 固定胞真实体系通过后扩展周期与联合变胞，再处理 RC 链。ASE Calculator 能提供 stress 只是必要条件，不表示联合变胞算法已经实现。

## 验收

- 每个模块先失败测试，再实现与针对性检查；数值测试只检查数学及接口。
- 无 native 二进制的真实金属团簇 EMT、分子势体系先作有界端到端贯通，随后上传作者体系同势面逐状态对照。
- 独立记录初始淬火、软面预淬火、方向旋转、偏置优化、最终真面淬火和失败成本。接口请求计数不冒充后端 SCF 或内部 force evaluations。
- 单步数值一致、完整执行、真实极小值认证与同成本搜索效率是不同验收层；完整 SSW/LS/GA 外层缺失时不宣布任务完成。

依据：本仓库 `ga-ssw-audit/` 中的 SSW、LS/RC 和 Java 审计；ASE 官方 Atoms 和 Calculator 接口。代码评审和跨体系科学验证持续追踪，不改变已有 PAM 默认策略。
