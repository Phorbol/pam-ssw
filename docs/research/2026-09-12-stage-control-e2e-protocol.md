# Stage-control 开发验证协议（运行前冻结）

目标是验证阶段停止能正确进入下一Gaussian或真实势落点/MC，并比较代价与
失败边界。不是证明完整LASP轨迹等价，也不是通用效率或论文结果复现。
不复制SSW外层循环，通过显式experimental biased-quench adapter复用
run_ssw。默认关闭，不扩展checkpoint；真实势资格标准保持不变。

## 设计

三类输入：已有Cu13金属团簇、Cu31固定胞空位、C4H6 bicyclobutane。
后端分别EMT、EMT、GFN2-xTB（accuracy=.001），使用已有保存输入。
一组seed11、两次outer attempts。两种Gaussian策略分别为现有reference
forward-force策略和现有PAMCurvatureGaussian(height_width)默认参数。
每种Gaussian策略内部只比较原偏置松弛和新stage-control，共12臂。
同初态、方向求解器和参数，不比较两种Gaussian策略的参数优劣。

共享操作点：dimer、rotation_bias100、rotation_hvp100、rotation_tol.02、
width.1Å、max_gaussians14、temperature150K、fd_step1e-4Å、
Safe-total（既有memory10）、relax_steps200、bias_fmax.1、真实fmax.01eV/Å。
方向global；周期Cu31使用cluster_frame=translation_only，非周期Cu13/C4H6
使用既有cartesian。公共入口明确禁止非周期translation_only，故在任何PES
运行前按现有几何域修正协议，不为本实验扩大接口。不增添全局反馈策略。
全部参数、PAM原有参数及单位在运行前保存，不在观察结果后更改。

stage-control 数值参数明确为独立研究选择：

- 首段与后段消费预算均200，取自共享relax_steps数值以限定开发范围，
  不是声称native默认值或accepted-iteration等价。counter起始1，每次
  完整modified E/F消费后+1；严格大于预算结束当段。
- climb_stopf=.1/sqrt(3) eV/Å，使原max-component判据足以保证每个原子
  力范数≤.1。该关系是范数上界，不是native阈值默认值。
- e_maxlimit=100eV、f_maxlimit=100eV/Å取已恢复preset数值；
  e_maxlimit_gm共用同一个100eV能量上限，不再独立调参；这是显式
  开发约定，其native默认未知，不冒充恢复值。
- 初始能量参照=当前已选真实minimum，GM参照=当前run已保存best energy，
  都是独立映射；native+b90的完整重置生命周期尚不能视为相同。
- .1eV低于初始的门、后段低于saved center 1eV的门、末段ng==H门均按
  已恢复谓词执行。center的基面能量若需额外求值，明确计费。
- stage_stop、release_all、optimizer converged、force-qualified分别记录；
  不能通过把停止标签改成converged来进入真实势淬火。

不使用LS，本轮先隔离SSW本体停止机制。基础quench原型已支持LS-inclusive
base分解，但本次不把那项接口能力当作native LS阶段生命周期证明。

## 成本与资格

每臂4000付费E/F请求、60秒上限，失败请求计费，拒绝预算请求不计费。
总搜索上限48000请求；无追加seed、重试或单例参数调整。新后端实例对所有
保存落点fresh复核，复核成本单列，拒绝MC落点同样保留。保存模型/库版本、
完整源码快照、实际import路径、输入、配置、请求账本、原始失败和结构。

报告：完成的真实落点、全局最佳能量及达到成本、预算/SCF/旋转/偏置松弛
失败、stage_stop/release_all次数与原因、同请求前缀结果。分子另列连接和
碎片诊断，力小不代表化学结构有效。周期固定胞要求cell exact，不要求
被禁止松弛的晶胞零应力。不以接受率或保存数声称新盆地或GM命中。

先审查新入口的默认/no-op路径等价和研究adapter单元检查，再执行本协议。
真实验证结果未形成前，stage-control仍为实验性原型，不升级通用默认。
