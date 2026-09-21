# 固定胞 SSW 阶段停止：实现闭合，未建立普适收益

> 2026-09-18 更正：后续独立审查发现研究监视器 `_MonitoredSurface._last` 与返回力数组共享内存，
> Gaussian 力被 SurfaceCalculator 原地加入后，监视器再次加入，导致门控实际使用重复偏置的力。
> 下文保留原记录；其中“恢复出的阶段谓词被正确用于真实轨迹”的解释及力门触发统计撤回。
> 原始账本、真实落点与独立无偏置复核仍保留，但不能作为正确阶段控制的效果证据。
> 错误不在原生标量谓词函数，也不影响未启用此研究 adapter 的 baseline。
> 独立复现和修复证据：`research/ga_ssw/evidence/stage-monitor-force-alias-20260918/`。

本轮通过显式 biased-quench adapter 将已恢复的单段停止与全部释放谓词接入
公共 SSW；默认路径不变，真实势淬火/MC资格不变。独立回归为520 passed / 1
skipped，研究monitor/adapter另有8项检查。没有运行LASP主程序。

## 实验与结果

按同目录 `2026-09-12-stage-control-e2e-protocol.md`，Cu13/EMT、固定胞
Cu31/EMT、bicyclobutane/GFN2-xTB，seed11、各两次outer attempt；每种Gaussian
策略内部仅改变是否启用stage adapter。内层fmax=.1、外层.01 eV/Å，
Safe-total memory10、relax_steps200、14 Gaussian、dimer。每臂4000请求/60秒上限。
参数、输入、源码快照与实际导入位置在PES前保存；非周期cartesian/周期
translation_only在运行前按现有公共几何域修正协议。

| 体系 | Gaussian | 原偏置淬火请求 | 阶段控制请求 |
|---|---|---:|---:|
| Cu13 | reference forward-force | 527 | 539 |
| Cu13 | PAM height_width | 727 | 733 |
| Cu31 | reference forward-force | 443 | 454 |
| Cu31 | PAM height_width | 1000 | 706 |
| C4H6 | reference forward-force | 901 | 903 |
| C4H6 | PAM height_width | 1080 | 1134 |

12臂全部完成，合计9147搜索请求+36独立复核请求=9183。无SCF/请求/时间
截断。每臂保存初始minimum及两个真实势落点，36个全部通过fresh新实例
能量差≤1e-8 eV、fmax≤.01 eV/Å、cell/PBC不变检查。所有付费请求的
composition/cell/PBC与输入相同；原始记录含完整E/F与失败/拒绝计费标志。
分子三落点均为10原子连通分量；这仅是几何连通诊断，不是TS或化学验证。

阶段控制六臂共有12次末段全部释放；另有31次力分量门与7次saved-center
能量门触发。计数预算、较初始低.1eV及能量/力上限门未在本轮真实轨迹触发，
其证据仍限于隔离原指令与数值检查。不能将全部分支称为真实体系验证完成。

相同请求前缀的最佳能量五组完全相同；C4H6/PAM仅差2.48e-6 eV，远不足以
支持科学效果排名。Cu31/PAM成本降低29.4%，但原方案两个MC拒绝落点分别
高于初始2.85755/2.92970 eV，新方案两个接受落点接近初始能量。
因此减少请求不能独立解释为更高探索效率：路径和所达到的结构范围也改变了。
本轮未做结构置换/对称去重，不以接受数或保存数宣称不同basin。

## 决定与证据边界

保留默认关闭的研究接口，不把原版停止逻辑升级为通用默认，不继续围绕Cu31
调参。生产公共接口已能在阶段停止后继续Gaussian、或全部释放后进入真实势
淬火；这与原版LBFGS/MCSRCH完整状态机等价是不同命题。反编译计数为外部
E/F的RC dispatch消费，不等于Safe-total accepted iterations；本实验200消费
预算及current/best能量参照是公开的独立选择，不能称native默认。

产物：`research/ga_ssw/evidence/stage-control-e2e-20260912/`，含每臂metadata、
raw-result、JSONL账本、fresh-checks。根智能体运行
`research/ga_ssw/audit_stage_control_e2e.py`，复核源码摘要、成本、几何、全部
存储落点与相同成本前缀，输出`root-audit.json`。子智能体报告一次直接child
启动因live import在PES前被拒绝；未提供独立日志路径。正式冻结runner
执行一次，无PES重试。

下一项转向已证实的固定胞周期GA契约缺口：TYPE1/2 proposals已有实现，但
现有periodic controller默认要求VC的objective/forces/stress结果，不能直接
接入固定胞run_ssw。先修结果契约与固定胞几何边界，复用已有三阶段算法；
不引入新的GA算子，不将固定胞资格错误地要求为零stress。
后续源代码复核进一步收窄为TYPE1：TYPE2的proposal本身会按分子extent
创建新cell，不能以接口适配假装固定胞；TYPE2固定胞请求需在PES前明确拒绝。
