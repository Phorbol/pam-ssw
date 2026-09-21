# 恢复方向与 LS 的真实体系衔接：结果和下一决定

## 目标、实际完成与结论

目标是将反编译恢复的局部方向/组选取/CBD作为显式配置接入同一独立Python/ASE SSW驱动，
并核验非零NativeLS、预淬火、真实响应更新和真实势能落点。不是二进制包装，不改变默认算法。
本轮已经在MACE-OMAT-0-small上完成衔接检验；完整LASP行为复现及通用性能优势仍未完成。

主要结果：恢复方向在水15的两个开发种子中找到了明显更低能结构；LS控制链确实工作，
但其收益受真实淬火的求解成本影响。原先seed11的八次LS落点失败都是300步maxiter，
不是LS预淬火失败或线搜索失败。不能因此判定LS公式错误或LS无效。

## 冻结协议和证据

- 续跑：[plan](../../research/ga_ssw/evidence/recovered-ls-materials-continuation-20260917/plan.json)，
  [逐臂核验](../../research/ga_ssw/evidence/recovered-ls-materials-continuation-20260917/analysis.json)，
  [几何检查](../../research/ga_ssw/evidence/recovered-ls-materials-continuation-20260917/geometry-analysis.json)。
- job1369567正常完成，6分37秒，1 V100；63项冻结源码测试通过。
  7/7未开始臂执行结束，41470次搜索、70次独立复核；成本核验无不一致。
  六个水臂均触及6000请求上限，属于预算截断，不是六个算法崩溃。
- MACE CUDA float64；水15来自上传example；Cu55为ASE二十面体构造。
  外步上限20、每外步25Gaussian、fmax=.03、bias_fmax=.1、真实/偏置淬火300步；
  LS软面预淬火=.1/50步，完整配置见唯一协议入口。

| 水15配置 | seed11合格新落点/最低能量变化 | seed29合格新落点/最低能量变化 |
|---|---:|---:|
| paper方向 | 2 / -0.01719 eV | 3 / -0.02137 eV |
| recovered方向 | 12 / -2.73390 eV | 16 / -3.24343 eV |
| recovered + NativeLS | 0 / 0 eV | 10 / -2.76883 eV |

落点数未去重，不是不同盆地数。所有返回帧均独立复核fmax<=.03、组分/固定胞/PBC不变。
两套方向的旋转停止规则、随机数消耗和每请求可完成外步数不同；本表是开发观察，不能归因于某个
局部方向或作为方法效率排名。Cu55恢复方向20步后仍主要回到近似初态能量，未证明金属体系收益。

两个LS种子均初始化30条非零键；分别9/12次真实响应更新，范围14.746–19.382和
14.746–28.978 meV/atom，更新表确实变化，真实能量差与meV/atom换算核验一致。
seed11预淬火全部达到软面力阈值；八次完整真实淬火失败，另一次被请求上限截断。
seed29有一次完整真实淬火失败、十次合格落点、一次请求截断。
20外步以内不验证100步以后的save_zero/restore周期。

冻结runner的`real_response_update`字段误读`observed_response`，原始false不可采信。
实际事件字段为`observed_response_mev_per_atom`；派生分析从原始事件重新核验，未覆盖原始产物。
当前研究runner已纠正该字段；核心LS计算未因此修改。

水15所有保存帧做几何筛查，最低能结构的每个O最近邻分配仍为两个H；恢复方向最佳结构
最大最近邻OH为1.021/1.023 Å。该检查不证明正定Hessian或模型适用域。
[MACE官方模型表](https://github.com/ACEsuit/mace-foundations)将OMAT-0目标列为materials；
这些水团簇结果仅为指定MLIP上的算法检查，不能声称DFT或真实水团簇稳定性已验证。

## 相同起点的真实淬火对照

为了区分优化器效率与300步上限，固定选取两个LS种子的前两个外步起点，再加一个Cu55起点。
使用原始`SSWStep.last_atoms`，即真实淬火前坐标；所有方法清空历史，从完全相同位置开始，
不从失败终点继续、不改力阈值。每臂最多600请求/600步，fmax=.03。

[协议与起点](../../research/ga_ssw/evidence/recovered-true-quench-replay-20260917/plan.json)、
[结果及前缀一致性核验](../../research/ga_ssw/evidence/recovered-true-quench-replay-20260917/analysis.json)。
job1369667正常完成，1分41秒，1 V100；20/20臂结束，8380搜索+12独立复核请求，成本/起点核验无错误。

| 优化器 | 力合格/5起点 | 总搜索请求（包括截断） |
|---|---:|---:|
| Safe-total/history10 | 4/5 | 1883 |
| Safe-total/history500 | 5/5 | 1731 |
| ASE LBFGSLineSearch | 2/5 | 2353 |
| SciPy L-BFGS-B | 1/5 | 2413 |

Cu对照四法均合格。水的非合格项均为600请求截断；本轮并非SciPy相对能量早停。
Safe500并非每个起点都更快，也不总落到更低能结构。该结果支持将其列入下一固定协议候选，
不支持立即改通用默认值，不证明原版LASP优化器的排名（未纳入本次对照）。

**反证与精度边界：**Safe10的重跑并未逐步复现原300步轨迹。起点完全一致、首力差约
2e-15–8e-15 eV/Å，但四个水起点到第200请求的最大坐标差约.043–.086 Å；Cu保持~1e-14 Å。
这证明长路径对微小数值扰动敏感，不能把重跑说成旧失败轨迹的严格延长。
旧失败的停止原因确为maxiter，重跑表明同起点能在更大预算下找到合格结构；
“只改变步数必然挽救原轨迹”仍不成立。不同GPU节点/浮点归约等可能造成首力差，未作机制归因。

## 行政事故和预算：不混入算法结论

原job1369369的运行目录被并行子任务错误改名，导致FileNotFoundError。
原始产物保留在`recovered-ls-materials-20260917-preinit-replay-superseded/`；
完成3臂，第四臂Cu-paper-seed29留下4342次请求、无最终结果，不重跑也不补造终态。
原运行合计15793次付费请求；错误重建的原路径prepared目录没有提交。
Cu-LS seed11零邻居来自已知Cu通用回退长度1.875 Å，另一个相同初态Cu-LS臂未运行；
不人为扩大键长，依据[既有Cu域审查](2026-09-12-cu-ls-table-domain-audit.md)。

合计本系列实际搜索=15793+41470+8380=65643，低于原72000上限。
回放plan的`prior_search=57793`为保守上界而非实际值，submission.json注明实际57263。
原始归档不覆盖；行政中断与算法失败分别记录。

## 决定和下一最高优先级

1. 保留显式recovered方向及NativeLS共享实现；不改默认，不扩展VC/RC/Q分支。
2. 不针对水失败改LS公式或放宽fmax。下一对照应把真实淬火的求解预算与历史长度作为独立因素，
   避免与方向/LS强度同时调整；以总请求预算和物理合格落点计收益。
3. Safe500进入候选，使用独立种子和另一个已具来源、可用LS表的体系验证；本次五起点只作开发诊断。
4. 方向收益仍需隔离旋转停止标准与局部生成器的贡献；不能用当前表格宣称组件因果优势。
5. 保留重要边界：恢复配置当前仅无约束非周期团簇；Q/compression、周期native路由及此配置checkpoint未闭合。
   现有其他ASE入口能力不等于这个恢复配置已支持全部体系。

复核命令：`python research/ga_ssw/analyze_recovered_direction_campaign.py <campaign> --output <analysis.json>`；
`python research/ga_ssw/audit_recovered_material_geometry.py <campaign> --output <geometry-analysis.json>`；
`python research/ga_ssw/analyze_true_quench_replay.py <replay> --output <analysis.json>`。
Python独立实现及研究脚本均为未提交工作区；未声称稳定发布。
