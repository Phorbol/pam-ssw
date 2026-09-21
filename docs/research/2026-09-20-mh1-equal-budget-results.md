# 等预算旋转器对照：更多落点尚不等于稳定的低能收益

2026-09-20；固定协议开发对照，不是独立成功率验证。

## 问题与已完成证据

此前同状态完整逃逸显示 recovered CBD 省调用，但不能由此推断全局搜索更好。
本轮固定两个 MH-1/omol C60 first 保存态，每态 baseline 与 recovered CBD 各12000搜索E/F。
两臂共用 Safe-total/history500、12 Gaussian上限、内层fmax=.1、外层.03、同随机种子和nativeMC。
改变的是整个已恢复旋转组件（包括预旋转与动态修正），不能把收益单独归因于某个常数或阈值。
此处baseline是当前Python旋转器；recovered CBD也是Python实现，不是本次运行LASP二进制。

作业1406860完成，40分43秒，48000搜索+63独立复核=48063 E/F。
四条均在请求上限终止；末段未完成成本保留，日志evaluation_failed在这里对应request_cap。
所有59个完成落点与4个初态独立单点合格；在1.64/1.7/1.8 Å三种键图阈值下均连通。
连通与小力不证明完整物理稳定性；本轮未计算C60完整Hessian。
完整笼、参考能量达标及其交集均为0/4条轨迹；不得据此估计总体成功率。

| 起点 | 旋转器 | 完成落点 | 搜索内最低能量/eV | 统一.01复核后/eV |
|---|---|---:|---:|---:|
| c60_17093-first | baseline | 11 | -62185.311131 | -62185.320455 |
| c60_17093-first | recovered_cbd | 18 | -62184.085472 | -62184.085797 |
| c60_17094-first | baseline | 11 | -62166.280165 | -62166.282581 |
| c60_17094-first | recovered_cbd | 19 | -62187.352276 | -62187.358706 |

最低能量包括初态与被MC拒绝但合格的候选；完成落点数不等于独立basin数。
第一组收紧力阈值后CBD仍高1.234658 eV；第二组仍低21.076125 eV。
各臂选搜索内最低合格候选再统一淬火至.01 eV/Å，仅用于核实排名，不计入原12000搜索收益。
该复核作业1407552耗时20秒，152优化+4独立单点=156 E/F；4/4合格、连通、仍无完整笼。
输入从原result逐项提取，.traj往返坐标精确核对；不对精度作额外性能解释。
这不是对所有候选重新排序，不能排除其他候选精修后改变局部排名。

## 决定与边界

- 保留 recovered CBD 为有实际成本价值的可选实现，不升级为通用默认，也不继续围绕这两个起点调旋转参数。
- 当前证据支持“同成本完成更多外步”，不支持“必然更低能”。两个C60起点和两个Cu55起点仍属很小的开发集合。
- 初始Cu55 octa辅助结果因负内部曲率撤回稳定basin解释；修正后两个正曲率真实落点对照及共同精修也给出混合收益，见下方金属报告。
- C60验收目标仍未完成。此结果不能解释此前Python与LASP完整轨迹的全部差距，也不能证明缺少LS是原因。
- 下一项优先级：在MH-1上核对已有原版启发偏置高度控制与当前自适应偏置的匹配对照；先审查现有协议与接口，再固定唯一变化因素。此前OMAT高度结果受到模型异常低能坑污染，不能代替此证据。不同时切换LS/GA/MC或增加旋转调参。

## 可追溯入口与验证

原始输入、冻结源码、完整配置、预算、逐步轨迹与分析：
[主实验](../../research/ga_ssw/evidence/mh1-equal-budget-rotation-20260920/plan.json)、
[原始汇总](../../research/ga_ssw/evidence/mh1-equal-budget-rotation-20260920/summary.json)、
[3000/6000/12000分析](../../research/ga_ssw/evidence/mh1-equal-budget-rotation-20260920/analysis.json)、
[统一精修](../../research/ga_ssw/evidence/mh1-equal-budget-refinement-20260920/plan.json)、
[精修结果](../../research/ga_ssw/evidence/mh1-equal-budget-refinement-20260920/summary.json)。

CPU作业1406910执行 `summarize_mh1_equal_budget.py --input <主实验> --output <主实验>/analysis.json`；
1407567用冻结的同一分析器处理精修目录。所有8条记录成本重建与实际账本一致、无缺失臂。
分析器与graph helper已保存在主实验目录，可复现分析；不修改原始轨迹。

[Cu55起点资格修正](2026-09-20-emt-rotation-boundary.md)；
[合格Cu55对照及精修](2026-09-20-emt-qualified-rotation-results.md)。
本轮无核心算法改动、未合并/发布；实质新增是成本公平的真实体系证据。
