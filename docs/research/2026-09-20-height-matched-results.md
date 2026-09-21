# 原版启发高度：固定预算对照与独立精修

## Cu55 EMT development comparison

This section records the two prequalified Cu55 development states. It is not
an independent success-rate study and does not pool these EMT results with the
C60 MH1 evidence.

The height run is
[`emt-cu55-qualified-height-20260920`](../../research/ga_ssw/evidence/emt-cu55-qualified-height-20260920/)
(`job 1414257`). It used seed 11, ordinary baseline rotation, and only
`ConservativeNativeHeightPolicy` as the changed component. Each candidate had
6000 search E/F calls and retained all fresh checks. Both trajectories reached
the search request cap and ended with an evaluation failure after the capped
search; the capped search cost is still included.

| state | best qualified height landing by search cost | 1500 EF | 3000 EF | 6000 EF |
|---|---:|---:|---:|---:|
| candidate 1 | 25.8110138873 eV (cost 4955) | 26.3648609658 | 26.1103724121 | 25.8110138873 |
| candidate 2 | 27.2691990877 eV (cost 5798) | 27.3430200881 | 27.3430200881 | 27.2691990877 |

All listed landings passed the run's fresh EMT force criterion of 0.03 eV/A.
At the same original .03 force threshold and 6000-call search budget,
candidate 1 improves from baseline 26.9527996973 to 25.8110138873 eV, while
candidate 2 worsens from baseline 27.0003597170 to 27.2691990877 eV.
These are the appropriate matched search comparisons; the .01 reference below
is used only for the separate refinement comparison. The two trajectories are fixed development
starting states, so these observations do not estimate independent basin
success rates.

The separate `.01` termination check is
[`emt-cu55-height-refinement-20260920`](../../research/ga_ssw/evidence/emt-cu55-height-refinement-20260920/)
(`job 1414290`). It selected the lowest fresh-qualified saved height minimum
from each trajectory and used the same EMT with `fmax=.01`. This is a
termination-sensitivity check, not additional search budget: candidate 1 used
12 search plus 1 fresh call (13 EF), and candidate 2 used 117 search plus 1
fresh call (118 EF), for 131 EF total.

| state | height before | height after `.01` | change | baseline `.01` reference |
|---|---:|---:|---:|---:|
| candidate 1 | 25.8110138873 | 25.8080832651 | -0.0029306222 | 26.9429430765 |
| candidate 2 | 27.2691990877 | 26.6302011435 | -0.6389979442 | 26.9984487349 |

The candidate-2 refinement reaches a lower energy only after its additional
117 search and one fresh call; that energy cannot be attributed to the
original 6000-call height comparison or treated as evidence that it shares a
basin with the baseline. The candidate-1 and candidate-2 `.01` baseline values
come from the corrected r2 reference run
[`emt-cu55-qualified-refinement-20260920-r2`](../../research/ga_ssw/evidence/emt-cu55-qualified-refinement-20260920-r2/).

The search-budget result and the `.01` refinement answer different questions:
the first records what the height trajectory produced within its fixed search
budget, while the second checks how two preselected landings move under a
tighter force termination. Neither result by itself supports a general
efficiency or basin-discovery claim.


## C60 / MH-1：完整等预算结果

1414139已完成（21分17秒）：两起点各12000搜索，32独立复核，总24032 E/F。
30个完成落点加2初态全部通过.03力阈值且在1.64/1.7/1.8 Å三个图阈值下连通。
两条均触及request_cap，终段成本保留；完整笼、参考能量达标及交集均为0/2。

| 起点 | 方法 | 3000预算最佳E/eV | 6000预算最佳E/eV | 12000预算最佳E/eV | 完成落点 |
|---|---|---:|---:|---:|---:|
| 17093 | baseline | -62162.749247 | -62171.087336 | -62185.311131 | 11 |
| 17093 | native-height | -62164.395809 | -62172.042911 | -62185.697160 | 16 |
| 17094 | baseline | -62153.353223 | -62160.051838 | -62166.280165 | 11 |
| 17094 | native-height | -62159.056543 | -62164.645855 | -62178.425926 | 14 |

同阈值预算内，这两个开发起点分别低.386029和12.145761 eV。不能由两个样本估计独立成功率。
各阶段first Gaussian的高度中位数均为5.6 eV；总共10次以maxw_exceeded结束高度调整，
最大权重12 eV符合既有“更新后检查max_weight=10”规则，并非裁剪错误。
该策略同时改变起始高度、角度停止和历史重写，不能只把观察到的差异归因于87度。

1414388统一精修至.01 eV/Å（43秒），21优化+2复核=23 E/F：
17093最低-62185.697518，17094最低-62178.426554，两者均通过新力阈值；
额外降能仅.000358和.000628 eV，未改变与基线的排序。
这是额外数值资格检查，不把23次调用计入原12000的搜索收益。
结构分析1414393已核实两帧在三个阈值下仍连通且均未成笼；详见下面分析入口。

证据：[固定协议](../../research/ga_ssw/evidence/mh1-native-height-equal-budget-20260920/plan.json)、
[预算节点与资格](../../research/ga_ssw/evidence/mh1-native-height-equal-budget-20260920/analysis.json)、
[阶段调用与高度](../../research/ga_ssw/evidence/mh1-native-height-equal-budget-20260920/stage-diagnostics.json)、
[额外精修](../../research/ga_ssw/evidence/mh1-height-refinement-20260920/summary.json)、
[精修结构资格](../../research/ga_ssw/evidence/mh1-height-refinement-20260920/analysis.json)。

## 当前决定

保留已有原版启发高度作为可选组件；不扫描这些开发起点的高度/宽度常数、不升为通用默认。
C60有同预算收益，Cu55原预算一好一差；小力阈值也不等于精确能量资格。
恢复CBD和本高度策略改变的路径不同，不能由分开的单因素结果推断组合必然更好。
下一项已执行既有NativeLS单因素验证：仍固定baseline方向、原高度和同预算，避免叠加归因。
本轮未改变SSW核心实现；修正了两个过时LS接口测试对mc转发的预期，16项相关测试通过。
