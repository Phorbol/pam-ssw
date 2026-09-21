# 原版方向恢复：行为一致性与算法设计的边界

后续进展：c4完整路由和受限恢复方向的共享ASE驱动已接通，并完成MACE接口检查，
见[现行集成状态](2026-09-17-run5-direction-integration-gap.md)。
下文“未实现c4/共享driver”保留为发现当时的记录；随机、半径mask和坐标chart的
证据边界仍适用。实验入口仍不等于完整原版控制器。

## 主线结论

本轮补齐选轴刷新和随机分量的独立 Python 实现，同时发现上传 ELF 的一些
具体行为不能直接当作物理上合理的默认设计。这里不要求停止恢复有用的方向
更新/CBD流程，也不以这些局部发现宣称 LASP 搜索整体较差。

事实、推论和待验证效果须分开：

| 已观察事实 | 支持的解释 | 尚不支持的结论 |
|---|---|---|
| VMB2 使用 `-int((u+1)*10)` 重置 RAN3；固定 N/mask 时同区间输出相同 | 原始随机分量不是连续各向同性采样 | 完整 SSW 只有十个方向、搜索效果必然差 |
| 半径筛选中第2或第3原子距中心13 Å时，清掉第4原子的mask；全近则全保留 | 该函数的写入依赖最后索引，不符合按各原子距离筛选的解释 | 全部真实运行必然触发、总体性能损失已有定量证据 |
| centered分子的forbidden检查可在整体平移入盒后改变 | standalone几何函数存在坐标chart前提/依赖 | 原生完整调用者必然违反此前提 |

第二项已完成实际 `get_dist`/`reci_latt` 指令复核；只替代floor2数学库，不替代距离。
三个输入均在40 Å盒内，中心(15,15,15)，仅第2或第3原子移至13 Å，最后原子距中心1 Å。
真实距离误差最大1.82e-12 Å；最初1e-12断言失败，因此记录实际误差并采用1e-10 Å
几何同一性判据，远小于12/13 Å筛选判别间隔。没有改变算法的12 Å阈值。
三个案例均正常返回并符合上述mask结果。此结论限定上传ELF，不推广所有LASP版本。

## 实现与验证

- `pamssw/standalone/native_pair_selection.py`：原版几何检查和pair刷新；明确区分
  几何接受与150个距离/fixatom拒绝计数耗尽。限定非周期、无ASE约束；不冒充通用约束接口。
- `pamssw/standalone/native_random.py`：独立RAN3/VMB2，无LASP运行依赖。
  十个主要种子区间为-10..-19；上端浮点加法舍入可产生-20，已单独核验。
- canonical24与edge24：pair、draw、acceptance及三种拒绝计数一致；主agent重跑edge24。
- 原生46条随机输出与Python比较，最大绝对误差1.39e-17。
- 规范坐标的选组→刷新→c6组方向→投影/混合9例均匹配，最大误差1.67e-16。
- 针对性测试36 passed：native group selection/local group/pair selection/random、
  native local pair及recovered CBD。没有新的PES或GPU作业。

原始随机分量的辅助几何检查：Cu13、Cu55、C60分别有33、159、174个内部Cartesian
自由度；固定几何、全active mask、十个主要种子输出经刚体投影后张成秩均为10。
这是有限向量集的几何性质，不是端到端搜索验证；局部混合和CBD可改变方向空间。

## 已作决定与仍需工作

保留上述原版行为的独立复现证据，不改现有ASE搜索默认，也不添加用于补偿这些
现象的经验回退。下一步集成不能把“按距离屏蔽各原子”的合理意图与上传ELF实际
“屏蔽最后原子”的行为混称；也不能把有限种子随机分量称为均匀采样。

c4还有与c6不同的group producer：c6为cross product，c4可为两组沿pair轴相反移动。
已确认c4先做forbidden gate，`vtable+0x160`为`find_atom_in_group`，不是已证明的
pair刷新。当前未实现整个c4路由、Q/compression分支或共享driver的完整native状态。
因此尚未交付完整原版方向控制器，更没有证明新组件提高SSW/LS效率。

研究路线建议：把原指令参考行为保留为明确的对照；主线ASE算法优先保持平移/
原子置换的一致性及连续随机采样，吸收可解释的局部模式、位移更新和CBD阶段。
这会是有明确偏离记录的LASP启发实现，不能标为全部行为一致。当前两套共享驱动
规则的既有设计继续适用；科学有效性仍需固定预算的多体系MACE-OMAT-0-small实验。

## 产物与复跑

位于 `research/ga_ssw/evidence/`：
`native-atom-neighbor-radius-actual-20260917.json`、
`native-pair-coordinate-chart-20260917.json`、
`native-random-python-comparison-20260917.json`、
`native-random-internal-span-20260917.json`、
`native-getpair-python-comparison-20260917.json`、
`native-getpair-edges-20260917.json`、
`native-axis-pipeline-canonical-20260917.json`。
旧centered输入产物保留，其接受率不再作为正常搜索行为的证据。

复跑（mace_env Python；instruction probes还需`PYTHONPATH=/tmp/pam-ssw-unicorn-probe:.`）：

```
python -m research.ga_ssw.compare_native_random
python -m research.ga_ssw.run_native_getpair_edges
python -m research.ga_ssw.probe_native_axis_pipeline --canonical
python -m research.ga_ssw.probe_native_atom_neighbor_radius --actual-geometry --output research/ga_ssw/evidence/native-atom-neighbor-radius-actual-20260917.json
python -m pytest tests/standalone/test_native_group_selection.py tests/standalone/test_native_local_group.py tests/standalone/test_native_pair_selection.py tests/standalone/test_native_random.py tests/test_native_local_pair.py tests/standalone/test_recovered_cbd.py -q
```
