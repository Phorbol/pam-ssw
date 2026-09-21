# 旋转预算放行：实现验收与材料结果（2026-09-17）

## 科学问题与结论

固定胞SSW需要一个可用于逃逸的方向，是否必须先把有限差分旋转残差收敛到给定阈值？
本轮只改变阶段退出契约，其他公式、参数、总搜索预算和真实落点门槛保持。

**实验证明存在这样的轨迹：旋转残差未达标，预算放行后仍到达真实力合格且结构不同于初态的落点。**
这支持把方向软化的预算结束与数学收敛分开；不支持普遍效率优势，更不能称完整LASP复现。
保留显式实验选项`force_or_budget`，默认`force`不变；不再围绕本组结果调整tol或预算。

## 已实现的范围

共同Python/ASE驱动增加`rotation_exit_policy`；Ritz、dimer、Euclidean Broyden及两阶段包装返回
`residual_converged`/`budget_exhausted`/`subspace_exhausted`等停止原因。
仅有限、已求值的预算出口可继续，保留`converged=False`及残差；子空间耗尽、未知原因、数值异常
不放行。真实落点仍须fmax≤0.03 eV/Å。独立atomic_climb/VC分块入口明确拒绝该选项。
旧配置默认保持；没有改方向迭代公式、Gaussian或局部优化器。

源文件：`pamssw/standalone/{direction,dimer,broyden_direction,staged_direction,paper_reference,atomic_climb}.py`。
测试：`tests/standalone/test_rotation_exit_policy.py`和`test_direction_stop_reasons.py`，加既有相关检查。
研究runner/分析器复用`run_matched_rotation_materials.py`和`analyze_matched_rotation_materials.py`。

## 协议与成本

AlOH26/brookite48 × seed11/29 × 三求解器，12/12臂完成，每臂6000搜索请求；
MACE-OMAT-0-small、CUDA float64、同一冻结输入，HVP预算100、fd_step=0.001 Å、
rotation_tol=0.02 eV/Å²、bias_fmax=0.1、真实fmax=0.03 eV/Å、Gaussian上限25。
完整参数和代码/模型来源见冻结plan.json及source快照。

作业1366235：COMPLETED，ExitCode=0:0，V100节点4v100n02，耗时17分09秒。
12实验臂搜索72000请求，独立复核77次；严格模式兼容性复跑另计6000搜索+5复核。
本轮总计78082请求。91项针对性检查在同一计算节点通过（1.63秒）。
所有77/77实验帧和5/5回归帧通过真实力、晶胞和原子信息检查。
12臂最后的`evaluation_failed`均明确来自预设搜索预算拦截，无新增PES数值异常；不能把该状态
当作完整外步成功。预算不足时最后一次逃逸保留为未完成。

## 分臂结果

结构数含初态，按较紧/较松两组周期匹配容差列出：(.1,.15,2°)/(.2,.3,5°)，
scale=False；近似结构身份不等于严格盆地或正定Hessian认证。能量为相对各自初始淬火的最低ΔE。
旧严格臂与新预算臂独立运行；箭头表示观察变化，不是无噪声因果效应。

| 体系/求解器/seed | 合格落点数，不含初态（严格→放行） | 结构数（严格→放行） | 最低ΔE eV（严格→放行） | 放行次数 |
|---|---:|---|---:|---:|
| aloh26-ritz-seed11 | 3→6 | 2/1→5/5 | -0.000530→-0.740858 | 2 |
| aloh26-ritz-seed29 | 6→7 | 7/5→8/6 | -0.588863→-0.589794 | 0 |
| aloh26-dimer-seed11 | 0→2 | 1/1→2/2 | 0.000000→0.000000 | 41 |
| aloh26-dimer-seed29 | 0→2 | 1/1→3/2 | 0.000000→-0.081813 | 53 |
| aloh26-broyden-euclidean-seed11 | 6→6 | 5/5→4/4 | -0.739982→-0.740681 | 0 |
| aloh26-broyden-euclidean-seed29 | 7→6 | 7/6→6/5 | -0.589887→-0.590208 | 0 |
| brookite48-ritz-seed11 | 8→8 | 2/2→3/3 | -0.000745→-0.000875 | 0 |
| brookite48-ritz-seed29 | 8→8 | 2/2→3/3 | -0.000893→-0.000726 | 0 |
| brookite48-dimer-seed11 | 2→2 | 1/1→1/1 | -0.000963→-0.000859 | 2 |
| brookite48-dimer-seed29 | 0→2 | 1/1→2/2 | 0.000000→-0.000658 | 1 |
| brookite48-broyden-euclidean-seed11 | 8→8 | 1/1→2/2 | -0.000952→-0.000953 | 0 |
| brookite48-broyden-euclidean-seed29 | 8→8 | 2/2→2/2 | -0.001050→-0.000952 | 0 |

99次预算放行中，7个完成逃逸的合格落点所在轨迹含有放行（不是99次独立成功试验）：
AlOH Ritz11为2个、dimer11/29各2个，brookite dimer29为1个。
这些轨迹的落点中，按两组容差不同于初态的结构分别为1/1、1/1、2/1、1/1。
其余放行可能同属一次逃逸或位于预算截断的末次逃逸，不能用7/99解释成功率。
关联映射与minimum索引见comparison.json；这能证明存在有效预算出口，不能恢复严格策略的反事实路径。
按落点而非不同结构组计数，7个关联落点中较紧容差5个、较松容差4个与初态不同；不是7/7全部不同。

## 物理与因果边界

全部77帧完成几何筛查。AlOH的H最近O距离0.971–1.068 Å、Al–O配位数4–6（2.3 Å截断）；
brookite最小周期原子间距1.745–1.871 Å、Ti–O配位数5–6。
这些检查没有显示明显原子重叠或H远离O；不等于化学稳定、模型适用域、DFT或Hessian验证。
两组匹配无未知项；全体混合去重19/16个结构不能替代分臂公平比较。

严格模式复跑并非逐位回归通过：第一请求坐标完全相同但力差约6.44e-15 eV/Å；
初始淬火后坐标差约2.13e-14 Å，首个随机anchor相同，后续差异逐步扩大并改变轨迹。
它支持数值敏感性的解释，不能证明所有后续差异都仅由GPU造成。
Broyden各臂及数个Ritz臂零放行，结构数仍变化，直接限制了新旧轨迹比较的因果归因。
因此不把AlOH Ritz11约0.741 eV改善完整归功于新策略，也不进行求解器普遍排名。

## 原版证据与下一步

原版`LCONVERGE`指针已从unbiasedrot经cbd_rotation追到rotate_dimer；它接收
`force<ftol OR rotnum>rotmax`（另有PreRot负曲率例外），并非纯数值收敛标志。
见[native-rotation-followup](native-rotation-followup.md)第8节。
Python外部HVP预算和原版内部rotmax不同；本轮吸收其阶段退出思想，不声称逐步等价。

本项契约有材料级保留依据，关闭这一有界任务。下一优先级回到已批准的固定胞复现主线：
恢复并接入完整方向生成/阶段更新（含局部组自动选择的真实证据），随后LS初始化和搜索验收；
不把进一步压低旋转残差或扩大本组种子数变成全部研发的阻塞条件。默认替换需更广泛独立证据，
当前不替换，不新增经验参数。GA/VC/RC不抢占该主线。

## 产物与复核

- 新实验：`research/ga_ssw/evidence/rotation-budget-exit-20260917/`，含plan、冻结source/tests、逐请求记录、完整结构、analysis.json、comparison.json和日志。
- 严格复跑：`research/ga_ssw/evidence/rotation-budget-exit-regression-20260917/`，含regression-comparison.json。
- 旧对照：`matched-direction-materials-20260917/`；其去重报告在`matched-direction-diagnostics-20260917/analysis.json`。
- 作业内验证命令：冻结环境`python -m pytest tests -q --tb=short`：91 passed。
- 离线分析：作业内`python analyze.py --root .`：12/12，0 PES；随后`python research/ga_ssw/evidence/rotation-budget-exit-20260917/compare.py`成功生成配对结果。
- 首次配对脚本误指旧analysis所在目录，报FileNotFoundError；已按旧结果报告修正到diagnostics目录并成功复核。未重跑或改动任何实验轨迹。
- 实现尚在研究工作树，未合并/发布；不将测试通过写成完整SSW系列生产验收。
