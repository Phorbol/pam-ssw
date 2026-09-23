# VC / RC 重审：独立工作流已贯通，完整性边界仍是物理适用域

日期：2026-09-23。只读核对当前 VC、RC、RC-VC 实现与已归档 2014 VC-SSW / 2025 RC-SSW 论文及 2026-09-09 LASP ELF 反编译证据。未运行计算、测试或作业，未改源码。本报告只列两个会影响“完整算法家族”或物理解释的缺口；它们不等于发现了梯度代码错误。

## 当前能力与证据边界

VC 现在有两条明确分开的路径：`run_block_ssw` 是晶胞方向位移、定胞离子松弛、可选固定胞原子攀爬、全自由度终淬火和 E+pV MC；`run_vc_ssw` 是显式 3N+6 对称对数应变联合搜索。二者均采用 ASE E/F/应力契约，输出独立物理力和完整残余应力证书。联合坐标图及 RC-VC 图对公开梯度做有限差分；RC-VC 的非仿射原子力修正也由实际 EMT 应力面检查。上述验证支持所声明坐标下的数值一致性，不证明 LASP parity 或跨体系效率。

RC 已包含单树、树森林、周期刚体中心/晶胞几何和完整 RC-VC 驱动；真实 butane/GFN2-xTB 与 S22 water-dimer/GFN2-xTB 路径都完成了带完整原子终淬火的运行。周期 XXXII/GAFF 证据也包括完整 RC-VC 路径及独立力/应力和有限胞 Hessian 检查。这些结果说明当前实现已超过“仅几何模块”的旧状态；但真实 RC 案例仍是少量、开发用途，不能据此宣称普适结构发现能力。

## 缺口一：VC 的物理坐标检查合格，论文/发布版算法身份仍须限定

2014 年论文 §2.2–2.3 描述的是连续 CBD-cell 块：每轮对 3×3 晶格方向作有限位移、定胞松弛原子，随后可运行固定胞原子 SSW，最后再做全原子和晶胞淬火。当前 `block_ssw.py` 实现了这一先后结构；另一个 `run_vc_ssw` 是原理上自洽但不同的联合原子/对数应变方向和 Gaussian 路径。不能把联合路径称作 2014 流程复现。

可核对位置：`pamssw/standalone/block_ssw.py:103–175`；`pamssw/standalone/vc_reference.py:65–85, 189–217`；论文对照 `docs/research/vc2014-native-crosscheck.md:14–65`；原版方向更新与状态分支 `docs/research/native-cell-direction-lifecycle-followup.md` 及所链 `native-cell-direction-evidence/ssw_move.asm`、`gen_randommode.asm`。现有 ELF 证据确认了 native 有独立晶体 walker、当前晶胞上的位移尺度和方向更新路径，但没有证明每个 cell cycle 是纯重新抽样还是保留/混合前一模式；原版停止量也不是当前 full-tensor 残余应力范数（见 `docs/research/native-vc-convergence-contract.md`）。因此 block 是有文献依据的独立实现，direction lifetime 和 release 执行语义尚非 parity。

这项不构成当前物理梯度的否决理由。当前更重要的边界是效果证据：已有 AlOH26 与 brookite48 四臂对照是单种子、有限调用的 feasibility gate；7 个 paid proposals 被预算截断、仅 AlOH joint 给出一个通过原始力/应力门的落点，brookite joint 没有落点。归档分析还指出其偏置路径体积扩张及短接触，但未证明扩张导致耗尽。见 `docs/research/complex-vc-feasibility-failure-geometry.md`。这能说明有限预算下的成本与失效阶段，不能给出 VC 胜过固定胞搜索的结论。

## 缺口二：RC-VC 的导数和终淬火完整，化学拓扑与周期 lift 仍是调用者前提

RC-VC 明确要求用户给出 `rigidbody/blist`、固定 body tree/forest 和连续周期 lift。实现拒绝 body graph 环路，不推断穿胞分子的 image，不会在 unrestricted 真淬火后重新判定拓扑是否仍有化学意义。终淬火释放所有刚体约束是正确的物理证书流程；但若过程中发生断键、成键或分子身份重排，后续搜索仍沿调用者给定的刚体分组重建 chart，数值上可继续而物理搜索子空间已经失配。

可核对位置：拓扑解析与环拒绝 `pamssw/standalone/rc_topology.py:23–73`；周期 forest map 的 lift 契约及刚体内部映射 `pamssw/standalone/rc_vc_geometry.py:12–20, 41–62`；驱动将同一 `trees` 用于每次选中最小值后的 chart `pamssw/standalone/rc_vc_reference.py:27–60, 67–74`；原生刚体旋转/力通道仍未由独立 JᵀF map 复现，见 `docs/research/rc-native-transmit-contract.md`。native `krot` 同时进入前向变换和力路由，不是可以在现有精确 Jacobian 上简单乘的单独缩放参数。现实现是保守的独立坐标/梯度替代，不应称原版 RC 路径 parity。

对**固定且化学稳定的分子/分子晶体拓扑**，显式输入不是算法缺陷；对反应、环闭合、跨胞图像自动识别或拓扑变化搜索则超出声明域。XXXII 的转换 GAFF 证据支持某一个周期体系的数值端到端可行性，但不替代其他化学拓扑、原生力传递或独立多种子效果验证。

## 优先级建议：XXXII 固定拓扑、成对 RC-VC / Cartesian VC 行走

撤回把 S22 水二聚体追加实验作为主线优先级门槛的建议。已有单次 S22 GFN2-xTB 运行仍保留为历史接口证据（183 E/F、没有 Hessian 或严格异构体判定），但不足以排序面向复杂分子晶体的 RC。新增案例不切换到 xTB：选已有 172 原子 TYPE2-XXXII 周期分子晶体，沿用转换自原始 GAFF 拓扑的同一 LAMMPS/ASE 模型、原始输入构型、四分子 `rigidbody`/`blist` 图和显式连续周期 lift。仓库保存的输入是 `tests/standalone/fixtures/type2_xxxii.extxyz`，SHA256 `ecfb6775cc0d9dae9ae05b432a32a91b7f88da12a387be4b5ca39dffd50457c`；来源拓扑在研究归档 `GA-SSW_examples_run/global_exploration/input-templates/TYPE2-XXXII/mc/{rigidbody,blist}`。当前模型文件及来源清单由 `research/ga_ssw/evidence/xxxii-rc-vc-replicated-completion/plan.json` 固定；它指向原始 converted `lmp.data`、`in.simple`、`manifest.json` 的哈希。`XXXIIReplicatedCalculator` 校验这些来源/输出哈希，内部 1×1×2 表示每次做 344 原子 E/F，再折回原始 172 原子。它可复用于有界研究对照，但要把数据/输入/manifest 一起冻结并按适用域使用；这不是原生 GAFF 引擎的逐位复现，也不是额外 DFT 物理背书。不要拿 stock-LAMMPS 资格化路径替代：归档明确指出其 finite-approximation 导数残差尚未闭合（`docs/research/xxxii-lammps-real-qualification.md`）。

最直接且当前驱动已存在的消融，是同一初态、同一模型、同一 E+pV 目标、同一细胞自由度与落地淬火条件下，对照 `run_rc_vc_ssw` 的固定刚体森林 chart 和 `run_vc_ssw` 的完整 Cartesian 3N+6 chart。它比较的是 RC 对固定拓扑分子晶体搜索的影响，不应写成原生 LASP parity 或 2014 VC 算法复现。建议预登记至少 5 个共同初态/随机种子配对，每臂每配对运行至少 20 个**尝试**的 proposal；每对按相同累计 E/F 请求上限截断，超限失败、偏置旋转失败、无有效落地及真淬火失败都进入分母。每臂每配对先设 100,000 E/F 的硬上限，只有在正式运行前的资源估算证明达不到 20 次尝试时，才同步下调两臂的 proposal 数和预算并作为方案版本记录，不得看候选能量后修改。100k 是计划预算，不是物理参数/已验证阈值。共同 seed、压力、温度、接受规则、cell metric（strain length 5 Å）、终止阈值、E/F 总请求定义和 wall cap 均写入冻结配置；RC 与 Cartesian 的原生自由度维数差异是处理因素，不通过把维数强行配平消除。

预定主要结果为每配对的有效且去重极小值数、独立结构数/相同 basin 复访率，以及按累计 E/F 成本得到的有效新盆地曲线；同时逐项报告接受/拒绝轨迹、偏置阶段与失败位置、调用数和墙钟。所有比较端点都用同一 GAFF 真 PES 做全自由度 cell+atomic quench，并用相同力/应力门及独立端点复算；按周期置换、物种、cell 及分子构象检查等价，保留未绑定、拓扑失效和非收敛样本。不得只比较最低能量或只纳入成功轨迹。由于 GAFF 拓扑固定，此实验回答模型内 PES 覆盖/刚体 chart 约束的相对搜索表现，不回答真实化学稳定性、断键反应或 DFT/实验相稳定性。

上述规模只是候选预算，尚无事件率或成本证据证明其足以区分方法；不能据此承诺可重复覆盖。只有事前成本估算和明确的目标事件支持时，才考虑将该对照纳入后续里程碑。当前无需提交作业，也不在本审查中启动此计划。VC 的既有 AlOH26/brookite48 证据仍未显示一致优势；这项分子晶体对照若不能改善相同计算预算下的有效盆地覆盖，优先推进固定胞 SSW/LS 主线，不增加 VC 参数或启发式。

## 决策界线

保留 VC 的 block 与 joint 路径为两个实验性算法选项；保留 RC/RC-VC 的固定显式拓扑和 lift 契约。当前没有证据要求改写其数学内核。若后续目标转为“复现 LASP release”，须分别恢复 VC cell-direction/state/stop 链以及 RC 完整前向坐标和力传递，再逐项对照；不能靠现有一致梯度测试补齐 parity。若目标是独立 PES 搜索方法，先完成同预算、真实体系的固定胞主线，再评估上述 XXXII 多配对对照是否足以把 RC 提至下一里程碑。S22 水二聚体只作为既有接口历史记录，不作为新增实验或优先级门槛。

## 主agent整合后的执行边界

以上XXXII成对方案是后续候选，不是已批准/已提交的百万EFS计划。
“至少20步×每步5000”只提供预算上界估算，不能证明20步有统计判别力；执行前必须先
用已有实际成本和目标事件率确定最小有效协议。当前不以RC/VC精确native parity作为
固定胞主线的前置条件。当前执行次序与预算以2026-09-23-mainline-reassessment.md为准。
