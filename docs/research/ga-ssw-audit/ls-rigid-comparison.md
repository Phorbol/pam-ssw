> 原始报告位于 `/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/analysis/ls-rigid-comparison.md`；下文的原资料相对路径按该外部研究目录解析。

# LS / rigid-body：原版 LASP、论文和 PAM 的边界与 ASE 复现规格

日期：2026-09-09。范围：只读代码/原始论文/ELF 静态研究；不运行优化、能量评估、GPU 或 HPC。PAM checkout：`/home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity`，复核时 HEAD `e5de210faee8b1f82a16f402ac3197801a8c51a8`。本文件及 `ls-*.asm` 写入外部研究目录，不改 PAM 共享源码。

**结论：完整 ASE GA-SSW 不能直接把 PAM 的 LocalSofteningModel 与 rigid.py 当作原版 LS / RC 内核。** 原论文 LS 有每个外层 SSW step 的固定邻居、预软化淬火、按真实能量响应更新的成键对势；PAM 则每个 walk 内步重建对势，参数与自适应语义不同。PAM rigid.py 是整体零模投影，并未实现分子刚体链自由度。水 TYPE3 的既有原版 smoke 只证明普通固定胞 SSW 接口；不能拿它证明 LS 或 rigidssw。

## 1. 原始资料与证据等级

- **论文事实**：Tong Guan, Cheng Shang, Zhi-Pan Liu (2024), *Local-Softening Stochastic Surface Walking for Fast Exploration of Corrugated Potential Energy Surfaces*, JCTC 20, 11093–11104, [DOI](https://doi.org/10.1021/acs.jctc.4c01081)。[作者正文](http://www.lasphub.com/publication/215.pdf)保存为 `literature/215.{pdf,txt}`；[ACS SI](https://doi.org/10.1021/acs.jctc.4c01081.s001)保存为 `literature/ct4c01081_si_001.{pdf,txt}`。正文 pp. D–E / §§2.3–2.4 / eqs.11–15 已渲染目检：`analysis/ls-paper-page{4,5}.png`。
- **论文事实**：Tong Guan, Xin-Tian Xie, Xiao-Jie Zhang, Cheng Shang, Zhi-Pan Liu (2025), *Global Optimization of Large Molecular Systems Using Rigid-Body Chain Stochastic Surface Walking*, JCTC 21, 5757–5770, [DOI](https://doi.org/10.1021/acs.jctc.5c00350)。[作者正文](http://www.lasphub.com/publication/225.pdf)为 `literature/225.{pdf,txt}`；[ACS SI](https://doi.org/10.1021/acs.jctc.5c00350.s001)为 `literature/ct5c00350_si_001.{pdf,txt}`。两篇 SI 均用 Figshare 官方 API 按 `resource_doi` 找到，完整文件元数据为 `literature/{4c01081,5c00350}-figshare.json`。
- **二进制事实**：实际 `GA-SSW_program/lasp` 的符号、指令、数据常数，见 `analysis/lasp-symbols.txt`、`selected-dwarf.txt` 和本次新增 `ls-*.asm`。静态恢复不是动态运行轨迹证明。
- **代码事实**：`decompiled/sgn/lasp/LaspCalculation.java`、实际模板与 PAM 源码。反编译源码和论文不是同一版本的自动保证。

## 2. LS 的科学目标和准确公式

目标是改善强共价键体系的局域/硬振动模式与高能垒导致的搜索滞留；不是修改真实热力学目标，也不是保证每个模都变软。令外层第 i 个 SSW step 的真实极小值为 R_i；只在该点建立无序邻居集合 B_i，并冻结其参考距离 r^0_pq。论文 eq.11 的成键对势可等价写为

\[
 V_{LS}^{(i)}(R)=\sum_{(p,q)\in B_i} A_{pq}^{(i)}
 \exp[-(r_{pq}(R)-r^0_{pq})/(\xi r^0_{pq})],\quad \xi=0.2.
\]

无序对形式已消掉论文逐原子、双计邻居的 1/2。ξ **无量纲**；A 为能量，距离以 Å 表示。初始 A=0.03 E_pq，E_pq 是标准对键能；正文给 C–C 约 3.61 eV 的例子。该初值是论文设置，不是由公式严格推出的通用最优值。

每步执行：真实极小值 R_i → 在 E+V_LS 上预淬火到 R̃_i → 在 E+V_LS+累积 Gaussian 上作 SSW climb → 同时去掉 LS 与 Gaussian → 真实 E 上完全淬火 → 真实能量 Metropolis 决策。LS 和 Gaussian 在数学及程序状态上必须分别保存。

定义论文真实响应量

\[
 P_i=[E(\widetilde R_i)-E(R_i)]/N,\quad
 \bar V_i=N^{-1}\sum_{B_i}A_{pq}^{(i)}.
\]

它**不是**当前 V_LS/N。论文 eqs.14–15 的更新用无序对可写成

\[
 \bar V_{i+1}=\bar V_i-\lambda(P_i-\Upsilon),\quad\lambda=1.8,
\]
\[
 A_{pq}^{(i+1)}={E_{pq}\over\sum_{(u,v)\in B_{i+1}}E_{uv}}
 \left[\sum_{(u,v)\in B_i}A_{uv}^{(i)}-N\lambda(P_i-\Upsilon)\right].
\]

Υ 的单位为 eV/atom，是搜索策略目标，不是温度；λ 无量纲。SI 提供体系设置：C60 用 0.02 eV/atom、C70/C90 用 0.04、C4H6 路径探索用 0.7、Fe7C3 用 0.01；LASP `SSW.soft.SAbiasAtom` 输入对应 meV/atom（20/40/700/10）。这些不能互换为通用默认值。SI 中 C70 的 20/40/60 meV/atom 对照是灵敏度证据，不能证明跨体系最优。

**数学推论**：对于正指数排斥势 U=A exp[-(r-r0)/a]，U'=-U/a，U''=U/a²；相对坐标 Hessian 为 U'' uuᵀ+(U'/r)(I−uuᵀ)。径向曲率正、横向曲率负，且预淬火改变真实 Hessian 的评价点。因此“加 LS 必然降低所有模频率”不是定理。

## 3. 实际 ELF 中已恢复什么

| 项目 | 直接证据 | 可下的结论 |
|---|---|---|
| 指数对势 | `ls-pot-bond-add.asm` 0x6c615d–0x6c61a4；`.rodata` 0x4a4c160=0.2 | exp[-(r−r0)/(0.2r0)]，与论文形式一致 |
| 势强度组成 | 同文件 0x6c60d0–0x6c6141 | 有效 A=amp_c × bond_ener_list(type_p,type_q) × atom_filter_p × atom_filter_q；DWARF para+93704 标为 atom_filter |
| 编译期 amp_c | `.data` 0x5520450 的 double 为 2.0 | 这是乘子，**不是**可直接写作原版固定 A=2 eV；元素表初始化/运行期改变仍须一起解释 |
| 邻居与参考距离 | `ls-bond-counter.asm` 0x6c57cc–0x6c582b | 接受距离小于元素对 bond_len_list + len_toller 的对；保存当步实际距离和原子索引。不是 ASE covalent_radii ×1.25 |
| 激活 | `ls-bond-info-init.asm` 0x6c6a8c–0x6c6ae5；`.rodata` 0x4a4c1b0 | 初始 l_softmode = LselfAdapt OR 文件 `pot_bond_input.txt` 存在。单凭 absence of keyword 还不足以判定关闭，须同时检查该文件 |
| 循环/更新控制 | 同文件 0x6c6aef–0x6c6b2f、0x6c7642–0x6c768b、0x6c7ebc–0x6c7f20 | nsoftstep=nearest_integer(softmodecycle×softratio)；存在 npreselfadapt/freqselfadapt 的预阶段与间隔控制，并读取 biasperatomaim/steplenselfadapt。不是每次坐标评估时按距离偏差更新 |
| 添加/剥离 | `ls-pot-bond-add.asm` 起始保存 raw_ene/raw_fa/raw_s33；`ls-del-pot-bond.asm` 0x6c6450 起恢复这些量 | 真实与 LS 修饰的能量、力、应力分开；恢复函数不是仅把能量减去一个标量 |

新增反汇编还包括 `ls-ssw-move.asm`（固定胞 SSW 中多处 LS 势调用）及 `ls-bond-info-init.asm`。原 ELF 的策略还含软化周期、元素过滤、受限更新等分支；本次**没有**声称完整恢复其逐分支调度、键能表出处/完整数值、周期 PBC 邻居边界、无邻居分母为零/负强度保护、所有参数缺省值，以及某个动态轨迹的实际 LS 开关。paper mode 与 binary parity mode 应明确分开，不能默默把未恢复分支替换成“合理”的 PAM 逻辑。

## 4. PAM 当前 LS 的实质差异

以 live `pamssw/config.py:430`、`softening.py`、`walker.py:3205` 为准；`docs/theoretical-analysis.md:465` 仍称默认 Gaussian，与当前 config 的 Buckingham 默认不一致，属于文档漂移。

| 维度 | 论文/实际 LASP 核心 | PAM 当前实现 |
|---|---|---|
| Buckingham 长度尺度 | ξ r0，ξ=0.2 无量纲 | exp[-(r−r0)/xi]；默认 xi=0.3 Å |
| 初始强度 | 按标准元素对键能缩放；随后按真实预淬火响应更新 | 所有被选对相同 strength=0.6 eV |
| 对集合 | 外层 SSW step 初始邻居，step 内冻结 | 默认 covalent-radius neighbor_auto、scale1.25；`_build_softening(current,...)` 在 `max_steps_per_walk` 循环每轮执行，参考距离也随 current 更新 |
| 独立预软化淬火 | Gaussian/dimer 前 E+V_LS 预淬火并测量 P_i | inspected walk 先初始化方向，随后构造 ProposalPotential 并做 direction scoring；没有与论文一致的独立预软化淬火/响应控制状态 |
| 自适应 | step-to-step，目标真实能量响应 Υ，邻居总键能重新归一化 | 默认关闭；可选按 abs(r−r0)/(0.25r0) 逐评估放大，最大3倍，含导数；不是论文式更新 |
| 截断 | 本次恢复的 pot_bond_add 遍历冻结成键列表，未见距离硬截断 | 默认超过 r0+2 Å 直接跳过项，未 shift/switch，端点能量和力有跳变 |
| 其他核 | 此次核心为 Buckingham 排斥项 | 可选正 Gaussian bump，虽名 gaussian_well 实际不是负势阱 |

默认 Buckingham 在硬截断前仍有 0.6 exp(−2/0.3)≈0.000764 eV/对；这是直接计算出的不连续量，不能仅用有限差分远离截断通过来证明全域光滑。是否改截断应另立实验；为复现原版不得自行加新 switching 或自适应启发式。

## 5. rigid-body 的三层含义不得混用

1. **Java GA 的分子操作**：依据 `seg` 拆分、组合、旋转/平移分子构造候选，随后交给 LASP 淬火。它只决定 offspring，并不决定 native SSW 每个优化坐标。
2. **LASP rigidssw/RC**：在分子刚体或刚体链广义坐标上做方向、二阶信息、Gaussian climb 和局部优化；周期版还含晶胞联合自由度。ELF 中有 `rigidssw_`(0x4c0a00)、`rigid_f_c2r_`(0x8132a0)、`rigid_x_r2c_`(0x81c0f0)、`rigidreset_`(0x818880)、`super_rigid_`(0x80bc10)。这些符号支持真实 native 独立模块的存在，但本次不声称每个变换/导数已与论文一致。
3. **PAM rigid.py**：`project_out_rigid_body_modes` 去掉整体平移/旋转方向，返回的仍是 3N 笛卡尔方向。它不施加分子内部刚性，不携带分子链拓扑，不生成每个分子的六自由度，不实现 RC 与晶胞耦合。

实际 Java `LaspCalculation.java:121` 将 seg 的 zero-based 索引加1写 `input/mc/rigidbody`；但只有 TYPE2 分支（约行108）复制 rigidbody、blist、lmp.data、in.simple 到 LASP task。水 TYPE3 的 `lasp.in` 是 `explore_type ssw`、`Run_type 5`，无 rigidssw；因此水 GA 使用分子块不意味着水 SSW 以分子刚体运行。TYPE2-XXXII 模板确实使用 rigidssw、movecell=T、nrigid_step=10、rotstepsize=7、transtepsize=0.2、latstepsize=0.2；这些是该模板事实，单位/归一化不能仅从名字猜测。

RC 原始方法需同时保存：刚体组、连接键/父子图、每步参考构型、中心体平移/角轴、子体扭转、广义力、允许晶胞自由度与尺度。SI §§7 / S9–S10 显示 `rigidbody` 的组允许**共享旋转键端点原子**；不能强制 disjoint partition。`blist` 是 1-based 原子连接对，应与 input.arc 的原子顺序完全一致。

论文 SI §§1–2 / S1–S4 给 Rodrigues 变换与完整阶段：初始化链坐标 → 刚体链 climb → 刚体链 l-BFGS → 去掉刚体约束的全原子 reopt → 真实能量 MC。只在刚体面收敛的结构不能称真实极小值。RC transmit λ（正文§2.2）是链上力/转矩传递参数，与 LS 更新 λ=1.8 无关；SI 用0.5–0.9比较chignolin，不能据单体系把0.7称通用最优值。

## 6. ASE 复现顺序和可证伪验收

**先恢复原版语义，再比较搜索效果；复用 ASE calculator/结构容器，不直接复用 PAM 策略。**

- P0：普通固定胞 SSW 和 Java GA/NNA行为层分开；能量/力符号、单位、Gaussian/LS分解、真实淬火证书和每阶段调用账本统一。已有水普通SSW可以作接口基线，不能替代LS/RC验收。
- P1：paper-faithful LS 最小实现：按已记录标准键能表初始化、ξ=0.2、冻结当步对列表/r0、独立软面预淬火与真实响应测量、eq.15跨步控制、去掉所有偏置后真实淬火。明确策略参数 provenance；未恢复的二进制调度不得隐藏。给出每步 B_i、r0、A_i、E_before/after、P_i、target、真/偏置调用数，才能做 native vs ASE 单步审计。
- P2：独立刚体几何层：先不连接的分子刚体，再连接链，再晶胞耦合；每级确认 forward transform、Jacobian/广义力、旋转图的零角/共线极限、PBC跨界和共享端点一致性。独立刚体是中间实现，不应提前冠名完整RC。最后接相同SSW外壳与全原子最终淬火。
- P3：完整 GA 调用链再开启原版类型映射；优先 TYPE3 普通水 + 一个 LS 共价体系 + TYPE2 真实分子晶体。若要面向完整RC论文范围，另加带可旋转键分子的链体系；独立刚性分子晶体不足以验证父子链。

**最小真实体系 E2E（计划，未启动）**：使用作者已给结构与相同模型/后端，LS优先C60+CHO_carbon_pf.pot（先确认该势文件实际可用），RC优先TYPE2-XXXII现有输入/GAFF后端并补一个作者RC连接链例子。每个用固定结构和不少于三组预登记随机种子，先短流程贯通：输入→候选/GA→局部优化→LS或RC搜索→去偏置、去刚体的真实淬火→NNA/几何去重→选择→最终结构导出与独立E/F/允许应力复算。短贯通预算只能证明接口，不足以证明效率；正式预算和成功率目标必须在运行前登记，不能用后验更换成功分母。

对照至少包含原版相应内核、ASE复现、当前PAM稳定设置；LS再做相同SSW无LS与固定A/自适应A分离，RC再做同后端全原子SSW与断开链传播的合理消融。比较有效且不同的真实极小值、低能覆盖/达到目标成本，同时报告所有E/F/stress调用、软面预淬火、dimer Hessian向量近似、失败/回退、全原子reopt和墙钟时间；不以archive数量或接受率代替成功。

停止/修正条件：单步 E/F/坐标或邻居/amplitude不一致则不得宣称行为复现；LS产生解离平台、RC最终全原子淬火失败或越出势适用域均单列失败；相同调用预算下没有跨种子覆盖/成本收益则不升级默认。收敛测试、运行正常结束、物理稳定与搜索性能四类结论分别报告。

## 7. 本次已完成与未完成

已完成：两篇正文+官方SI归档，LS关键公式目检，实际ELF指数核/参考距离/部分开关和调度静态恢复，Java rigidbody传递边界、PAM LS及零模投影差异审计，以上ASE规格与验证顺序。

未完成：完整LS binary调度与参数表动态取证、native rigid-chain坐标/广义力逐函数恢复、ASE新内核实现、LS/RC真实体系E2E、相同预算PAM效率比较。当前材料只支持研发规格和差异结论，不能称完整ASE GA-SSW、LS或RC已复现/科学有效。
