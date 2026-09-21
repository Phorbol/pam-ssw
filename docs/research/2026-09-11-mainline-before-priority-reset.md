# 当前主线：独立 ASE SSW 家族与可验证的收益

更新于2026-09-11。本文是当前执行计划；历史阶段决定存于
`2026-09-10-mainline-before-evidence-consolidation.md`，不再作为待执行任务。
实现清单与结果边界见 `ssw-family-current-status.md`。

## 目标与设计取舍

完整实现可接入 ASE Calculator 的独立 Python SSW/VC/LS/RC/GA 本体。
SSW 与 VC 是公共核心，LS/RC/GA 的已知真实缺口可并行补齐。完整交付以各变体
算法本体闭环为目标，不能以接口存在或测试通过替代完整性。正式搜索不调用
LASP/JAR；原程序仅作为静态与隔离数值实验的行为证据。沿用 PAM Safe-total
作为工作数值后端，任何更换都需要可比成本下的证据。

数学/物理一致性优先于照抄原版：Gaussian 能量与力必须配对，偏置在一次
局部优化内冻结，变胞优化 E+pV 并使用同一坐标的精确力/应力链式法则。
原版旧 Gaussian 力重复累加、辅助 LJ 角导数和非旋转不变 Broyden 内积等
已证实问题不作为生产默认。独立修正、数值替代和原指令等价分别标注。

当前不推荐普遍开启 LS，也不建立新的混合策略调度器。C4H6 与高度策略
对照已经表明成本、完整分子覆盖和不同极小值覆盖可能互相冲突。有限实验
不能证明不存在任何未来收益；每条研究线以明确、可证伪的问题推进。

## 当前执行任务

### 最新优先级纠正：偏置停止与VC分块（2026-09-11）

本轮已实现 `VCSSWConfig.bias_release=numerical_stop`（实验选项，strict保留），
正常数值停止可进入真实淬火，最终证书与预算不变；SciPy异常退出已单独分类。
33项相关回归通过，另block回归3项通过；Cu/EMT六流程952EFS，未触发释放。

Fe7C3同预算释放实验job1261354完成，8493EFS含10fresh：Safe得到1个不同
高能候选（+13.44eV，拒绝），SciPy三个证书合格点均匹配初态，ASE无新合格点。
无低能增益，不能据低成本重复初态推荐SciPy。详见`fe7c3-vc-numerical-release.md`。

既有block实现job1261467完成，3998EFS含4fresh：两个cell-only提议各222/221EFS，
全部中间原子松弛25步maxiter仍可真实淬火合格，但终点+15.11/+18.80eV，均拒绝。
第二步含atomic SSW，两组已完成9/8个偏置梯度合格阶段后耗尽预算，属于截断，
不是内层不收敛。详见`fe7c3-block-baseline.md`。Fe累计60940EFS。

反编译已定位`fixcell_climb`在climb阶段清零尾3行cell-force输入；是否完全冻结
坐标还必须核对LBFGS耦合历史和输出投影，不能由零cell梯度直接推断。
接下来聚焦atomic climbing成本/停止生命周期、native约束及cell方向状态寿命；
不再把joint严格biased gate失败泛化为VC原算法缺陷，不继续盲扫stress阈值。

用户指出的两点已通过代码/全文和真实体系检查确认：内层偏置未达到梯度
证书，不代表不能去偏置淬火；2014 VC使用cell扰动、有限步固定胞原子松弛、
按频率加入固定胞SSW，最后才做全自由度淬火。现有block实现必须重新成为
复现主线，joint log-strain保留为独立对照，不能将其失败泛化为原版VC问题。

此前六臂10个有计算的accepted偏置终点，全部通过no-bias quench及独立
真实力/应力证书，新增835EFS（含20fresh），作业1259727单V100/27秒。
因此零落点受到我们自己的严格gate影响。无低于初态的结构：三点在三组
周期结构匹配容差下均匹配初态，其余七点均不匹配初态或彼此，高2.94–13.44eV。
终态Hessian未确认，近似结构匹配不等于严格盆地身份，不能声称新稳定相。
Fe7C3登记成本48449EFS，诊断成本不可放回旧实验预算或冒充完整搜索效率。

原版真实频率消费者已定位到get_random_mode0：正Ratio_atomcell按nsswstep
取模等于1选cell，负值取反，零值禁用；35个隔离原指令检查通过。随后
moveds按lcellmove选ds_cell/ds_atom。这个结果证明模式调度，尚不证明原子
模式下每次inner relax冻结cell。继续闭合允许DOF和NG/NG_cell的消费，
并将正常数值停止与允许真实淬火分开；不再优先扫LBFGS历史或逼偏置梯度。
完整证据、参数和限制见 `vc-schedule-and-release-correction.md`。

用户最新重排（2026-09-11）：优先尽可能完整实现各类SSW本体。Luna承担边界
清楚的实现和缺口审查，主agent负责原理、证据核验与优先级，不转交全项目上下文。
执行顺序如下：

1. 共用SSW/VC核心：方向、偏置续接、联合坐标/梯度、局部淬火、真实落点与选择。
   优先修真正缺失的生命周期或数学契约，而不是继续在Fe7C3上调参数。
2. 成熟局部优化基线是上述核心的验证依赖：SciPy L-BFGS-B与ASE
   LBFGSLineSearch；Safe-total和恢复的LASP数值版本是待比较版本，“优化版”
   不预设更快。统一目标/坐标/证书/请求账本，明确各自线搜索和步长差异。
3. LS、RC/RC-VC与GA：按代码核验的本体缺口补齐；已有实现但未真实验证
   与未实现必须分栏。反编译证据不完整时标明独立替代，不能声称完全等价。
4. 完整调用链落地后开展有界真实体系端到端比较。冻结目标仅用于定位问题，
   不再用大量局部参数试探代替全流程进展。禁止bias-separated/历史transport重启。

注意 `history500` 是最多保留500个secant pair，不是 `maxiter=500`；此前
Fe7C3 history10/500对照均为maxiter300。history500在四冻结目标仅1/4收敛，
不能据此宣称完整VC成功。成熟线搜索基线尚未接入公共联合VC调用链；原版
LBFGS/MCSRCH/MCSTEP目前仍是隔离原指令研究工具，不是独立Python生产后端。

本轮两种成熟基线已通过研究bridge接入完整联合VC流程，仍未成为公共生产
选项。主审在user-site与隔离环境分别验证16项相关测试。Cu4完整流程揭示
停止范数不一致；按明确范数不等式转换后，三算法各得到一个通过fresh物理
证书的落点，成本64/67/96EFS。未确认不同盆地，不能据此排名。已冻结
Fe7C3-80非LS原始设置做三优化器×两seed完整对照。作业1257474单V100已完成，
总8398EFS、12requested/10paid attempts、0新合格落点；六fresh全为初态。
Safe/ASE各3996EFS且预算受限；SciPy406EFS但四次native_stop均为相对能量停止，
共同梯度0.001991–0.006016未达到0.001。不能据此排名或宣称换优化器解决VC。
见 `fe7c3-vc-mature-baselines.md`；Fe7C3登记累计成本47614EFS。不继续调参。
执行目录 `research/ga_ssw/fe7c3-vc-mature-baselines/`，额外CPU准备检查33EFS。
细节和全部626EFS探索成本见 `vc-mature-baseline-norm-contract.md`。

native moveds四条retry分支与三例selected-minus-center归一化/width写回
局部原指令检查均完成；后者已修正探针指针方向与绝对坐标输入错误，该错误
不是LASP行为。未覆盖完整caller。固定胞表+1f8/+200已解析到Allopt及其
收敛判断；实际run_ssw描述符0x53ca680的相关表项也已核验相同目标。
外层正常顺序已闭合为cal_pes→move→run_ssw→返回→下一轮cal_pes。
剩余是停止状态到释放坐标/force记录的条件映射，不能再称首个外层E/F未知。论文规定在
对应终止条件后去偏置再淬火，并未支持maxiter自动当作有效逃逸。见
`ssw-paper-native-release-boundary.md`、`native-moveds-success-probe.md`。
不先做checkpoint，也不把已闭合的CSSW dispatch重列为缺口。

用户要求主开发只考虑Safe-total，不重启bias-separated。偏置历史transport
草案已退出执行队列，未实现/未启动对应实验。继续审查已有联合VC Safe-total
的坐标、梯度尺度、正曲率历史与步长，并对照原版VC调用链。

2026-09-11 CuO64 VC 线已完成当前登记的有界证据。原始资格检查为806
E/F/stress；PQC/joint 四臂比较为7,992；冻结失败点历史诊断为997；两 seed
history500 whole-SSW 控制为3,996，总计13,791。四臂比较为8 requested、6
entered、2 not-started、2 maxiter 和4 budget-censored，0 search landing。
history500 控制两次实际尝试均在biased quench耗尽1,997 search请求，阶段
13/12后无 landing。完整 CuO 成本、stage、失败范数和请求分母见
`cuo64-vc-comparison.md` 及各 evidence summary JSON。

这些结果是 CuO64/MACE 的受预算限制诊断，不支持 PQC/joint 排名、history
500 的 whole-SSW 收益、CuO 相身份或 native VC parity。冻结目标对照已完成：history10两次均达300迭代上限，history500分别在
161/193步收敛，并通过fresh偏置梯度检查；这只证明两个局部淬火得到改善。
全流程最后两段是总预算截断，因此取消所谓后期优化器故障的Hessian诊断。
原版释放审查只闭合optimizer stop，不能据此释放未收敛点；CuO本轮不再扩容或调参。

固定基底SSW与LS显式组合现已完成公共入口与Cu28/EMT端到端检查，含真正
二维PBC；ls=None轨迹保持，二维/三维真空表示的954次调用逐项相同。登记成本
3,215 EF，2seed以下的小样本不支持搜索收益。见constrained-ls-results.md。

LS的化学pair energy filter及跨步保存已实现，默认Cu4/EMT39次调用完全一致。
静态证据表明原版Fe–Fe过滤置零能量贡献、保留几何候选计数；独立paper响应
与原版Nb归一化响应分别标注。Fe7C3-80/MACE初态检查990EFS、PQC/joint
比较7994EFS、两高能PQC候选严格检查1982EFS均完成。两候选有正的有限胞
联合曲率但均高能且MC拒绝；joint无候选。LS全pair/过滤pair比较5502EFS，
8次尝试无候选，6次maxiter与2次预算耗尽。输入不是论文精确初态，表值是
版本提取值，不宣称模型/化学通用性。见fe7c3-80-*.md各证据记录。

冻结完整LS+Gaussian失败目标的history10/500诊断现已完成（job1253325），
2435EFS。四个原失败点梯度均复现；八个局部优化仅过滤pair/seed101/history500
达到阈值，不能推断whole-SSW收益。解析87度联合VC高度对照也已完成
（job1253271），3996EFS、两seed、四requested，0候选；已准备阶段的力夹角
均为87度，但两次实际付费尝试都耗尽预算。因此不替换现有高度或history默认。

进一步的零PES审查发现：旧LS预淬火只优化原子，八个prepared状态的软化
势cell梯度仍为15.543–27.969 eV/A。现已实现显式`ls_prequench="joint"`，
联合优化E+LS+pV并按真实ΔH更新响应；没有新数值参数，固定胞默认保持。
对应Fe7C3四臂对照job1254087已完成，5777EFS、八requested、0有效候选；
软化势联合驻点恢复并没有带来这组搜索的成功。完整逐臂核验见
`fe7c3-80-joint-prequench-comparison.md`。这项能力保持实验性，不扩大预算
挽救该算例。Fe7C3当前六组实验加本次共28676EFS，失败和fresh检查均包含。
最近完整接口/数值回归397passed/1skipped只支持实现检查，不支持搜索收益。

原版普通CSSW的descriptor消费链现已静态闭合：run_ssw明确初始化类型表
0x53cacc0，ssw_move的+0x198进入update_forcepara，+0x48再进入refresh；
refresh的+0x98实际调用stress2dedlatt。原先将Fortran descriptor与其对象
混淆导致的“未知slot”不再是缺口。详见native-vc-consumer-followup.md。
恢复的仍是已有压力/体积/逆晶胞力变换，不据此新增搜索策略。

同四个冻结Fe7C3 LS失败目标的隔离原版LBFGS配置对照job1254745已完成，
1151EFS，0/4满足共同联合梯度；3例300接受步上限、1例原版线搜索失败。
已有Safe10为0/4、Safe500为1/4；不同scalar/block步长与history使它属于
配置比较，不能归因于MCSRCH单项。四目标初始E/g、接受坐标与fresh末态
均独立核验。见fe7c3-native-lbfgs-results.md，不替换默认优化器或调参。
job1254721的依赖路径失败发生在PES前，0EFS，另行保留。Fe7C3全部登记
实验成本在该阶段为29827EFS，停止此组优化器与LS选项的重复试探。

Safe-total 四个冻结失败末态的完整偏置 Hessian 检查 job1255150 已完成，
新增3892EFS，总登记成本33719EFS。两种差分步长下三个目标有可靠负曲率，
一个局部正定；四个重建的 Safe-total 方向仍均为下降方向。最低模式的 cell
分量为0.4%–2.2%，不能把失败简单归为缺少变胞支持或 cell 尺度不当。
这不是物理极小值稳定性检查，也不证明负曲率就是迭代失败原因。
详见 vc-safe-total-failure-curvature-results.md。后续只审查 Safe-total 的已有
收敛契约，核对原版偏置阶段 ftol/strtol 与本实现联合范数的定义；偏置阶段
精度与最终真实势力/应力证书分开。未闭合单位和量的定义前不改阈值，不新增
优化器、历史搬运或经验回退。

原版VC停止算术已用27个隔离原指令组合及4个压力案例闭合：max(abs(sfa))
达ftol，或原子分量力与平均正应力残差分别达ftol/strtol。这不是完整张量
应力证书，也不同于本实现的联合范数；不照抄该OR分支或数值0.05。
已有默认gtol=.005对原实验.001的单变量完整Fe7C3检查job1255870完成，
5497EFS、8requested/7paid、0候选，对照5502EFS/0候选。四fresh均为初态。
登记Fe7C3总成本39216EFS；内层容差恢复未解决本组问题，停止此轮扫描。
详细证据见native-vc-convergence-contract.md。后续核心反编译只继续闭合原版
原子/晶胞坐标和力的对偶映射及reference重置时机；不再扩展LS选项或替换
Safe-total。任何映射差异先证明其数值意义，再决定是否值得实现和真实体系对照。

原版climb-release单层后续审查未证明optimizer maxiter意味着有效逃逸，
因此不实现未收敛偏置点释放。具体边界见native-climb-release-followup.md。

TYPE4、XXXII 和 TiO2 的此前实验均已按其各自模型、预算和成功定义封存：
TYPE4 的掩码对照、XXXII 的复制表示修正和 TiO2 的 central-Ritz 对照都提供
实现或局部数值证据，但没有升级为跨体系搜索有效性或物理相稳定性结论。
完整实现状态与科学边界见 `ssw-family-current-status.md` 和
`2026-09-11-core-gap-audit.md`。

## 历史记录

修改前的长篇历史主线保存在 `mainline-before-cuo64.md`。

## 已有依据与保留决定

| 问题 | 当前证据 | 决定 |
|---|---|---|
| 工作局部优化器 | 同 31 个冻结 Cu13 偏置问题，Safe-total 31/31、隔离原指令 LBFGS 29/31；共同成功集原版略省调用 | 保留 Safe-total，不宣称普遍优胜，也不等同完整 BFGSDRIVER 比较 |
| 原版 87° 高度 | 两体系 132 个 stage 没触发增长；效应来自初始化高度。Cu13 出现一个额外正内部 Hessian 结构组，但总成本更高，能量高于初态 | 保留明确可选 profile；不默认替换，不把更大高度归功于动态角度调节 |
| 解析最小角度高度 | 无迭代增长参数；当前 Cu13 成本更低但覆盖更少，C4H6 无低于初态新结构 | 保留实验性简化策略，不能声称 dominates |
| 胞自由度 | 后验胞淬火、2014 启发的胞/原子分块、联合 log-strain 三个可执行不同算法；八组对照已完成 | 不再把 local cell-relax 冒称联合 VC，已通过所列有限胞资格；有限样本不支持效率排名 |
| RC 的 krot | 已恢复有限域转动继承；树形可达空间可由现有 torsion 重参数化覆盖 | 不因原版存在参数就新增参数；优化条件数收益需单独证据 |
| LS | 生命周期、强度响应与周期 image-resolved 扩展已实现；已有分子结果无一致优势 | 困难 C60 检验具体失败机制，保持普通 SSW 基线 |
| GA | TYPE0–4 算子/控制器及初始化已实现；部分真实 EMT 集成结果，复杂分子晶体效果未证实 | 不以 Java 身份推定优于 ASE-GA/USPEX，不把辅助 LJ 调用隐藏为免费 |

## 后续推进与停止条件

每次新增计算先写明初态、模型适用域、参数来源、种子、总预算、失败定义和
保留/删除判据，再冻结源代码与输入。用户已授权常规有界 CPU 资源安排；
旧的长 campaign 不因文档更新自动启动。当前有界 GPU 作业状态以上方当前执行任务及对应证据目录为准。

若闭合原版语义显示它只是相同可达空间的重参数化或与一致目标冲突，就结束
该条忠实移植路线并保留证据。若真实对照没有收益，不靠增加奖励/阈值或反复
使用同一测试集调参挽救。若失败集中于确定核心原因，先隔离该原因再最小修正。
多个预算受限样本仍不足以确定有效性时，明确“证据不足”，不伪造饱和结论。

全部单元测试只说明公式/接口；小力不是稳定相，有限胞正 Hessian 不是完整
声子稳定性，结构描述符不是动力学连通性。论文/模型/源码无法访问时保留
具体缺口；目前可执行任务不因这些局部缺口停止。
