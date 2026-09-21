# 当前主线：先闭合 SSW/VC 核心，再验证收益

2026-09-11，按用户“自行评估优先级并自行推进”更新。
此前主线全文保存在 `2026-09-11-mainline-before-priority-reset.md`，其中旧任务、
阶段成本和“尚未实现”文字仅供追溯，不再作为执行队列。

## 判断依据

目标是独立 Python/ASE SSW 家族，不依赖 LASP/Java 执行搜索。
优先级按依赖关系、已知实现偏差、现有数据能否判别、补证成本确定，不按反编译
函数数量或变体数量衡量进度。没有依据证明所有 native 细节都能提高算法效果。
修复一致性问题优先；研究性的性能机制必须做独立、同预算真实体系比较。

最新证据改变了问题定义：Fe7C3 的25步原子预松弛虽未收敛，仍可得到最终
真实力/应力合格的候选；原子 climbing 完成9/8个合格Gaussian后被预算截断。
联合VC放行正常数值停止也能得到落点，但SciPy三个落点重复初态，Safe一个
不同点高能且被拒绝。因此不能再把这些现象统称为“偏置淬火不收敛”，更不能
只通过增加history、放宽stress或放行更多落点来宣称改善全局搜索。

## P0：SSW释放生命周期与VC自由度契约（立即并行）

A. 原子 climbing 的停止和释放：
- 从现有两条block轨迹提取每个completed Gaussian的真实能量、参考能量、
  方向变化、偏置强度、旋转与局部优化成本，先做零PES诊断。
- 核查“低于外层初始能量”的终止条件在cell预扰动后的具体作用；明确这是
  一种停止规则，不是盆地身份判据。不能未经证据改成相邻步能量下降即停止。
- 定向追 native `climb_convg` 到实际释放结构的选择，区分原子/晶胞模式、
  正常数值停止、Gaussian上限及请求预算截断。
- 完成标准：状态—坐标—能量/力—真实quench调用的可核查映射，以及失败时
  accepted与pending点的界限。只修证实的问题；不增加任意提前释放阈值。

B. VC允许自由度与方向状态：
- 最新零PES审计把位移尺度提为首项：0.15晶格Frobenius位移在Fe7C3实际形成
  单步约31–40%的最大主伸长变化；seed101进入atomic前体积约为初始54.2%，
  高能/高应力已经产生。先追native ds_cell归一化和cell mode质量，再讨论
  atomic优化器。详见 `vc-cell-deformation-priority.md`，不据此直接调小参数。
- 在已解析CSSW表基础上追 gen_randommode/update_mode0 的调用条件，回答
  方向何时重抽、何时续接；不能再次停在“表项未知”的旧结论。
- `fixcell_climb`尾9个cell-force分量归零已由原指令验证；继续核查BFGS
  历史初始化和实际位移，判断这是否足以严格固定晶胞。
- 25步来自2014论文算例，不是已恢复的binary默认；原子/晶胞外层频率已恢复，
  不能把该频率直接等同于所有inner relax的自由度选择。
- 完成标准：每种阶段允许的坐标/力投影/参考点和状态生命周期明确；能闭合的
  native分支有原指令对照，不能闭合的独立替代有清楚数学定义和证据边界。

## P1：CBD旋转核心的剩余恢复及可替换实现

重点是BRZERO4历史矩阵、历史裁剪、旋转步输出与约束投影，并明确输入输出力
属于何种势面。现有dimer/Ritz保留作成熟对照，不能预设CBD必胜。
已证实的非欧氏退化内积、Gaussian力重复累加等局部行为不得直接变成生产默认。
完成标准：可独立调用的旋转模块、完整调用成本、原指令算术对照和真实PES上
的匹配起点比较；随后用完整SSW验证，而不是仅比较特征值残差。
若连续有界切片不再增加可用证据，记录阻点，继续独立实现/真实测试，不等待
恢复整份Fortran源码才推进。

## P2：固定实现后的真实体系与优化器比较

复用现有Cu/EMT、C4H6或C60、氧化物和复杂周期体系。先用少量相关体系检验
P0/P1具体改动，再冻结多seed最终比较；不机械重跑所有变体。
Safe-total为工作后端，SciPy L-BFGS-B和ASE LBFGSLineSearch为对照，native
LBFGS/MCSRCH/MCSTEP目前是隔离研究内核。是否需要完整Python移植由缺失能力
和实测收益决定，不优先逐指令移植整套优化器。暂不继续扫history/阈值。
指标为不同且物理有效的候选、低能发现、达到目标成本；包含初始化、失败、
旋转、quench、独立资格检查成本。不得把同一结构的廉价回访当成更好覆盖。
预算内无法完成与科学失败分别报告。诊断性追加quench不得并回原预算。

## P3：LS、RC及GA的剩余本体与集成

SSW/VC已确认的核心修复同步传递给各变体；新增研究按LS→RC/RC-VC→GA次序。
LS先解决势面/参考对/响应生命周期一致性；RC优先真实用例暴露的约束缺口；
GA最后比较DCCD、归档调度与ASE-GA等替代。既有可运行实现保留，不声称完整
native等价，也不继续扩展多kernel调度器、奖励函数或更多启发式操作。

## 本轮已完成与执行分工

- 独立joint VC正常数值停止释放选项已实现，strict保留；SciPy异常退出单列。
- Safe-total原始telemetry已透传到固定胞quench、atomic和paper参考事件，
  无新增停止策略；最后相关集成51项通过。
- Fe释放实验8493EFS、block实验3998EFS均完成；Fe累计60940EFS。
  科学结论见 `fe7c3-vc-numerical-release.md`、`fe7c3-block-baseline.md`。
- Luna A：P0 VC位移/方向规范化反编译，以及block跨cycle论文契约核查。
- Luna B：P0四个已存Fe7C3构型的cell Hessian诊断脚本；主agent审核并运行。
- 主agent：核查证据、闭合释放条件、决定最小修复与实验，审查所有子任务结果。

目前不需要新增论文或用户决策即可推进P0。未承诺本轮一定需要新增长时间计算。
用户已有授权范围内的有界工作继续；发布、清理或其他未授权事项单独处理。

本轮P0首轮产物：
- `vc-cell-deformation-priority.md`：20个真实cell位移的有限形变审计。
- `research/ga_ssw/block-cell-energy-diagnosis.md`：逐request精确对齐的
  能量/应力/结构连续性，两个seed共17个completed atomic边界核验。
- `native-release-snapshot-followup.md`：固定胞normal release在optimizer
  callback前保存并在Allopt前恢复坐标/力/能量快照，未扩推到全部失败分支。
- `native-cell-direction-lifecycle-followup.md`：真实CSSW表和post-climb分支
  已明确，引入新随机量已知；与旧方向混合关系仍待闭合，不能声称完全重抽。
以上分析零新增PES请求、零GPU；无未经验证的算法默认值修改。

随后有界补证：`fe7c3-block-cell-hessian`诊断1269206已完成，单V100、
81秒、148EFS，Fe累计61088。四点方向软三模权重99.5/96.8/97.9/75.3%，
两个差分步长一致；目前不支持把方向求解误差列为主要瓶颈。详见
`fe7c3-cell-hessian-diagnosis.md`，这不是新的搜索成功或全体系稳定性证据。
`native-vc-moveds-dscell-followup.md`确认scalar ds_cell路径，但上游cell
方向度量仍未闭合；两次有界切片后暂不继续扩展同一链。
`block-cbd-paper-crosscycle-review.md`确认每cycle当前晶格范数与论文一致；
跨cycle随机anchor是否继承缺乏明确论文/native证据，保持显式独立策略。

P0下一对照：已新增实验`cell_step_metric=deformation_rms`，无新增数值参数，
立方晶胞上与旧规则等价；默认`lattice_frobenius`不变。相关15项检查通过。
作业1269321对两规则各seed7/101进行同时期完整block搜索，每arm2000EFS、
两步含独立证书，总上限8000。现已完成7996EFS、234秒，Fe累计69084；四arm均只有一个高能且MC拒绝的
新合格落点，combined步均耗尽预算，无best-energy改善。详见
`fe7c3-deformation-rms-results.md`。不推广或继续调节新度量，转回释放参考状态。
此次是既有体系/seed上的诊断pilot，不能作为独立泛化验证；不增加history或
改动stress/25步/rotation预算。结果不支持时不推广新度量。


释放参考状态复核：`block-atomic-reference-energy-review.md`区分候选快照与
reference字段。当前`E_candidate < H_old-pV_work`严格等价于候选焓小于旧焓，
没有压力项代数错误。论文未明确cell→atomic接口的anchor；VC reference写入
链仍未闭合，不把fixed-cell tene0快照冒充参考能量，也不据此放宽释放条件。
后续只追真正energy0/初始状态的生命周期；无需再写一个只证明已知传参的toy测试。


最新定向反编译已定位VC具体consumer，且本轮纠正漏跟栈变量造成的误判：
`energy0`既用于能量增量，也参与`tene0 < energy0-0.1`的all-stop条件；
trajectory记录减1.0比较首先影响stage-stop。48个隔离原指令case已通过，
零PES。详见`native-vc-climb-convg-reference-followup.md`。因此不能再据此
怀疑Python使用外层参考这一点本身错误，更不能改成相邻能量下降直接Allopt。
当前最小缺口转为biased stage的停止与next-record保存、能量势面语义；
all-stop与stage-stop的状态层级需要在任何移植中保留。


本轮追加零PES诊断：四arm的36个completed Gaussian边界中，14个相对本阶段
中心真能量下降超过1eV，而0个低于外层初态0.1eV。完整统计在
`fe7c3-block-deformation-rms/energy-trigger-summary.json`。这只是Python真实
轨迹上的两个能量不等式，未重放native全部mask/能量势面，不估计节省请求数。
两种停止层级确实值得区分；没有证据把阶段级下降直接升级为整段释放。

下一具体工作是闭合stage-stop允许在何种力/优化状态下保存下一条记录，尤其
原版crystal_opt前的tene0/work快照与优化后trial的区别。普通CSSW实际call-site本轮已纠正：优化调用5f2642走+0x108=crystal_opt，
不是邻槽+0x110=noncrystal_opt；+0xe8=scart2cart不重新计算真实能量。
详见`native-vc-optimizer-handoff.md`。
不再为排查reference问题追加大预算搜索，不改变25步/history/阈值。


2026-09-11阶段预算闭合：`climb_convg`按trajectory索引N==1选择
ngaus_relax_ini，否则ngaus_relax；climbstep严格大于预算才触发。
广义sfa最大绝对分量严格小于climb_stopf也可结束阶段，不等于Python的
每原子力向量范数。108个隔离原指令检查通过，零PES；具体默认预算及
完整reverse-communication调用计数仍不等同于Safe-total已接受步数。

Python差异已定位：block的25步仅用于cell cycle后无偏置原子松弛；
combined atomic Gaussian实际使用relax_steps（当前Fe配置300），并强制
偏置力收敛才生成completed boundary。下一有界实现为显式bias_stage_steps：
独立限定偏置阶段Safe-total已接受步数，允许正常maxiter交接已接受点；
保留未收敛标记、真实能量评估和最终联合淬火证书。None保持现有行为。
此为native阶段控制思想的独立适配，不是原计数/阈值逐步复刻；不移植
未经完整势面语义核验的0.1/1.0能量条件，不改最终力/应力、history或25步。


阶段预算对照已完成（job1270433，1V100，213s，6609EFS，Fe累计75693）。
strict两seed的combined均在1997搜索请求处截断；显式stage25分别在1034/1571
搜索请求内完成两个outer提案，额外各3请求独立复核。两combined落点均通过
原fmax/stress标准且三组结构匹配均为新结构，但高于初态17.32/17.58eV，
MC全部拒绝，无best-energy改善。详见`fe7c3-stage-budget-results.md`。
因此保留实验接口，不推广默认；本组算例的主要卡点已从biased阶段完成转为
高能提案。下一优先级是native首/后阶段调度及偏置高度/力、参考态语义对照，
再做预先登记的跨体系低能发现测试；不继续盲调history、stress或cell步长。


用户纠偏后重新聚焦PAM自适应Gaussian与数值停止的混杂因素。已恢复旧Git报告：
固定height、固定macro target及whole-walk首次下降早停不是同一机制。
旧proposal.05/80与当前Fe.001/300也不能混为同一数值基线。
下一对照按`pam-gaussian-controlled-plan.md`固定全部停止/淬火设置，只比较
forward-force高度、PAM曲率高度、PAM曲率高度+宽度；不把core-only无反馈
适配叫完整PAM。后续再解耦并交叉检验inner maxiter/fmax，保留统一最终证书。

PAM偏置×内层阈值的12-arm受控pilot已完成：job1270644/1270704，各1V100，
合计23556EFS，Fe累计99249。innerfmax.001时三策略combined均0/2；.02时
前向力高度0/2、PAM高度1/2、PAM高度+宽度1/2。新增落点+13.80/+14.13eV，
全部MC拒绝，无best-energy改善。C/seed101已完成14Gaussian但最终淬火被总
EFS上限截断（155接受步<300），不能错归因为外层maxiter失败。
新增bias_fmax已与真实淬火fmax解耦；36项相关检查通过。PAM height seed101
全部撞10eV上限，估计中心曲率仍为正；不因此调高cap。详见
`fe7c3-pam-gaussian-crossed-results.md`。用户2026-09-12进一步指定内层.05–.2、外层.01–.05 eV/Å，
因此优先执行四个端点组合×两种子（job1276906，单V100，上限16000EFS）。
固定PAM height_width core、maxiter300、stress .0001，新增partial_atom_fmax
覆盖只用于维持cell-interleave .001，避免外层阈值牵动第三变量。
结果前不推广新默认；宽松证书与原严格证书分别报告。实验位于
research/ga_ssw/fe7c3-user-tolerance-grid。随后再评估跨步反馈与独立真实体系验证。

用户阈值8-arm已完成（1276906，单V100，382s，11831EFS；累计111080）。
inner .05/.2 × outer .01/.05 的combined有效数分别2/2、1/2、2/2、2/2，
全部高能MC拒绝。固定outer .01，inner .2较.05成本降低17–32%，但GPU重复
公共前缀有数值分叉，不能把单个basin差异完全归因于阈值。下一诊断暂用
.2/.01，不改通用默认；主线转回偏置/方向/释放语义与精确阶段输入复播。
详见fe7c3-user-tolerance-results.md，不继续扩大阈值扫描。
