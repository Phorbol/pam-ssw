# 当前主线：完整实现固定胞 SSW → LS-SSW → GA-SSW

2026-09-12，按用户要求重新收拢。此前执行记录保存在
2026-09-12-mainline-before-fixed-cell-reset.md，仅作历史证据，不再是任务队列。
本文件取代其中继续 VC/Fe7C3 诊断及采用 .2/.01 操作点的安排。

## 目标与边界

交付独立 Python/ASE 的完整固定胞 SSW：明确的算法状态、可替换的势能后端、
可复现的输入与成本记录，以及跨体系的盆地间探索证据。固定胞既包括非周期
团簇/分子，也包括固定晶格的周期原子体系。全局低能优化是主要效果指标；
有效不同盆地覆盖是单列指标，不能把两者、动力学或平衡采样混称。

用户最新明确顺序为 SSW → LS-SSW → GA-SSW，之后 VC-SSW → RC-SSW。
当前暂停 VC/RC 的新增实现与参数实验，保留全部已有产物。用户新增授权
QE/VASP/ABACUS cell-relax独立源码调研，见2026-09-12-dft-cell-relax-comparison.md；
这不改变固定胞实现优先级。
LS 与 GA 是当前阶段必须完成的交付，不作为可无限延期的可选增强；先闭合
共享 SSW 契约，再各自补齐本体，并同步做端到端验证。
Fe7C3 退出当前开发决策集，历史数值结论保留但不决定默认值。

## 最新用户要求：ASE约束一致性优先

FixAtoms/Hookean已接入显式约束SSW及paper/native LS，不以外层ASE投影替代
内层方向/HVP和目标函数处理。五条真实短流程完成，一条Cu/native LS预淬火
失败保留，共605搜索+11fresh；见2026-09-12-hookean-constraint-results.md。
后续公共入口选项对齐与GA约束索引语义仍未闭合，不宣称任意ASE约束支持。
VC调研发现QE显式metric、ABACUS cg2的1/N cell缩放及联合line search，
未来先检验规模一致的cell预条件，再考虑新增阈值/历史策略；不立即修改默认。

约束入口PAM Gaussian现已闭合：plain/paper/native LS可使用已有曲率高度/宽度
规则，默认6臂逐E/F ledger与改前一致。真实12臂1424搜索+22fresh，PAM未显示
本组收益，不改默认。见2026-09-12-constrained-pam-gaussian-results.md。
VC后续metric方案见2026-09-12-vc-metric-followup-protocol.md，尚未执行。

用户澄清：旧PAM的Safe-LBFGS/adaptive-bias参数未细致验证。“沿用PAM”仅是
实现来源，不是数值依据。当前优先复核周期LS光滑性及原版image语义、预淬火
停止规则，随后核验各后端FD步长稳定区间，再做优化器恢复与bias参数对照。
详见2026-09-12-parameter-basis-and-recovery-review.md。已有短流程结果不用于
宣称PAM参数劣于参考策略，亦不以调松阈值替代上述核查。

## 本轮原版证据更新

隔离原指令的9组边界测试确认 LS 正常退出条件为 force_measure < ftol 或
counter >= LSoptsoftmax；解析器缺省为0.1和50，实际输入可覆盖。不能由此推断
line_search_failed也可继续。见2026-09-12-native-ls-prequench-stop-audit.md。
论文/SI默认dimer间距0.005 Å，当前0.0001 Å不是论文默认；详见
2026-09-12-ssw-ls-primary-parameter-audit.md。

原版 LS helper 每次重选镜像，有直接距离快捷分支与有限27镜像搜索；
ASE exact MIC并非逐项相同。先确认真实失败几何上的区别，不把镜像求和当成
原版补丁。最小实现先分离预淬火精度/预算与真实起点、外层终点证书，保持旧
配置与必须收敛策略；预算出口和失败恢复须另有可观测的状态语义后再比较。

2026-09-12实现和受控检查：共享LSPrequenchSettings已接通固定胞入口，旧None
行为保留；VC对非None显式拒绝，避免静默忽略。完整standalone回归581 passed / 1 skipped；
随后VC边界拒绝与LS专项34 passed。水二聚体GFN2与Cu表面EMT四LS臂仅改软面
fmax .03→.1（steps300、外层.03不变），332搜索+8fresh，四臂完成且8证书合格；
对应旧四臂447搜索+7fresh并保留Cu/native失败。这是参数分离资格检查，
不提升为效率或通用参数结论。见ls-prequench-decoupled-multicase.md。

新的反编译纠正：mirror_min_dist比较在best_squared中减去0.001后再与候选比较，
并非裸最短距离/仅等距不替换。真实Cu request132/152 helper保留同镜像，
ASE选择切换。此容差是原版数值规则，不是光滑性证明；后续需评估它是否值得
独立复现，不能将其当作无参数修复或推断原版完整优化器不会失败。

## 路线选择

不等待逐指令复刻整套 LASP 才交付，不围绕旧 PAM 不断增加组件，也不先搭建
多 kernel 控制器。选择：一个完整固定胞流程，分别保留参考规则与 PAM 规则
的来源和差异，在共同输入、后端、预算及最终物理标准下判断值得吸收的设计。
已有 run_ssw 等流程可调用；工作重点是核实契约和补证实的缺口，不从头重写。

## 当前优先级补充（用户确认后）

固定胞SSW阶段停止和native-LS适用域已取得下述阶段结果。
停止继续扩展持久化与阶段停止调参；VC/RC仍后置。固定胞TYPE1 GA结果
契约已闭合并完成下述两臂验证，不展开变胞GA研究。

- SSW：核对 `climb_convg` 的最大力分量、首段/后续段计数以及双停止标志。
  先恢复纯条件，再检查公共迭代语义；不能因原版存在数值停止就把失败当作合格极小值。
- native-LS：保留现有 ASE 最小镜像模式（原版仅有限镜像搜索，不能称精确复现）；新增显式 periodic-images 独立扩展，
  用相同周期键集合做势能求和及 N/Nb 归一化，包含非零自镜像。
  不混用 MIC 计数与镜像求和，不引入新物理阈值；先验证原胞/超胞一致性，
  再做Cu/Al EMT端到端对照。原版自身MIC限制与Python实现错误分开报告。
- 周期镜像扩展已接入。Cu/Al完整晶体和空位晶体8臂完成：2880搜索+20复核；
  两个短胞MIC预淬火失败，镜像模式均完成两步；全部保存结构fresh合格。
  保留默认MIC和显式扩展，不作效率排名；见2026-09-12-native-ls-periodic-results.md。
  本次完整standalone回归511 passed / 1 skipped。
- native-LS 已接入既有约束SSW活动坐标流程，保留固定端点键并复用native
  cycle runtime；完整standalone回归513 passed / 1 skipped。Cu/Al(111)真实检查已完成，
  连续与续跑付费序列一致，总2356搜索+12复核，固定坐标/晶胞及活动力合格。
  Al两个落点MC拒绝保留；见2026-09-12-constrained-native-ls-results.md。
  SSW原版阶段计数已定位到RC dispatch，生产接入不能直译为Safe-total maxiter。
- 以下编号保存已完成证据链，并非要重新执行的实验队列。

- 阶段停止/全部释放公共实验接口完成，默认关闭；独立standalone回归520
  passed / 1 skipped，研究adapter8项通过。三体系×两Gaussian×两停止方式
  12臂实际9147搜索+36fresh，全部存储落点合格，无普适最佳能量改善。
  Cu31/PAM少用294请求但改变高能落点覆盖，不能独立声称效率提高。
  见2026-09-12-stage-control-e2e-results.md。不再围绕该例调参。
- 已闭合：原periodic GA默认VC结果契约要求stress/objective；直接传
  固定胞SSW既缺字段也错误要求stress收敛。复用已有TYPE1 proposals和
  三阶段控制，显式支持固定胞真实E/force证书与共同cell，保持VC默认。
  TYPE2 proposal按分子extent重新构造cell，不能作为纯接口适配强行冻结；
  固定胞模式明确拒绝，留待独立设计。TYPE4已有固定support流程，不重复实现。
  fixed_cell=True现已复用固定胞run_ssw及真实max_force证书，输入方向不变。
  Cu31实际交叉+两变异+fine，Al31近重复归为一个代表、如实无proposal；
  两臂521搜索+13fresh，全部观察力/cell合格。quick/offspring仅真实淬火、
  fine一步SSW，不能称长程性能验证。527 passed / 1 skipped，详见
  2026-09-12-fixed-periodic-ga-results.md。
  随后四臂quick/offspring/fine各1个SSW attempt验证完成：4912搜索+42fresh，
  每个实际walker有一个完整record；Cu31两seed实际子代/fine，Al近重复
  两seed保持no_proposal。全部观察fresh合格，不作长期效率结论。

## 当前执行队列（2026-09-12，依据已完成证据更新）

1. 优先闭合已定位的 Ritz 提前退出：直接残差认证失败后，在原预算内继续已有
   Krylov 迭代，不放宽阈值。三类两seed旧/新12臂已完成72114总请求：
   金属四组调用账本完全相同，分子旋转失败3→0，最佳能量不变。
   已纳入公共实现；LS公共入口六臂36042请求、GA两seed24012请求复核均完成；
   GA实际生成子代，fine受预算截断，全部保存归档fresh合格。
   详见2026-09-12-verified-ritz-stopping.md。
   固定胞独立氧化物验证：TiO2 两种初态的固定/分阶段 Ritz 对照已经完成，
   两个高能落点增加了正曲率证据；CuO64 同协议 CPU 四臂已结束于墙钟预算，分别报告
   预算截断、完整落点、搜索成本与复核成本。无需回到 VC 或为单例改参数。
2. GA 单盆地启动证据现已补齐。此前五个 C4H6 已知异构体共同初始化只证明了已有
   种群后的生命周期，不能证明从一个盆地自然建立种群。使用单一 bicyclobutane
   初态，由 quick SSW 产生父代；描述符参考保持固定但不注入 archive。
   两个种子均建立父代并实际生成四个交叉子代；其中一条子代阶段进一步降低
   最佳能量。已补相同搜索请求数的纯 SSW 对照：seed11 的 GA 最佳能量低
   0.1433 eV，seed29 无实质进一步改善；两seed不足以证明普适效率优势。
   保留预算截断、失败和碎片记录。
3. 已完成固定初态 trans-butadiene 的 SSW/paper-LS/native-LS 两seed对照，
   每臂40000搜索请求/120秒封顶。依据LS原文§3.1，0.7 eV/atom用于反应空间
   探索，分别统计fresh合格连接类别、碎片组成和能量；GFN2不是论文PBE，
   不能据此声称论文数值复现。共131177搜索+129复核；同请求前缀连接类别
   为4/5/6与4/4/4，最佳能量无实质差别。两控制器实际响应不同，不能用
   相同target冒充相同软化程度。保留此前全局优化负结果，不重新解释为优势。
   原版ASE BasinHopping + Safe-total三类两seed对照已结束：30536搜索+388复核；
   四个金属臂请求封顶，两个分子臂实际SCF失败。所有已保存落点fresh合格，
   同成本前缀最佳能量无实质优势。更多淬火不等于更多不同盆地，保留失败。
4. 维持已验证的共享接口：显式 staged rotation、PAM Gaussian adapter、LS 软面
   工作几何及 GA 逐子代短 SSW 已接通。跨体系结果未支持修改通用默认值。
   几何改变、局部极小值覆盖与最佳能量改善分开评价。
5. Broyden 历史递推、谱超限删除/最小历史重启、旋转调用 flags 和 CBD.fact
   来源已取得原指令及独立 Python 重放证据。研究版 ASE 方向求解器已完成：
   Cartesian 度量与原版退化块缩并度量分开，直接残差停止仍属独立替代。
   三类两seed固定几何24臂1388请求、完整SSW18臂22022请求均已结束；
   完整比较52个保存极小值fresh合格，Ritz/Cartesian/native成本6841/6879/8250，
   未证明Broyden性能优势，原版metric在一个分子seed未找到能量下降。
   保留Ritz默认，不为原版metric补调参。公共Cartesian-Broyden入口已完成，
   六组SSW公共重放与研究实现账本逐字节相同，6879搜索+17复核请求。
   公共LS六臂36000+42请求复核完成；GA两seed24000+16请求复核完成，
   两者均有明确预算边界。GA实际产生子代，generation-short耗尽预算、未到fine，
   每个归档各有一个碎片结构，不能将force合格当作完整分子资格。
   不以反编译数量替代算法价值。
   详见2026-09-12-broyden-direction-reconstruction.md、
   2026-09-12-broyden-full-update-recovery.md。完整native ftol/CBD_PreRot
   特例与独立公共求解器的收敛语义仍分别标识，不宣称逐指令全程序复现。

6. 用户批准转入外层断点续跑和公共流程统一：先SSW/paper-LS/native-LS
   完成attempt边界，再GA完整quick/generation/cycle边界。保存RNG、LS状态、
   种群/阶段与累计成本，不重复初始化，不把预算耗尽当作可恢复暂停。
   见2026-09-12-outer-resume-plan.md；后续才整合约束/表面与结构归档。

   外层续跑已实现；截至 2026-09-12，`pytest -q tests/standalone` 为 507
   passed / 1 skipped。Cu13 SSW
   和Cu13 GA的quick/generation边界重放逐调用一致；Cu31更换EMT实例后
   有微小浮点差，GFN2三臂在恢复首call相同几何上已有力差，随后轨迹分开。
   五臂所有保存落点fresh合格，但不宣称任意calculator电子状态或轨迹精确重现。
   详见2026-09-12-outer-resume-evidence.md及2026-09-12-ga-boundary-checkpoint.md。
   约束体系公共Ritz/dimer/Broyden/staged方向适配器已完成，保留active
   Cartesian chart、原约束quench与默认旧路径。Cu/Al(111)八臂两步诊断
   共3360搜索+24独立复核请求；24落点活动力合格，固定基底/晶胞不变。
   四个Al臂各拒绝第二落点，拒绝结构仍保留；单seed短轨迹不支持效率排名。
   SSW/两种LS现有可选结构identity view，保存全部观察、由调用者提供
   几何比较器；Cu13离线副本映射[0,0,2,2]，并非通用身份算法。
   约束入口attempt边界续跑已实现：Cu连续443=70+373逐调用一致，
   Al连续401=111+290但非精确重放；12次fresh检查合格，总成本1700请求。
   paper-LS运行态已有接口级续跑检查；native-LS及GA约束接口仍待闭合。
   详见2026-09-12-constrained-resume-audit.md。


Cu13 两项静态轨迹回归已定位到依赖栈差异；改为同环境下公共路径与归档研究
包装器的严格对照，保留历史坐标。GA偏置策略前置校验已复用共享helper；原生高度的非空历史路径有实际回归覆盖。
历史Ritz修正时回归为465 passed / 1 skipped；当前完整回归见上述507/1。
这只是实现验证，真实体系结果与科学边界见进展文档。

## 验收要求（保留完整主线约束）

1. 固定胞行为契约审计：从实际 public entry point 跟踪初态淬火、随机方向、
   刚体零模处理、soft rotation、Gaussian 保存/更新、短松弛、阶段停止、
   whole-walk release、真实势淬火、MC 与状态恢复。逐项列出代码入口、
   native/论文证据、独立替代和影响行为的未决问题；旧报告不得当作当前代码。
   特别核查 paper_reference 与 atomic_climb 是否存在有意或意外语义差异，
   不假设 block-only 的 PAM Gaussian 适配已经覆盖固定胞公共接口。
2. 只闭合决定上述流程的缺口。反编译每次回答一个会影响实现选择的具体问题；
   native 数值怪癖不自动移植。已明确数学契约且有成熟替代的内部求解器不阻塞
   基线交付；CBD 优于标准 dimer、Safe-total 优于成熟优化器均是待检验假设。
3. 整理并冻结最小跨体系开发集与独立评估集。按物理困难选择，而非按哪个case
   更好跑：金属团簇/EMT、强共价分子或碳团簇、固定胞周期材料。真实输入与
   calculator 的适用性先核验；MACE-OMAT 不因能运行就被视为通用分子势。
   复用已有 C4H6/C60/氧化物与上传案例清单，最终名单以输入和后端核验为准。
   LJ38 可补充多盆地模型证据，不能替代真实体系。未读取案例不宣称已可运行。
4. 冻结版本做完整固定胞轨迹：相同初态组/种子组、相同 oracle 和总预算；
   比较参考 SSW、既有 PAM，以及必要的成熟简单搜索基线。先厘清全流程差异，
   再仅对有必要的方向/高度/宽度/停止模块消融；不做参数×优化器×体系全笛卡尔积。
   优化器比较固定 proposal 规则，proposal 比较固定优化器；history不同时混调。
5. 完整 LS-SSW：核查软化 pair 集合、固定/更新时机、软势预淬火、方向与偏置
   所用势面、依据真实能量响应更新强度、丢弃软化后的真实势淬火及选择。
   已有实现先审计复用；区分论文/native规则与保守独立替代。共同 SSW 对照
   验证完整轨迹与强键体系行为，效果不佳也如实交付，不追加补偿性启发式。
6. 完整固定胞 GA-SSW：初代生成与合法性、短 SSW、父代选择、交叉/变异、
   子代真实势淬火、结构去重与竞争、历史归档/分区、长 SSW 与循环、预算和
   随机状态续跑。核查已有实现哪些仅有可调用函数、哪些已由公开入口串通；
   Java/native仅作恢复规则的证据，Python运行不得依赖其二进制或LASP pot。
   先闭合明确支持的固定胞体系流程，不借本阶段扩展变胞GA或RC分子流形。
   GA与普通SSW用相同总oracle预算比较；不预设原Java优于ASE GA。
7. SSW/LS-SSW/GA-SSW 的缺口表、完整ASE用例、跨体系结果与实现边界全部交付
   后，另立 VC-SSW，再 RC-SSW 任务。完整实现不等于逐指令复刻，也不等于
   已证明所有变体在所有体系上优越；这三个完成标准必须分别报告。

## 判断与停止规则

记录有效不同盆地、低能发现与目标达到成本；所有初始化、旋转、失败和最终检查
计入预算。新势能后端、复杂约束、独立资格诊断成本另列，不隐藏为零。
未知全局极小值的体系不伪造 GM 命中率；高能 MC 拒绝不是单次算法失效证明。
固定胞最终检查力和物理结构，stress仅作诊断，不能要求禁止松弛的晶格达到零应力。

单个case失败先区分接口/公式错误、预算截断、后端适用域、轨迹随机性与真实搜索
不足。可证明的一致性错误立即修；仅单case表现差则保留失败并转向其他体系检查
是否共有，不能接连围绕它增加阈值或分支，也不能通过剔除难例美化结果。
最终评估集不用于调参；发生开发反馈就重新标明其开发集身份。

阶段交付以可用固定胞流程、闭合行为契约、跨体系冻结对照和清楚失败边界衡量，
不以反编译函数数、变体名数、测试数或单点淬火成功衡量。当前不启动新GPU任务。

2026-09-12执行进展：见2026-09-12-fixed-cell-progress.md。LS高度接口与GA独立
结构身份已接通，跨五类输入18条SSW/LS和两金属4条完整GA已有真实CPU结果。
有效力证书与有效分子/不同basin分开报告；不以GA archive=8声称多样性。
GA总请求预算与fine回流多轮闭环已完成；默认单轮四条真实轨迹逐调用不变。
该阶段曾暂缓持久化；现已完成SSW/LS/GA外层边界持久化，
后续以顶部执行队列为准，保持本体顺序，不回到VC或Fe调参。

2026-09-12进一步闭环：五个G2 C4H6异构体共同种群的六臂GA/GA+paper-LS/
GA+native-LS均完成实际交叉子代和fine；解离与未复核归档分别报告。mixed TYPE0
零整数配额已提前拒绝，不用新增概率/补偿算子修补。公共ASE线搜索/SciPy基线
已接通并保持默认，跨金属/周期缺陷/分子/非Ih C60有界对照已完成；C60两臂SCF失败、一臂时间截断，保留失败，不继续围绕该例调参。


2026-09-12新的核心证据：LJ38全对势的两seed×两高度规则相同12000调用预算，
均生成力合格而高能的落点、全部MC拒绝。87°高度条件未解决该点的搜索效率；
不继续围绕该case调高度。隔离原指令已定位CBD_PreRot且curv>-1e-6时rotation weight=curv的
赋值，详见native-rotation-bias-field-trace.md。前置旋转、动态强度与anchor保存的公共耦合接口及逐offspring短SSW取best分支
现已接通；固定100和分阶段方案均是明确的开发选项，不冒充完整native。
不扩展新的archive策略；后续以顶部当前执行队列为准。

该续跑阶段的根智能体独立回归：`/tmp/pam-root-constrained-checkpoint-final-20260912.log`，
指定 mace_env、PYTHONNOUSERSITE=1、PYTHONPATH=.：507 passed / 1 skipped。
唯一跳过项为需显式设置 PAMSSW_NATIVE_MC_ELF 的原程序数值片段测试。
