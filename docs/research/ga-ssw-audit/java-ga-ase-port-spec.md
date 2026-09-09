> 原始报告位于 `/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/analysis/java-ga-ase-port-spec.md`；下文的原资料相对路径按该外部研究目录解析。

# Java GA 外层的 ASE 移植规格（来源审计，不是已完成实现）

2026-09-09；范围为上传版本 `sgn.jar` 的 CFR 反编译代码、原论文、当前独立工作树的行为重建模块。不运行计算，不修改共享实现。Java 外层与 SSW 内核职责分开：本文件定义候选生成、population/archive、分区和调度；Gaussian bias、局部软化/爬坡、淬火与 Metropolis 由 `analysis/ssw-kernel-comparison.md` 定义，不能把当前 PAM walker 当原版 LASP 内核替身。

## 交付范围判断

当前代码位置是 `/home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity/research/ga_ssw/behavior.py`（不在根 checkout 的同名路径）。已实现的是非周期 NNA 描述符、相似度、投影去重与 archive merge、能窗、非退化 Compete 累计权重。**交叉、变异、初始化、KMeans、完整 quick/fine 状态机、SSW 运行接口均尚未移植。**现存 Java 外部运行证据也不等于 Python GA 已移植。

| 组件 | 实际来源行为 | ASE 复用/新增 | 与论文或通用性的边界 |
|---|---|---|---|
| NNA/投影/归档 | 三层描述符，count-only 稳定排序；投影每维差值不超过阈值；同桶留较低能结构 | 可复用 `behavior.py`，包装 `Atoms`；冻结元素对表、reference 顺序和权重 | 不是论文声称的通用置换不变 DCCD；componentwise 阈值不同于论文 Euclidean 距离；非周期分支而已 |
| Compete | energy-span 自适应累计权重，底数 2.7183 | 可复用有限、n>2、span>0 分支 | 当前 Python 主动拒绝退化输入，原 Java 未保护；不能称全域 byte-for-byte parity |
| TYPE0 交叉 | flatten 各簇后构建 gamete pool；平面近等分、组分匹配 | 须独立实现 | 不是直接选两结构各切一次；候选配额不是论文固定 1/3 |
| TYPE0 变异 | best region 优先，原子扰动、元素置换、低配位原子内部重插 | 须独立实现 | 固定 Å 尺度与计数是经验来源值，不是通用最优参数 |
| TYPE3 交叉/变异 | 分子单元切割、重组、旋转、可变单元重构、几何 docking | 须独立实现；附独立 monomer membership | 原程序每阶段无条件调用水专用后处理，不能作为任意分子簇实现 |
| KMeans/父代截断 | 前三个投影坐标；随机初始中心；每簇最多20个成员 | 须独立实现精确规则 | 活跃流程不是论文 grid partition；不宜直接换 sklearn 默认 KMeans++ |
| fine region score | 每个已截断簇的 `0.5 Emin + 0.3 Emean - 0.2 Evar`，升序 | 可独立纯函数，但目前尚无 | 论文谈 grid 内全部 minima；源码统计抽样后的最多20个，且 E 与 E² 混合，需保持原单位作兼容值 |
| quick/fine scheduler | 明确倍率、carry terminal、不同批次数与 floor 运算 | 须独立状态机 | 不能简化为每代 GA 后固定长度 SSW，否则不是上传版复现 |
| 局部优化/SSW | `SSWOpt` 本身运行 OPTSSWStep 个 SSW steps，取每个 all.arc 的最低能结果 | 内核接口必须提供 minima 序列、terminal 和数值状态 | 常规 optimizer.minimize 不等价；原版 normal exit 不保证 ftol |

## TYPE0：实际交叉和变异

来源：`decompiled/sgn/ga_Interface/TYPE0.java:24,39`；`ga_cluster_cell/{Cross,Cut,CutBasicAbstract,Mutate,Compete}.java`。

设 G=MinGA、R=region 数，所有 `/` 为 Java 整数除法：每个 batch 对所有父代 flatten 后生成 floor(G/4) 个 crossover；对最低能 region 调用 mutate(floor(G/2))，其他每个 region 调用 mutate(floor(G/8))。根据第一结构元素种类选择 pure/doping。`getFinalGAStructure` 整批追加直到数量 >= G；空批次提前结束；**不截断为 G**。

交叉父代抽样使用 Compete。`Cross.getParentModelIndex()` 产生 n_parent×2 索引，但 `genePool` **只用第0列**；每个选中父代作 `10^(n_element+1)` 次 Cut，将正/负 gamete 放入全局两池；joint 各池均匀抽一个，不必来自不同父代，直到总原子数和每种元素计数匹配。匹配循环无上限。二元体系每个 parent 抽样槽就是1000次切割，属于应计总成本的实质操作。

Cut 先按几何中心（不是质心）平移原父代坐标；旋转角取 `atan(tan(3.14159*(U-.5)))`，按原矩阵及行向量约定依次绕 x/y/z，不能替换成均匀 SO(3) 就声称同分布。取 `pl=2*(U-.5)`，以 `z+x/pl` 分正负，重复直到两半原子数差<2；转动切面至统一方向后两半 z 分别加/减0.3 Å。Cut 中心化存在修改输入 Model 的副作用；ASE实现应输入copy并记录这一坐标等价性选择，避免破坏 archive 结构与 calculator cache。

`Mutate.disturbance(n,range)` 以有放回抽样选 n 次原子，每次各坐标加 `range*(U-.5)`，不是移动 n 个不同原子；range 是全宽而不是最大位移。

| pure mutate(n) 子项 | 数量 | 父代/操作 |
|---|---:|---|
| 小扰动 | floor(n/4)+1 | 簇内最低能结构，选 floor(N/10) 次，range=.3 Å |
| 中扰动 | floor(n/4)+1 | 最低能结构，选 floor(N/2) 次，range=.5 Å |
| 随机父代较大扰动 | floor(n/2) | 均匀父代，选 floor(N/2) 次，range=.7 Å |
| 随机父代稀疏扰动 | floor(n/2) | 均匀父代，选 floor(N/10) 次，range=.7 Å |
| 内部重插 | floor(n/4)+1，另加1 | 均匀父代重插5原子；另对最低能重插10原子 |

因此 pure 返回 `3*floor(n/4)+2*floor(n/2)+4`，连 n=0 仍返回4个。doping 返回 `3*floor(n/4)+5*floor(n/8)`：第一项是10N次随机原子对的元素标签交换；剩余五项分别对应前述四种扰动及随机父代重插5原子。小G会导致部分分支空，但不是按论文比例补足。

`interMu` 用3.2 Å内邻居数升序找低配位原子，删前5或10个后重新插入，保留元素；每次重心归零，用剩余簇尺度R抽半径 `(0.1+0.5U)*R` 的球内点，以collisionDetection(...,.3)拒绝；超过10000尝试插在原点。N<重插数会越界；没有必要把这类失败静默变成新物理策略。

## TYPE3：分子单元语义和水专用边界

来源：`ga_Interface/TYPE3.java:32,47`，`ga_molecular_crystal/{CrossMC,CutMC,MC_Base}.java`，`ga_monomer/{MutateMonomer,MonomerBase}.java`。

读取 seg 定义的 monomer atom indices 和 changeTypes，并写入父代 Model。所有 region flatten；每批 floor(G/4) 个 CrossMC；令 M=G-floor(G/4)，三种变异各 floor(M/4)，并不是三项平分剩余 M。过滤后按整批补到>=G。默认最小键长表是 `(元素半径和)*0.3`，自定义 BLLimit 完全覆盖该表；必须提取原表，不自行换 covalent radii。

CrossMC 的父代抽样槽数为10*n_parent，每槽10次 CutMC，共100*n_parent 次切割，仍只使用抽样对第0列。CutMC 把 monomer 几何中心暂当带序号的虚拟原子做 Cut；gamete 持有完整单元及原单元编号。joint 要求合并的编号排序恰为0..n_monomer-1，再以 `MC_Base.fitBinding(1.5, ..., accuracy=5)`进行几何 docking，最后恢复单元编号顺序和生成cell。不能按单个原子切开水再以通用ASE crossover替代。

需要保留的代码疑点：CutMC用于切分的是经过随机旋转的虚拟中心，但restoreToMonomer只对原完整原子组施加切面 y rotation；初始随机旋转没有以同一变换显式传入，须原jar几何fixture确认，不能凭常规算法直觉修正。

三种变异：(1) 均匀随机父代，把所有单元各自中心化、随机旋转后以 `fitMonomerCombination(1.5, ...,10)`重组；(2) 对changeType=1的单元跨父代独立构建原子GA（2n/3交叉，其余doping mutate），changeType=0单元取父代列表第一项，重新组合；(3) 从最低能父代中随机挑一个非单原子单元旋转。若所有单元均单原子，第三项的while不终止。第二项固定单元选取受flatten顺序影响。

**TYPE3目前实际上有水专用后处理。**主程序每次原生优化/搜索后依次 `handleClusterTem`、`TemH2O.handleH2OCluster`、`checkMonComplete(...,3.0)`。TemH2O对每个O找最近两个H，仅接受两条OH都<=1.1 Å，按OHH重建列表；没有跨O的H唯一性指派保证，并可能丢掉其他元素。随后按seg检查3 Å连通。这不是通用分子分组，不能偷偷推广成水以外的TYPE3。原程序wrap/中心移动也要与真实非周期Atoms语义区分。

与LS/Rigid审计交叉核对：TYPE3水模板调用固定胞`explore_type ssw`，Java的分子整体变换不表示native rigidssw。仅TYPE2复制rigidbody/blist等文件；原生rigid-chain允许单元共享旋转键原子，因此本文件的TYPE3 monomer映射不能直接充当通用RC拓扑。RC应另有atom-index topology、rigidbody/blist、周期/晶胞自由度与阶段预算，参见`analysis/ls-rigid-comparison.md`。

## 父代权重、KMeans与region score

`Compete.java`：设 span=Emax-Emin，n=父代数，

`T = -span*1000*2625 / (ln(2/n)*8.314)`；`w_i=2.7183^(-(Ei-Emin)*1000*2625/(8.314*T))`。

非退化时指数化简为 `(Ei-Emin)/span * ln(2/n)`，所以这是依赖population跨度/大小的相对权重，并非输入SSWTemper的热力学Boltzmann分布。两个单位换算常数在正常分支相消；n=1、2或span=0仍必须单独记录原异常语义。现Python函数还缺按累计权重抽索引的统一随机数接口。

`SSWGaSupport.java:529`：n<=k时原序返回singleton（没有后续能量排序）；否则过滤没有至少3个sims的结构，仅用sims[0:3]做未标准化平方欧氏距离。k=min(k,有效数)，随机不重复样本初始化中心；最多100轮Lloyd；tie选较早中心，labels初始全0，empty center随机重置，`changed=false`立即结束。每非空簇先按能量排序；>20时保留最低能，再按 `exp(-(Ei-Emin)/10000)`有放回抽样、去掉重复对象直到20个。此10000是源码硬常数，**不是SSW温度，且没有kB**。最后各簇内和簇列表按最低能排序。

Fine scoring在该函数返回之后进行，所以统计对象是截到最多20的簇，而非全archive region。按 `0.5 Emin+0.3 Emean-0.2 population_variance`升序，给每个region的第一个最低能成员启动fine，源码没有按score限制成固定top3/top16。此式改变能量单位会改变variance相对权重；只可保留为论文/发行版经验打分，不能包装为无量纲普适规律。

## 完整quick/fine状态机及隐式倍率

来源 `app_ssw_ga/SSW_GA.java` 与 `SSWGaSupport.java:129,174,190`。记基础温度T，O=OPTSSWStep，Q=QuickSSWStep，F=FineSSWStep，C=TaskNum。

| 阶段 | 实际调用/选择 |
|---|---|
| 初始化 | 系统专用生成器；随后 `SSWExploreInitial(...,3O,5T)`，保留所有all.arc结构，再postprocess→投影→batch去重→排序/能窗→archive |
| quick i=0..Qiterations-1 | KMeans archive→GA；TYPE0每i%3==0追加一次完整初始化生成；GA `SSWOpt(...,O,5T)`；merge；重新KMeans取每簇最低能，i>0另加上轮carry；SSW长度Q，偶数i温度T、奇数4T |
| first fine | 按region score排序所有簇最低能，长度F、温度T；源码 `for(i=0;i<1;i++)`，即便FineSSWIterations=0也执行 |
| later fine odd i | 先GA+O步5T优化并merge；carry加随机region代表；源码`--ssw_n - 1`导致最多C-2个region，加carry总计最多C-1；长度F、温度5T；C<2可能负subList端点 |
| later fine even i | GA同前；每簇最低能，不额外carry；长度`floor(F*C/9)`，温度T；这个9是硬编码，不是当前region数 |

`SSWOpt` 从每个任务 all.arc 中取最低能结构；不能假定 O=1等同纯minimize（LASP步数语义和force cap另见内核审计）。`SSWExplore`输入种子先按能量排序；合并并排序全部任务all.arc；若全局最低能比输入最低能低超过1e-4 eV，carry取产生该最低能的任务的最后一个all.arc结构；否则取排序后任务0的末结构。carry追加到已排序结果末尾，可能重复且不是最低能；**末尾不是自动代表原Metropolis当前状态**，需要验证all.arc内容与内核状态的对应。主程序据此取result[-1]作为carry，不能让 archive dedup 改写它。

TYPE0初始化来源 `GaInitialStructure.generate0`：Unlimited、TripleTangencyBallsPacking、SimpleCubicPacking、IrregularBall/IrregularBallOri各floor(count/2)、IrregularCage、RegularRing、RegularCage、CustomStructure全部按配置追加；每个簇包围盒加50 Å cell并中心移动。TYPE3 generate3：输入addition→seg→C个CrossMC加三类各C变异→以getLimit(.4)过滤→shuffle→可选LJ单元优化/能量排序→包围盒加50 Å→取前C。初始化C是CombineMultiUnitNum，切勿与TaskNum混淆；生成预算明显大于保留预算。MinGA、population region count、TaskNum、SSWTaskNum也不是同一个参数。

## 以ASE Atoms为边界的最小接口（规格，尚未写代码）

对外顶层应接受 `run_ga_ssw(initial: Sequence[Atoms], calculator_factory, config, rng_state, references, monomer_map?) -> GARunResult`。initial显式提供用户授权结构；若使用原初始化生成器，则作为同样返回Sequence[Atoms]的独立模块，不能在默认run中悄悄全原子建模。

1. `propose_type0(regions, config, random_stream) -> list[Candidate]` / `propose_type3(..., monomer_map) -> list[Candidate]`。Candidate包含独立Atoms、parent IDs、operator及所有随机选择/预算；Atoms承载numbers/positions/cell/pbc，元素序、约束索引和单元映射独立保持。几何生成不隐式调用E/F。
2. `project(atoms, frozen_references, descriptor_config) -> vector`，`update_archive(records,batch) -> archive_delta`。record持Atoms、energy、numeric certificate、projection、lineage。legacy行为archive与科学通过集合分别记录；不因force失败抹掉发行版行为证据，也不把未通过者叫certified minima。
3. `partition(records,k,random_stream) -> Regions`；`rank_regions(regions) -> RankedRegions`。保留完整成员ID和20成员抽样ID两套，令score分母可追溯。
4. `run_ssw(seed: Atoms, steps, temperature, kernel_config, random_stream) -> WalkResult`。结果提供按原all.arc语义的序列、真正kernel terminal Atoms、E/F调用计数、接受事件、stop reason、force certificate；不要只有best Atoms。不同结果字段不得共用可变Atoms对象。
5. `run_opt_batch(candidates, O,5T) -> list[OptResult]`；scheduler据原规则选择最低archive输出；force失败作为显式状态，后续兼容运行与科学分析分离。
6. `plan_stage(state,archive,config) -> StagePlan` / `consume_stage(state,results) -> new_state`。state保存phase、i、carry ID、所有温度/步数倍率、随机流、预算、E/F累计及stage artifact IDs；可在不启动计算下完整展开/审查计划。

随机数应注入，不能在不同函数里各造随机种子。Java当前混用Math.random、new Random、Collections.shuffle；相同seed跨语言本身不足以逐步相同。第一级用冻结draw stream与原jar候选fixture进行行为验证，第二级比较分布；不能凭相似最低能量声称trajectory parity。

## 推荐首个完整复现体系及下阶段边界

**首个完整外层+内核复现优先上传的非周期 TYPE3 (H2O)15**，原因是现有addition/seg/势、冻结reference和完整原Java短程链路产物；可沿用已生成输入，避免引入新模型。目标是完整stage状态、候选计数、结构组分、投影/归档、E/F预算、失败状态可追溯，**不是以现有14结构档案宣布物理有效**。当前水native E/F精度及ftol失败证据使最终科学质量仍需独立验证；推荐不授权提交任务。

TYPE0 Au10Ag10作为紧接着的跨类型验收：已有上传AgCuAu势与参数，验证原子切割/doping变异，且无水后处理依赖。LJ75适合数值与预算诊断，但不能替代真实体系有效性。

有界可实现模块顺序：(A) KMeans+score+stage planner，以冻结archive和假数据stage results验证倍率/carry/预算，不运行模型；(B) TYPE3水分组/cross/三变异全部真实实现及原jar候选fixture；(C)原生SSW接口接入并完成一个预声明小矩阵的端到端运行；(D)TYPE0实现及真实AuAg验证。A/B均不是完整GA；只有候选生成→优化/搜索→投影/去重→归档→下一代选择→fine→最终结构证书与完整预算都接通，才可标记“完整ASE GA-SSW实现”，其科学有效性仍单列。

## 证据与范围

主证据均在本研究目录：`decompiled/sgn/...`上述类，`literature/GA-SSW-user.txt:189–216,269–288`（论文GA比例、grid/score描述），`analysis/example-matrix.json`（已有真实体系资料）。上传论文与作者244.pdf非同一字节文件，见`analysis/paper-comparison.json`；本报告论文对照明确采用上传版文本。交叉/变异尚未新增原jar随机fixture，因此反编译疑点保留为待核验项，不把重构出来的源代码当原始公开源码。报告没有运行任何模型、优化或搜索。
