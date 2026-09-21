> 历史快照：其中“缺失/尚未接入”的判断可能已被后续实现取代；当前队列与覆盖状态以 [2026-09-12主线重评](2026-09-12-mainline-reassessment.md) 和当前代码为准。

# 固定胞 SSW / LS-SSW / GA-SSW：跨体系推进

本轮遵循 SSW→LS→GA 顺序，未修改 VC/RC，未追加 Fe7C3 计算或依据其结果选默认值。
所有新增计算为 CPU；ASE EMT 用于 Cu/Al，GFN2-xTB 用于两种 C4H6 分子。

## 已修复的共性接口

1. `run_ls_ssw` 与 `run_native_ls_ssw` 现在显式接收并透传现有 `height_policy`
   和 `height_update_budget`。不改变 soft prequench、方向 anchor、强度更新或旧默认。
   Cu13 与固定胞 Cu4/EMT 用非默认高度策略实际完成 true landing。
2. GA archive 对非有限能量显式拒绝；这是输入资格一致性修复，不是搜索增益。
3. `run_ga_ssw(structure_matcher=...)` 将结构身份与描述符分区分开：不同投影也
   可以匹配，同投影不强制合并。低能代表替换、异常拒绝和已付成本均保留；
   matcher只调用一次，不先试调用再重算。None保持legacy projection模式。

未接受的推测性修改：最终淬火失败时禁止LS更新、强制从soft geometry采样anchor。
前者现有独立caller明确允许已完成prequench的失败climb更新；native响应来自真实
prequench能量差，未证明必须等待landing。后者native顺序证据不足以直接规定
当前paper局部方向规则的坐标选择。保留差异，不新增无依据分支。
出处：native-ls-cycle-state.md、native-ls-prequench-exit-pair.md；论文 Guan等，
JCTC 2024, DOI https://doi.org/10.1021/acs.jctc.4c01081（在线原始出版页面已复核）。

## 完整固定胞轨迹：五类输入，18条运行

脚本 `research/ga_ssw/run_fixed_cell_multicase.py`，产物
`research/ga_ssw/evidence/fixed-cell-multicase-20260912/`。

- Cu13、Al13：ASE二壳层icosahedron，SSW，各seeds11/29。
- Cu31：固定2×2×2常规FCC晶胞删首位点，SSW，seeds11/29。
- 丁二烯、环丁烯：ASE G2，普通SSW、paper-LS、native-derived LS，各seeds11/29。
- 每条2个outer提案、最多25Gaussian/提案、3000总E/F上限（含最多3个fresh检查）。
  采用现有C4H6 runner的width .1、rotation100、maxiter400，inner .1、outer .01；
  inner值来自用户允许范围，是诊断操作点，不是调得的最优默认。LS表/target来自
  已有C4H6对照。两分子用同一组参数，没有依据结果调参或重试。

实际27471次E/F请求（含54次独立fresh），18/18运行保留完整两步记录，36/36
非初态落点通过真实力检查，全部固定cell不变。累计运行约90.49 CPU wall秒。
逐调用日志、结构、配置与源码快照保留。42项针对性回归检查通过（不是效果证据）。

物理检查揭示：24个分子非初态落点中存在解离片段，不能将小力等同于目标分子的
有效构象。丁二烯普通SSW/seed29第二落点比初态低0.143325 eV，同时新增C0–C3
连接且保持单连通，不能叫纯扭转；未验证正Hessian或高精度反应能。其他17条运行
没有降低best energy。LS没有在这组短诊断中体现一致优势，不能推广。
详细图比较在 `physical-summary.json`，固定阈值HC_BOND_LENGTHS+.1 Å；
图同构不区分键级或全部立体构象，也不是完整basin身份。

## GA完整流程：两种材料，四条运行

`research/ga_ssw/run_fixed_ga_multicase.py` 与
`research/ga_ssw/evidence/fixed-ga-multicase-20260912/`。
Cu13/Al13，各seed11/29，初始icosahedron+cuboctahedron，TYPE0一代，quick/
generation-short/fine各1步、每步3Gaussian。四条均实际完成初代淬火、两次quick、
proposal、两个offspring淬火、generation-short和fine；没有模拟这些核心函数。
搜索请求分别242/258/196/210（906总计），32个archive记录独立复核均满足力阈值，
总成本938E/F。不是与上述25Gaussian/两步SSW的公平效率对照。

所选ASE geometry.distance是惯性轴对齐+贪心置换近似，高对称团簇可能过度拆分。
因此每条archive=8不意味着8个不同basin，本轮仅证明身份回调贯穿完整GA流程。
不得据此宣称发现率、多样性、GA优于SSW或全局最低能结构已验证。

## 尚需推进

GA公共总请求预算已补齐并验证截断与恰好完成的区别；当前受控runner通过surface外部限额有效约束总成本，
所以不能说过去完全不能做公平预算实验。可持久化续跑状态仍是独立交付缺口。
固定胞公共API尚未完整接入旧PAM的跨步反馈；native CBD也仍是明确独立替代，
不宣称逐指令复现。GA TYPE0/TYPE3之外的固定胞周期适配需单独核查，不能把
已有VC controller自动算作固定胞GA。下一步按这些共享缺口推进，不挑一个失败
案例无限调整。测试集当前用于开发，后续科学评估必须另用未参与开发的输入/种子。

## 上传复杂案例：TYPE3-(H2O)15

继续使用上传45原子水团簇（source IfPer=0，segmentation.non的15个连续三原子组），
通过已恢复water.json读取原始坐标、描述符表和reference。使用GFN2-xTB而非原NN，
SSW/GA各seeds11/29，共4条计划内运行，合计1632E/F，约53.06秒CPU；无追加重试。
产物 `research/ga_ssw/evidence/uploaded-water15-fixed-20260912/`。

四条均在初始真实势淬火止步，未进入SSW/GA搜索。GA保留的初态记录明确显示400
优化步后fmax=.08805998 eV/Å，未达.01；因此是此后端/数值设置下的初始化资格
失败，不是TYPE3交叉或SSW逃逸失败。SSW抛InitialQuenchError，GA返回
no_eligible_minima，二者都没有把未收敛初态列为搜索成功。案例保留为负例，不因
方便而换掉它，也不围绕它加大预算/放宽阈值。本轮TYPE3真实完整搜索证据尚缺。

本轮三套有界CPU运行合计30041 E/F（含独立复核），不是三种方法的同预算性能
排名。两个种子共享同一确定性初始淬火，不能把水团簇两seed失败当成独立统计证据。

GA总请求预算实现经审查后使用单一proxy阻塞状态，避免SSW捕获异常后控制器失去
预算信号；仅在下一需要PES的阶段前或实际请求前拦截，不把自然完成恰好用满预算
误报为截断。Cu13/Al13真实EMT覆盖初始化和quick阶段截断，原有默认回归保留。

进一步论文核查定位了本体缺口：GA-SSW用户PDF的2.4.3节明确fine minima应回流
下一轮GA并重复操作（GA-SSW-user.txt:236–252，DOI10.1021/acs.jctc.6c01078）。
旧控制器fine已入archive但立即return，缺少迭代。现已加入显式cycles默认1，
共享归档/RNG/总预算，补这个算法闭环，不照搬Java的隐藏倍数和carry规则。


## GA闭环与默认轨迹复核

`PaperGAConfig.cycles` 现驱动完整 generation/fine 循环；initial/quick仅一次，
后续轮次复用fine更新后的archive、RNG和同一个总预算。各阶段记录零起始cycle。
这是独立论文级闭环，不声称复刻Java隐藏调度。

`evidence/fixed-ga-cycle1-regression-20260912/default-parity.json` 对Cu13/Al13、
seeds11/29逐行比较旧运行与cycles=1的真实evaluation JSONL，四条完全相同；
搜索906次，加32次fresh，总938 E/F。
`evidence/fixed-ga-cycle2-20260912/ledger-audit.json` 核对四条真实两轮运行：
搜索379/399/305/309次，共1392，另40次fresh，共1432 E/F。每条均有两轮
proposal和fine，每条archive12；只独立复核最低10条，尚有2条未fresh复核。
已复核40条全部满足力阈值；近似结构matcher的限制仍在，不能据此计数不同盆地。
本轮五套实验总计32411 E/F（含独立复核，不包含软件测试开销）。

## 完整软件回归边界

相同mace_env、PYTHONNOUSERSITE=1、OPENBLAS_NUM_THREADS=1、PYTHONPATH=.：
`python -m pytest -q tests/standalone` 得到411 passed、1 skipped、2 failed。
失败均为test_direction_only.py的public_direction_only_replays_first_cu13_escape
（ritz/dimer）历史终点逐坐标比较。将pamssw导入路径切换至本轮之前冻结的
`research/ga_ssw/fe7c3-user-tolerance-grid/source`，同环境重跑这两项，仍均失败，
已确认实际module.__file__来自冻结源码。本轮未修改容差；不把已存在差异称为
本轮引入的回归，也不把其环境/浮点原因未经定位便写成定论。以上为新增GA高度
转发接口之前的完整测试结果，后续针对性测试另列。

续跑checkpoint设计已完成只读审查（fixed-cell-continuation-design.md），暂缓较大
持久化实现；当前优先补SSW/LS与GA公开入口衔接及真实组合流程证据。


GA→SSW高度策略衔接已补齐，quick/generation-short/fine均显式传递height_policy与
height_update_budget，默认None/1000不变；ls设置原样传递。根agent复核该测试与
相关GA/LS回归共43 passed（2.69s）。LS跨GA阶段独立restart边界经Java文件接口
复查，尚无证据要求跨parent继承自适应强度；不新增该科学策略。


## GA+LS组合：真实C4H6种群及输入配额边界

`fixed-ga-ls-g2-20260912/` 是子agent错误脚本产物，GA力阈值错为.1、native臂
误用paper LS、描述符膨胀reference误入种群。12条错误协议运行耗8020搜索+27
fresh=8047 E/F。保留INVALID-EXPERIMENT.md，完全排除科学比较，计作开发开销。

根agent修正为butadiene+cyclobutene共同种群、2seed×3臂、outer .01、真正的
NativeLSSettings，产物`fixed-ga-ls-g2-corrected-20260912/`，共2946搜索+13fresh
=2959 E/F。quick/fine实际运行；五臂不足3个父代，另一臂empty_batch，故无完整
遗传阶段。根agent进一步核对原Java TYPE0/Mutate代码，定位mixed元素G=1时
所有整数配额为零。现公共GA提前拒绝mixed G<4、generations>0配置，未改变算子；
父代数不足仍在真实proposal处判定，因为quick本可产生新父代。

单独预登记的新协议`run_fixed_ga_ls_g2_population.py`使用ASE G2全部五个C4H6
异构体（不是按结果挑选子集），三种真实异构体作为descriptor references。
G=4/max_batches4来自恢复出的非零交叉配额；此配置没有mutation配额，不能称作
全变异覆盖。其余fmax、inner、步数、LS参数、预算保持已声明值。
产物`fixed-ga-ls-g2-population-20260912/`：六臂全部completed，各有4个真实交叉
子代淬火、generation-short和fine。搜索7922，fresh56，总7978 E/F；所有账本
逐JSONL复核相符。56个fresh均force合格，其中一个native/seed29落点解离为两片；
native/seed11另有一个archive点未fresh复核。归档数不是不同basin数。

各臂native LS有7次response/7次显式update，paper LS有7次response但当前
last_update字段为空（不代表公式没有执行）；ordinary无LS。原先2步native独立
轨迹已有8次normal update，target实际为700 meV/atom，前100步每步更新。默认
nsoftstep110>cycle100导致save/zero/restore分支inactive，不是等待100步再更新。

physical-audit.json保留元素标记图比较、解离和初始最低能对照；没有实质降低五个
初态的最低能量（最大差约1.5e-7 eV）。存在新连接图但尚无Hessian/准确basin证据，
不能用这组短开发验证宣称GA或LS效率优势。两种种群协议不可合并为同条件统计。

## 成熟优化器公共基线

SSWConfig新增ase-lbfgs-linesearch与scipy-lbfgsb，贯穿SSW/LS以及GA所有显式新
基线阶段。历史GA的ase-lbfgs设置在初态/子代仍用BFGS，其兼容边界明确保留，
不可将其称为全程同一LBFGS。新两基线和Safe-total可做显式一致对照。

SciPy直接调用minimize(L-BFGS-B,jac=True)，仅设maxiter、gtol=fmax/sqrt(3)，
保留已安装SciPy1.16默认ftol=2.220446049250313e-9/maxls20/maxcor10；不用旧
Relaxer中ftol=0版本冒充标准基线。默认相对能量停止合理地保留为数值终止，但
最终力未合格仍converged=False。零steps不移动/不调用SciPy，Eckart在PES前拒绝。
局部telemetry扩展保留message/nit/nfev，不改变全局RelaxTelemetry schema。

根agent在最后的配额检查与基线接口修改后跑完整tests/standalone：424 passed、
1 skipped、2 failed（23.58s）。两失败仍为之前已在本轮前冻结源码复现的Cu13
历史逐坐标轨迹，不改容差、不宣称全绿。四案例×三优化器的一步真实流程测试
已结束，产物fixed-optimizer-baselines-20260912。3312搜索+21fresh=3333 E/F，
1502.10秒。Cu13/Cu31固定空位/cyclobutene九臂均完成且fresh力通过；非Ih C60
Safe与ASE线搜索分别952/518搜索调用后GFN2自洽失败，SciPy1031调用后600秒
墙钟截断。三者都没有C60最终落点，不能据更少调用判断赢家，也不能把SciPy
截断称为SCF失败。账本逐项复核相符。

本轮以上全部实验合计60915 E/F，包含错误协议的8047开发开销；其余52868。
该合计不含单元测试、旧实验和随后独立的LJ数据资格检查。


## 含氧LS：补全参数来源与真实链路

原函数bondeneval_(0x6cb660)/bondlenval_(0x6cc0b0)的18个H/C/O有序pair返回
已记录于native-ls-pair-table-hco-20260912；OH和CO为本次补全，OO与既有extended
查表记录重复核验。原始OH/CO/OO键能返回分别4.817279815673828、
3.384550094604492、1.515779972076416；键长返回分别.9599999785423279、
1.4299999475479126、1.4800000190734863。原始值不是最终LS振幅或最终cutoff。

H2O/CH3OH、scale5与2.5共四项原版隔离初始化前缀与Python比较，由根agent
独立重跑：bond count分别2/5、能量表和长度表max_abs_error均0。记录及可重跑
脚本见native-ls-initialization-hco-20260912/root-verification.json和run_check.py。
未运行LASP主程序/PES/保护逻辑。96-byte运行时对象是Fortran descriptor，不能
误读成12个数值槽；先前agent据此称缺表的判断已撤回。

Python新增显式HCO_BOND_ENERGIES/HCO_BOND_LENGTHS供调用方选择，不改变既有
默认或在运行时调用二进制。全部18个lookup条目由根agent逐项核对相等。
随后甲醇(G2)、水二聚体与甲酸二聚体(S22)×三臂×两seed，共18条真实GFN2
完整两步SSW/LS流程完成：6133搜索+54fresh=6187 E/F，约17.04 CPU wall秒。
36个非初态落点及18个初态的fresh force证书均通过，固定胞未改变；没有实质
best-energy降低，最大量级7e-6 eV。二聚体初始本来有两个共价组，不能以
components=2声称解离；离线结构变化另行复核。不是LS优势或准确结合能证明。


## LJ38：有完整资格核验的负面搜索结果

新research FullPairLJ显式全对、无截断；epsilon=1 eV，sigma=2.7 Å。
旧benchmarks/lj_cluster_compare.py使用ASE默认截断，不能直接对公开GM数值。
OPTIM 03目录的odata/finish真实两端经独立计算为−173.928426244/−173.252377601，
fmax分别0.000809/0.001582 eV/Å。02目录coords.1/2/8为随机初态，曾被agent
错误关联05目录min.data编号；能量检查揭示错误，保留原文件与纠错记录。
资格检查共18 E/F（首次7、坐标排列诊断6、正确两端及力差分5），不是搜索预算。

lj38-height-comparison-20260912：同SLM起点、两个seed、固定Safe-total，比较
既有forward_force和minimal_angle87；后者是原87°条件的独立解析隔离，非完整
native高度增长/历史算法。全对LJ、width.6、NG14、kBT.8eV；其他数值参数继承
已有fixed-cell协议，global方向是显式变体。配置与源码运行前冻结，不调参。
四臂均用满12000搜索调用，分别9/9/11/10 fresh，总48039 E/F，约59.46秒。
35个非初态落点均fresh力合格，但全部MC拒绝，无最低能改善，无GM命中。
落点能量约−167.20至−152.63，明显高于SLM。这不是淬火不收敛导致的失败；
87°规则未解决该操作点的高能proposal。样本不足以排名两规则或复现文献MFET。

anchor-overlap-audit.json还从已有轨迹无新PES地提取方向与初始随机anchor重叠：
LJ各臂中位数.954–.960；Cu31约.983–.988；分子/LS一般更低。高重叠是测量，
不是固定rotation_bias100错误的证明。100仅是继承的开发操作点，未声称论文
普适默认；下一步核查已知方向曲率/rotation参数来源，不对LJ追加高度参数。


## 逐子代SSW细化已接通

`offspring_steps=0`保留旧direct quench；正值对每个candidate独立run_ssw，内含
一次初始淬火，全部落点保留观测，仅最低合格者进入archive。可显式传入另一个
SSWConfig表达原Java的温度安排，不硬编码倍率、不新增候选策略。原Java
SSWGaSupport.SSWOpt:190–205与SSW_GA:88等提供逐任务取最低的源码证据。
根修正了一个会被去重掩盖的测试，并补充walker内部捕获预算异常后不再执行
后续candidate的测试。最后test_paper_ga为29 passed；此前连同optimizer为34
passed，后又新增上述一项。不是宣称历史全套两失败已修复。

`fixed-ga-offspring-refinement-20260912`以原五异构体协议、两seed、三臂运行，
只启用offspring_steps1。六臂均completed，各有4个真实offspring_ssw阶段，
无重复controller offspring淬火。12307搜索+56fresh=12363 E/F，约58.94秒；
逐JSONL、stage sum、public请求计数一致。native每臂11响应/11显式更新；
paper11响应，无native-update事件不代表未自适应。
56fresh全力合格，native29一个解离；native11另1个archive未fresh复核。
没有实质最低能改善；这是完整流程和选择语义证据，不是效率优势。
本轮记录的实验及LJ数据资格检查累积121335 E/F，包含8047错误协议开发开销，
不包含单元测试/历史实验；反编译隔离指令执行不含PES调用。

## 核心反编译新结论

根重跑12个原指令分支案例：CBD_PreRot且curv>-1e-6时，原版设w=curv、
rotstep=1、mode=CBD_biasedRot；零和小负值也可通过，不能写成严格curv>0。
细节与纠错见native-rotation-bias-field-trace.md。尚未将完整前置旋转/anchor
状态机接入Python；这一共享kernel差异优先于继续调高度、VC或GA分区策略。


## 两阶段方向的跨体系配对开发实验

研究helper two_stage_dimer_direction先做5 HVP的无偏plane-dimer预算阶段，
再以所得Ninit和a=max(Ce,0)做主旋转。两个中心调用均占用原1+100总请求上限；
main仍要求原residual证书。负Ce继续无偏plane-dimer是独立SSW选择，不是native
-1e-6或TS搜索复刻。a=Ce使起点Crot=0，不保证严格负值；更不能保证全局搜索。
Cu13投影掉刚体零模的真实资格检查：初始quench6调用、rotation38调用，
Ce=2.99167534376454，main residual=.01152，通过且输入未改。另有agent使用
平移方向的4调用检查只作零模接口检查，不是有意义的软化效果证据。

`two-stage-direction-comparison-20260912`：Cu13/Cu31固定空位/bicyclobutane/LJ38
×两seed×fixed100或two_stage，16臂均6000搜索请求。所有搜索账本相符，
总96000搜索、215.99秒。Cu13没有最低能降低，Cu31仅1e-5量级数值差异；
两阶段同预算完成更少落点。LJ两阶段分别12/13次rotation_failed，无非初态落点；
fixed100各4个落点，但无MC接受或能量改善。不得升级两阶段为默认。

双环丁烷有正面但有边界的信号：fixed100两seed约降.3057 eV，连接图匹配
methylenecyclopropane；两阶段分别降.3304/.4227 eV，连接图匹配butadiene，
均单组分。但两阶段随后42/47次rotation_failed，只有一次非初态落点。
这支持继续隔离主旋转求解器的数值限制，不支持泛化、完整CBD或GM成功率。
匹配只用元素连接图，不是键级/构象/动力学验证。

原冻结runner的一个缺陷：bicyclobutane-fixed100-seed11在fresh循环SCF异常时
没有保存已运行fresh计数。其搜索result完整；原fresh总数只能报告100–107
而非伪造精确值。后续每步持久化且finally记数。单独对该臂8个minimum各用
新GFN2 calculator做一次冷启动复核，8次均通过，能量差<3e-13 eV；这保留了
原warm-start序列失败，并证明8个几何可独立取得力证书，没有重新优化或调SCF。
新测试统一逐点新calculator复核，防止前一个minimum的电子猜测影响独立检查。

## 同预算 Ritz 主旋转及正式实验接口

`two-stage-ritz-comparison-20260912` 保持上面预旋转、曲率律、初态、种子和
每臂6000请求，仅将主旋转替换为现有前向差分 Ritz。8臂共48000搜索请求，
68次逐几何冷启动证书，全部力合格、晶胞未改变，逐请求账本匹配。
Cu13无能量收益，Cu31仅约1e-5 eV差异。双环丁烷两seed分别降低
0.4227531887和0.5660495816 eV，最低结构连接图对应butadiene和cyclobutene，
单组分、fmax分别0.004958和0.005347 eV/Å；仅1/2次旋转失败，对比plane的
42/47次。LJ各5个落点，没有更低结构，旋转失败0/1次。连接图不是键级、
构象身份或动力学证书；这些是开发数据，不支持通用优势或完整CBD复现。

据此保留显式实验接口：SSWConfig(rotation_bias=None, pre_rotation_hvp=5,
rotation_solver='ritz')。5是本次独立求解器的开发预算，不是通用最优值或
原版5步等价；固定正高度曲率参数路径不变。实际偏置强度和预旋转anchor
进入后续高度策略曲率恢复，LS/GA通过共享SSW配置使用，不增加额外控制器。
新接口的15项定向契约检查通过；跨真实体系逐请求回放及完整回归正在验证。

正式接口接入后的完整standalone回归：434 passed、1 skipped、2 failed，23.79秒。
两失败仍为旧Cu13 direction_only终态精确坐标回放；此前已在本轮修改前的
冻结源码复现，不能报告全绿，也未用更新参考坐标掩盖差异。新staged定向检查
和原paper_reference检查共15通过。

`public-staged-direction-replay-20260912-v3`：Cu13/Cu31固定空位/双环丁烷/LJ38，
两个seed，各前1200次搜索请求的energy/fmax逐行与冻结研究原型完全一致，
9600搜索+20独立fresh=9620请求。原型日志仅能支持energy/fmax精确回放，
不能声称完整force数组逐项一致。边界截断记录为evaluation_failed并明确
request_cap，不是计算器数值失败。此前v2 runner误保留monkeypatch，造成
预旋转递归套入研究helper；56次初始化开销保留，不归因于生产预算公式。

`fixed-ga-staged-integration-20260912`：把新staged Ritz配置送入既有五异构体
GA/GA+paperLS/GA+nativeLS，各两seed；其余offspring_steps1协议不变。
9981搜索+50cold fresh=10031 E/F，根核验ledger和各stage求和完全一致。
两臂completed，四臂completed_with_failures；失败为空proposal批或rotation，
没有隐去。plain offspring分别2/4，paper4/4，native4/2；因此不能把这六臂
全部称为完整四子代成功。50fresh全力合格，49单组分，native29一个碎片。
这证明共享配置贯通主要生命周期，也表明方向改变会影响后续候选可用性；
不能将减少请求数本身当作效率改善，更不能据此给LS排序。

现有PAMCurvatureGaussian已通过gaussian_policy显式接入SSW/LS/GA；与height_policy
互斥，不改变默认值或停止条件。先选择width/weight再位移；历史仅传Gaussian，
LS已包含于旋转面曲率，不重复加入。staged使用实际a和presweep anchor恢复曲率。
4个解析适配检查验证宽度位移、实际偏置力及历史Hessian；它们只证明接口公式，
不证明搜索效果。完整standalone回归439 passed、1 skipped，仍为两项已复现的
旧Cu13终态精确坐标失败。atomic_climb子阶段尚未接入staged，现显式提前拒绝，
避免共享SSWConfig的None在途中产生误导性失败；VC/block不在本次扩展范围。

## 固定胞氧化物独立检验

`tio2-fixed-ritz-staged-holdout-20260912`：原SI的12原子rutile/anatase TiO2，
MACE-OMAT-0-small CPU float64单线程，两个seed，fixed100 Ritz/staged Ritz。
每臂1200搜索，20独立fresh，总9620请求、850.06秒（含各臂模型加载/验算）。
根逐ledger序号、initial+records和surface总数核验一致；20fresh均fmax<.01，
能量复核误差0，cell和验算几何未改变。fixed各2个非初态落点，staged各1个。
没有实质最低能收益，唯一−3.72e-6 eV差异属当前容差下数值尺度；因此不升级
staged为默认。固定晶胞不要求应力为零；局部结构不同也不等于新相或动力学。
最小Ti–O距离只是几何描述，不凭单一长度判定无效。此前physical-audit为6臂
进行中快照，最终8臂以root-final-audit.json为准。模型能量不等同DFT相稳定性。

`pam-curvature-gaussian-comparison-20260912`：Cu13/Cu31固定空位/双环丁烷×2seed，
复用v3同配置baseline，不重复收费；新增adaptive共7200搜索+10fresh=7210。
175次PAM选择，width0.44693–1.5 Å，其中47次width clipping，weight0–5.05485 eV、
无weight clipping。没有最低能收益。它是已有height_width组件对照，不代表完整
PAM walker的per_atom_rms默认；本轮不调宽度界限或叠加新回退，保留显式选项。

`hco-pam-ls-integration-20260912`：甲醇/甲酸二聚体×paper/nativeLS×两个seed，
staged Ritz+PAM Gaussian组合，8臂均完成2步、共48Gaussian阶段。4733搜索+24
cold fresh=4757；全部力合格。48阶段明确标LS-inclusive曲率，原始raw pair表不变。
没有通用效果/物理结构有效性结论；需要区分甲酸二聚体原有两共价组分与解离。
runner启动时导入live源码后复制snapshot，记录此界限，不宣称isolated-import。

LS几何采样的一处一致性修复：在soft-quench后的work生成paper pair proposal，
而非仍被接受的true-minimum current。源于LS论文2.4步骤2–3和实际搜索状态约定；
不改变沿Gaussian序列的anchor寿命。Cu13/EMT针对性检查及相关参考接口12通过，
近期global采样轨迹只依赖mass，因此此修复不改变它们。见ls-direction-sampling-geometry.md。

## Native随机向量算术与显式分布对照

根新probe `probe_native_vmb2.py` 执行原VMB2/VELO_LOC/RAN3指令，仅以host cos/log
代替数学库调用；5组零PES案例全部通过，误差<=1.74e-18。VMB2传给VELO_LOC的
分母是内部常数1.0，不是输出buffer或原子质量；最终减去笛卡尔算术平均。部分
mask的未写入分量在均值扣除前保留初值。不能把这条子程序证据当作完整native
邻域选择、global/local混合或随机种子逐轨迹复现。

据此增加独立显式direction_sampling='isotropic'：Cartesian normal不乘
1/sqrt(mass)，刚体投影仍由已有frame完成。global/paper保持原分布。它检验
Born–Oppenheimer能量搜索是否需要质量预条件，不声称质量加权数学错误。
Cu13/EMT质量变更后整步exact replay及相关参考检查9通过。全PBC允许isotropic
配translation_only；atomic/block仍保持原global支持域，没有扩展VC。

`isotropic-direction-comparison-20260912` 新增6臂7200搜索+8fresh=7208；双环丁烷
global对照复用v3，甲酸二聚体global两臂新算。双环丁烷isotropic seed11降
.56603161 eV、seed29无合格新落点；同预算global分别0和−.42268831 eV。
甲酸二聚体各臂没有最低能收益。结果显示方向分布会改变路径和种子表现，
2seed不能证明普适优势或据此改默认。冻结runner沿用标题/EMT文字已以
PROVENANCE-NOTE澄清：实际两体系均GFN2、无PAM Gaussian，不重跑修饰结果。


### Latest regression and next independent fixed-cell holdout

After isotropic sampling and LS working-geometry correction: 442 passed, 1 skipped,
2 pre-existing coordinate-snapshot failures in 23.93 s. Log:
`/tmp/pam-ssw-standalone-20260912-isotropic.txt`. No all-green claim.
The next holdout reuses the qualified AFLOW CuO64 endpoint, keeping its cell fixed;
its historical V100 joint-cell qualification is provenance only, not this test.
The new comparison is CPU MACE-OMAT-small, fixed100 Ritz vs staged pre5 Ritz,
seeds 11/29, unchanged TiO2 protocol and 1200 requests / 300 s per arm.
No GPU job, per-case parameter retuning, or VC development is involved.


### TiO2 coverage clarified by structure and Hessian checks

The eight-arm fixed/staged campaign did not improve best energy appreciably;
this does not mean every landing was the original basin. Offline same-species
assignment / periodic translation comparisons are recorded in
`tio2-fixed-ritz-staged-holdout-20260912/root-structure-distance-audit.json`.
The assignments provide constructive distance upper bounds, not globally
certified identity or kinetic connectivity.

To check actual local curvature, the highest-energy completed landing within
the fixed-Ritz baseline was selected for each of anatase and rutile. This is
post-hoc qualification, not parameter selection. CPU MACE-OMAT-small, central
force differences at 0.001 A, fixed cell, three translation modes removed:

| Input / selected landing | E above initial (eV) | fmax (eV/A) | Smallest internal eigenvalue (eV/A^2) | Requests |
|---|---:|---:|---:|---:|
| Anatase / seed11 fixed, minimum2 | 2.2137802 | 0.0062483 | 0.7138613 | 73 |
| Rutile / seed29 fixed, minimum2 | 2.7851104 | 0.0058659 | 0.7026057 | 73 |

Both reduced Hessians have no negative eigenvalues. Hessian skew spectral
norms are 6.23e-5 and 3.16e-5 eV/A^2, respectively; translation residuals
are 1.29e-5 and 1.53e-5. Total qualification cost: 146 E/F requests,
15.72 s; no relaxation or search parameters changed. The finite gradient
and single finite-difference step bound the conclusion: these are strong
local-minimum indications on this fixed-cell MLIP surface, not rigorously
certified exact stationary points, identified experimental phases, or DFT
validation. They support nontrivial basin exploration despite no lower-energy
discovery. All other landings remain at their original force-only evidence level.
Evidence: `tio2-fixed-landing-hessian-20260912/{manifest,results}.json`.

### Native neighbor-mask branch closure

The root independently reran the 12-case isolated atom_neighbor_radius probe.
Exactly radius 12 leaves the mask unchanged; a non-center candidate beyond
12 zeros the final atom triplet, even when the distant candidate is not last.
get_dist was hooked only to supply controlled scalar distances; the wrapper
executes original instructions, with no LASP main/PES/protection execution.
Static caller checks show center=object.atompair[0], N=object.natoms; run types
5/6 return to the common center read. The distance branch must actually be
taken for this behavior to matter; small structures within the radius do not
exercise it. This native behavior is not copied into the Python sampler.
See `native-global-random-workspace-audit.md` and the root-verification JSON.


### Cu13 regression environment boundary resolved

The former static coordinate fixture matched ASE 3.29 / NumPy 2.5.2 / SciPy
1.18.0, but not the compute environment ASE 3.26 / NumPy 2.0.2 / SciPy 1.16.0.
With the latter, the first direction still agrees, but subsequent quench
trajectories differ (maximum endpoint differences 0.1628552 A Ritz and
0.0338590 A dimer). These observations isolate the dependency-stack distinction,
not which individual library change caused it.
The regression now compares the archived research projected_solver wrapper
against the public direction_only path within the same environment, keeping
the static evidence unchanged. Both paths must still report biased_quench_failed,
agree on direction, endpoint and request count, and exclude the failed move
from minima. This is an actual Cu13 EMT paired replay, not zero-PES testing.
Root verification: **444 passed, 1 skipped in 25.58 s**, entire standalone suite,
`/tmp/pam-ssw-standalone-20260912-live-reference.txt`. No production algorithm,
default, fixture coordinate or assertion tolerance changed for this fix.


### CuO64 independent fixed-cell holdout completed (CPU)

Evidence: `cuo64-fixed-ritz-staged-holdout-20260912/root-final-audit.json`.
Both seeds per arm use the previously qualified 64-atom AFLOW-derived geometry,
fixed cell, MACE-OMAT-small. The historical V100 qualification is not a new GPU
run. All four current arms reach the declared 300 s wall cap before 1200 E/F
requests; the reported `evaluation_failed` is therefore a budget boundary,
not a MACE calculator failure.

| Seed | Arm | Search requests | Completed noninitial landings | Fresh requests |
|---|---|---:|---:|---:|
| 11 | fixed Ritz, a=100 | 562 | 1 | 2 |
| 11 | staged Ritz, pre5 | 539 | 0 | 1 |
| 29 | fixed Ritz, a=100 | 565 | 1 | 2 |
| 29 | staged Ritz, pre5 | 545 | 0 | 1 |

2211 search + 6 fresh = 2217 requests. All fresh checks pass .01 eV/A,
energy error is zero, and cell/geometry are unchanged. All ledger sequences
and initial+record cost sums match. Fixed-arm landing energy changes are
+0.00015859 and +0.00051784 eV; no lower-energy discovery. Staged arms stop
during their first escape, after 22 Gaussian records, before true landing.
Fixed arms complete 25-Gaussian escape and enter their next attempt.
These are time-limited workflow observations, not equal-request performance
rankings. Other bounded CPU checks briefly ran concurrently, so wall time
is not an isolated benchmark. No parameter or default was changed in response.

### GA from one supplied basin: actual population bootstrap and crossover

Evidence: `fixed-ga-single-basin-20260912/{root-final-audit,root-graph-audit}.json`.
Only ASE G2 bicyclobutane is supplied as initial geometry. Three fixed G2
descriptor references are declared prior structural information; they are
never injected into the archive. Plain SSW only, no LS. quick_steps=10 is
an explicit bounded short-exploration choice; max_gaussians=25 matches the
ordinary staged-Ritz walker rather than the old three-Gaussian integration
smoke test. These are not fitted defaults. Other GA operator/identity settings
are inherited, including the .1 A approximate geometry matching criterion.

Both seeds establish parent populations from actual quick SSW landings,
including force-qualified MC-rejected observations. Both execute four
atomic_crossover offspring short walks, a generation short walk and fine stage.
The sole initial observation is id0 for each seed; actual crossover parent
IDs include newly discovered quick observations, not descriptor references.

| Seed | Search + fresh requests | Quick best change (eV) | Offspring best change (eV) | Terminal state |
|---|---:|---:|---:|---|
| 11 | 12000 + 7 | -0.4227532 | -0.5660808 | budget_exhausted during fine |
| 29 | 11154 + 6 | -0.5660496 | -0.5660291 | completed_with_failures |

Total 23154 search + 13 fresh = 23167 requests; every ledger sequence and
phase cost sum agrees. All 13 final archive entries pass fresh force checks,
but one seed29 entry has two molecular components. All others have one.
Both best structures are single-component and match cyclobutene's G2
species-labelled graph (graph test does not resolve stereochemistry/bond order).
Archive sizes 7 and 6 are approximate representatives, not 13 proven distinct
minima. Seed11 gets further improvement in the offspring phase; seed29 had
already reached the same low-energy structural class during quick SSW.
This closes the missing single-input population-establishment example; it
does not establish GA superiority at matched cost or molecular validity of
every force-stationary archived structure. No extra seed injection or
compensating proposal fallback was added.


### Anchor and width conventions: hypotheses kept separate from bugs

Root re-read the actual SSW2013 full text (`literature/74.txt`, p1840,
Eq6–8 and Overall Algorithm). It explicitly states that each Gaussian's
direction is refined from the original random N0; the rank-one rotation bias
is added to the real PES. Therefore the public paper path's unchanged outer
anchor and exclusion of Gaussian history from its rotation surface have
source support. A native continuation difference, if found, must be recorded
as a later-release difference rather than silently labelled a paper bug.

Native n_normal at 0x578e20 sums the complete 3N squared components and applies
one inverse L2 norm; gen_randommode calls it at 0x5d6733. This is not per-atom
RMS scaling. A delocalized unit direction consequently has typical per-atom
amplitude proportional to 1/sqrt(N), in both conventions being compared.
The CuO/TiO2 difference does not by itself diagnose an incorrect width unit
or establish that changing scale will improve search. No extra scale parameter
or automatic continuation branch has been introduced.


### Matched-request SSW control for the single-basin GA result

Evidence: `fixed-ssw-single-basin-matched-20260912/root-final-audit.json`.
The pure SSW control copies the exact frozen GA source and asserts identical
SSWConfig, initial coordinates/species and paired realized request caps before
any PES call. No GA descriptors/proposals or LS are used by the control.

| Seed | Search requests per method | SSW best change (eV) | GA best change (eV) | GA minus SSW (eV) |
|---|---:|---:|---:|---:|
| 11 | 12000 | -0.4227532 | -0.5660808 | -0.1433276 |
| 29 | 11154 | -0.5660906 | -0.5660496 | +0.0000410 |

Pure SSW adds 23154 search + 26 fresh = 23180 requests. Both ledgers have
exact sequential request IDs, and all initial+record sums agree. Every fresh
check passes .01 eV/A; maximum energy difference is 3.98e-13 eV. The two
controls end explicitly at their paired request caps, with 12 noninitial
force-qualified landings each. These are not all distinct basins.

Seed11 provides a concrete case where GA reaches a lower-energy structural
class within the same search-request budget; seed29 provides no material
evidence of additional improvement. The 4.1e-5 eV reverse difference is
reported as measured, not interpreted as an algorithm ranking. This comparison
was designed after observing the GA runs and has only two seeds; it is a
bounded paired control, not an independent or statistically sufficient
efficiency benchmark. Fresh validation costs differ and remain separately
reported (GA 13, pure SSW 26); only search requests are exactly matched.


### Target-aware C4H6 SSW / LS comparison

See `2026-09-12-c4h6-ls-reaction-coverage.md` and frozen evidence
`c4h6-ls-reaction-coverage-20260912`. Six CPU arms: 131177 search +129 fresh;
all129 force checks pass. Common-prefix graph counts SSW/paper-LS/native-LS
are4/5/6 (seed11) and4/4/4 (seed29), including initial connectivity. No
consistent best-energy advantage. Paper LS reaches .7 target; native-rule
LS remains .078–.110 under its bounded ramp. This is GFN2 reaction coverage,
not paper PBE reproduction or a reason to alter defaults.


### Shared policy validation before GA initial PES work

GA now shares SSW height-policy type/frame/update-budget validation before
initial relaxation, including offspring-specific configuration. Valid inputs
retain their existing flow. Added zero-request invalid-input checks and a
real EMT two-Gaussian MinimalAngleHeightPolicy history regression. Root
full suite in mace_env with PYTHONNOUSERSITE=1:450 passed,1 skipped,26.67s
(`/tmp/pam-ssw-standalone-20260912-policy-final.txt`). Total search accounting
was not broken: zero-cost GA failure diagnostics overlap stage/observation
records and must not be summed as independent search costs.


### Mature ASE BasinHopping control completed

`ase-basin-hopping-baseline-20260912-v2`:30536 search+388 fresh=30924.
All388 returned-quench force checks pass; two molecule trajectories stop on
GFN2 SCF failures, four metal arms at6000 requests. Matched-prefix best energies
show no material SSW advantage; BH landing counts do not establish basin diversity.
Full costs/limits in `2026-09-12-ase-bh-baseline-results.md`.


## Ritz认证与跨体系配对（2026-09-12）

已完成三类两seed旧/新12臂：72000搜索+114fresh，全部fresh力合格。
金属四组全账本相同，分子旋转失败3→0且最佳能量不变。决定只纳入已定位的
认证退出修正，保留全部搜索参数。局部LS失败重放31/39复现旧失败，新规则39/39
认证通过，额外62请求；不能称39次精确失败复现。详见2026-09-12-verified-ritz-stopping.md。


公共Ritz修正的组合流程验证结束：LS六臂36042请求、GA两臂24012请求。
LS全部42落点、GA全部12归档冷启动力合格；GA实际4子代/seed，fine预算截断，
不声称精搜索完成或GA优势。全量回归465pass/1skip。后续聚焦原版Broyden
完整历史更新及删除条件；隔离多步探针已走通，尚非完整CBD旋转轨迹复现。
