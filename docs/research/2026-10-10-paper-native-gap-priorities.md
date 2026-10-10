# 论文、LASP 与当前代码：进度、具体缺口和优先级

审查日期2026-10-10，代码基线222a078。实际研发checkout为
`research/c60-local-defect-qualification`，不是用户工作目录的main。
目标仍是独立Python/ASE算法，吸收可解释、可消融、有效的设计。
目前主要流程已贯通，处于组件效果验证阶段，尚未完成通用搜索效率或生产验收。
本轮新增证据是两个已完成C60面板的原始账本/冻结源码/检查点审核；重新阅读相同
论文和旧反编译记录不算新增独立科学证据。没有改变搜索参数或核心算法。

## 审查范围与证据入口

问题是哪些缺口限制有效结构发现，而不是还差多少LASP开关。比较单位为SSW、LS、
VC、RC、GA的论文机制、当前入口与实际实验版本。优先看状态/坐标/目标一致性、
有效转移及全部成本；原版数值行为一致性与算法效果分别判断。
沿用[来源台账](../../research/ga_ssw/evidence/mainline-reassessment-20260923/source-ledger.csv)
与该系列的阅读计划，不复制另一套文献目录。检索截止本日。
ACS的SSW/LS页面仍返回403，作者VC PDF的web工具访问失败；本地已归档完整正文/SI
可读，不因此声称关键论文缺失，也不以检索摘要代替公式。

- SSW2013：Shang/Liu，DOI10.1021/ct301010b；74.txt，§2、Eqs.3–9、steps1–8。
- LS2024：Guan/Shang/Liu，DOI10.1021/acs.jctc.4c01081；215.txt §2.1–2.4、
  Eqs.11–15及Table1；ct4c01081_si_001.txt §7.3–7.4。
- VC2014：Shang/Zhang/Liu，DOI10.1039/c4cp01485e；§2.2–2.3、
  pp.17847–17849、Fig.1b、Eqs.3–7；[逐式对照](vc2014-native-crosscheck.md)。
- RC2025：Guan等，DOI10.1021/acs.jctc.5c00350；225.txt §2.1–2.2及SI。
- GA2026：Liu/Liu/Shang，DOI10.1021/acs.jctc.6c01078；GA-SSW-user.txt §2，
  与Java NNA反编译分开；[描述符契约](2026-09-22-ga-dccd-contract-audit.md)。
- LASP证据只适用于上传ELF，SHA256以各指令审查为准；静态切片、隔离函数执行、
  完整程序的同势运行是不同证据层级。两个有界只读子审查负责SSW/LS与VC/RC/GA，
  主agent复核了关键代码入口、公式和下列实际产物。

## 刚完成的结果改变了什么

### 两个新随机C60：方向组合有范围的收益，目标仍未验收

冻结算法1445a53，GPU1664068四臂及CPU1664073已结束。
同一MH-1/omol、每臂60000收费E/F，唯一预定因子为恢复CBD旋转与完整方向组合。
后者含局部pair/group、逐Gaussian位移反馈及持久选择状态；不是单独记忆项的消融。

| 初态/搜索seed | 方向 | 合格观察数，含初态 | 最佳能量相对Ih / eV | 收费E/F | 实际search calculate |
|---|---|---:|---:|---:|---:|
|26100781|恢复CBD|59|22.28235|60000|58074|
|26100781|完整方向|122|10.51517|60000|55887|
|26100782|恢复CBD|68|19.18861|60000|57906|
|26100782|完整方向|133|4.87790|60000|55490|

完整方向的最终最佳分别低11.76718/14.31072eV，八次初态/最佳独立复算均力合格。
两条最佳完整方向结构在三个键截断下均为60个三配位原子、三连通平面图，但分别
含4/7元环或7元环；未达到预定义12五元环+20六元环的完整富勒烯拓扑，也未到能窗。
笼目标、能量目标及联合目标均0/4，不能叫成笼验收成功。
15k前缀反而均是仅旋转臂能量更低；30k两初态的优势不同。因此不是所有成本区间
普遍改善，不能升为通用默认，更不能把观察数翻倍解释为不同盆地数翻倍。

四条终态的`evaluation_failed`均为预定60000请求限额，不是MACE失败。
原始240000 search无付费后端错误，加8fresh=240008；资格准备720另列，合计240728。
search实际calculate227357，fresh实际calculate未独立计数，不补填为8。
已记录的偏置淬火占四臂收费成本70.4%–77.5%，旋转10.7%–16.0%；这定位了
E/F成本重点，尚无分阶段耗时证据证明L-BFGS代数是walltime瓶颈。

进一步零PES回放：用保存的单边FD真实曲率，加所有旧Gaussian解析Hessian贡献，
再减新Gaussian的w/sigma²，4160个有完整记录的中心中4157个估计沿当前方向
曲率为负（各臂636/636、1346/1348、690/690、1485/1486）。这是局部FD诊断，
不是新完整Hessian或逃逸证书。它不支持“普遍未能使选定方向失稳，所以应增高
偏置”这一简单解释；有限位移/模式选择/非线性路径/真淬火回返仍可能限制搜索。

原始与派生结果入口：[固定协议](../../research/ga_ssw/evidence/c60-direction-transfer-20261007/protocol.md)、
[读出](../../research/ga_ssw/evidence/c60-direction-transfer-20261007/readout-1664073/analysis.json)、
[主agent闭合审核及曲率诊断](../../research/ga_ssw/evidence/c60-direction-transfer-20261007/root-audit-20261010-v2.json)。
没有把更早的开发初态或其他版本成绩算进这两组新输入。

### 文献C60异构体#3：原版也未修复，不再把它当单独实现错误

Python SSW/论文LS四臂先前实际付费63077+8fresh，目标0/4；LS的预淬火和响应
更新均执行成功，末响应接近0.02eV/atom，不能解释为LS未打开。
原版参照87b5b91、GPU1665016、读出1665018和cold1665019现也已完成：

| native seed | 收费E/F | 实际calculate | 合格minimum事件，含初态 | 停止 | Ih目标 |
|---|---:|---:|---:|---|---:|
|26100791|9621|9592|29|900秒监督上限|0|
|26100792|9878|9848|31|900秒监督上限|0|

19499search+4fresh=19503，实际search19440及fresh4次calculate；四次cold均合格。
全部请求通过真空邻接等价/元数据门，无后端错误、拒绝请求或遗留子进程。
未完成尾段268/231请求仍计费。两臂均没到10000前缀，不能虚报16k完整运行。
只有5000共同前缀可直接按预设前缀比较；两套方法最佳仍接近原缺陷笼、未到Ih。
小于约0.0001eV的原笼优化差异不作算法排名。

这是一种初态上的两个native seed，不是两个新结构；原版与Python的RNG、MC、
优化器、停止条件不同，仍是整套配置参照。结果降低了“仅我们的实现导致此笼停滞”
这一解释的可信度，但不证明模型无法跨越、LS无效、长预算必然失败或Python优于LASP。
此案例按有界开发目的收口，不自动延长或改参数。
[原版读出](../../research/ga_ssw/evidence/c60-source3-native-reference-20261007/readout-1665018/analysis.json)、
[独立复算](../../research/ga_ssw/evidence/c60-source3-native-reference-20261007/qualified-1665019/summary.json)、
[Python/LS结果](../../research/ga_ssw/evidence/c60-source3-paper-ls-transfer-20261007/decision.md)。

## 对照论文和实际代码的缺口

| 模块 | 已经实现 | 剩余缺口及其性质 | 顺序 |
|---|---|---|---|
|SSW本体|随机/局部方向、恢复CBD、逐Gaussian爬坡、去偏置淬火、MC、方向/池恢复；独立ASE E/F入口|新方向组合尚未隔离局部选轴与位移反馈贡献；不同盆地转移、回返及目标成本还没有稳定跨体系优势。这是效果证据缺口|P0|
|Gaussian与局部优化|解析一致Gaussian；BP forward-force、原版启发87°、PAM曲率策略均为显式选项；Safe-total和ASE/SciPy基线|原版移动后的实测宽度/保护重试和释放时刻不是固定宽度策略的同义词；完整caller补偿未闭合。需在真实多Gaussian逃逸中检验收益，不按开关数量补齐|P0内有界问题|
|LS|论文指数pair势、当步冻结邻居/r0、软面预淬火、真能响应、移除全部偏置；另有native表/周期控制|论文总幅度控制与native元素表限幅不是同一算法；达到响应目标不保证采到化学重排方向。跨体系覆盖/降能收益仍混合|P0并列|
|VC|2014式cell/固定胞原子交替block；另有联合log-strain；E+pV和真力/完整应力核验|联合TiO2方向残差求解的预算效率是真实阻塞；cell方向生命周期、周期/度量native契约未完全闭合。2014交替已实现，不以joint失败否定它|P1|
|RC|刚体链/森林、周期lift、RC-VC、完整原子终淬火|调用者须提供固定化学拓扑和连续映像；native krot同时改前向几何及力传递，尚无对应完整JᵀF；跨分子晶体搜索收益未验收|P2|
|GA/池|短SSW、遗传候选、archive、分区、长SSW、恢复；池LS重启/去重可选模式|当前Java NNA不是论文canonical DCCD；GA调度是显式替代；同成本提升尚未成立，不能把工程可恢复当性能验收|P2|
|Q描述符方向|S1/S2研究原语及已批准方案|当前研发入口仍Q-off，没有完整S1–S6公共walker；即使补齐也需单独证明方向收益。MACE/SOAP特征梯度可作不同机制，不是直接替换论文公式的等价实现|P2|
|接口/集成|无约束固定胞、FixAtoms/Hookean受限入口、独立周期方向与显式几何|各入口组合不是全支持：普通run_ssw只接受非周期pair Hookean，FixAtoms/点/平面Hookean须走受限入口；block原子段限定global/translation_only/Safe-total，尚不能直接继承完整方向组合|随阻塞处理|

代码定位：`paper_reference.py:460,905,1020,1047,1184,1246`；
`softening.py:94,167,209,228`、`ls_cycle.py:59`；
`block_ssw.py:53,103`、`cbd_cell.py:28`、`vc_reference.py`的`mode.converged`分支；
`rc_vc_reference.py:27`；`paper_ga.py:473`及`legacy_descriptor.py:39`；
`constrained_reference.py:361`、`ssw_restraints.py:11`。
受限入口在显式`periodic_local`模式下已可用周期完整方向，不能重述成“只有非周期”。
但某项源码存在不表示所有组合/所有Calculator已端到端资格化。

### LS的原理不能简化成把所有模变软

论文Eq.11的冻结对势为v(r)=A exp[-(r-r0)/l]，l=xi*r0。相对位移的Hessian为

    D = (v/l²) uuᵀ - (v/(l*r))(I-uuᵀ),  u = r_vector/r.

径向曲率为正、两个切向曲率为负；完整体系还经过预淬火改变几何，继而改变真实
Hessian和模式耦合。因此应检验局域反应模式是否改变及产生更多有用落点，而不是
以总响应到达目标或平均频率下降替代科学效果。当前公式/生命周期已有实现，
无需因某个C60失败再发明额外pair奖励。[原理和控制律](2026-09-23-ls-mechanism-and-budget.md)。

### 原版数值细节：已知不等于应该照抄

1. 原版LBFGS→MCSRCH→MCSTEP调用链已有直接证据和保存态对照。Safe-total仍是
   带曲率历史筛选和Armijo回溯的独立L-BFGS；原版数值内核尚不是Python生产后端。
   保存态结果没有普遍赢家，历史500也不保证更好。
   [实际差别](2026-09-17-safe-total-versus-native-lbfgs.md)。
2. Gaussian指令oracle早已完成，**不是尚未执行的下一任务**。它发现旧Gaussian
   力重复两次、能量一次；“第二遍缺投影因子”是已撤回的早期误读。
   完整程序补偿未证明，不能叫整个LASP错误；保留Python一致E/F。
   [已完成指令证据](native-addgaussian-instruction-oracle.md)。
3. 直接native移动与Python均可采用全3N单位方向，不能再用“原版每原子固定步长”
   解释差距。原版实测宽度、0.95重试、成功/失败选点才是有证据的不同。
   [移动证据](native-moveds-scale.md)。
4. 2014 VC用9个lattice分量扣除3个转动模式，交替cell和原子段，不要求联合原子/
   应变Gaussian。25步是其固定胞部分原子松弛上限，不是全SSW的统一优化上限。
   论文调度相位、Eq.7曲率符号有明确歧义，当前正Hessian约定须标注。
5. native VC的已恢复stress诊断为abs(trace(stress)/3+p)，不控制纯剪切/无迹应力。
   完整驱动还有其他路径，不能说它最终一定接受任意坏应力；独立落点保持完整允许
   stress核验。[应力指令证据](native-vc-convergence-contract.md)。

## 下一步顺序与可改变计划的信号

**P0：SSW/LS有效转移，先解释现有轨迹再做单因素实验。**
新随机C60说明局部方向组合值得保留为研究候选，也暴露“组装出三配位缺陷网络
之后的拓扑修复”这一剩余困难。不能只追加外步/换GA来掩盖它。首先沿用既有
结构身份/拓扑分析，按共同成本区分重复优化、有效不同结构和回返；观察数仅作
吞吐量。当前偏置淬火70%–78%的成本足以优先审查真实多Gaussian连续变形，
但不足以立即替换优化器或声称AutoSella必要。

已完成中心曲率回放，背景含旧Gaussian，未直接把裸PES曲率当背景。4157/4160
估计为负，故不安排无依据的增高/曲率阈值扫描。下一最小检查是沿现有Gaussian
中心和真淬火落点区分两种解释：方向/局部形变没有触及有用键重排，还是爬坡
已改变键图而真淬火回返。结构图变化仅作诊断，不称反应/盆地证书。
前者优先检验现有局部方向和LS在不同体系的贡献，后者才可能涉及有限连续路径
或释放交接；按实测信号选择一个已有机制，不同时改宽度、LS、方向、MC与optimizer。
复用保存态和既有身份检查，保留全部成本，不重开已收口单Gaussian扫描。
若进入真实多Gaussian对照，覆盖C60与非碳Cu/EMT或已有周期TiO2；保留物理
资格门和失败成本。没有新的决策信号或端到端收益则结束该机制，不继续扫阈值。

LS并列推进：C4H6关注连通化学候选覆盖，C60关注富勒烯拓扑/能窗；保留正反
结果。source3适应控制已工作却无修复，且native也无目标，不再将其当调参靶子。
应使用不同论文来源/已资格输入验证方向与真实转移变化，不机械扩大此笼预算。
MH-1、OMAT或论文PBE/G-NN是不同PES，明确模型边界，不声称纸面数值复现。

**P1：VC优先论文block和可识别相目标。**
已有TiO2 block48在995请求/outer4找到anatase，cold力/应力/结构资格通过，MC拒绝
不抹掉发现；block最佳另匹配TiO2-B，不能称anatase为该势GM。joint两臂共40次
全rotation_failed，保存态Ritz在40EFS仍未过0.02门，而12原子全Hessian能过门。
这支持方向求解预算/停止契约问题，不是先修局部L-BFGS或继续加严stress。
已有atomic force_or_budget局部补接有范围地改变落点，48原子却比直接quench差，
不推广成通用收益。joint公共退出契约若要改变，先按AGENTS第8节讨论。

后续LJ bulk先完成势截断和fcc/hcp目标资格，已有rc2.7与3/6/10相序翻转；不能
为配合论文选截断。SiO2的OMAT轨迹不等于2014 BKS结果，SI目标输入未取得时
不冒造。原版cell方向生命周期/度量的反编译仅在它能区分具体失败解释时继续。

**P2：RC、GA、Q与外部优化器继续后排。**
它们不是已完整验收，但当前收益证据不足以优先于上述内核问题。RC优先明确
固定拓扑域和同坐标力映射；GA优先结构身份与同成本对照，不复制所有Java权重。
并行分支只做互不冲突、有验收价值的任务，不再扩建全部外围机制。

## 工程交付和本轮验证

当前各变体实验散落多个版本；LS/VC/optimizer面板提交并非当前HEAD祖先，不能
合称同一版本完整验收。Q分支99a881f实际是方案文档，1037dae是S1研究原语修复，
不是公共模式入口。当前选定配置按冻结源复现，不机械合并全部分支。
固定胞使用说明的旧分支及atomic入口边界已更正；其他旧报告保留为日期明确的历史。
核心`paper_reference.py`承担较多状态/持久化职责，仍有工程债务；本轮未借此重构。

主agent零PES审核重查117个冻结core文件、实际plan/source清单、连续请求、全部
收费/检查点成本、共同前缀、原版Minfound对应的E/F及cold结构哈希；结果见上链。
重复审核相同产物不是独立新实验。重跑审核命令：

```bash
PYTHONPATH=research/ga_ssw/evidence/c60-direction-transfer-20261007/prepared-1664056/source \
PYTHONNOUSERSITE=1 /home/gengjianrui/.conda/envs/mace_env/bin/python \
research/ga_ssw/audit_oct10_completed_panels.py --output /path/to/new-audit.json
```

`--output`要求新文件，保护旧证据。Slurm结束状态仅与运行日志合并解释；native是
有界监督停止，Python是收费上限，二者都不是算法完成1001/100外步或达到目标。
科学阶段仍未验收，不报完成百分比，不以现有证据要求用户解决普通代码/模型问题。
