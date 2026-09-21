# 当前唯一执行队列：固定胞 SSW → LS-SSW → GA-SSW

## 2026-09-14 当前评估（优先于下方历史记录）

本日进一步诊断已完成：[OMAT有效性诊断](2026-09-14-omat-validity-diagnosis.md)。
23次直接ASE/MACE请求复现异常低能（误差<4e-13 eV），所查方向的能量差分
趋近模型力；孤立H2未复现同样深井。优先解释为复杂环境相关模型异常，不能
归因为PAM偏置泄漏，也不据此添加双体排斥。全部1334观察离线几何审计完成；
C4H6六臂的单连通观察最低能量均约-55.488 eV。维持OMAT-small和实验性LS出口，
不把异常低能计为算法优势、不继续扩展同类C4H6搜索。配对首个oracle输入相同但E/F已有约1e-15差异，第二条外步记录调用数即分化；
新策略未触发，优先解释为数值误差被非线性搜索放大，不归因为退出策略收益。

已重新读取 `/home/gengjianrui/.codex/AGENTS.md`。总体阶段为独立实现与
真实搜索验证，尚不是完整LASP行为复现或生产级算法效果交付。以下历史队列中的
GFN2/MH-1选择及“正在运行”状态已被替代；后续研究后端遵照用户选择统一为
MACE-OMAT-0-small，ASE通用Calculator接口保留。

最新12臂OMAT实验的Slurm任务均结束；产物实际为8臂完成150次尝试、2臂严格
LS预淬火失败、2臂搜索预算截断。1305395次搜索调用，1334个归档观察均通过
已有fresh能量/力/组成/固定胞不变量检查；不等于1334个不同、物理可靠极小值。
水15严格LS在第2/3次尝试的50迭代maxiter停止；允许正常预算出口的两臂形成
141/144条记录，分别70/11次step_limit出口。机制已在真实体系触发，但搜索
优势未建立。C4H6两策略均未触发step_limit却有轨迹差异，不能将差异归因于
退出规则；需排查数值可重复性/分支放大或隐藏状态差异。

新增优先阻塞：C4H6 SSW seed11最低观察-258.1012 eV（初态约-54.6133 eV），
最近H-H距离0.5153 Å，真实模型力最大0.02254 eV/Å。小力证书不足以说明
化学合理性；先区分短程模型异常/外推与接口、能量/力一致性问题，不将异常
低能作为算法胜利，也不直接改模型或叠加排斥势。其他结构尚需系统几何审计。

下一队列：
1. 对现有完整产物做结构有效性和同预算比较；先完成离线审计，再选最小的
   独立E/F或局部一致性检查区分模型与实现。不得由极低能直接给方法排名。
2. 核对未触发策略时的配对轨迹差异，隔离数值重复性与代码状态差异；同时
   将水15结果限定为正常maxiter释放的机制证据，默认不自动放开。
3. 固定SSW/LS参考配置、支持边界和可解释指标后，再做GA相对多起点SSW增量。
4. VC/RC继续后置；反编译仅解决能改变上述决策的anchor/history、状态退出、
   LS响应计数问题，不无限恢复无关原程序细节。

本次仅审查、离线读取产物并更新本文件；未改算法代码、未提交或重跑计算。
下文为历史证据，不能把旧测试数量或旧运行状态作为当前完整验证。

---

2026-09-12重新核对当前代码、论文全文/SI、LASP反编译和真实产物。
完整理由与证据：[主线重评](2026-09-12-mainline-reassessment.md)。
此前混合了历史结果与旧卡点的版本保存在
[旧主线快照](2026-09-12-mainline-before-reassessment.md)，不再作为执行队列。

## 当前判断

共享SSW/LS主要流程、ASE计算后端、Gaussian预算/释放选项、checkpoint和
带约束专门入口已存在。Cu预淬火、1e-4是否必然过小、是否有checkpoint等
问题退出最高优先级。581通过/1跳过等回归结果证明实现契约，不证明搜索效率。

真正的科学卡点是：尚无可靠的跨体系、跨seed、同总成本低能盆地发现优势；
已有BH对照亦未显示明确SSW优势。实现parity缺口与科学证据缺口分别处理。

## 顺序

1. 冻结共同SSW流程和加LS的对应流程，明确paper/native-derived差异。
   对方向重入/anchor、LS跨步response、正常预算退出只做有界必要核查。
   不重新移植整套旧PAM启发式，不以原版镜像容差充当光滑化。
2. 推进多步、等预算真实搜索：C4H6异构体GFN2与非平凡金属团簇/固定胞
   缺陷EMT作为首组；目标/seed/结构身份/预算提前冻结。C60是困难复核，
   后端稳定性须先满足，不作为阻塞所有体系的门槛。Hookean小例保留回归。
   统计有效不同盆地、目标结构达到成本、best-energy曲线、所有失败成本。
3. 根据真实轨迹瓶颈固定LS/PES后比较Safe-total、ASE LBFGSLineSearch、
   SciPy L-BFGS-B，需要时用LASP隔离原指令作行为参照。不同时修改history、
   FD、回退与Gaussian，不把history500单独发展成主项目。
4. 再检验GA相对同预算SSW多起点的增量；补必要入口/约束语义，避免新GA算子。
5. VC/RC后置，保留已有代码与证据，不回到Fe7C3单case调参。

## 交付边界

- 普通、constrained、GA入口支持域分别说明；slab/FixAtoms/Hookean已有专门
  路径，不等于任意ASE约束统一支持。
- LS软面预算出口仍与当前严格收敛政策有差异；不得把line_search_failed、
  后端错误或未评估trial当作成功，不放宽真实起点/最终证书。
- 原版LS镜像候选以平方距离改善>.001替换，当前ASE MIC不是逐项parity；
  原版容差是经验数值规则，保持独立记录，暂不改默认。
- 每次反编译须明确将改变哪条状态转移；已知数学怪癖不自动吸收。
- 调参与最终评估分开。没有GM真值不报告GM命中；力小不代表新盆地或稳定分子。
- 本次仅更新计划，没有启动新PES/GPU/HPC任务或改变算法默认。

## 新增复杂案例的实际进展

见[三体系12配置结果](2026-09-12-expanded-case-results.md)：水15/GFN2、
bicyclobutane/GFN2、Cu55/EMT，43,126搜索+74fresh，72观察独立合格、
2fresh SCF失败保留。10条搜索按预算截断、2条CuLS零邻居初始化停止。
水与C4H6未显示一致LS优势；C4H6非初态27观察中10个解离。Cu55 SSW
两seed降低约2.18eV，但raw Cu LS fallback不适用，不能给LS效率排名。
下一步优先有效化学结构/失败成本和元素表来源，不继续为单例调截断。

AlOH固定胞三优化器诊断见[结果与执行边界](2026-09-12-aloh-optimizer-results.md)：
两个真实晶体输入，正式六臂1799请求均被时间截断，无完整外步落点/fresh，
不能给优化器排名。前置中断版本额外574请求也计入总成本2373。先修research
runner的预算退出记录，再按实际吞吐量设计完整外步预算；不据此调fmax或Gaussian。
Cu LS静态长度倍率已由原版11,664个double全为1确认，默认1.975 Å cutoff
不足以覆盖当前Cu55邻居；此为参数适用域问题，不将fallback提升为金属LS默认。

## 用户纠正预算后：独立长预算已完成

用户指出旧预算不足以测出有效结果。新的
`research/ga_ssw/evidence/aloh-optimizer-independent-budget-20260912/` 已冻结并提交
Slurm数组1288809：两初态×三优化器，各50外步/50000搜索+51fresh请求；
每臂60分钟搜索+5分钟核验，单V100/CUDA float64，最多2并发，Slurm每臂70分钟。
每臂独立进程与时限，消除共享全组预算造成的顺序偏差。SSW参数保持上一轮值，
CPU试运行不混入GPU效率统计。禁止自动重提交；任务目录有独占执行标记。

六任务现均结束，见[完整结果](2026-09-12-aloh-independent-budget-results.md)。
264146搜索+288fresh请求，288观察独立能量/力/几何不变量检查合格；尚非
288个不同盆地。Safe/SciPy四臂各完成50步，ASE两臂各41步后触及请求上限。
Safe调用更少，但同预算低能发现不占一致优势：初态0 SciPy最低，初态1
ASE/SciPy接近。各自初始淬火可落入不同区域，不能把此端到端比较当固定目标
优化器提速。没有线搜索故障证据，也没有SciPy相对能量提前停止。

因此保留Safe及成熟基线，不调AlOH参数。下一项回到有来源H/C/O LS案例的
跨步响应、软面预算出口和真实落点验证；复杂体系测试预算须覆盖完整外步。

## LS normal iteration-limit exit: implementation and prospective execution

The independent fixed-cell Safe-total path now supports explicit
`LSPrequenchSettings(exit_policy="force_or_step_limit")`; default remains `force`.
Only finite evaluated normal `maxiter` endpoints qualify. Raw nonconvergence is
preserved, true-PES energy response is recorded, and archive force certificates
are unchanged. Unsupported constrained/backend entry points reject before PES.
Checkpoint diagnostics preserve actual qualification and backward-compatible
strict defaults. Verification: 591 passed, 1 skipped (26.84 s). Frozen runner
integration used 8 EMT search + 1 fresh request, accounted separately.

HCO campaign: `research/ga_ssw/evidence/hco-ls-exit-policy-20260912/`.
Source, inputs, all 12 plans and runner frozen before submission. C4H6 array
1290465 uses rush-cpu tasks 0–5; water15 array 1290466 uses huge-cpu tasks 6–11.
Both DSPRHBM, 1 CPU/task, max 2 concurrent each, 110-minute Slurm limit. Actual
startup confirmed for both cases with tblite imported and paid ledgers created.
Each arm: 150 outer attempts, 150000 search requests + 151 fresh, 90-minute
search and 15-minute fresh reserve. No automatic requeue/retry. Source provenance
and exact submission commands are stored with the campaign. These are running
experiments, not a demonstration of LS benefit.

Native soft counter increments per `bfgsdriver` return, not a proven accepted
optimizer-iteration count. Python 50 is a provisional numerical budget, not
native LSoptsoftmax=50 parity. See the counter audit and step-limit design.
Only the LS exit policy differs between the two LS arms for a given case/seed.
Do not shorten budgets or tune controller/bias parameters from live results.

## User backend switch: MACE-MH-1 OMOL (2026-09-12)

At the user's explicit request, the GFN2-xTB arrays 1290465/1290466 were
cancelled; terminal and partial ledgers/checkpoints are preserved, without
retries or pooling their energies with MACE. `campaign-stop.json` and
`sacct-user-switch.txt` record this administrative stop, not scientific failure.

The replacement is frozen in
`research/ga_ssw/evidence/hco-mh1-omol-ls-exit-20260912/`: same 12 inputs,
seeds, algorithm settings and 150-step/150000-search+151-fresh budgets.
Explicit MACE-MH-1 `head="omol"`, CUDA float64, no accelerated equivariance
backend; the runner checks the resolved head before any PES request. The local
weight contains omol and is float64; calculator initialization selected omol
successfully with zero PES calls. SHA256:
`a522eb7f59c7879963d41586528f4980baf33e086c94aa92e3eafdeccad3be47`.
Official model source: https://huggingface.co/mace-foundations/mace-mh-1 .

Submitted C4H6 array1290601 (rush-1o2gpu, tasks0–5) and water15 array1290602
(flood-1o2gpu, tasks6–11), 4V100 single GPU per task, one concurrent task per
array (two total), 110-minute allocation each. Waiting/running/finished states
must be read from scheduler and artifacts, not inferred from submission.
The new PES requires new trajectories and independent final certificates;
removing SCF iteration does not establish physical validity of all proposals.

## Current backend decision: MACE-OMAT-0-small

User explicitly selected MACE-OMAT-0-small for all subsequent tasks, superseding
the MH-1/OMOL choice. Keep this as the research backend unless the user changes
it; general ASE Calculator support remains unchanged. Cancelled MH-1 arrays
1290601/1290602; retained all paid ledgers and checkpoints separately.

Replacement campaign: `research/ga_ssw/evidence/hco-omat-small-ls-exit-20260912/`.
Same original structures, seeds and SSW/LS settings; CUDA float64, no OMOL head.
12 arms, 150 outer attempts/150000 search +151 fresh per arm, 90-minute search
and 15-minute fresh reserve. Source/runner unchanged, weight SHA256 matches the
completed AlOH comparison. Each array runs at most one single-V100 task, two
GPUs total. See submissions.json for actual job IDs and allocation parameters.
Do not merge energies or success rates across the discontinued backends.
