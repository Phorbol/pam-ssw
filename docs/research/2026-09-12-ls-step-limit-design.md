# LS软面预处理正常步数出口：实现与验证约定

要补全的是正常数值预算出口，不是将预淬火失败一律忽略。
已恢复原版判断 `force_measure < ftol OR counter >= LSoptsoftmax`，正常出口
存储 `1000*(E_after-E_before)/N`；解析器缺省LSoptsoftmax=50。
原始证据见`2026-09-12-native-ls-prequench-stop-audit.md`及对应隔离指令产物。
进一步审计确认该计数器在每次bfgsdriver返回后递增；它不是已确立的accepted
iterations映射，原版力测度也不同，因此不称逐步parity。见
`2026-09-12-ls-prequench-counter-audit.md`。

在LS势P固定时，有限次优化提供一个被实际求值的几何R_k。真实能量响应
`(V(R_k)-V(R_0))/N`仍有明确定义，不要求R_k严格满足`grad(V+P)=0`。
但R_k不能被称为软面极小值，改变它会改变后续方向/逃逸提议和自适应强度；
这不是可无条件证明有效的加速。最终归档仍须去掉所有人工偏置并通过真实力证书。

有界接口：`LSPrequenchSettings(fmax,steps,exit_policy='force')`保持严格默认，
显式`force_or_step_limit`仅在Safe-total可靠telemetry表明正常maxiter、到达
预算且终点有限时允许继续。保留`soft_quench.converged=False`，实际出口记录
为`step_limit`，不是伪造收敛。SCF失败、请求/墙钟截断、线搜索失败不属于
这个出口。暂不支持该选项的后端/约束入口须在oracle调用前拒绝。
旧checkpoint缺字段视作force；改变策略不可静默续跑。

受控真实实验：C4H6/bicyclobutane（ASE G2）和上传水十五聚体，GFN2-xTB，
seed11/29，三臂SSW/LS-force/LS-force-or-step-limit，共12配置。
LS两臂只有exit_policy不同，软面fmax=.1、steps=50，真实fmax=.03；其他
参数继承前次同案例实验，不根据本次结果调参。50在本实验中只是明确暂定的Python迭代预算；原版数值50不能证明它们
等价，更不是普适最优值。若观察到策略实际触发，后续需在独立验证中检查
25/50/100迭代的敏感性，而非挑出表现最好的值后宣称通用。每臂150外步/150000搜索+151fresh，独立90分钟搜索与
15分钟核验余量，避免旧共享短时限。target为C4H6 700meV/atom、水20meV/atom，
H/C/O查表沿用已有来源。这不能替代其他物质类别或其他能量后端验证。

必须统计实际force/step-limit出口、真实响应、跨步强度变更、完整搜索/失败
成本和独立核验后的低能落点。若政策未触发，结论是本批未测到增量作用；
不事后缩小steps制造优势。若仅增加解离/高能落点或损害同预算发现，则保持
默认关闭，不用更多阈值修补。若触发且有效，仍需独立案例和seed确认才考虑默认。

本实验与已完成AlOH优化器对照分开，不再扩展AlOH调参；金属Cu LS原始fallback
不进入此验证矩阵。实现/接口回归与科学有效性分别报告。

150外步的设计目的还包括跨过默认100步控制器边界，检验此前短轨迹未覆盖的
更新频率分支；不改变presteps/frequency/cycle/ratio来强行触发。若请求或模型
失败先阻断，保留该未覆盖边界，不把计划长度当实际验证长度。

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

### First completed arm (interim, no LS comparison)

C4H6 SSW seed29 completed 32 outer attempts and stopped on attempt 33 with
`true_quench: CalculationFailed: SCF not converged in 250 cycles`. Cost is
27119 search requests plus 33 independent fresh attempts; 31 passed the
energy/force/invariant certificate and 2 failed SCF. Offline HCO-table-plus-0.1 Å
connectivity marks 20/33 stored observations connected and 13/33 fragmented.
This is an empirical graph diagnostic, not bond-order or basin certification.
All costs and failed structures remain in the denominator; no retries and no
parameter change. Remaining arms continue unchanged. Evidence:
`bicyclobutane-ssw-seed29-geometry.json` inside the campaign directory.

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

Startup verified: both replacement arrays run on Tesla V100-SXM2-32GB and
produced finite first E/F evaluations from their frozen source. Old xTB ledger
retains 109915 search requests plus 33 fresh attempts; integration EMT 8+1
is separate. Old denominator: 1 completed arm, 4 user-stopped, 7 unstarted.

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
