# 独立 ASE SSW 家族实现计划

目标：Python 独立执行 SSW、LS-SSW、GA-SSW；用户通过 ASE Calculator 提供势面。运行搜索不得要求上传的 LASP/Java，也不得调用旧 PAM walker 冒充原版。

这是用户已明确授权的架构与开发方向。原程序保留在 research 中作为对照。新的代码在 pamssw/standalone；尚未完成完整搜索时不导出误导性的 SSW/GA_SSW 入口。

## CBD/BP-CBD 上传全文后的优先级更新

用户已提供 CBD 2010 与 BP-CBD 2012 正文，见 `cbd-upload-review.md` 与
`bp-cbd-upload-review.md`。已有 paper_reference SSW/LS 与受限 TYPE3 GA 可运行；
下列原始步骤是架构记录，不代表当前各项均未实现。

主线补独立标准 dimer rotation，保留论文的 rank-one 方向偏置，和现有 Ritz
分别命名与对照；不要求先完成整个 CBD 的 TS translation 或原 ELF BRZERO4。
CBD2010 §2.1 明确其旋转残差求根不保证最低模，因此替换属于算法选择，不能
把相同力差接口当作模式选择等价性。研究参数和真实体系验收要求保持不变。

反编译线继续以 CBD eq7–11 的 Johnson modified Broyden 公式定位历史矩阵，
并用原指令核验；原版块分量和算子等异常只保存在明确标注的兼容模块。
尚未解释的 scratch、历史重启或原版控制分支，不再阻塞数学定义清楚的 SSW
主线开发。下一阶段需同时报告方向求解正确性、模式选择差异及真实逃逸收益。

## 接口与实施顺序

1. `surface.py`：接收任意提供 energy/forces 的 Calculator；显式 energy/free_energy 选择，独立真实势与偏置势评估。ASE 局部优化后检查真实力，记录失败及所有评估请求。第一阶段固定胞、无约束，其他域明确拒绝。
2. `gaussian.py`、`direction.py`：投影 Gaussian 与静态恢复高度控制；有限差分软方向作为独立数值组件。原版 biased rotation、climb termination 和 MC trapping 状态尚需恢复，不以近似组件声称完整原版。
3. `softening.py`：调用者提供元素对键能和成键长度表；冻结邻居/r0，soft-only 预淬火、真实能量响应和跨步幅度更新分别保存。论文模式与发行版调度分开。
4. `ga_operators.py`：原 TYPE3 可确认的刚性分子操作；依次补 docking、父代竞争、描述符/网格及 quick/fine controller。不能用泛化随机交叉代替未恢复算子。
5. 完整 SSW 外层接通后，将 LS 接入相同 escape 生命周期，再接完整 GA controller。所有参数记录出处、单位与使用域；未知规则继续反编译。
6. 固定胞真实体系通过后扩展周期与联合变胞，再处理 RC 链。ASE Calculator 能提供 stress 只是必要条件，不表示联合变胞算法已经实现。

## 验收

- 每个模块先失败测试，再实现与针对性检查；数值测试只检查数学及接口。
- 无 native 二进制的真实金属团簇 EMT、分子势体系先作有界端到端贯通，随后上传作者体系同势面逐状态对照。
- 独立记录初始淬火、软面预淬火、方向旋转、偏置优化、最终真面淬火和失败成本。接口请求计数不冒充后端 SCF 或内部 force evaluations。
- 单步数值一致、完整执行、真实极小值认证与同成本搜索效率是不同验收层；完整 SSW/LS/GA 外层缺失时不宣布任务完成。

依据：本仓库 `ga-ssw-audit/` 中的 SSW、LS/RC 和 Java 审计；ASE 官方 Atoms 和 Calculator 接口。代码评审和跨体系科学验证持续追踪，不改变已有 PAM 默认策略。

### 2026-09-10: prioritize cluster coordinate consistency

The longer paired Cu13/EMT test (two seeds, Ritz/dimer, 15 steps, 14 Gaussians)
produced no demonstrated new structure: all 62 stored initial/landing records
strictly requench to equivalent labelled structures within 2.7e-6 Angstrom
proper-aligned RMSD. Consecutive climbing centers move predominantly through
rigid motion. See `2026-09-10-cu13-escape-diagnosis.md` for costs and failures.
This supersedes workflow success as a rationale for expanding controller tests.
Next resolve cluster gauge/alignment and objective-consistent forces through
paper/native/PAM comparison; do not add heuristic escape rules or claim dimer
alone fixes the kernel. TYPE0 crossover remains a separate geometric module.

### 2026-09-10: coordinate-section experiment yields new local minima

Implemented opt-in `cluster_frame='eckart'`: consistent fixed-section coordinates,
full modified-force pullback, trial-domain checking and failure records; final
true quench remains unrestricted. Final four-run Cu13/EMT experiment plus strict
post-quenching yields 12 fingerprint-distinguishable structures (including
initial), all representatives with positive internal FD Hessians at two steps.
Cost rises from 31,097 to 65,585 search requests; 10/60 moves fail. No lower energy
or accepted interbasin transport is established. Full scope/cost/provenance is
in `2026-09-10-cluster-frame-progress.md`.
Next isolate direction-only versus full-section contributions before promoting
this geometry, and recover native rigid projection arithmetic. Cross-system,
LS and GA effectiveness are still open; do not skip those validations.

### 2026-09-10: direction-only control changes geometry conclusion

The direction-only Cu13 control also yields 12 strict fingerprint groups, with
positive internal FD Hessians. Full fixed-section geometry is not necessary for
escape in this case. Direction-only uses 57,671 requests but fails 31/60 biased
quenches; full section uses 65,585 and fails 10/60. Only five groups overlap.
Native warm-cache setconstraints execution now matches orthogonal projection
for tested nonlinear geometries; the earlier static nonorthogonal suspicion is
superseded. `cluster_frame='direction_only'` is available explicitly, with first
escape regression against both research solver runs. Keep defaults unchanged.
Next investigate local-quench convergence at matched E/F budgets; do not enlarge
caps post hoc or add heuristic penalties. See `2026-09-10-direction-only-ablation.md`.

### 2026-09-10: frozen-quench replay rejects a single-cause optimizer explanation

All 31 direction-only biased-quench failures replay exactly from original stage
starts. With the same 201 E/F cap, LBFGSLineSearch converges 7/31; ordinary LBFGS
0/31. Only three failures exhibit negative curvature history/uphill steps.
The other 24 line-search runs exhaust requests, so switching optimizer alone is
not established as a complete fix. These are modified-surface subproblems, not
new successful SSW moves. Keep the original failure denominator and defaults;
see `2026-09-10-biased-quench-replay.md` before further optimizer changes.

### 2026-09-10: native local optimization and LS initialization are distinct contracts

Native fixed-cell BFGS actually invokes MCSRCH/MCSTEP, unlike our original ASE
LBFGS numerical substitute. Inspected GTOL starts at900, not a verified standard
strong-Wolfe configuration; runtime global overwrites are not exhausted. Recover
caller tolerance/step/history settings before claiming parity. Do not copy the
large constant into the mathematically consistent implementation.

The default no-custom-file LS initialization prefix now executes under an oracle
with real geometric bond counting on C60 and trans-C4H6. The matrix includes
N/Nbond normalization and divides by the raw C-C table value, which cancels in
pure carbon. It is NOT yet the final pair amplitude: downstream amp_c and atom
filters still apply. Therefore raw-table3.44684 versus paper3.61 alone cannot
explain actual bias strength. Preserve paper and release initialization as
explicitly different contracts; complete runtime amplitude/update tracing before
parameter-parity or LS-efficiency claims.

### PAM optimizer correction after reviewing existing implementation

The recent ASE LBFGS/LBFGSLineSearch replay does not evaluate PAM's existing
custom optimizers. The research checkout already contains `safe-lbfgs-total`
and `bias-separated-lbfgs` in `pamssw/relax.py`: both use total-objective Armijo
backtracking, total-gradient descent checks and positive-curvature history gates.
The separated variant learns secants from total_gradient-bias_gradient, uses the
full objective/gradient for steps and acceptance, and explicitly rejects LS.
It does not add an analytic bias Hessian or retain history across separate quench
calls. Historical C60/PdO evidence includes history/scale ablations and supports
safe-total as a comparator, not universal superiority of bias separation.

Next frozen-subproblem comparison MUST include these existing PAM numerical
kernels under identical E/F accounting, rather than treating ASE LBFGS as PAM's
optimizer or inventing a replacement. Reuse only the isolated optimizer as a
clearly labeled comparator; do not wrap the PAM walker and call it independent
SSW. An eventual port/adaptation should preserve component-gradient semantics,
MIC-history reset behavior and explicit LS limits. Existing historical runs are
not a substitute for validation on the present Cu13 Gaussian subproblems.

### Mainline decision after existing PAM optimizer control

User requests focus on delivery. `MAINLINE.md` is the authoritative short next-step
plan. Safe-total solves all31 frozen failed stages under the same201-request cap
(3058requests), versus bias-separated1/31 (6149); all31 Safe-total endpoints pass
fresh modified-force checks. Use Safe-total as the next full-SSW working backend;
stop new optimizer/Hessian branches. Proceed SSW -> LS -> TYPE0 GA -> fixed-cell
release validation. The selected local result is not a full-search efficiency claim.
