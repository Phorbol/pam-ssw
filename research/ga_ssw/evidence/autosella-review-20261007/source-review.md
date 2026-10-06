# AutoSella 源码审查：作为 PAM-SSW 局部优化器的适配边界

只读源码审查，AutoSella checkout 固定为 `b964218eed6efdac55cb023f5a18170a6aed29e1`（2026-10-07）。审查范围：AutoSella-F 的公开 `minimize_func`、核心最小化路径、`sella_wrapper.py`、分子示例，A/G 的入口差异；对照 PAM `Safe-total` 与 standalone surface。未安装依赖、未运行 PES、未改源文件。

## 结论

**AutoSella 的公开分子入口不能直接作为 Safe-total 的等价后端。**它接受的 `calc(positions_nm)->(E_kJ/mol,F_kJ/mol/nm)` 只是无晶胞的分子坐标接口；内部明确重建 `Atoms(numbers, positions)`，既不携带 `State.cell/pbc/fixed_mask/metadata`，也不接收 ASE Calculator 对象或应力。F/A/G 三份 `minimize_func` 都走此接口。核心 `Sella` 构造函数仍暴露 `optimize_cell/cell_mask/scalar_pressure/smax` 等参数，但本 F 源码快照在 `initialize_pes` 中对 `optimize_cell=True` 明确抛 `NotImplementedError`；不能从签名推断本 checkout 已有可用的变胞优化实现。

**局部复用并非不可能，但分两种成本。**若仅把 AutoSella 最小化当作 Cartesian、无约束分子优化器，包一层受限 evaluator 在形式上可接入，且现有 PAM `Relaxer` / `standalone.surface.quench` 已展示了 ASE callback 适配边界。若要在 SSW 里复用其实际高价值算法（内坐标、模型 Hessian、trust-region、secant history），需要维护自己的 Sella 算法对象/状态桥、坐标与约束投影、评估计数和退出/返回语义；这已经是一个新的优化器后端，而非换一行 Safe-total backend。周期性、固定原子或 cell 优化尤其不能靠其公开入口表达。

现有源码支持“机制可借鉴、需隔离验证”，不支持 AutoSella 普遍优于 Safe-total，也不支持只因其论文/实现组件多便把它们整体复制。若要比较，应先固定同一初态、同一完整 evaluator、相同真实 E/F 请求预算与终止判据；特别记录 F 入口的预处理，因为初态可能在首次计费前被改变。

## 1. 实际入口、单位、回调时序、预算与返回几何

- `examples/optimize_molecule.py` 通过 `entrypoint()` 加载所选实现；示例计算器是 GFN2-xTB。它把位置以 nm 交给 `system.compute`，记录每次回调的能量/力并更新收敛状态，返回 kJ/mol 和 kJ/(mol nm)：[example lines 98-124](../../../../../../pam-ssw-research/autosella-review-20261007/examples/optimize_molecule.py#L98)。公开契约与默认 200 次预算写在 [README lines 48-50, 96-100](../../../../../../pam-ssw-research/autosella-review-20261007/README.md#L48)。
- F 的 `_WrappedCalc.calculate` 将 ASE Å 转为 nm，调用用户 callback，再把 kJ/mol 与 kJ/(mol nm) 转成 ASE eV 与 eV/Å；成功返回后才增加 `call_count` 并保存 `last_positions_nm`：[autosella_f.py lines 8312-8332](../../../../../../pam-ssw-research/autosella-review-20261007/minimizers/autosella_f.py#L8312)。
- F 的入口先预处理位置，再构造无 cell 的 ASE `Atoms`，设 `internal=True, order=0, allow_fragments=True`；`irun(fmax=0, steps=max_force_calls-1)` 每次 yield 后检查外部 `converged()`；最终返回 wrapper 保存的最后一次**已评估**位置和 callback 数：[autosella_f.py lines 9114-9158](../../../../../../pam-ssw-research/autosella-review-20261007/minimizers/autosella_f.py#L9114)。ASE 的优化器 step 中 `kick()` 设置候选位置并立即取新的能量、梯度，随后更新 Hessian/trust radius：[lines 8134-8155, 6143-6167](../../../../../../pam-ssw-research/autosella-review-20261007/minimizers/autosella_f.py#L8134)。
- 预算边界：F/A/G 的 wrapper **没有**在 calculator 层拒绝超额请求；只用 `steps=max_force_calls-1` 作为循环步数上限（A/G 可见 [A lines 6908-6931](../../../../../../pam-ssw-research/autosella-review-20261007/minimizers/autosella_a.py#L6908)、[G lines 7502-7540](../../../../../../pam-ssw-research/autosella-review-20261007/minimizers/autosella_g.py#L7502)）。因此它们不像 `sella_wrapper.py` 那样由 wrapper 硬性拦截预算：[wrapper lines 32-49, 64-76](../../../../../../pam-ssw-research/autosella-review-20261007/minimizers/sella_wrapper.py#L32)。在普通入口单步一次 E/F 的路径下，意图是初始一次加 `max_force_calls-1` 个 step；但“硬预算”不是 F/A/G wrapper 自己保证的契约。遇到另行触发 calculator 的路径时，不能仅凭参数名保证上限。
- `converged()` 的调用发生在 optimizer `irun` yield 后；示例的状态在 calc callback 内更新，故检查看到的是刚结束的那次评估。首次起始能否在任何 step 前作为 yield 暴露，需结合其父类 ASE `irun` 版本确认；源码本身未把“初始已收敛则零步退出”写成 F 的保证。
- **F 入口改变初态。**调用顺序是 `_prerelax_rotors`、`_pyramidalise_centres`、`_break_start_symmetry`、`_dock_start`，之后才第一次 callback：[lines 9116-9132](../../../../../../pam-ssw-research/autosella-review-20261007/minimizers/autosella_f.py#L9116)。对 XH3 转子和特定平面三配位中心做类别几何变换、对多片段作确定性刚体扰动，以及对部分多片段做经典势 docking，都是模型/化学类别相关的起点启发式；例如对称扰动参数固定为 0.02 Å 和 0.02 rad、seed=0：[lines 8334-8339](../../../../../../pam-ssw-research/autosella-review-20261007/minimizers/autosella_f.py#L8334)；相关选择性过滤和移动上限见 [lines 9082-9110](../../../../../../pam-ssw-research/autosella-review-20261007/minimizers/autosella_f.py#L9082)。这些计算不计入目标 PES callback 次数，但会改变优化初态。因此同样 E/F 预算的 benchmark 不等于从同一初态开始的 optimizer-only 对照。
- 返回几何是最后 callback 真正评估的位置，不一定是 `Atoms` 离开 `irun` 时的位置。F 与 A 的注释明确指出 loop 结束后 `Atoms` 可能位于未计价 proposal；G 当前入口也返回 wrapper 保存值。此设计与 callback 的最终能量/力对应，适合“末次已评估几何”契约。

## 2. F 的关键方法：可借鉴原理与需谨慎的成分

**相对可迁移的数值原理（不是直接收益承诺）：**

1. Sella 的冗余内坐标把键长、键角、二面角映射到 Cartesian 梯度/Hessian；初始 Hessian 由内坐标 Jacobian 投影并由解析/经验模式刚度构造：[F `InternalPES` lines 6181-6221](../../../../../../pam-ssw-research/autosella-review-20261007/minimizers/autosella_f.py#L6181)。几何坐标可以在强刚性键、柔性扭转主导的分子上改善条件数。此依据对分子适用，不自然推广到有周期跨边界、金属键网、表面吸附或 topology 改变的通用 SSW。
2. F 保存最多四组多割线对，筛掉近线性相关或不满足对称 Hessian 一致性的旧对，并对几何相关解析模型传输 secant 数据：[F lines 5749-5758, 5952-6034](../../../../../../pam-ssw-research/autosella-review-20261007/minimizers/autosella_f.py#L5749)。它还把旧 secant 对映射到当前分子的对称像；源码假设真实目标能量在相同核置换与正交点群操作下不变：[F lines 5965-5998, 7093-7135](../../../../../../pam-ssw-research/autosella-review-20261007/minimizers/autosella_f.py#L5965)。因此完整被优化目标也必须满足这些操作。仅化学骨架/模型本身对称不够：固定中心、方向性外场或非对称 Gaussian/LS bias 可能破坏不变性，此时 image secants 不再是该目标的有效曲率数据。有限内存拟合多个已观测曲率方向的有效性还取决于坐标拓扑稳定和局部二次模型近似。PAM Safe-total 已有不同机制：Cartesian 总目标 L-BFGS、正曲率 secant 筛选和 Armijo 回溯，[relax.py lines 202-250](../../../../../../pam-ssw-worktrees/c60-local-defect-qualification/pamssw/relax.py#L202)。两者差异必须通过配对总成本验证，而不能把更大的 secant history 自身当作更好。
3. F 以模型预测与真实能量下降比值更新 trust radius（`kick` 算 `rho`; `step` 收缩/扩张 `delta`）：[F lines 6143-6173, 8187-8223](../../../../../../pam-ssw-research/autosella-review-20261007/minimizers/autosella_f.py#L6143)。模型可信度控制步长具有普适数值动机；F 的步长约束、增长条件与上限则是在它的内坐标和分子任务上设定。
4. F minimization 默认 `_default_kwargs['minimum']` 设为 `method='qn', eig=False`，即默认 minimum 路径不是 RFO：[F lines 7680-7711](../../../../../../pam-ssw-research/autosella-review-20261007/minimizers/autosella_f.py#L7680)。RFO/P-RFO 实现存在（[F lines 7325-7383](../../../../../../pam-ssw-research/autosella-review-20261007/minimizers/autosella_f.py#L7325)），但 P-RFO 默认用于 saddle 配置，不能将它描述成 AutoSella-F 默认分子 quench 的关键优势。

**经验和范围约束：**

- F 的模型 Hessian含键、角、扭转和非局域接触刚度，且会针对离子/线性中心等类别分配经验刚度；`guess_hessian` 注释和实现显露化学分类：[F lines 5335 onward](../../../../../../pam-ssw-research/autosella-review-20261007/minimizers/autosella_f.py#L5335)。F 还为 connected molecule 的最大原子位移设置 `cart_ratio_mol=2.0`，并启用沿步路径修正接触模型的 predictor-corrector：[F lines 7680-7700, 8050-8132](../../../../../../pam-ssw-research/autosella-review-20261007/minimizers/autosella_f.py#L7680)。这两个机制依赖 Sella 内坐标、指数距离接触模型和分子分类型；它们不是 Safe-total 上可独立贴上的通用公式。
- F 的预处理明确改变初态，并将一些已知软模式直接移至某类构象/相位；docking 使用经典 surrogate，仅对特定片段/电荷情形启用。它们可能是研究基准上有效的分子专用初始化器，但不能算成局部优化器本身的数学进展，也不适合未经验证用于真实势能面上需要忠实探索的 SSW walker。
- A/G 的入口表明三份冻结版本并非一个算法加小参数差异：A 另对无片段 connected case 把初始半径设为 0.2；G 按 connected、原子数和特定官能团/化学类别设置多个内坐标刚度开关（[A lines 6913-6917](../../../../../../pam-ssw-research/autosella-review-20261007/minimizers/autosella_a.py#L6913)、[G lines 7507-7518](../../../../../../pam-ssw-research/autosella-review-20261007/minimizers/autosella_g.py#L7507)）。源码 diff 还显示 A 有曲率度量相关调整，F/G 的实现历史不同。不能用“AutoSella”单一名称推断所有变体共享同一参数化或迁移成本。

## 3. 目标、ASE/PBC/约束和架构接口

- AutoSella 核心 `Sella(Atoms, ...)` 的 API 比公开 callback 更广：类签名列出 `optimize_cell`、`cell_mask`、`scalar_pressure`、`smax`，PES 也可读取 ASE constraints；`Internals.merge_ase_constraint` 对 `FixAtoms` 有映射处理：[F class options lines 7714-7779](../../../../../../pam-ssw-research/autosella-review-20261007/minimizers/autosella_f.py#L7714)、[internal constraint handling lines 3170-3195](../../../../../../pam-ssw-research/autosella-review-20261007/minimizers/autosella_f.py#L3170)。但 `minimize_func` 不传这些信息，且该 F 快照的 `initialize_pes` 对 `optimize_cell=True` 明确抛 `NotImplementedError`，错误消息称此 build 不包含 cell optimization：[F lines 7896-7956](../../../../../../pam-ssw-research/autosella-review-20261007/minimizers/autosella_f.py#L7896)。准确结论是：**公开入口不表达 cell/PBC/stress/fixed-mask；class signature 暴露过 cell 参数，但本 checkout 不支持据此调用变胞优化。**源码另有部分依据 `atoms.pbc` 的投影分支；本审查没有验证直接构造带 PBC 的 `Sella(Atoms)` 对任意周期目标是否可用，故不将其扩大宣称为普遍不支持或已支持周期优化。
- 三个 `minimize_func` 都只构造 `Atoms(numbers, positions)` 且 callback 只收位置数组；没有 `cell/pbc/fixed_mask/stress` 参数。因此调用者若把完整 Gaussian/LS objective 封在 closure 中，形式上可以让 callback 调 evaluator，但 AutoSella 自身每次只改变 Cartesian positions；坐标、固定原子与 cell 均不能由该入口正确管理，也无法要求/校验应力。入口也不接受 ASE calculator 实例。[F lines 9114-9142](../../../../../../pam-ssw-research/autosella-review-20261007/minimizers/autosella_f.py#L9114)。
- PAM `State` 明确包含 `cell`, `pbc`, `fixed_mask`, `metadata`，[state.py lines 10-61](../../../../../../pam-ssw-worktrees/c60-local-defect-qualification/pamssw/state.py#L10)。Safe-total 优化 movable Cartesian coordinates，并以整组 evaluator 做 Armijo line search；候选 acceptance 后才更新状态，末态再真实评估：[relax.py lines 669-820](../../../../../../pam-ssw-worktrees/c60-local-defect-qualification/pamssw/relax.py#L669)。其 evaluator 通过模板 State 可保留晶胞/PBC等目标上下文（ASE surface adapter 也保留 work 上的 cell/PBC/metadata，见 [surface.py lines 24-52, 136-153](../../../../../../pam-ssw-worktrees/c60-local-defect-qualification/pamssw/standalone/surface.py#L24)）。当前 standalone surface 的 Safe-total 桥接显式使用无 cell `State` 做连续 Cartesian 变量，属于固定晶胞 quench，不代表联合变胞搜索。
- 在 `standalone/surface.py` 已有通用真/改造表面 adapter 与固定晶胞 quench 接口；函数可传 ASE Calculator-derived surface，并可指定 Safe-total 后端：[surface.py lines 24-52, 97-153](../../../../../../pam-ssw-worktrees/c60-local-defect-qualification/pamssw/standalone/surface.py#L97)。若目标只是“让另一优化器获得同一无约束完整 objective”，接口级 wrapper 可行，不需要重构 Safe-total 架构；但该 wrapper 必须真实统计所有 callback，固定相同完整 evaluator，返回/重新评估 accepted/last-evaluated 的正确点，并明确初态预处理开关。AutoSella 当前公开接口预算只有步数近似而非硬回调保护，难作为严格预算对照的现成后端。
- 若目标是复制 AutoSella 的内坐标曲率和 trust 模型，则要为 PAM 的 `State` 和 complete evaluator 实现独立 mapping：坐标重建、PBC/MIC/周期键定义、fixed-mask 投影、内坐标重建与 Hessian 状态失效、预算拒绝语义、迹线与终态验收。这个接口/状态适配范围足以构成新的 backend / 研究实现，而非低风险粘合层。文档讨论前应先界定想验证的是 Safe-total 的成本/收敛差异，还是 AutoSella 的某个独立数值机制。

## 4. 依赖与许可事实

- README 与 `LICENSE` 声明软件为 LGPL-3.0-only；F/A/G 是修改版 Sella 2.5.0，`NOTICE` 列出上游 Sandia/NTESS copyright 与改动来源：[LICENSE lines 1-12](../../../../../../pam-ssw-research/autosella-review-20261007/LICENSE#L1)、[NOTICE lines 11-37](../../../../../../pam-ssw-research/autosella-review-20261007/NOTICE#L11)。源码复用/分发前需遵守 LGPL 与保留 notices；本结论是文件事实，不替代法律意见。
- `environment.yml` pin Python 3.12、NumPy `<2.5`、JAX/JAXlib 0.4.38、Sella 2.5.0，并包含 ASE、SciPy、Torch CPU/MKL、OpenFF、OpenMM、RDKit、xTB、DFT-D4、ORCA PI 等全仓库 benchmark/data pipeline 依赖：[environment lines 1-43](../../../../../../pam-ssw-research/autosella-review-20261007/environment.yml#L1)。核心 `autosella_f.py` 是自包含 Sella-derived 源文件，但 import JAX/ASE/NumPy/SciPy；若只移植其中核心实现，可以避免整套分子数据工具依赖，却仍须承担 JAX/Sella-derived 代码兼容与 LGPL义务。AutoSella 的 README 建议用整套 Conda 环境，故直接环境复用会带来明显依赖面扩张。

## 建议给主线的决策

1. 当前可接受的决策是：**不把 AutoSella-F 直接指定为 Safe-total 的通用替代实现；可以将它列为分子 Cartesian/内坐标 quench 的候选研究对照。**
2. 若需要最小接口核查/公平 benchmark，先选定不含 AutoSella-F 私有预处理的调用配置或明确承认其预处理是 treatment 的一部分；在相同 Safe-total surface/evaluator 上添加可选的 AutoSella 接口 adapter，并用调用计数器硬停，报告每个目标请求与最终 accepted state 的费用。不要将 optimizer 内部“steps”当作相同 E/F budget。
3. 若比较后要吸收机制，一次只检验一个可辨认假设（例如 PBC-free molecule 上的坐标/metric 对条件数的影响，或 trust-ratio 对失败淬火成本的影响），在当前真实 SSW 目标上做配对初态、配对调用预算的 ablation；在完成验证前不得宣称通用提升。Safe-total 的 line-search 接受语义、PAM 的 trace 与终态 certificate 仍应保留。
4. 周期体系、fixed atoms、表面 adsorption 或 Gaussian/LS 带额外 bias/components 的总目标：采用前应先证明 callback 在每次候选上调用的是同一个完整 objective，并保留 cell/PBC/constraints；当前 AutoSella public entrypoint 不满足这些表达能力。不要用它重定义已有 SSW 的 `State` 或 evaluator 架构。
