# Constrained SSW 接入 Native LS 的最小方案

本审计只覆盖 fixed-substrate/partial-PBC 的 `run_constrained_ssw`。目标是让
Native LS 复用已经实现的 bond-table/cycle arithmetic，同时保留
`ReducedCartesianChart` 的约束语义。这里是接入设计，不是生产实现，也没有
启动 PES 计算。

## 当前边界

`pamssw/standalone/constrained_reference.py` 的 `run_constrained_ssw` 在初始
真实 quench 后调用 `ConstrainedLSRuntime.initialize_at`，之后由
`prepare`、公共 `_run_reduced_ssw` 和 `update` 完成一轮 LS。当前 runtime
只接受 `paper_reference.LSSettings`，`ConstrainedLSSurface` 将物理 E/F 与
冻结的 `FrozenBondSoftening`/`FrozenPeriodicBondSoftening` 相加。真正的
landing quench 仍通过 `constrained_quench`，其证书只计算 active force。

`pamssw/standalone/ls_native_reference.py` 的 `NativeLSRuntime` 已经复用
`native_ls.initialize_native_ls`、`freeze_native_ls`、
`response_mev_per_atom`、`NativeLSCycleState.advance` 和
`update_native_table`。但它的构造函数立即初始化冻结势，且没有 constrained
chart 的 `prepare`；`update` 假定完整无约束原子，并把 step 定义为已完成
outer attempt 的一基计数。`native_ls._bonds`/`freeze_native_ls` 对带 ASE
constraint 的输入明确拒绝。因此把现有 native wrapper 直接传给
`run_constrained_ssw` 会在类型检查、初始化时机和约束对象三处失配。

公共 `_run_reduced_ssw`（`rc_reference.py`）已经提供需要的生命周期钩子：
`factory` 建 chart，`quench_callback` 做 constrained landing，
`ls_runtime.initialize_at` 在初始 quench 后执行，`prepare` 在每次 climb
前执行，成功的 prepared attempt 后调用 `update`。默认无 LS 和默认
generalized-dimer 分支应保持逐调用不变。

## 最小实现边界

建议只新增一个 `ConstrainedNativeLSRuntime`（放在
`constrained_ls.py`），并在 `run_constrained_ssw` 以 `isinstance` 分流：

1. `ls is NativeLSSettings` 时创建该 adapter；`LSSettings` 继续走现有
   `ConstrainedLSRuntime`。不改变 `run_native_ls_ssw` 或 GA。
2. adapter 的构造只保存 settings、physical surface、fixed index 和
   chart-independent reference；Native 初始化延迟到
   `initialize_at(initial.atoms)`。这可以通过 Native runtime 的一个保持
   默认构造行为不变的 deferred/classmethod 入口实现；不要在 constrained
   入口提前冻结初态。
3. `initialize_at` 对内部 `Atoms.copy()` 清除 constraints 后调用
   `initialize_native_ls`。清除只发生在内部副本：物理 surface 仍接收
   chart 产生的无约束副本，用户的输入对象和 fixed mask 不被修改。
   `_bonds` 的 `N_b` 必须包含 fixed-fixed、fixed-mobile 和 mobile-mobile
   所有合格 bond；fixed 端点不能因 chart 而删掉，也不能将其振幅置零。
   `atom_filter` 仍只表示显式 Native 设置，不可偷偷映射成 fixed mask。
4. `prepare` 复用当前 `ConstrainedLSRuntime.prepare` 的流程，只替换冻结
   势为 Native frozen potential。`ReducedCartesianChart.evaluate` 先重建
   full atoms，再由原始物理 E/F 加 analytic LS E/F，最后返回
   `g_q=-F_raw_active`。因此固定原子可参与 LS 势，但优化和资格判断只看
   active force；不得用 full-force 阈值替代 active-force 证书。
5. true response 必须调用已有
   `native_ls.response_mev_per_atom(E_before,E_after,N)`，其 eV 输入到
   meV/atom 的转换不能在 adapter 中重复实现。`update` 将所选 current
   landing 的新 bond count（仍包括所有 fixed/mobile bonds）传给同一个
   `NativeLSCycleState.advance(self.steps + 1, ...)`，并用 returned table
   调用 `freeze_native_ls`。MC reject、completed failed climb 的计数规则
   以现有 Native runtime 的 `completed_outer_attempts_selected_current`
   约定为准；LS prequench/initial/update 的真实异常保持 terminal。
6. 周期结构使用 `settings.bond_geometry` 原样传递到初始化、更新和冻结。
   `native-mic` 与 `periodic-images` 不混合计数；后者的 canonical image
   records（包括 self-images）就是 `N_b`。非 PBC 走现有 ordinary-pair
   路径。所有 fixed/mobile bond 都保留在 Native potential 中。

不新增 cutoff、幅度、周期、响应或控制器参数，也不改变 Gaussian、dimer、
MC、true-quench 或 direction-fixed 的控制流。`ConstrainedLSSurface`
可以抽成接受两种 frozen potential 的小公共容器，但不需要新的通用 LS
框架。

## checkpoint 合同

现有 `ConstrainedCheckpoint` 的 `ls_settings` 可保存
`NativeLSSettings`；`ls_state` 应带明确 discriminator，例如
`kind='native'`，不能把 native 状态伪装成 paper 字段。native state 至少
包含：

- calculator-free 的 post-initial reference 与 frozen potential（包括
  `bond_geometry`、canonical pairs/image shifts、reference distances、
  strengths）；
- `NativeLSCycleState` 的完整 table、old/new bond count、cycle/ratio、
  frequency/presteps、response notes 和 step phase；
- `steps`、`last_update` 诊断及当前 response；
- 外层已有的 initial/current/best/minima/raw records、next index、RNG、
  evaluation requests、fixed/mobile masks 和 chart reference。

不得 pickle physical surface、calculator 或带 calculator 的嵌套 Atoms；沿用
现有 calculator-stripping copy/save machinery，并在 load 时验证 native
settings kind、bond geometry、fixed mask、composition/cell/PBC/masses、
RNG class、records/cost continuity。恢复时新 surface 仍由 caller 提供，
`checkpoint.current` 是权威可续跑几何。terminal initial/LS initialize/
prequench/update failure 只允许 load 供诊断，run resume 必须拒绝；普通
`completed_with_failures` attempt 仍是可恢复的边界。`steps` 表示新增
attempt 数，而非成功 climb 数。

## 未确定的 caller 约定

需要在实现前由现有 constrained caller 测试固定三点：

- Native response 的 `energy_before/after` 必须对应同一次 prepared
  attempt 的真实物理 E，不能使用含 LS 的 energy；landing 被 MC reject
  时仍使用该 attempt 的 selected current 规则。
- 若 `prepare` 未收敛，现有 paper 路径在写入 paid ledger 后 terminal；
  Native iteration-cap 是否可作为 measured response 不能从
  `NativeLSCycleState` 推断，应继续采用 constrained driver 的显式
  convergence policy。
- `NativeLSSettings.atom_filter` 与 fixed substrate 的关系只能保持独立；
  没有 caller 证据时不能把 fixed atoms 自动变成 Native filter。

## 有界验收

使用真实 EMT 的小周期 Cu(111) 与 Al(111) slab，各固定底层、移动上层和一个
吸附原子，采用已有 qualified 输入和现有 Native/SSW 配置，不引入新阈值。
每个材料只做两步连续运行与 `1+1` 存盘恢复运行；paper LS 可作为接口
对照，Native LS 为目标臂。记录并逐项比较：

- 初始及每次 attempt 的 full geometry、固定坐标、active-force certificate；
- 物理 E/F ledger 的请求数、失败/拒绝付费数和每个 raw record；
- Native `N_b`（分别列 fixed-fixed/fixed-mobile/mobile-mobile）、table、
  cycle phase、response(meV/atom) 和 `steps`；
- continuous 与 split 的第一次 resume 前缀、最终 current/best/minima、
  RNG state 以及 checkpoint load/resume 的零额外初始 quench；
- fixed positions/cell exact、周期边界 bond/image 计数和新 calculator
  fresh E/F 重算。

额外做一个 Cu 与 Al 的有限差分检查：只移动一个 mobile atom，固定端点仍
参与 frozen potential，验证 chart 梯度是 analytic full force 的 active
切片；同时检查固定原子坐标恒定。若 native MIC 与 periodic-images 的
`N_b`、table 或轨迹差异出现，只报告 domain/成本差异，不把它解释成搜索
质量提升。该验收只证明 constrained caller 的状态、力投影和成本账本接线，
不证明 Native LS 的科学优越性或 binary caller parity。
