> 原始报告位于 `/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/analysis/ssw-kernel-comparison.md`；下文的原资料相对路径按该外部研究目录解析。

# LASP SSW kernel 与 PAM-SSW 的可复用边界

日期：2026-09-09。只读静态审查；没有运行优化、提交任务或修改共享实现。PAM 工作树 `/home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity`，HEAD `e5de210faee8b1f82a16f402ac3197801a8c51a8`；本报告涉及的 `pamssw` 文件为本次读取的现场文件。原版证据根目录为本报告上两级目录，ELF 为 `GA-SSW_program/lasp`。本文的源码行号由 ELF DWARF 得到，**不代表已获得原始 Fortran 源文件**。

## 结论

可以复用 PAM 的 ASE 势能/力适配、结构容器、调用计数、Gaussian 数学原语及独立终点验证基础设施。不能把现有 `SSW` 配置或 Gaussian 偏置的相同名称当作原版 kernel 等价性：方向旋转、Gaussian 高度更新、步长更新、逃逸停止、Metropolis 状态以及 LS 项均须单独恢复/对照。特别是原版最新 Gaussian 的高度依据合力与方向夹角更新；PAM 依据曲率目标赋值并通过模型误差控制调整，两者不是同一计算。

来源级别：A = 本次直接核对 Python/汇编及 DWARF；B = 已有论文全文说明；U = 尚未恢复，禁止作等价性承诺。

## 1. Deform：投影 Gaussian 形式可复用，方向生成不可直接复用

### Gaussian（A）

原版 `ssw_fixlat_mp_addgaussian_`，入口 `0x5cda70`：`kernel-ssw_fixlat_mp_addgaussian_.asm` 的 `0x5cdf20–0x5ce048` 累积位移与保存方向的点积；`0x5ce073–0x5ce0bd` 对该标量平方、除保存宽度平方、乘 0.5、改符号后调用 exp。常量地址 `0x4a45e40` 经 GDB 静态读取得到 0.5。DWARF 将 exp 定位到 `Class_ssw.F90:1215`。不是各原子距离平方直接求和的各向同性 Gaussian。

因此可核对的标量形式为：

`p_j = (R - R_j) · n_j`

`D_j = exp[-p_j²/(2 σ_j²)]`

`V_j = W_j D_j`, `F_j = (W_j D_j p_j/σ_j²) n_j`。

最新项分支 `0x5ceb34–0x5ceb92` 再次计算指数及 `p/σ²`，随后将它们以 `d1,d2` 传入 `set_thisgaussw`。PAM `pamssw/bias.py:28–47` 明确计算相同投影 Gaussian 的能量与**梯度**（ASE 力需取负号）；`hvp_contribution` 给出解析偏置 Hessian-vector 项。复用前仍要核对原版方向规范化、坐标排列、周期分支与中心更新时机，尤其不能自动把 PAM 的 MIC 分支宣称为 LASP 的周期原式。

### 软方向（A/B/U）

正式论文 `literature/GA-SSW-user.txt:162–176`（Liu, Liu, Shang, 2026, DOI `10.1021/acs.jctc.6c01078`，Section 2.2）说明：从归一化随机方向出发，biased rotation 向 Hessian 的小特征值方向软化，**不保证最低特征向量**；沿更新方向连续添加 Gaussian 并在变形 PES 上产生局部极小值。

二进制调用证据：

- `unbiasedrot` 在 `0x5c40ca` 调用 `newssw_basics_mp_cbd_rotation_`，DWARF `Class_ssw.F90:798`。
- `biasedrot` 先在 `0x5c4c45` 调用 `add_rotation_bias`，再在 `0x5c4cea` 调用同一 `cbd_rotation`，DWARF `Class_ssw.F90:857`。
- `cbd_rotation` 在 `0x6e55f6` 调用 `newssw_basics_mp_rotate_dimer_`，见 `kernel-cbd_rotation.asm`。

这条真实调用链支持“在方向旋转力中加入偏置后旋转 dimer”，不能仅凭共同使用 HVP 就等同于 PAM 的候选方向池/评分/Krylov 选择。PAM `walker.py:3205–3285` 调用配置化方向选择，含 anchor、Krylov、oracle 等可选路径；`3475` 起初始化方向上下文。最小 parity port 应提供独立原版旋转策略；可以共用 E/F evaluator，不默认复用方向选择器。原版 rotate_dimer 的全部更新式、有限差分点位置、归一化/投影顺序、随机流和停止判断本次未完全恢复（U）。

## 2. Continue：高度的角度控制与 PAM 曲率控制不同

### 原版最新 Gaussian 高度（A，静态推导）

`newssw_basics_mp_set_thisgaussw_` 入口 `0x6e1730`，DWARF `newssw_basics.F90:166`。`analysis/selected-dwarf.txt:177704–177815` 给出按序参数：

`na, d1, d2, fa0, fa2, n, e2, w, e, anglef_n, maxw, step, scalefact0`。

首个循环 `0x6e17f4–0x6e1812` 的可读代数是：

`fa0 <- fa0 - fa2 + (d1*w*d2)*n`

同时 `e = e2 + d1*w`。对该合力计算 `acos((fa0·n)/||fa0||)*180/pi`，见 `0x6e1cba–0x6e1cf7`。门槛是二进制常量 **87 度**（地址 `0x4a4ccb0`，见 `kernel-static-probes.txt`）。若角度不大于 87 度，直接返回原 W。

若超过门槛，撤销本轮所加项并尝试新 W。`0x6e1fce–0x6e1ffe`（DWARF 行 190 起）给出：

`scale = scalefact0`（进入循环前）

`w_new = min(w_old*scale, w_old + 2)`

`scale <- scale*step`

重新计算合力/能量/角度。`0x6e257b–0x6e258a` 表明：若 `w > maxw` 则停止，否则角度仍大于 87 度时继续。因此 **maxw 是循环终止检查，不是每次 min 裁剪到 maxw**。常量 2 来自 `0x4a4ccb8`，不是本项目参数建议；其物理单位必须随 LASP 能量单位确认后再映射。

这些是汇编和参数位置给出的实际代数；尚未对整个函数做执行级逐点等价测试。特别是零合力、acos 数值越界和特殊 W 初值不要未经 oracle 即改成新回退规则。

### 原版初值（A）

`set_initial_gaussw`，DWARF `newssw_basics.F90:207`，见既有同名 asm：读取 `para.maxw/w_initial/w_step/w_scalefact`。结构偏移分别为 `0x2dce0/0x2dce8/0x2dcf0/0x2dcf8`。还读取 `control` 偏移 `0x40` 的符号及另一个初值字段。`w_level`（`0x2dcd8`）等于 1 时将 `w(1)` 设成 5.6；等于 2 时将 `w(1)` 以及存在时的 `w(2)` 设成 0.5。不能把这些分支常数误报为所有体系统一默认值；运行配置和 preset 选择仍决定是否走到这些分支。

### PAM（A）

- `walker.py:3598–3607`：默认能量尺度步长 `σ = clip[sqrt(2 E_target/max(|κ|,1e-4))*σ_scale]`；另有 RMS/energy-bounded 路径。
- `walker.py:3680–3682`：`W = clip[σ² max(κ_inner + κ_target,0), W_min,W_max]`。
- `walker.py:3366–3386`：乘当前 `weight_scale`，在当前位置存中心，显式位移 `σ*n` 后进行偏置局部优化。
- `walker.py:3446–3459`：按真实 PES 能差、局部二次模型误差、边界活跃比例等更新步长和高度尺度。

原版 Gaussian 数学形式可以直接比，但上述“赋高→更新→停止”的策略不能调用 PAM 控制器后声称复现。

## 3. Continue → Quench：结束一次 escape 与优化收敛是不同判断

原版 `climb` (`0x5ca8e0`) 调用 `addgaussian`/优化/判断的部分是 Fortran type-bound 间接调用。本次没有完整恢复 vtable 对应，不能把所有间接调用靠名字猜完。已保存相关 asm 供进一步静态解析。

`climb_convg` (`0x5cd130`) 明确不是简单“跑满固定 Gaussian 数”判断：

- `0x5cd1ca–0x5cd238` 从数组计算最大绝对分量；后面与 `f_maxlimit` 比较。
- `0x5cd27f–0x5cd2d1` 计算并更新 `control.maxe_height`。
- `0x5cd4eb–0x5cd548` 使用 `para.e_maxlimit (0x2dd38)`、`e_maxlimit_gm (0x2dd40)`、`f_maxlimit (0x2dd48)`，以及是否是首个 Gaussian，组合退出/限制标志。
- `0x5cd5fe–0x5cd658` 又把 Gaussian 数等于 `para.ng` 加入退出组合。
- 后段还引用 `stop_gauss_lowlimit/medlimit/stop_anglefn` 等参数对应域；这些全分支逻辑本次未完整恢复。需区分“限制触发/失败”与“成功离开盆地”，不能用任一 true 返回代替成功。

PAM `walker.py:3205–3473` 中循环上界是 `max_steps_per_walk`，提前结束主要来自结构无效、非有限能量和总位移裁剪；没有原版同构的逐 Gaussian `climb_convg` 状态机。之后独立 true-PES quench 的证书契约可以用于物理数值验证，但与原版 kernel 停止含义不同。本报告不重复另一审查的 ftol/quick_setting/local optimizer 恢复工作。

## 4. Select：kernel MC 与 GA/DCCD 外层选择必须分开

原版 `make_decision` 在 `0x5d195c` 调用 `ssw_commsub_mp_metropolismc_`。该函数有静态 `NSAME` 状态，不只是裸 `exp(-ΔE/kT)`：`0x57e877–0x57e899` 根据近等能容差增加 NSAME，减去 `maxtrap`；`0x57e89e` 计算整数幂 `10^(NSAME-maxtrap)`，随后加入传入温度；`0x57e8ac–0x57e8d2` 做单位转换和 exp；接受非近等能转移才在 `0x57e919` 清零 NSAME。精确 eV/K 常数和整数负指数的运行行为应由局部 oracle 验证后再移植。

PAM `walker.py:3540–3557` 的链接受先拒绝 `is_new=False`，然后 downhill 接受，uphill 按配置能量温度 `exp(-ΔE/T_energy)` 接受；不含同构 NSAME/maxtrap 加热状态。原版 `str_select` 还有 rigid/chiral/slab/bond 检查调用（`0x5de080/0x5de372/0x5de3a8/0x5de3f0`），不能将 PAM archive 去重或 GA 的 DCCD 竞争当作其完整替身。

## 5. LS 与变胞边界

本报告未重复 LS 公式恢复；独立 LS 审查的中间证据为 `analysis/ls-pot-bond-add.asm`、`ls-bond-counter.asm`。PAM `softening.py:80–97,124–173` 可见：Gaussian pair term 与 Buckingham-like `strength*exp[-(r-r0)/xi]` 可选，默认宽度/强度来源和动态 scaling 都是 PAM 策略。即便选择后者，也须核验原版逐 pair 振幅/长度尺度/atom_filter、pair 生命周期和应用时机；不能凭“repulsive exponential”归类就声称 LS 一致。

`run_ssw_.asm` 的 `0x530822` 调固定胞 `ssw_fixlat::ssw_move`，`0x535ddf` 调独立 `ssw_crystal_basic::ssw_move`。PAM 的 posterior cell quench 不能充当原版联合变胞 escape。最小版本应先明确限定 fixed-cell parity。

## 6. 最小 ASE 接口与逐阶段原版对照方案（建议，不是已实现）

```python
class KernelEvaluator:
    def evaluate(self, atoms) -> Evaluation: ...  # energy[eV], forces[eV/A], counts

class NativeSSWState:
    # atoms, direction, gaussian centers/widths/weights, rotation history,
    # gaussian_index, climb flags, MC NSAME, exact native option snapshot
    ...

class FixedCellSSWKernel:
    def deform(self, state, random_input) -> NativeSSWState: ...
    def continue_step(self, state) -> NativeSSWState: ...
    def termination(self, state) -> Termination: ...
    def quench(self, state) -> QuenchResult: ...
    def select(self, current, landing, uniform_input) -> SelectionResult: ...
```

接口建议不需要四个独立生产类；重点是保存可观察状态和可注入的随机数，使原版与 ASE 端能在相同输入上对照。`Termination` 要保留原版数值/逻辑 flags，并独立给出 reached limit、failed、candidate ready 等解释，避免一个 `converged` 布尔吞掉不同含义。

依次执行最小对照：

1. **无 PES 静态/代数函数 oracle**：固定 R、Rj、n、σ、W，比较 Gaussian E/F；给定 d1/d2/fa0/fa2/n 测试角度 <87、=87、>87、maxw 前后与 W_level 分支。不需要运行全局搜索即可找到高度更新错误。
2. **direction replay**：截获原版一次随机方向及全部旋转入口/出口状态，使用同一 E/F 后端比较每次方向、力差、角度、范数、步长和评价次数；随机种子相同不保证 Fortran/NumPy 随机序列相同。
3. **one-escape replay**：比较每一 Gaussian 中心、n、σ、W、biased E/F、true E/F、climb flags 和退出原因；在这一层才决定哪些 Relaxer 部分可共用。
4. **quench/select replay**：将已记录候选交给独立恢复的原版 quench；对同一温度、NSAME、能量差、uniform 输入对照接受/拒绝，检查被拒结构是否仍向外层 GA 数据库输出。
5. **有界真实体系端到端**：在明确授权预算后，使用原始水簇/金属簇等合法输入与相同模型；保留全失败分母，报告力调用、旋转调用、偏置优化、quench、LS 构建与时间。逐阶段实现一致不自动证明科学搜索性能。

最终比较应分别列出 original LASP GA-SSW、ASE parity GA-SSW、PAM 原稳定配置；在同 E/F 后端、初态/随机输入、预算和终点验证下做成本与不同有效极小值覆盖比较。保留 mature baseline，不能把更高 E/F 消耗换得的覆盖直接归为效率改善。

## 可重查证据

- 本次新增 `analysis/kernel-ssw_fixlat_mp_*.asm` 和 `kernel-cbd_rotation.asm`：纯 objdump 产物。
- `kernel-static-probes.txt`：只读 GDB 常量/源行查询，没有 run/start。
- `kernel-dwarf-member-offsets.txt`：从既有 DWARF 提取的成员偏移索引，含不同类型相同偏移，使用时必须结合类型上下文。
- `selected-dwarf.txt`、`run_ssw_.asm`、`newssw_basics_mp_set_thisgaussw_.asm`、`newssw_basics_mp_set_initial_gaussw_.asm`、`ssw_commsub_mp_metropolismc_.asm` 为已有证据。
- 正式论文全文已在输入中提供，本次无额外网络文献结论。报告仅对上述静态证据作结论，没有新增科学有效性结论。
