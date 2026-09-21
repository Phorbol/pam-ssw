# Broyden 多历史更新：原指令与 Python 重建边界

2026-09-12，root核验。公共Ritz修正及SSW/LS/GA端到端衔接已完成，本项继续
恢复不同数值求解器，不改变公共默认值。

## 探针范围

`research/ga_ssw/probe_native_broyden_full.py` 直接执行上传ELF的BRZERO4以及
INVERS/LUDCM/LUBKS。分配、复制、清零和输出是隔离运行时hook；DGEGV用实际
SciPy DGGEV求解原程序传入的矩阵，保存输入、输出、alpha/beta和INFO。
不是伪造本征值，也不是原MKL逐指令复现；谱阈值附近的控制可能依赖数值库。
未进入LASP主程序、保护逻辑或任何PES计算器。

证据目录 `research/ga_ssw/evidence/native-broyden-full-probes-20260912`：
- n6-seed11：非均匀G0，4次历史更新；
- n9-seed29：非均匀G0，4次历史更新，实际历史删除；
- n9-seed29-spectral：只读增加谱控制记录，返回状态与前一运行相同；
- n6-seed29-quartic：保守非线性力 F=-H x-.2 x³，3次历史更新，用于区分
  在纯线性算例中不可辨识的矩阵转置。

这些是算术/状态验证，不是原子体系搜索效果测试。保存每个版本的执行脚本，
原始数组、输入力和G0均在JSON中。iniangle=.5、langle=0、rotmode=0等调用参数
显式在脚本中；尚未证明这些参数覆盖实际SSW所有CBD调用路径。

## 目前闭合的矩阵关系

设D和U的列分别为已有的归一化DF、U历史，Z_old为对齐且最新列为零的旧Z。
原版差分归一化及历史投影使用块求和半内积q，不能当作物理Cartesian内积。

实测每步WI均为1000；静态初始化常数也为1000。对于此等权分支：

```
A = I + 1000² * FINF,        FINF_ij = q(D_i,D_j)
beta = inverse(A)
Z_new ≈ (1000² * U + Z_old) @ beta
X_new = X + G0*F - Z_new @ q(D,F)
```

最后一式在多步完整输出中得到数值核验。原指令实际先计算BETAQ再递推Z，
其旧历史列在等权下约为beta对应列，最新列为单位列；由于旧Z最新列为零，
得到上述数学简化。它不是相同浮点运算次序；高条件数/大Z的消减误差需报告，
不能只用宽松绝对容差称逐位一致。历史删除时旧Z必须先随对应DF列对齐。

## 广义谱矩阵的符号已进一步辨识

完整调用的SMAT等于 `D.T @ (G0[:,None]*D)`，但最终GMAT不能写成D.T@U。
在未删除历史的已测路径上，令C=D.T@U：

```
GMAT = triu(C,1) - tril(C)
GP = GMAT; SP = SMAT  # 再按当前候选历史删除作对应缩减
```

第二个构造循环在0x6faaa5翻转符号；因此早先只在0x6fa815停止的第一矩阵
probe没有覆盖最终矩阵。非线性算例中C与C.T相差.0037106154，以上公式仍与
实际GP符合约5.6e-17，排除了纯线性情形无法区分的转置候选。
这项三角符号约定不应被自动解释成真实Hessian，也不能直接推广成物理预条件器。

## 真正的历史删除判据

原版读取原始复数alpha/beta，计算最大 `abs(1+alpha/beta)`。阶数小于50且
该最大值严格小于1e7时接受；否则可以尝试删除旧历史后重新计算。
这是单独于打印EIGENVAL的消费者，早期“DGEGV只有诊断用途”的判断已撤回。
详见 `native-broyden-dgegv-spectrum-audit.md`。

n9实例第4次历史添加的最大值为2.7592684e7，尝试删除最旧历史后降到17.70358，
实际阶数4→3。尚需核验多列删除、最小历史触发重启及真实CBD调用标志的全部
边界；不能据此宣称完整原版CBD/SSW逐指令复现。

## 下一步

先形成明确标识的Python历史算术重建，验证多步与删除对齐，随后再判断原版
角度/步长控制如何接入现有dimer壳层。物理Euclidean版本与原版半内积兼容版本
必须分开；是否保留这些数值规则最终依赖跨真实体系旋转和完整SSW对照，
不能因反编译恢复了常数就把它设为通用默认值。

## Independent state reconstruction and offline native replay

`research/ga_ssw/broyden_state_reconstruction.py` now implements the recovered
raw fixed-G0 state, including normalized secants, the signed triangular GP,
ordinary SP, actual SciPy DGGEV and oldest-prefix deletion. Weight, metric,
history bound and spectral limit are explicit caller choices. This is research
code, not a public CBD driver. Zero seminorm and failed DGGEV remain explicit errors. The minimal-history
spectral reset below is implemented from its own native probe; angle branches
are not inferred.

`audit_broyden_state_replay.py` replays native input X/F while carrying Python
history independently, so history discrepancies propagate rather than being
silently replaced with native arrays. Evidence: `native-broyden-full-probes-20260912/root-state-replay.json`.

| Probe | Maximum absolute X error | Native/Python deletion agreement |
|---|---:|---|
| graded G0, n6 seed11, four updates | 9.8530e-9 | no deletion, same history sizes |
| graded G0, n9 seed29, four updates | 5.8410e-12 | one oldest column removed on update four |
| quartic force, n6 seed29, three updates | 2.0393e-11 | no deletion, same history sizes |

These are numerical comparisons, not bitwise equivalence. The n6 history has
large Z values and cancellation; the evidence records both absolute and
relative Z errors and inverse residuals. Native probes substitute SciPy DGGEV
for the embedded eigensolver, so branch agreement is qualified by that backend.
Eight research tests pass, including an independently derived one-dimensional
quadratic first secant, strict history bound, and explicit null-seminorm error.
No new PES calls and no production solver change were made for this recovery.

Next: recover actual rotation caller flags, force scaling and radius restoration
before using this state in a CBD direction solver. Keep the physical Euclidean
variant distinct from the degenerate native compatibility form. Neither raw
state replay nor matrix unit tests establish an SSW efficiency improvement.


### Minimal-history spectral restart

A separate bounded native probe uses G0=1e-8, diagonal quadratic H=(.5,1,2),
and three raw calls. This scale intentionally triggers the branch; it is not a
proposed SSW parameter. Order-one spectral maxima are 5.0171056e7, exceeding
1e7. The executable sets INI=-1 (0x6ff8b1), returns through initialization
(0x6f8d41), resets ITER=1 and computes X+G0F in the same call. Two consecutive
such restarts were observed. See `probe_native_broyden_restart.py` and
`native-broyden-full-probes-20260912/minimal-spectral-restart.json`.

The independent raw state now exposes `restarted=True` and resets its active
history on this branch; all three output coordinates match native exactly.
This closes this particular raw reset, not the still separate angle-dependent
restart rules. Nonzero DGGEV INFO is an explicit unsupported error rather than
being silently interpreted as a native spectral restart.

### Rotation caller flags versus isolated raw inputs

Root verified production ROTMODE=-1 (true); NORDER is output/workspace written
before read; LANGLE has no body read. INIANGLE is copied to CURV_LAST, with no
CURV_LAST body consumer found. The separate force/step-angle restart is bypassed
when ROTMODE is true. This is narrower than implementing the complete caller.

`probe_broyden_rotation_flags.py` now replays the same three archived input
sequences with ROTMODE=-1, LANGLE=-1, output unit6 and deliberately varying
INIANGLE. All 14 output coordinate vectors are numerically exactly equal
to their corresponding earlier false-mode probe vectors, and the n9 spectral
prefix deletion remains present. Evidence: `production-rotation-flags.json`.
This resolves a potential numerical-body mismatch for the tested sequences;
it does not execute rotate_dimer itself or validate a physical trajectory.

The next research direction runner will use these history equations, endpoint
force geometry, native algebraic first-rotation scaling and angular cap, with
explicit Cartesian/native metrics. For comparison with Ritz and standard dimer,
its directly evaluated residual stopping is in common curvature units. This
stopping choice is explicitly independent of native ftol and CBD_PreRot's
budget override; no complete native driver equivalence will be claimed.
