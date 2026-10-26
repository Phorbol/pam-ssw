# Safe-total L-BFGS：获取、原理、使用与参数说明

核查日期：2026-10-27。面向希望阅读、复用或迁移本项目局部优化器的开发者。

## 1. 先读结论

Safe-total 是**标准有限内存 BFGS + Armijo 回溯 + 正曲率历史筛选 + 位移上限**，不是已证明的 SOTA 优化器。它独立于 LASP 二进制，可以通过 ASE Calculator 获取能量和力，也可以接收一般的 `q -> (E, gradient)` 回调。

它不是旧的 `bias-separated-lbfgs`：Safe-total 对完整目标的梯度建立历史，不扣除 Gaussian 梯度，不单独使用 Gaussian 解析 Hessian。`safe` 指具体数值保护，不保证任何目标都收敛。`history500` 只是较长历史配置，不是另一种更新公式，也不是普适最优参数。

## 2. 远端版本与获取

仓库：<https://github.com/Phorbol/pam-ssw>

本次用 `git ls-remote` 实时确认以下远端研究分支存在，指向算法源码提交：

- 分支：`research/c60-local-defect-qualification`
- 固定源码提交：`c57fcb839b49b91990bac9d242351fad82d5b879`
- 当前 `main` 的 `c798ff79aa39924b9bab522ca9cce774725a329e` **不包含这里介绍的 Safe-total 实现**。不要只克隆默认分支后搜索。
- 这是研究实现，不是已经发布的稳定独立优化器软件包。仓库访问权限由仓库设置决定。

```bash
git clone --branch research/c60-local-defect-qualification --single-branch https://github.com/Phorbol/pam-ssw.git
cd pam-ssw
git switch --detach c57fcb839b49b91990bac9d242351fad82d5b879
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Python >=3.11；项目声明依赖 NumPy >=1.26、SciPy >=1.11、ASE >=3.23、PyYAML >=6。使用 MACE 等后端时另行安装相应软件与模型。上面固定提交用于复现源码；本说明文档在其后的文档提交中添加。

### 阅读顺序（固定提交链接）

| 文件 | 应看内容 |
|---|---|
| [pamssw/relax.py](https://github.com/Phorbol/pam-ssw/blob/c57fcb839b49b91990bac9d242351fad82d5b879/pamssw/relax.py#L192) | 常量；`_lbfgs_inverse_product` 双循环；`_accept_lbfgs_curvature`；`Relaxer._relax_with_safe_lbfgs` 主循环 |
| [standalone/generalized_numerics.py](https://github.com/Phorbol/pam-ssw/blob/c57fcb839b49b91990bac9d242351fad82d5b879/pamssw/standalone/generalized_numerics.py#L61) | 最容易复用的平坦坐标 `safe_lbfgs` 回调接口，明确请求预算及失败返回 |
| [standalone/surface.py](https://github.com/Phorbol/pam-ssw/blob/c57fcb839b49b91990bac9d242351fad82d5b879/pamssw/standalone/surface.py#L97) | ASE Calculator 到固定胞优化的接线 `ASESurface` / `quench` |
| [standalone/cell_relax.py](https://github.com/Phorbol/pam-ssw/blob/c57fcb839b49b91990bac9d242351fad82d5b879/pamssw/standalone/cell_relax.py) | 广义坐标变胞使用方式；不要直接拼接 cell 与 positions 后沿用 Å 单位 |
| [tests/standalone/test_generalized_numerics.py](https://github.com/Phorbol/pam-ssw/blob/c57fcb839b49b91990bac9d242351fad82d5b879/tests/standalone/test_generalized_numerics.py) | 数值接口与失败契约 |
| [tests/standalone/test_generalized_lbfgs_memory.py](https://github.com/Phorbol/pam-ssw/blob/c57fcb839b49b91990bac9d242351fad82d5b879/tests/standalone/test_generalized_lbfgs_memory.py) | 显式历史容量及跨入口传递 |

## 3. 数学设计

在一次优化调用内固定目标 U(x)。SSW 内层通常为 U=V+B，LS 打开时还包含相应软化项；无偏置淬火为 U=V。若外层更改了 Gaussian 或 LS 势，就开始新的优化调用；当前实现不跨调用继承拟牛顿历史。

令 g=∇U，保存最近 m 对：

\[
s_k=x_{k+1}-x_k,\qquad y_k=g_{k+1}-g_k,\qquad \rho_k=(s_k^Ty_k)^{-1}.
\]

采用标准 L-BFGS 双循环得到 p=-H_k g，不显式建立完整 Hessian。初始逆 Hessian 缩放有历史时为 γ=(sᵀy)/(yᵀy)，空历史时为 1/70。BFGS 的逆 Hessian 更新形式是：

\[
H_{k+1}=(I-\rho sy^T)H_k(I-\rho ys^T)+\rho ss^T.
\]

实现的保护规则：

1. **曲率筛选**：仅当 `s@y > sqrt(machine_epsilon)*norm(s)*norm(y)` 才入历史。负曲率或近零曲率对被丢弃；不是把负曲率取绝对值，也不是阻尼 BFGS。
2. **步长上限**：将整个方向等比例缩短，固定胞 Cartesian 版确保单原子最大位移不超过 0.2 Å；保留方向而非逐分量截断。
3. **Armijo 回溯**：从 α=1 开始，要求 U(x+αp)≤U(x)+c₁αgᵀp；不满足则 α减半。这里只要求充分下降，没有强 Wolfe 曲率条件。因此需要另外筛选历史曲率。
4. **明确退出**：非下降方向、无效值或线搜索耗尽报告失败；没有自动切换 FIRE、任意扰动或无条件重启。
5. **目标一致性**：E 与梯度必须属于同一个目标。仅在优化器看得到周期偏置映像标记的原 PAM component 接口中，映像分支改变会清空历史；这不是所有入口都自动具备的机制。

每步双循环成本和历史存储约 O(md)，d 是活动坐标维数。广义坐标接口另保存每个接受点的坐标/梯度轨迹，长运行还存在 O(iterations*d) 的轨迹存储成本。

这些是标准拟牛顿方法的工程组合，不能据此宣称新的普适收敛定理。参数与适用域必须通过实际目标检验。

## 4. 最短 ASE 固定胞使用示例

下面是可运行的 Cu 团簇/EMT 接口示例。换成其他 ASE Calculator 时，替换 `EMT()` 即可；计算器必须提供有限且一致的能量和力。示例不负责验证替换后势函数的适用域。

```python
import numpy as np
from ase.cluster.icosahedron import Icosahedron
from ase.calculators.emt import EMT
from pamssw.standalone.surface import ASESurface, quench

atoms = Icosahedron('Cu', 2)
atoms.positions[0] += [0.08, -0.04, 0.02]
surface = ASESurface(EMT())
result = quench(atoms, surface, optimizer='safe-lbfgs-total',
                lbfgs_memory=500, fmax=0.05, steps=300)
print('converged:', result.converged)
print('requests:', surface.requests)
# 输出结构不带 Calculator；在真实势上重新核验。
check = result.atoms.copy()
check.calc = EMT()
fmax = np.linalg.norm(check.get_forces(), axis=1).max()
print('fresh fmax (eV/Angstrom):', fmax)
assert result.converged and fmax <= 0.05
```

注意：`steps` 是接受的优化步数上限，**不是 E/F 调用预算**，线搜索可能需要多次请求。上述 `ASESurface` 不支持 Atoms constraints，会明确拒绝；不能把这个短例子当成通用 FixAtoms/Hookean 接口。约束应使用仓库已有受限几何入口，或自行在下述回调中定义正确的活动坐标与投影。单个 surface/calculator 供串行使用，并行任务应各自持有实例及文件输出目录。

电子展宽后力若对应 free energy，可显式使用 `ASESurface(calculator, force_consistent=True)`；必须确认后端支持，不能混用不同目标的能量与力。

## 5. 脱离 SSW 使用、如何拿走

一般数值优化的最小调用：

```python
from pamssw.standalone.generalized_numerics import safe_lbfgs

# evaluate(q) 必须返回 (scalar_energy, gradient)，不是 forces。
result = safe_lbfgs(q0, evaluate,
    gradient_norm=gradient_norm, step_norm=step_norm,
    gtol=0.05, max_step=0.2,
    maxiter=300, max_requests=3000, lbfgs_memory=500)
if not result.converged:
    print(result.status, result.error)
```

Cartesian 无约束原子例中，两个 norm 可定义为 `lambda v: np.linalg.norm(v.reshape(-1, 3), axis=1).max()`，力转梯度用 `g=-forces.ravel()`。这是接口模板，须提供 q0/evaluate/norms；3000 是示意请求上限，不是最优配置。变胞或内部坐标时，由调用者定义链式法则、度量和物理停止条件；`convergence_norm(q,g)` 可覆盖停止度量，但不得偷偷额外调用势函数。

返回 `q` 是最后接受点；`requests` 计入失败 evaluator 调用；第一次评估失败时能量/梯度可为 None。一般接口捕获评估异常，需检查 `status/error`，不能只看返回对象存在。

**推荐复用方式：固定版本安装整个 pamssw，再只导入优化器函数。** 不需要运行 SSW，不需要 LASP/Java/MACE。当前不是单文件零依赖库：generalized_numerics 从 relax 导入双循环、曲率检查和常量，还包含 dimer 工具，包初始化也会导入其他模块。

若必须移植到另一个项目，最小数学部分是 `safe_lbfgs`、`GeneralizedRelaxResult`、其输入/评估辅助函数，以及 relax 中的双循环、曲率筛选、历史校验与对应常量。按实际引用关系抽取，而非只复制一个函数；保留符号、失败状态、预算及停止语义，并用原有合同测试核对。迁移后是新的实现版本，不能自动继承本仓库性能结论。仓库采用 MIT License，复制/再分发时保留版权和许可文本；模型和 LASP 二进制不因此获得同一许可。

## 6. 参数表：没有已证明的“SOTA 参数”

| 参数 | 当前值/配置 | 含义与依据边界 |
|---|---|---|
| `lbfgs_memory` | None→10；显式500 | 10 是兼容默认；500 在部分开发起点表现较好，非普适最优，额外 O(md) 成本 |
| 空历史缩放 | 1/70 | 固定数值初始尺度；Cartesian eV/Å体系对应 Å²/eV，非从每个体系物理刚度推导；广义坐标需注意尺度 |
| `max_step` | Cartesian主路径0.2 Å；一般接口显式传入 | 位移保护；一般坐标单位由 step_norm 决定，不等于晶胞元素逐项0.2 Å |
| Armijo c₁ | 1e-4 | 无量纲充分下降系数，已有实现选择，未做通用最优证明 |
| 回溯系数 | 0.5 | 无量纲步长减半 |
| 最大试探 | 20 | 每次线搜索的评估上限；最小α为2^-20 |
| 曲率相对阈值 | sqrt(float64 eps)，约1.49e-8 | 相对数值保护，非物理 PES 曲率阈值 |
| `fmax` / `gtol` | 示例0.05 eV/Å | 任务精度而非优化器“性能参数”；SSW内层可使用较松标准，真实落点必须独立核验 |
| `steps` / `maxiter` | 示例300 | 接受步预算；需与总E/F预算分别记录 |

建议初次使用保留默认10作基线，另用500作长历史对照；不要悄悄把较松停止阈值换来的省调用解释成算法优势。bias_fmax 等属于 SSW 驱动的优化任务定义，不属于 L-BFGS 更新公式。

## 7. 当前性能证据能说明什么

固定胞开发对照：4个水团簇起点+1个Cu55起点，同一起点、MACE-OMAT-0-small、fmax=0.03、每臂最多600请求。无偏置淬火结果：

| 方法 | 力合格/5 | 总搜索请求，含截断 |
|---|---:|---:|
| Safe-total10 | 4/5 | 1883 |
| Safe-total500 | 5/5 | 1731 |
| ASE LBFGSLineSearch | 2/5 | 2353 |
| SciPy L-BFGS-B | 1/5 | 2413 |

来源：[固定胞归档报告](../research/2026-09-17-recovered-ls-integration-results.md)。这是积极开发证据，不是独立广泛排名：历史容量等配置未统一，水团簇占多数，OMAT水结果不代表DFT资格，且本次本地该报告链接的原始plan/analysis未找到，不能声称重新核算完成。报告本身已经在远端，不代表所有原始轨迹均已上传。

两个48原子TiO2冻结变胞偏置目标中，Safe500/SciPy500的内层E/F/stress请求分别82/81、107/111，未显示稳定优势。结果记录在另一分支 `experiment/vc-optimizer-frozen-20261012` 的 `research/ga_ssw/evidence/vc-frozen-optimizer-20261012/decision.md`；本说明核查远端的范围仅为上面的固定胞源码分支和main。以上结果不能证明全局搜索成功率提升，也不支持通用SOTA声明。

## 8. 与 LASP 原版的关系

反编译确认固定胞原版链：`BFGSDRIVER → LBFGS → MCSRCH → MCSTEP`。L-BFGS产生方向，MCSRCH管理一维搜索，MCSTEP用能量和方向导数作受保护插值。Safe-total则用Armijo减半回溯。

已检查二进制的GTOL初始化及隔离数值执行使用900；其导数判据形似强Wolfe，但不满足标准c₂<1的前提，不能照搬并宣称正曲率保证。这不是对所有完整LASP运行的参数断言。

当前**没有完整纯Python移植该原版优化器**。`research/ga_ssw/probe_native_lbfgs_emt.py` 通过Unicorn执行LASP机器指令，再由ASE提供E/F，仅为研究对照。它不是上面Safe-total的依赖；`native_stage_quench.py` 也不是原版L-BFGS实现。CBD方向旋转中的Broyden是另一层算法。

详见：[差异与对照](../research/2026-09-17-safe-total-versus-native-lbfgs.md)、[调用链反编译依据](../research/native-local-optimizer-linesearch.md)。

## 9. 复用后的最小检查

```bash
PYTHONNOUSERSITE=1 python -m pytest -q tests/standalone/test_generalized_numerics.py tests/standalone/test_generalized_lbfgs_memory.py
```

再执行第4节EMT例子；这只证明基本接入与数值接口，不证明新Calculator/新体系的科学效果。比较成熟优化器时统一起点、目标、力资格、预算和历史容量，保留失败与截断成本；有偏置淬火与无偏置淬火分别比较。

### 本文示例核验记录（2026-10-27）

在上述源码版本的既有 `mace_env` Python 环境中，设置 `PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`：直接提取第4节代码块执行，返回 converged=True、6次请求，独立EMT复核 fmax=0.0210826938 eV/Å。上述两份pytest文件合计16 passed（1.63 s）；`git diff --check`通过。未验证全新环境安装流程，未重跑历史性能实验；本次只新增使用文档与README入口，不改变优化器。
