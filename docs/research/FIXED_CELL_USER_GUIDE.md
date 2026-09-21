# 固定胞 SSW / LS：当前可用版本

目标是用统一ASE能量/力接口探索可靠的低能结构。当前是研究版本：核心搜索独立
使用Python运行，不调用LASP/Java；尚未证明跨体系优于成熟全局优化方法。
当前工作分支为`research/ga-ssw-behavior-parity`，包含未提交研究改动；仅报Git HEAD
不足以复现实验，应使用对应实验保存的源码。通用Calculator接口保留，后续研究
后端按具体固定协议选择：材料主线保留MACE-OMAT-0-small，C60已按用户决定改用
MACE-MH-1/omol；CUDA float64。当前运行与模型边界以[主线记录](MAINLINE.md)为准。

## 选哪个入口

| 需求 | 入口 | 当前边界 |
|---|---|---|
| 无约束团簇或固定胞三维周期体系 | `run_ssw` + `SSWConfig` + `ASESurface` | 固定cell，能量/力后端；不等于变胞搜索 |
| 在同一SSW流程上加论文LS | `run_ls_ssw` + `LSSettings` | 显式提供有依据的pair表与目标响应 |
| 使用反编译恢复的LS强度更新 | `run_native_ls_ssw` + `NativeLSSettings` | 原版启发的独立实现，不是完整原版时序复现 |
| FixAtoms、Hookean或slab | `run_constrained_ssw` + `ConstrainedSSWConfig` | 专门入口；不承诺任意ASE约束 |
| GA、变胞、刚体 | 各自实验入口 | 本阶段不作为固定胞SSW/LS默认组合 |

Safe-total是目前研究配置的局部优化器；ASE LBFGSLineSearch和SciPy L-BFGS-B
保留为基线。Safe-total在已有AlOH实验中调用更少，但没有一致低能发现优势。
不要把history500、PAM adaptive Gaussian、两阶段rotation等选项同时打开进行比较。

方向退出可显式设为`SSWConfig(rotation_exit_policy='force_or_budget', ...)`。
它只放行已求值且有限的`budget_exhausted`方向，记录`rotation_converged=False`，
并继续Gaussian爬坡；不是放宽真实落点力阈值。默认`force`仍要求方向残差达标。
这里`force`是旋转退出策略的名称，检查量仍为`rotation_tol`对应的HVP残差
（eV/Å²），不是原子力`fmax`，也不是原版的`ftol`。
`subspace_exhausted`、未知原因和数值异常不放行。此选项适用于`run_ssw`及调用它
的LS流程，当前不用于独立`atomic_climb`/变胞分块入口；不支持的入口显式拒绝。
AlOH26/brookite48已观察到预算放行后得到不同的力合格结构；尚无普遍效率优势，
仍是显式实验选项。见[材料结果与适用边界](2026-09-17-rotation-budget-exit-results.md)。

## 如何开始

复用已经冻结的案例配置，避免误用代码默认值。下例加载AlOH已执行过的设置；
这是该案例的参考设置，不是通用推荐参数。运行前在计算节点选择相应资源。

```python
import json
from pathlib import Path
import numpy as np
from ase.io import read
from mace.calculators import MACECalculator
from pamssw.standalone import ASESurface, SSWConfig, run_ssw

# 位于研究checkout根目录；这里只读取已有配置与输入。
case = Path("research/ga_ssw/evidence/aloh-optimizer-independent-budget-20260912/frame0-safe-lbfgs-total")
plan = json.loads((case / "plan.json").read_text())
atoms = read(case / "inputs/aloh.add.arc", index=0)
config = SSWConfig(**plan["config"], quench_optimizer="safe-lbfgs-total")

# 以下会启动实际计算；本段展示公开API，正式等预算评估使用下述runner。
calculator = MACECalculator(model_paths=plan["model"], device="cuda",
                            default_dtype="float64", enable_cueq=False,
                            enable_oeq=False)
result = run_ssw(atoms, ASESurface(calculator), steps=plan["outer_steps"],
                 config=config, rng=np.random.default_rng(plan["seed"]))
print(result.status, result.evaluation_requests, len(result.minima))
```

完整且实际执行过的入口是`research/ga_ssw/run_aloh_optimizer_comparison.py`
的`run_arm`，其中展示了读取输入、构造Calculator、配置SSW、记录失败和独立复核。
请使用对应冻结目录的runner复现实验，不对已执行目录再次运行。

## 三个必须区分的限制

- 外层`steps`：尝试多少次盆地逃逸；不等于优化器迭代或计算调用数。
- `relax_steps` / `fmax`：每次真实面淬火的迭代上限及最大原子力阈值。
  `bias_fmax`独立控制含Gaussian的内层优化。
- `LSPrequenchSettings(fmax=.1, steps=50, exit_policy='force')`：LS软面准备。
  `.1 eV/Å`和50次为当前对照配置，不是通用最优值；50是Python优化迭代，
  不能等同LASP的50次driver返回。

`force_or_step_limit`仅在无约束固定胞Safe-total路径中作为显式实验选项：正常
达到迭代上限且终点有限时继续；不会把该点标为收敛，不接受线搜索或后端失败。
水15已证明它能解除部分早停，但尚无一致搜索优势，默认仍是`force`。

## 什么算成功

程序结束、接受一步、归档增加和小力都不单独构成成功。至少检查：独立E/F复算、
组成/约束/固定cell、结构物理合理性，并用适用的结构比较判断是否发现不同盆地。
同预算比较必须计入初始化、失败、搜索和复核开销；不同模型能量不能合并。
本轮C4H6存在直接MACE可复现的异常低能结构，已停止将其原始best energy用于排名。

下一项交付是材料案例的公平效果验证，而不是增加入口。当前决定与证据见
[MAINLINE](MAINLINE.md)和[OMAT有效性诊断](2026-09-14-omat-validity-diagnosis.md)。
