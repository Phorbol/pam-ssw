# CBD stage 间 Broyden history reset 审查

日期：2026-09-17。范围是冻结 ELF 的限定符号静态审查；未运行 LASP main、保护代码或 PES，也未修改实现。

证据 ELF：`/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp`，SHA-256 为 `bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704`。主要已有证据为 `native-rotation-bias-field-trace.md`、`native-cbd-reentry-frequency-audit.md`、`native-rotation-followup.md`、`2026-09-12-native-brions4-caller-flag-audit.md` 和 `native-broyden-prefix-recovery.md`；反汇编来自外部 `analysis/` 目录。

## 结论

在已核实的调用链上，新的 CBD stage **逻辑上重置 Broyden history**，不会把前一 stage 的 secant history 作为当前 stage 的有效历史继续使用。更精确地说：CBD setter 清零的是 rotation-stage 控制状态；首次 `BRIONS4` 调用由 `rotnum-1=0` 触发 `ITER=1`，随后向 `BRZERO4` 传 `ini=-1`，`BRZERO4` 进入第一迭代分支，使 active history order 为零并重新保存当前输入为 first-stage 的 last state。

这不是“每次 stage 都释放并重新分配所有数组”。若维度不变，`BRIONS4`/`BRZERO4` 的 allocatable arrays 和旧数值可能仍留在静态工作区；但 active `ITER`/history 状态被重新起算，旧 secant 列不应进入新 stage 的数值递推。**同一 CBD stage 内**的后续 outer rotations 则保留 Broyden history，直到普通 history removal 或 stage reset。

## 调用链与 `for_cpstr` 修正

1. `climb_` 的 `run_type == 5` 分支（`0x5cc4c3–0x5cc4cc`）调用 `update_mode0`（`0x5ccb9e`），随后在 `0x5ccba8–0x5ccbc6` 将 `object+0x1b34` 与 `Allopt` 比较。这里的调用参数 `r8d=3`（`ecx=6` 为右字符串长度）表示 `for_cpstr` 的不等比较；因此条件是 `status != Allopt`。
2. 不等时在 `0x5ccbe7` 通过 method-table slot `+0x188` 调用 `set_status`，参数为 `CBD`（字面量 `0x4a42980`）。这只证明条件 re-entry，不证明每个 Gaussian 必然走到该分支；频率仍由外层 `run_type`、mode update 和 convergence 状态决定。
3. `set_status(CBD)` 在 `0x5c0c30–0x5c0c7a` 的 CBD 分支中，`0x5c0c94` 将 `control+4`（`rotstep`）写为 0；随后在 `0x5c0cae–0x5c1224` 保存当前坐标/力快照，在 `0x5c1249–0x5c1314` 写入 `CBD_PreRot`，并在 `0x5c131b–0x5c1325` 将旋转曲率字段设为 1.0。`0x5c1330` 经表项 `+0x1b8` 调到 `unbiasedrot`（`0x5c3fa0`）。
4. `unbiasedrot` 的一次可见 `cbd_rotation` 调用位于 `0x5c40ca`；旋转完成后的 `CBD_PreRot` 判断和 `CBD_biasedRot` 过渡见 `0x5c44b7–0x5c45f7`。`biasedrot` 的一次可见调用位于 `0x5c4cea`。这些是 stage 内旋转调用边界，不应直接当作 Gaussian 频率。

## `rotnum=1 → BRIONS4 → BRZERO4`

`rotate_dimer` 在 `0x6e635a–0x6e639a` 将 `rotnum-1` 的地址作为 `BRIONS4` 第六个（`hist_len`/`input_step`）参数；调用目标是 `0x6f6440`。在新的 rotation stage 的第一次 outer rotation，已恢复的调用状态是 `rotnum=1`，所以该输入为 0。

`BRIONS4` 的关键指令如下：

| 地址 | 事实 | 含义 |
|---|---|---|
| `0x6f6471–0x6f6480` | 检查保存的 X descriptor 与 `3*nions` 是否匹配 | 仅维度变化时进入 alloc/dealloc 路径 |
| `0x6f6510`, `0x6f65b5`, `0x6f6661` | X、F、G0 的旧 allocatable descriptor 可被释放 | 这是维度/初始化路径，不是每个 stage 的必然释放 |
| `0x6f69fb–0x6f6a13` | 初始化 `STEP=1`、`ITER=0` | 只在 BRIONS4 workspace 初次建立/维度重建路径可见 |
| `0x6f6a1f–0x6f6a3c` | 读 saved `ITER`；若 `input_step==0` 取 0，再加 1 并写回 ITER | 新 stage 首次调用得到 `ITER=1`；非零 outer step 延续 saved ITER |
| `0x6f6b3c–0x6f6b4a` | `ITER==1` 时 local `ini=-1`，否则 `ini=0` | 把 stage 首次调用标给 BRZERO4 |
| `0x6f6b8d` | 调用 `BRZERO4` | 将该 `ini` 与 X/F/G0 传入 history kernel |

因此，不能把 `BRIONS4` 的静态数组生命周期误读成 stage 间 history 保留；判定 active history 的是 `input_step`/`ITER`/`ini` 组合。

## `BRZERO4` 的 reset 语义

`BRZERO4` 入口为 `0x6f6c00`。它在 `0x6f6c52–0x6f6c7c` 只按 descriptor 状态和 ndim 决定是否重建 X/F 等工作数组。`ini` 被保存于 `rbp-0x420`（`0x6f6c3d`），随后在 `0x6f8d41–0x6f8d65` 使用：

```text
if ini != 0:
    ITER = 0
else:
    ITER = saved_ITER
ITER = ITER + 1
```

在生产首次 stage call 中 `ini=-1`，所以 `BRZERO4` 的 `ITER=1`。`0x6f8d79` 对 `ITER==1` 跳到第一迭代块 `0x7007ae`。该块从 `0x700804` 起处理 `X_LL/F_LL`、`X_LAST/F_LAST` 与当前 X/F 的保存/移位；随后第一迭代返回使用 `X + G0*F` 的 elementwise 初始步（对应 source line 840）。由 `ITER-1=0`，后续矩阵/历史列不作为 active secant history 参与本 stage。

相反，在 `ITER>1` 的路径，`0x6f8d83` 起计算 `dF=F-F_LAST`、`dX=X-X_LAST`，并在 `0x6f96b8` 调 `inproduct_`、随后构造 `DF/U`；这正是同一 stage 后续 outer rotation 的历史累积路径。已有 4→3 history removal 证据（`native-broyden-dgegv-spectrum-audit.md`）也属于该连续 stage 内行为，不是新 stage reset。

所以本审查支持以下状态转移：

```text
new CBD stage:  rotstep=0 → rotnum=1 → BRIONS4 input_step=0
               → ITER=1 → BRZERO4 ini=-1 → active history order=0
same stage:    rotnum>1 → input_step≠0 → saved ITER/history continue
```

“active history order=0”是由首次迭代计数和第一迭代分支得到的语义结论；数组物理清零/释放并不是该结论的必要条件，也没有在每次 stage 的调用链中被证明。

## 与 Python 实现的差异

当前 Python `pamssw/standalone/broyden_direction.py` 在 `paper_broyden_direction`/`broyden_direction` 每次函数调用内新建 `BroydenState`（`broyden_direction.py:147–151`）。因此，当外层为每个 CBD/Gaussian stage 再次调用 solver 时，Python 也不会跨调用共享 Broyden state；这与 native 的 active-history reset 方向一致。

Python 的 algebraic FACT1 retry 在 `broyden_direction.py:183–198` 重新构造 `BroydenState`，与 native 第一次 outer rotation 的 retry 不能把未改变 endpoint 当成新的 secant observation 的边界一致。两 stage wrapper 在 `staged_direction.py:77–85` 分开选择并调用 main solver；其 pre stage 是独立 dimer solver，不是 native BRZERO4 pre-rotation。

仍有三个不可混同的差异：

- Python reset 是对象级新建 state，native 通常保留静态 allocatable 数组，只通过 `ITER=1/ini=-1` 让旧列失去 active 资格；不能据此宣称 native 做了物理内存清零。
- Python `BroydenState` 的默认公共路径使用 Euclidean 或显式 `native_block_sum` 选项，而 native BRZERO4 的完整矩阵/历史更新、Fortran descriptor stride、`WI`/`DF`/`U`/`Z` recurrence 尚未完全移植；因此这里只比较 stage 生命周期，不比较数值轨迹。
- Python 的 pre/main 调度预算和终止判据是独立 API 设计，不能从它的“每次调用新 state”推断 native 每 Gaussian 的 re-entry 频率。

## 最终边界

已核实的是：满足 `status != Allopt` 的 outer re-entry 可以进入新的 `CBD` setter；该 setter 清零 `rotstep`，并在首个 `rotnum=1` rotation 通过 `BRIONS4`/`BRZERO4` 重置 active Broyden iteration/history。尚未核实的是该条件在每个 Gaussian 的实际访问频率、所有 alternate `rotate_dimer` 分支是否同样传入 0、以及 native arrays 的完整 deallocation 时序。

因此建议在 parity 说明中使用“每个新 CBD stage 逻辑 reset Broyden active history；同一 stage 内跨 outer rotation 保留；底层 workspace 可能复用”这一表述，不使用“每个 Gaussian 都重新分配/清零全部 Broyden 数组”。
