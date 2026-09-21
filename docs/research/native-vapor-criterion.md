# `check_vapor_new_` 静态判据审查

本审查只针对保存的原版 ELF，不运行 LASP 主程序、不调用 PES，也不修改核心实现。原始对象为
`/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp`；完整反汇编保存于
`research/ga_ssw/evidence/native-vapor-criterion.asm`，caller 片段见
`/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/analysis/judge-convergence.asm`。

## 已闭合的输入和输出边界

`ssw_commsub_mp_check_vapor_new_` 起始地址是 `0x5950f0`。从入口保存和后续解引用可确认如下局部 ABI：

| 调用寄存器 | 静态确定的含义 |
|---|---|
| `rdi` | 对象指针；其首字段是原子数，`0x595128` 读取 `(%rbx)` |
| `rsi` | 3-double/atom 坐标数组；`0x595877–0x595891` 读取三个坐标 |
| `rdx` | 标量判据指针；`0x595f2d` 读取 `(%rdx)` |
| `r8` | 标量输出指针；`0x596463`/`0x5966c4` 写入 `(%rax)` |
| `rcx` | mode 逻辑参数地址；`0x596016` 测试其最低位 |

`r8` 指向的输出是一个 double。函数没有调用 calculator/PES。但在特定模式分支中，`-0x80(%rbp)`（入口保存的 `rsi` 坐标指针）被重新装入 `rdx`（`0x596062`），随后 `0x5963ea–0x596454` 明确把三个 double 写回坐标槽。因此不能概括为只读几何判断：它至少有一个会改写输入坐标数组的内部路径。该位移的 caller 触发条件和物理意图尚未完全闭合；没有观察到力数组写回或 PES 调用。

## 几何计算

第一段循环明确计算输入点的成对欧氏距离。以坐标差 `dx=x_i-x_j`、`dy=y_i-y_j`、`dz=z_i-z_j` 表示，其指令对应

```text
d_ij = sqrt(dx*dx + dy*dy + dz*dz)
```

证据为 `0x595940–0x595979`（标量路径）以及 `0x5958ce–0x595905`（两路 SIMD 路径）。随后 `0x595f2d` 将调用者给出的 double 载入 `xmm0`；`0x595f7f–0x595fa5` 对候选点再次形成上述距离，`0x595fa9–0x595fad` 与判据比较。`comisd %xmm6,%xmm0; jbe` 在 `xmm0 <= xmm6` 时跳过更新，所以连边条件是严格 `d_ij < vapor_cri`（unordered 也跳过）。满足严格不等式时，`0x595faf–0x595fc4` 更新整数列表和组件标记，形成阈值连通组件。

该边界来自机器级 flags；Python 复刻必须保留严格边界和 unordered 行为，而不是使用 `<=`。

`0x5964a6–0x596659` 遍历组件索引，重新计算组件间欧氏距离（例如 `0x5964e3–0x59650e`、`0x5965f2–0x596625`），并用 `minsd`/比较指令持续归约。该段最终在 `0x596463` 或 `0x5966c4` 将一个标量写给 `r8`；空/特殊路径在 `0x59666c–0x596670` 写零。可以确认“连通组件后的距离归约”，但仅凭该片段尚不能唯一证明输出是最近跨组件距离、第二近距离还是某个分支特定的统计量。这个输出语义是当前关键未闭合点。

坐标和判据在 ELF 中都以 double 直接相减/比较，故单位至多能确定为“坐标数组所用的长度单位”。SI 文件给出 `vapor_cri 1.7`（`/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/literature/ct4c01081_si_001.txt:252–260,275–283`），但 ELF 本身不编码 Å；不能仅由反汇编证明该值在每个归档运行中生效。

## caller 的动作边界

在 `ssw_fixlat_mp_allopt_judge_converg_` 的 `0x5cfb64–0x5cfbd5`：

1. `0x5cfb64` 读取 `para+0x2de60`，与地址 `0x4a45e28` 的常量比较；随后 `0x5cfb70–0x5cfb74` 要求局部 step 大于 `0x32`（50）。
2. `0x5cfb76–0x5cfb9e` 将原子数指针、坐标指针、`para+0x2de60` 和局部输出地址传给 `0x5950f0`。
3. `0x5cfba3–0x5cfbbf` 将返回标量再与 `para+0x2de60` 比较；`comisd vapor_cri, output; jbe` 在 `output <= vapor_cri` 时跳过，只有 `output > vapor_cri` 时由 `0x5cfbc1` 写 `-1` 到 judge 结果。
4. `0x5cfbd5` 后进入已有 judge 的后续状态分支。该 caller 片段没有坐标缩放、力改写、MC landing 删除或“拒绝碎片”的直接指令；其传入的零 mode 也不进入已定位的坐标写回分支。

这里的参数字段是 DWARF 标注的 `vapor_cri`。邻近的 `testb $1,0x2dad8(%r12)`（`0x5cfb0d`）对应 `lts_extra`，不是 vapor 开关；不能把 `para+0x2dad8` 解释成 vapor 控制。`rcx` 是 mode 逻辑参数：judge caller 传入 `0x4a45894`，其最低位为零，进入 `0x596470`；Allopt caller 传入 `0x4a45654`，低 32 位为 `0xffffffff`，进入非零模式。

在非零模式（`0x596016` 不跳转）中，`0x596062` 重新加载原始坐标指针，`0x596329–0x596347` 由 `NSAVE` 方向计算位移，随后 `0x5963ea–0x596454` 对选中的成员坐标逐分量加写。`0x596030` 从 rodata 载入 `0.7`，`0x59604a` 执行与 `vapor_cri` 的乘法，所以可确定出现了形如 `NSAVE * (|NSAVE|-0.7*vapor_cri)/|NSAVE|` 的平移尺度；成员集合和完整触发条件仍未完全恢复。judge caller 的零模式在 `0x596470` 所见为读坐标和输出归约，未见坐标写回；Allopt caller 则确实进入可写的非零模式。

另一个已确认的 caller 是 `ssw_fixlat_mp_set_status_`：`0x5c26bf–0x5c26e0` 用长度 6 的 `for_cpstr` 将状态与 rodata `0x4a45ed4="Allopt"` 比较；成功后，`0x5c2837–0x5c2857` 将 `rdi=[...]` 对应的结构、`rsi=[rdi+0x170]` 坐标、`rdx=para+0x2de60` 判据和 `rcx=0x4a45654` 传入例程。原 ELF 中 `0x4a45654` 的低 32 位为 `0xffffffff`，即该 caller 明确走非零/true 模式，于是例程内部坐标写回路径可达。调用返回后，`0x5c2875` 写控制状态 `-1`，`0x5c287f–0x5c28e0` 构造记录描述，`0x5c28e1` 经 `r10+0x100` 间接调用后续动作；本片段没有直接 calculator/PES 调用，后续间接目标和是否重新求能未闭合。该序列证明存在 Allopt 真模式的原地坐标变换和后续状态动作，但不单独证明“碎片重连”或重新求能。

## `globalcompress` 定位结果

SI 明确列出 `globalcompress 0.0001`，但本次限定静态范围内没有名为 `globalcompress` 的 ELF 符号或直接 consumer。能定位的是 `newssw_basics_mp_compress_mode_`（符号地址 `0x6e36f0`）以及 DWARF 字段 `compress_mode`；这不等同于 `globalcompress`。因此当前不能说 C60 运行读取了该参数，更不能由它推断坐标压缩动作。

## 可证伪的 Python 接口规格（不实现）

若后续要复刻这条边界，最低接口应保持纯几何、无 PES 副作用：

```python
def check_vapor_new(positions, vapor_cri, mode):
    """Return the native scalar and apply the native mode-dependent geometry action."""
```

实现必须验证 `positions.shape == (N, 3)`、使用同一坐标单位，并显式建模模式是否允许原地坐标写回。可证伪测试包括：严格 `d < vapor_cri` 的连边边界；mode=0/1 的坐标写回差异；输出与 native scalar 在相同分支逐点相符；空/特殊路径输出按 `0x59666c` 分支复现。由于最终 scalar reduction、模式描述和 caller 是否触发写回尚未完全闭合，当前不应实现任何 `vapor_cri` 启发式或新增碎片阈值。

## 结论与缺口

已确认的是：`check_vapor_new_` 读取坐标和 `vapor_cri`，按严格 `d < vapor_cri` 构造阈值组件，再做组件间距离归约；其 Allopt 真模式 caller 会触发计算平移并写回部分输入坐标，随后写状态 `-1` 并调用一个尚未解析的间接后续动作；另一个 judge caller 使用零模式，未见该调用点触发写回。judge caller 在 step>50 且返回值严格大于 `vapor_cri` 时也会写 `-1`。未确认的是返回标量的精确定义、移动成员集合、后续间接目标是否重新求能、参数启用链，以及 `globalcompress` 的 consumer/action。因而本证据足以约束未来接口形状和测试，不足以授权 Python 添加 vapor stop、压缩或连接约束策略。

## 隔离 oracle 尝试（2026-09-11）

为执行完整原指令例程，保存了 [probe_native_vapor_oracle.py](../research/ga_ssw/probe_native_vapor_oracle.py)。首次默认环境缺少 Unicorn，后续已使用既有隔离环境完成执行；初始失败记录由后文说明。

补充：首次默认 Python 缺少 Unicorn 的尝试及随后 runtime ABI 失败均已保留在工作记录中。改用既有 `/tmp/pam-ssw-unicorn-probe` 环境，并让 `for_check_mult_overflow64` 在变参计数为 2 时将 `N*4` 写入 caller 的 size 输出、allocator 仅写 descriptor data pointer 后，6 个 2/3/4 原子 × mode 0/1 样例均完成原指令执行。结果见 `research/ga_ssw/evidence/native-vapor-oracle/result.json`：N=2/3 返回值均为 0 且无坐标写回；N=4 返回 `7.2`，mode=0 无坐标写回，mode=1 最大坐标改变量为 `6.01`。这些是合成坐标和 synthetic allocator 下的 instruction oracle，不能代替完整 native caller 或 C60 物理验证；allocator descriptor 仍是最小 stub，未证明所有 Fortran runtime 边界。

随后加入保存的 C60 碎裂 landing（来源为 `research/ga_ssw/evidence/hard-c60-mace-omat-single-step/whole-run/results/paper-seed3/result.json`，仅读取坐标，不重新求能）。在同一 synthetic allocator 下，N=60 的 mode=0 返回 `16.262497884656177` 且坐标不变；mode=1 返回相同标量，最大坐标改变量 `10.676467356171024`。这验证了保存结构可驱动两条指令分支，但数值属于原指令 + stub runtime 的离线算术结果，不能解释为 MACE/GFN2 能量或物理碎裂判定。

## 离线 reference 诊断

当前 oracle 脚本版本冻结于 `research/ga_ssw/evidence/native-vapor-oracle/v1/`；NumPy 诊断脚本为 `research/ga_ssw/analyze_native_vapor_reference.py`，结果为 `native-vapor-oracle/derived.json`。严格 `d<1.7` 的图 reference 在已测 N=4 和保存 C60 case 上与 mode=0 oracle 返回值一致：N=4 为 `[3,1]`、返回 `7.2`；C60 为 `[58,2]`、返回 `16.262497884656177`。这不证明 native mode=0 总是取全局最小跨组件距离；另一个 anchor/多组件构造已显示 native 归约可能先选几何 anchor。C60 mode=1 平移后，组件内部距离最大变化为 0，而所有点对距离最大变化为 `15.072497884656176`，说明写回主要是组件间平移。

## 输入矩阵与 mode=0 reference

`probe_native_vapor_matrix.py` 及其输出 `native-vapor-oracle/matrix.json` 增加了三分量、原子重排、整体旋转/平移和 `cri` 恰等及两侧邻近值。12 个 case（6 几何 × 2 mode）均完成指令执行。mode=0 的 NumPy reference 对所有 case 逐项相符：

- `d=cri` 返回 `cri` 且组件不连通；`d=cri-1e-8` 返回 0；`d=cri+1e-8` 返回该跨组件距离。
- 三分量输入组件为 `[2,1,1]`，返回 `3.2`。
- 原子重排和刚体旋转/平移保持 mode=0 结果（重排例为 `[3,1]`、返回 `7.2`）。

mode=1 是有状态的坐标动作，不能用 mode=0 reference 替代。三分量 case 的 mode=1 返回 `6.01`，与 mode=0 的 `3.2` 不同；这说明该模式会在最终归约前改变几何，不能把所有 mode=1 返回值解释为输入结构的原始最小跨组件距离。当前已闭合的是 mode=0 图和跨组件最小距离；mode=1 的精确成员选择、移动顺序及移动后归约流程仍需继续从 `0x596000–0x596470` 恢复。

追加的 `anchor_vs_global_min` case 为 `[0,10,11,13]`（`cri=1.7`），mode=0 组件 `[2,1,1]`，全局最小跨组件距离为 2，oracle 返回 2；mode=1 返回 10。该对照排除了仅凭先前几何猜测把输出称为“固定 anchor 到其余组件的距离”或“全局最小距离”：两个 mode 的流程不同，且 mode=1 明显先移动/重排后再归约。

坐标写回 hook 显示代表性四原子 mode=1 只写 offset 72/80/88，即 atom index 3 的三个坐标；`[8,0,0]` 被写为 `[1.99,0,0]`，其余原子未写。非共线三分量和刚体变换输入也已加入 `matrix.json`。这闭合了“确实写哪个坐标槽”的事实，但尚未闭合其成员选择在任意组件编号下的通用规则。

后续反事实 MACE 淬火已冻结输入和计划，但当前环境在导入 `mace` 时得到 `ModuleNotFoundError: No module named 'mace'`，因此没有发生任何 MACE/PES 请求。预处理坐标已保存于 `c60-native-vapor-mace-counterfactual/native-mode1.json`，计划和原始 work 分别见同目录 `plan.json`、`input-original.json`；阻塞结果为 `result.json`，物理请求数为 0。不能据此报告淬火能量、fmax 或 landing。

## 反事实 MACE 淬火结果

MACE 阶段现已在指定 `mace_env`、CPU float64、单线程下完成。输入是已保存的原 LS 最后偏置 work 经 native mode=1 离线预处理后的坐标；这不是重新运行 native trajectory。初始 MACE 评估为 `E=-507.2125704671109`、`fmax=5.108869008712894`；Safe-LBFGS memory=400 在 44 次 E/F、18.8135 s、41 iterations 后收敛到 `E=-510.3303865148641`、`fmax=0.009946495140055618`，因此 `DeltaE=-3.1178160477532 eV`。MACE 初始评估另计 1 次，反事实总成本为 45 E/F。

结果文件为 `research/ga_ssw/evidence/c60-native-vapor-mace-counterfactual/result.json`，输入、计划和初始证书在同目录。最终按 C--C 1.74 Å 图得到的组件大小及最小点对距离由离线命令计算保存前述数据；该诊断只是同一 MACE PES 上的结构/力结果，不能证明完整笼、native trajectory parity 或普遍优化收益。

v2 反事实已按正确的 MACE whole-run `records[0].last_atoms` 执行。native mode=1 预处理后的 MACE Safe400 quench 使用 79 次 E/F、75 iterations、32.7013 s，收敛到 `E=-508.82286162350294`、`fmax=0.006001311389608432`；独立 fresh calculator 1 次 E/F 给出同一能量和 `fmax`。总成本 80 E/F（quench 79 + fresh 1）。按原 fresh cutoff `1.6399999618530273 Å`，最终图为单一 60 原子组件，最小点对距离待以结果 JSON 为准。该 v2 才是请求的 MACE whole-run 末步反事实；上一版结果明确无效。
