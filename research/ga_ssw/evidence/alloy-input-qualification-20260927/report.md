# Ag30Au30 / Cu30Au30 OMAT 输入资格

## 结论与范围

两份 CC BY 4.0 文献坐标输入均在 `MACE-OMAT-0-small` (`omat_pbe`, float64) 下完成有界数值资格：source 与一次固定 seed 的物种交换初态，各自经 ASE `LBFGSLineSearch` 优化后通过 `fmax ≤ 0.05 eV/Å`，且末态在 3.0、3.5、4.0 Å 的距离图诊断中均为单一连通分量。该结果只说明这两个输入在指定 OMAT 模型下可收敛至数值合格且几何紧凑的结构；不证明 Gupta 模型的最低能结构在 OMAT 下仍是全局最低点，也不证明非局部物种交换算子有搜索收益。

两个 arm 的初态原子坐标相同，但 exchange arm 改变了 Ag/Au 或 Cu/Au 的原子占位，因此它们是不同的化学初态。两臂末态能量和计算成本不能解读为算子效率或普遍收益对比；每个组成也只有一个源结构和一个交换 seed。

## 可追溯输入

来源为 Du et al. (2019), *Theoretical study of the structures of bimetallic Ag-Au and Cu-Au clusters up to 108 atoms*, DOI [10.1098/rsos.190342](https://doi.org/10.1098/rsos.190342)，补充数据 [10.6084/m9.figshare.9164993.v2](https://doi.org/10.6084/m9.figshare.9164993.v2)，CC BY 4.0。SI 原文与 figshare metadata 保存在本目录。两份源能量都是论文 Gupta 势搜索给出的标签，不能与下表 OMAT 能量混用。

| 组成 | SI 行（1-based） | 源 Gupta 能量 / Rsuc | 导出输入 |
|---|---:|---:|---|
| Ag30Au30 | 4–65（header/energy 4–5；坐标 6–65） | −188.373819 eV / 12 of 100 | [`Ag30Au30_gupta_source.extxyz`](Ag30Au30_gupta_source.extxyz) |
| Cu30Au30 | 4451–4512（header/energy 4451–4452；坐标 4453–4512） | −199.346823 eV / 19 of 100 | [`Cu30Au30_gupta_source.extxyz`](Cu30Au30_gupta_source.extxyz) |

Cu 输入的 60 个元素符号和坐标按顺序与 SI 对应块逐项完全一致。Ag 的默认解析路径也与已有 Ag extxyz 的符号、坐标逐项一致。两输入均为 60 原子、1:1 组成、非周期且零晶胞。

## 数值结果

协议和完整逐臂记录见 [`protocol.md`](protocol.md)、[`run-1504820/summary.json`](run-1504820/summary.json) 与 [`run-1504872/summary.json`](run-1504872/summary.json)。模型路径为 `/home/gengjianrui/.cache/mace/mace-omat-0-small.model`，头为 `omat_pbe`；ASE LBFGSLineSearch、history=500、最大 300 步、力阈值 0.05 eV/Å。两次运行均 `status=complete`，实际调用全部完成，远低于每次 1000 调用上限。fresh fmax 由优化完成后的独立 calculator 调用复核。

| 输入 / 初态 | 优化步数 | 实际 E/F 调用（fresh） | 初态 OMAT E (eV) | 末态 OMAT E (eV) | fresh fmax (eV/Å) | 末态最短距离 (Å) | 末态 bbox (Å) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ag source | 10 | 20（1） | −149.610640 | −152.254381 | 0.045250 | 2.774108 | 11.299 × 10.121 × 10.141 |
| Ag exchange | 14 | 28（1） | −150.390806 | −152.876702 | 0.035113 | 2.754624 | 11.266 × 10.082 × 10.145 |
| Cu source | 14 | 29（1） | −179.719640 | −184.375382 | 0.048992 | 2.438897 | 9.398 × 9.287 × 11.049 |
| Cu exchange | 140 | 241（1） | −159.712076 | −178.962468 | 0.047074 | 2.384773 | 9.753 × 9.901 × 10.176 |

总调用数分别为 Ag 48 和 Cu 270，均为实际完成的 Calculator 调用（包含 fresh 复核）。每个末态在 3.0/3.5/4.0 Å 距离阈值下均为单个 60 原子连通分量；这是阈值敏感的几何描述，不是化学成键判据。源结构自身的最短距离分别为 Ag 2.668425 Å、Cu 2.436137 Å。两条臂末态的最终能量及最大原子力均由轨迹和 final extxyz 独立读取复核；与摘要值的浮点差在约 10⁻⁸ eV/Å 以内。

逐臂原始轨迹：

- Ag source: [`trajectory.traj`](run-1504820/source/trajectory.traj)，exchange: [`trajectory.traj`](run-1504820/native_exchange/trajectory.traj)
- Cu source: [`trajectory.traj`](run-1504872/source/trajectory.traj)，exchange: [`trajectory.traj`](run-1504872/native_exchange/trajectory.traj)

此外，各臂的 `optimizer.log`、`input.extxyz` 和 `final.extxyz` 同目录留存。

## 独立核验与限制

- 重新读取两份 summary：状态均为 complete，顶层 started/completed 调用数分别一致（48/48、270/270）；每臂 fresh 调用均为 1，`force_qualified=true`。
- 用 ASE 读取四条轨迹和四份 `final.extxyz`：末帧/终态能量相等；从终态力数组重算的 fmax 与 summary 复核值一致。轨迹帧数为 Ag source/exchange 11/15、Cu source/exchange 15/141，与优化器步数相符（含初始帧）。
- 从 SI 按 Cu 块边界重解析 60 行，验证 30 Cu + 30 Au、元素顺序和坐标与导出文件精确一致。
- 资格通过只针对该模型、输入、优化器与阈值。没有独立 DFT 验证、全局最低点证明、重复种子统计或 SSW/GA 搜索面板；Gupta 源能量不作为 OMAT 目标。Cu exchange 比 source 多用 212 次调用，也不能据此判断交换算子效率：初态占位不同、终态所在局部极小不同，且每类只有一个实例。
