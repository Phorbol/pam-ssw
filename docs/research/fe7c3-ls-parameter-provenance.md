# Fe7C3 LS 参数来源审计

日期：2026-09-11。本记录只核对论文/SI、冻结 ELF lookup 和已有 LS
初始化合同；没有运行 LASP 主程序或 PES。

## 已发表参数

正文 `215.txt` §2.3 只给出标准 C–C 键能约 `3.61 eV`，并说明初始
`A_pq` 为标准 pair bond energy 的 3%，随后由 self-adaptation 调整。Fe7C3
专节 §3.3 明确 LS 使用 `Υ=0.01 eV/atom`，penalty pair 为 Fe–C 和 C–C。
SI `ct4c01081_si_001.txt` §7.7 的 Fe7C3 输入还明确给出：

```text
SSW.soft.LselfAdapt T
SSW.soft.SAbiasAtom 10.0
%block SSW.soft.bondFilter
  26 26
%endblock SSW.soft.bondFilter
```

SI 没有给出 Fe–C 或 Fe–Fe 的标准键能表、标准键长表，也没有把
`3.61 eV` 扩展为 Fe–C 参数。因此不能从论文补造 Fe–C 数值。

## 冻结 ELF 的 raw lookup

核对的 ELF 为
`/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp`，
SHA-256 为
`bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704`。
隔离调用 `bondeneval_`/`bondlenval_` 的完整输出保存在
`research/ga_ssw/evidence/fe7c3-ls-parameter-provenance/raw-lookup.json`。

| pair | `bondeneval_` raw return | `bondlenval_` raw return | 解释 |
|---|---:|---:|---|
| C–C | 3.4468400478363037 | 1.5399999618530273 Å | release lookup；不是论文 3.61 的同值证明 |
| Fe–C | 13.779999732971191 | 1.9199999570846558 Å | release lookup；无论文独立校准证据 |
| Fe–Fe | 3.6298000812530518 | 2.630000114440918 Å | energy 为 generic fallback；长度为 release branch |

反向 Fe–C 查询返回相同数值。`len_toller=0.1 Å` 的 ELF 静态值已在既有
计数证据中核验，因此这些 raw length 若用于 native-style 几何计数，会给出
约 `2.02 Å` (Fe–C)、`1.64 Å` (C–C) 和 `2.73 Å` (Fe–Fe) 的距离门；这只是
释放版 lookup 加容差的算术结果，不是论文另行指定的 Fe7C3 化学 cutoff。

Fe–Fe 的 `26 26` `bondFilter` 会将 pair energy factor 置零，但现有
`selfadapt_nbondcounter_` 证据显示 `N_b` 按几何/fixatom 候选计数，未直接
读取该 pair filter。因此 Fe–Fe 是否仍进入 `N_b` 不能被过滤块本身排除；
应在实现/实验中把这个分母规则显式记录，不能把 Fe–Fe 从候选计数中悄然删掉。

## 可执行的最小测试合同与边界

若只做接口/算术测试，最小显式表可以记录为：

```text
bond_energies = {(6,6): 3.4468400478363037,
                 (6,26): 13.779999732971191,
                 (26,26): 3.6298000812530518}
bond_lengths  = {(6,6): 1.5399999618530273,
                 (6,26): 1.9199999570846558,
                 (26,26): 2.630000114440918}
energy_filter[(26,26)] = 0.0
length_tolerance = 0.1
target = 0.01 eV/atom
```

这组数值可复现 release raw lookup 与 `bondFilter 26 26` 的数据流，不能
称为论文 Fe7C3 标准化学参数，也不能证明 G-NN PES 上的 LS 收益。真正的
Fe7C3 端到端测试还需固定：元素/坐标/PBC、NN potential 文件、Fe–C/C–C
表值的选择依据、`N_b` 是否包含零 filter pair，以及完整预算和失败成本。
若使用当前 `LSResponseState`，过滤项必须随 response rebuild 一起传播；
否则其按下一结构全部距离合格 pair 重新分配总强度的行为会重新引入被
过滤的 Fe–Fe pair。当前不应把 raw lookup 直接接入生产默认。

复现命令（仅隔离 lookup）：

```bash
PYTHONPATH=/tmp/pam-ssw-unicorn-probe:. python -c 'import json; from pathlib import Path; import research.ga_ssw.probe_native_ls_pair_table_extended as p; p.PAIRS=((6,26),(26,6),(26,26),(6,6)); out=Path("research/ga_ssw/evidence/fe7c3-ls-parameter-provenance/raw-lookup.json"); out.write_text(json.dumps(p.run(Path("/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp")),indent=2)+"\n")'
```
