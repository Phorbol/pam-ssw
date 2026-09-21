# FixAtoms / Hookean：实现与真实体系证据

2026-09-12。当前实现入口是 `run_constrained_ssw`（plain / paper LS / native LS）
与 `constrained_quench`。普通 `run_ssw` 和 GA 尚未自动分派或支持 Hookean；
GA 的原子重排涉及约束索引传递，不能直接复制 parent 约束。

## 目标函数和自由度

- FixAtoms 仅删除对应原子的活动坐标。Hookean 为持久目标函数的一部分：
  `E_objective = E_calculator + E_Hookean`，通过 ASE 原生能量/力方法计算一次。
- LS 预淬火、方向/HVP、Gaussian 淬火、无偏置真实淬火和 MC 使用同一持久
  目标；真实淬火移除 Gaussian 和 LS，但保留用户 Hookean。
- 所有底层 oracle Atoms 都去掉约束；总力合成后才投影固定自由度。输出与
  checkpoint 的 Atoms 恢复 Hookean，然后放置 FixAtoms，避免 ASE 将固定原子
  的力重新加回来。固定原子的反力不作为活动力资格判据。
- `full_raw_fmax` 指投影前持久目标的总力；certificate 另外保存裸物理 E/F 和
  Hookean E/F，不追加 oracle。失败请求清除上次成功 snapshot。
- 支持 pair / point / plane 原生 Hookean，输入预检、约束规格保存及恢复校验。
  无固定原子也可使用全 Cartesian 活动坐标，不强行移除平移/转动模式；可用于
  点/平面约束破坏相应对称性的情况，也允许部分 PBC。其他 ASE 约束仍明确拒绝。

## 验证与局限

根智能体运行 standalone suite：549 passed, 1 skipped（28.30s，
`/tmp/pam-root-hookean-suite-20260912.log`）。随后新增真实继续一步的恢复测试，
根智能体运行 Hookean driver 文件：14 passed。覆盖 plain/paper/native 的
continuous 2 vs 1+resume1，固定+Hookean 和 Hookean-only，成本、坐标、RNG、
LS 状态匹配。解析接口测试不能作为科学效率证据。

真实体系固定六臂，seed11，每臂1个外层尝试；inner Gaussian 活动梯度L2阈值
0.1 eV/Å，outer最大活动原子力0.03 eV/Å，relax_steps300。
LS 预淬火当前沿用 outer 的0.03，尚无独立 LS 预淬火精度参数。
水二聚体为已有ASE S22结构/GFN2-xTB，Cu111为已有13Cu固定8原子EMT结构。
Hookean显式用于检验持久约束，不是优化所得参数；完整参数来源见plan。

| 体系 | 变体 | 搜索E/F | fresh E/F | 外层结果 |
|---|---|---:|---:|---|
| 水二聚体 | SSW | 109 | 2 | valid_landing |
| 水二聚体 | paper LS | 119 | 2 | valid_landing，LS状态更新 |
| 水二聚体 | native LS | 120 | 2 | valid_landing，LS状态更新 |
| Cu111 | SSW | 49 | 2 | valid_landing |
| Cu111 | paper LS | 56 | 2 | valid_landing，LS状态更新 |
| Cu111 | native LS | 152 | 1 | LS预淬火line_search_failed |

共605搜索+11fresh=616次E/F请求。11个已存储极小值观察全部通过新calculator
检查：持久目标能量一致、活动力≤0.03、约束元数据保留、固定坐标及cell不变。
这些观察包含重复初态，不能称11个不同盆地。Hookean能量均非零，因此不是
“约束没有激活”的空检查。GFN2重新初始化的最大fmax差2.39e-7 eV/Å，原有和
fresh证书均满足预定0.03；不是逐bit oracle等价。

Cu/native LS失败保留：预淬火48步后线搜索失败，没有外层落点，只有初态
资格证据。既未增加预算也未据此调整LS参数；不能声称六臂全部走完或native
LS适用于所有表面。无需因此停住整个主线。

研究runner汇总原用错成本字段 `evaluation_requests`，引发记录异常；实际
`ConstrainedSSWResult`字段是`requests`。每臂result、逐调用ledger和fresh
检查均已保存。原始summary和runner保留，独立audit从这些产物恢复真实程序
状态与成本，**没有重跑PES**。未来runner已修正字段。原始记录异常不应被
误写成算法失败，也不能掩盖Cu/native LS的实际失败。

证据目录：`research/ga_ssw/evidence/hookean-multicase-20260912/`，以
`plan.json`、各臂`result.json`/`evaluations.jsonl`和`audit.json`联合解释。
审计脚本：`research/ga_ssw/audit_hookean_multicase.py`。
本结果证明有限支持域中的实现一致性与五条完整短流程，**没有证明全局搜索
效率、不同盆地覆盖提升、无约束稳定性、任意calculator或VC约束支持**。
