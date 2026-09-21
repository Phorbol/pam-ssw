# 固定胞 SSW/LS 示例候选盘点

**核查日期：** 2026-09-12
**范围：** 只读核查 `ga-ssw-20260909/GA-SSW_examples_run`、`example-preview` 及 `analysis/example-matrix.json`；没有运行 PES，也没有修改示例源文件。

## 结论

指定目录没有 Cu、Pd、Pt、Ni 的已核实模型设置，也没有对应 GFN2 模型设置或已验证轨迹。可直接识别的金属例子是 Au10Ag10 的配置模板，以及 AlOH 和 TiO2@Au24O4 的周期 ARC；后两者随附的是项目自定义势文件，不能直接当作 ASE EMT 输入。目录中的水15例子是 45 原子非周期分子簇的模板，但随附 `H2O_pf.pot`，不是 GFN2。若要用当前项目的 ASE EMT 或 tblite GFN2，必须把它们标成“换后端的独立验证”，不能称为原论文数值复现。

## 可核实候选

| 候选及来源 | 结构事实 | 示例用途与输入/结果性质 | 当前固定胞 SSW/LS 适用性 |
|---|---|---|---|
| `.../input-templates/TYPE0-Au10Ag10/`；结构配置为 `configure.non`，势文件为 `AgCuAu.pot` | Au10Ag10，共 20 原子；`IfPer=0`、`SearchType=0`。模板没有 `addition/` 结构，type-0 由程序生成初始团簇。 | `analysis/example-matrix.json` 将其列为 50 个 GA 个体、每步 3000 SSW 步、500 K 的非周期团簇例子。244 主文第 6 页 Figure 3 将 Au10Ag10 纳入二维 PES 图，并标出 GM、第二低 LM 和其他关键异构体；该页未给出结构文件或势的完整定义。 | 适合作为**小型非周期接口候选**，但当前目录没有可复用的确定初态。ASE EMT 可覆盖 Au/Ag 的元素域时仍需另行检查初态与能量基准；`AgCuAu.pot` 与 ASE EMT 不是同一后端，不能混算或作原论文复现。 |
| `.../input-templates/TYPE1-AlOH/addition/add.arc` | 15 个 ARC frame；每 frame 为 H4Al8O14，共 26 原子。`PBC=ON`；首 frame cell 为 `[7.26771434, 3.04838616, 12.06238490, 85.23086962°, 85.24340140°, 91.32503758°]`。 | `configure.non` 为 `SearchType=1`、`IfPer=1`、`NNPotName=AlOH.pot`。这是周期晶体的 addition 初始结构集合，不是本项目的 SSW 结果归档。 | 结构规模和固定胞边界适合做周期入口检查；但 H/O/Al 体系不能直接用 ASE EMT 代替 `AlOH.pot`。只有在明确提供可用后端并重新核查能量、力和 cell 后才适合有界实验。 |
| `.../input-templates/TYPE4-TiO2@Au24O4/addition/add.arc` | 514 原子：O328Ti162Au24；1 frame；`PBC=ON`；cell `[26.7156, 19.7083, 35.0, 90°,90°,90°]`。`configure.non` 的 `FixAtom=0-296`。 | `SearchType=4` 的 supported-cluster 初始结构，随附 `AuTiO.pot`。同一 ARC 也被复制到 `GA-SSW/input/addition/add.arc`；它不是优化结果。 | 适合检查大周期 supported-cluster 的固定原子契约，但 514 原子和自定义势超出当前 EMT 优先级；不能以 ASE EMT 或缺失 MACE 参数替代而声称论文体系结果。 |
| `.../input-templates/TYPE3-(H2O)15/addition/add.arc` | 1 frame，H30O15，共 45 原子；ARC 写有 `PBC=ON` 和 50 Å 立方 cell，但模板 `configure.non` 明确 `IfPer=0`、`SearchType=3`，因此应按非周期分子簇模板解释其大盒子。 | 244 主文第 6 页 Figure 3 将 `(H2O)15` 用于二维 PES 展示。随附 `H2O_pf.pot`；配置是初始 addition，不是优化结果。 | 适合作为中小分子簇的**换后端候选**。当前项目若改用 tblite GFN2，必须保存新后端/参数并单独验证；不能把 `H2O_pf.pot` 结果转换成 GFN2 或把 GFN2 运行称为 244 的数值重现。 |
| `.../input-templates/TYPE2-XXXII/addition/add.arc` | 1 frame，H68C84N4O8Cl8，共 172 原子；`PBC=ON`；cell `[13.30658765,11.75354942,13.99321669,107.46552917°,79.35936423°,109.47796017°]`。 | `SearchType=2` 分子晶体模板，另需 `segmentation.non` 和 `mc/`；`NNPotName=lj.pot` 且 `potential_exists=false`（见 `analysis/example-matrix.json`），更像流程模板而非可直接复现实验结果。 | 只推荐作后续结构/输入契约审计，不推荐当前有界真实后端实验。172 原子、多组分、周期分子晶体且缺少可核实物理后端。 |

## 例子目录与 preview 的完整性限制

`GA-SSW_examples_run/global_exploration/README.md` 说明了六类模板：LJ75、Au10Ag10、AlOH、XXXII、(H2O)15 和 TiO2@Au24O4。`example-preview/PREVIEW_ONLY.txt` 明确写着示例归档仍在上传，内容不是完整核验的正式例子。当前实际 preview 只包含配置/README 的裁剪副本；不能从 preview 推断缺失的结构、优化轨迹或后端参数。

`Path-LJ38/1. Structural dimensionality reduction representation/all.arc` 是 LJ38 补充数据；旧 inventory 已说明其中使用 Au 标签表示 LJ 模型原子。静态记录约 1,335,776 个 Au 原子行，对应约 35,152 个 38 原子 frame。它可作为 LJ 方法学补充，但不能当作真实 Au 势或金属体系候选。

## 244/SI 对水15和 Au10Ag10 的边界核对

`literature/244.pdf` 第 6 页 Figure 3 的图注只确认两者被用于二维 PES 展示，并说明图中标注 GM、第二低局部极小值和其他关键异构体；该本地 PDF 是短版/图稿形态，没有在可检索正文中给出水15或 Au10Ag10 的完整计算方法、结构坐标或后端参数。`literature/244-SI.pdf` 的 S2--S10 是图/公式页；可检索文字明确的 S2 示例是 biphenyl 和 LiZrCl，S5 给出 LJ 势公式，S11--S14 讨论 DCCD 层数、权重、处理成本和每步能量评估数。它没有提供可据以断言“水15 使用 GFN2”或“Au10Ag10 使用 ASE EMT”的证据。

所以：水15的论文用途是构象/PES 图示，Au10Ag10的论文用途也是团簇 PES/异构体图示；本地 GA 模板分别提供 `H2O_pf.pot` 和 `AgCuAu.pot` 名称，但没有足够原文信息证明它们就是论文图的完整数值设置。当前项目选用水15/GFN2、bicyclobutane/GFN2 或 Cu55/EMT 时，应在实验记录中写成**替换后端和输入来源的独立接口/搜索验证**，并保留与原论文不可直接比较的限定。

## 建议的最小后续选择

若只做固定胞 SSW/LS 的输入契约和有限成本验证，优先级为：

1. `TYPE1-AlOH`：26 原子、周期、明确 cell，前提是先确认可用的 Al/O/H 后端；
2. `(H2O)15`：用 tblite GFN2 作为明确换后端的小分子簇案例，单独记录模型域；
3. `TYPE2-XXXII`：保留作后续周期分子晶体输入审计；
4. `TYPE4-TiO2@Au24O4`：仅在需要验证 supported-cluster 固定原子边界且接受 514 原子成本时使用；
5. `Au10Ag10`：只有先补充确定初态后才可用于固定初态的 SSW/LS 对照。

这些候选支持的是输入、后端和账本契约检查；它们尚不足以支持跨体系搜索效率、全局最优发现率或与论文数值相同的结论。

## 来源

- `/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/analysis/example-matrix.json`
- `/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_examples_run/global_exploration/README.md`
- `/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/example-preview/PREVIEW_ONLY.txt`
- `/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/literature/244.pdf`（第 6 页）
- `/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/literature/244-SI.pdf`（S2、S5、S11--S14）

## 本轮实际选择（取代上面的候选排序）

水15已作为本轮新增真实输入执行；AlOH26作为下一批周期候选。XXXII172与
TYPE4 514后置，Au10Ag10需要单独生成并冻结初态。实际结果见
[三体系结果](2026-09-12-expanded-case-results.md)，不能将目录盘点视为PES验证。
