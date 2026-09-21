# 上传 examples 的真实结构与后端盘点

2026-09-10。仅文件扫描、ASE结构解析和配置核对，没有PES调用、结构生成或任务提交。
主线用途是SSW→固定胞PBC→VC，不启动新的GA/LS/RC开发。

**最直接的已有VC结构输入是TYPE1-AlOH：15个26原子晶体帧，配有原生AlOH.pot。**
本次两个扫描根目录没有C60、PdO或CuO实际结构，也没有可直接当作EMT真实金属
算例的坐标。TYPE0金属仅给组成；LJ数据中的Au标签不能当作真实Au体系。

## 完整扫描范围与解析

根目录均位于 `/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/`：

- `GA-SSW_examples_run/`：77文件。
- `example-preview/`：24文件，只有文本配置/说明/脚本，没有结构或势文件。

101文件全部列入 `research/ga_ssw/benchmark_inventory.json`。候选结构为8个ARC、
2个LAMMPS data、1个Gaussian输入模板。所有ARC帧逐帧使用ASE `read_dmol_arc`
解析；字节相同的文件共享解析结果，重复来源仍全部保留。没有把大数据集复制到仓库。
普通ASE路径读取可用 `read(path,index=0,format='dmol-arc')`；流式字符串块采用
ASE直接reader，避免通用格式分派对StringIO的AssertionError。

机器清单保存每文件路径、大小、结构摘要、完整configure/lasp参数、势文件引用和
解析失败。扫描脚本为 `research/ga_ssw/inventory_examples.py`，不依赖势能后端。

## 各case及可用性

下表路径均相对于 `GA-SSW_examples_run/`；模板目录前缀为
`global_exploration/input-templates/`。

| case / 实际结构路径 | 真实文件内容 | 边界与晶胞证据 | 原输入后端 / 势文件 |
|---|---|---|---|
| TYPE1-AlOH/addition/add.arc | 15帧，均H4Al8O14，26原子 | SearchType1、IfPer1；Run_type15、NG_cell7、ds_cell1.6，联合变胞意图明确 | NN；AlOH.pot提供，2,193,538 bytes |
| TYPE2-XXXII/addition/add.arc | 1帧C84H68Cl8N4O8，172原子 | SearchType2、IfPer1；rigid.movecell=T，LAMMPS boundary p p p | LAMMPS；mc/in.simple、lmp.data、blist、rigidbody提供；不是通用SSW首选 |
| TYPE3-(H2O)15/addition/add.arc | 1帧H30O15，45原子 | SearchType3、IfPer0：孤立团簇；ARC 50 Å盒与PBC=ON仅为存储形式 | NN；H2O_pf.pot提供，2,515,239 bytes |
| TYPE4-TiO2@Au24O4/addition/add.arc | 1帧Au24O328Ti162，514原子 | SearchType4、IfPer1；Run_type5固定胞，含基底固定原子输入 | NN；AuTiO.pot提供，1,054,184 bytes |
| TYPE0-Au10Ag10 | **无实际坐标**；配置Au10Ag10，20原子 | SearchType0、IfPer0，原程序生成初态 | NN；AgCuAu.pot提供，1,079,701 bytes |
| TYPE0-LJ75 | **无实际坐标**；配置Au75占位标签，75个LJ位点 | SearchType0、IfPer0 | potential lj；lj.pot为0字节占位，不能当成参数文件 |
| Path-LJ38/1. Structural dimensionality reduction representation/all.arc | 35,152帧，均38位点，以Au38标注；xy/all.arc、zw/all.arc为字节相同副本 | README明确LJ38孤立体系；其中DCCD type2不是GA SearchType2 | 路径数据集没有lasp.in或明确LJ sigma/epsilon/cutoff来源 |
| global_exploration/GA-SSW/input/addition/add.arc | 与TYPE4的514原子ARC字节相同 | 当前configure为TYPE4/IfPer1，固定胞 | AuTiO.pot在同目录提供 |

`GA-SSW/input/`还混放水/AlOH/合金pot、TYPE2的mc文件和Gaussian模板，**不能把
这些辅助文件都解释成当前TYPE4案例的组成或后端**。其mc/lmp.data与TYPE2版本相同。

势文件提供不等于已可接入ASE：这些NN `.pot` 是LASP格式，本盘点未发现对应独立
ASE模型适配器，也没有验证模型适用域。替换MACE/GFN2等calculator应另标势面变化。
LAMMPS的in.simple使用 `units real`、`atom_style full`、`lj/amber/coul/long`、
`dihedral_style amber` 等特定设置；有data文件不等于本环境的LAMMPS构建可直接运行。

## 关键晶胞与输入参数

| case | 首帧cellpar：a,b,c Å；α,β,γ ° | lasp原SSW设置 |
|---|---|---|
| AlOH | 7.26771434, 3.04838616, 12.06238490；85.23086962, 85.24340140, 91.32503758 | T400、NG10、ds_atom0.6、ftol0.05、strtol0.05、MaxOptstep500、steps10000；cell NG7/ds1.6 |
| XXXII ARC | 13.30658765,11.75354942,13.99321669；107.46552917,79.35936423,109.47796017 | rigidssw；T1000、NG6、ftol/strtol0.005、MaxOptstep2000、steps1000000、latstepsize0.2 |
| 水15聚体 | 50,50,50；90,90,90，非物理周期证据 | T50、ftol0.001、quick_setting1、steps100 |
| 支持簇 | 26.7156,19.7083,35；90,90,90 | T100、ftol0.05、MaxOptstep500、steps300；fixed-cell及fixatom块 |
| LJ38首帧 | 58.55814362,59.77612686,59.55655289；90,90,90，孤立LJ盒 | 没有同路径SSW输入，不猜原生成参数 |

Au10Ag10的lasp输入T200/ftol0.01/steps10000；LJ75为T200/ftol1e-5/steps5。
不要将这些值和configure中的外层覆盖混合：例如AlOH configure的SSWTemper600、
SSWStep2000不同于lasp的400/10000；XXXII为700/30000，而lasp为1000/1000000。
机器清单保留两套原值，实际运行时由哪个参数生效需检查调用端，不能从模板宣称确定。

TYPE2的 `mc/lmp.data` ASE解析也得到172原子及相同元素组成，但其盒为
41×32×51 Å正交胞，**不同于addition/add.arc的倾斜晶胞**。二者不能无核查地作为
同一个周期构型互换；这里保留差异，不擅自修正。

## 解析失败与缺口

`GA-SSW/input/temp.gjf`不是填好的分子结构，仍含 `#charge_spin`、`#coordinate`
占位。ASE Gaussian reader因缺少charge/multiplicity失败，完整错误留在清单。
`ssw_gaussian/gaussian.inp.pre`、`.after`及external脚本也是模板；没有可直接运行的
完整Gaussian输入。所有ARC实际帧解析成功，LAMMPS data几何解析成功；后者元素
来自ASE对Masses的推断，未执行force field。

`coninfo.base`和allSim等文件是描述符/相似性数据，没有据此恢复或发明额外结构。
C60/PdO/CuO的缺失结论只限上述两个完整扫描根目录，不否认仓库其他已有C60实验。

## 对SSW→PBC→VC主线的建议

1. 用现有LJ38坐标做补充内核检查，先明确独立LJ单位/参数；不把其当真实材料验证。
2. AlOH是最小且具实际周期/变胞输入依据的上传案例。先独立核验所选ASE calculator
   的E/F/stress与应变导数，再用于固定胞PBC和VC对照；native pot存在不免除此步骤。
3. TYPE4需要固定基底约束与表面边界处理，TYPE2涉及分子晶体和专用LAMMPS/rigid接口；
   二者保留作后续验证，不阻塞最小联合原子/晶胞内核。

本文件仅盘点与排序，不授权或启动PES计算；实际预检由主线单独记录。
