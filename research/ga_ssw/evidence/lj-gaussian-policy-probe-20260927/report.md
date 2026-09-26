# 已有Gaussian策略：单次完整逃逸结果

2026-09-27，开发机制检查。源码162da5b，CPU1505559，dpn01，Slurm COMPLETED/0:0，
11秒（runner9.92秒），7801搜索+16独立复核，无截断、无GPU。
四个初态/首旋转方向配对一致；LJ38基线两条首外步的状态、接受、成本、能量与旧记录一致。
16/16初始/落点force≤0.01eV/Å，但一个基线落点碎裂。力合格与物理完整分开计。

| 体系/seed尾号 | 策略 | 搜索调用（含初始） | 落点ΔE/eV | 完整簇 | Gaussian数 |
|---|---|---:|---:|---|---:|
|LJ38/501|forward|1399|+6.1982|否|14|
|LJ38/501|PAM height_width|1186|−1.8924|是|6|
|LJ38/502|forward|955|−9.3541|是|1|
|LJ38/502|PAM height_width|1024|−12.3697|是|2|
|LJ55/501|forward|782|−6.6579|是|2|
|LJ55/501|PAM height_width|815|−6.8987|是|3|
|LJ55/502|forward|811|−1.0063|是|1|
|LJ55/502|PAM height_width|829|−1.0062|是|2|

连通性在1.3/1.5σ_LJ两个固定阈值下判据一致。LJ55/502两落点同标签刚体对齐
RMS0.00808Å且能差约0.00014eV，不当作有意义的不同落点收益。
其余配对的RMS较大只排除了同标签刚体一致，未用它证明置换后不同盆地；表中降能是独立复核值。
未做Hessian稳定性或长期全局搜索验收。

最明显异常例LJ38/501首width从0.6变0.274875Å，首weight从114.180变1.20378eV。
PAM各臂未发生宽度/高度clipping。该整策略纠正了一条碎裂轨迹并以更少调用降能；
但其它三对调用上升，不能只报告这个正例或宣称效率普遍提高。

## 决定

保持所有默认。因为策略同时改变高度与宽度，现有结果不能把效果全部归因于缩短位移。
下一项仅补已实现的height_only选项（固定width0.6），沿用这四个初态、不重跑旧臂，
见[新跟进协议](height-only-followup.md)。它是结果驱动的开发诊断，不冒充独立验证。
这组单逃逸完成后不扩展参数扫描；旧C60/Cu55/water15 PAM混合/负结果保持效力。

## 证据与验证

- [预定协议](protocol.md)、[零PES预检](preflight.json)、[提交脚本](run.sbatch)。
- 原始目录 `run-1505559/`：execution、runner及input-generator副本、8臂输入/初态/落点/记录/有效配置；大型原始记录不复制进Git。
- [派生读出](analysis-1505559.json)、[配对几何](landing-comparison-1505559.json)。
- 复核命令：`python readout.py --run run-1505559 --previous ../cluster-paper-reproduction-20260925/stage-probe-repaired-runs --output NEW_ANALYSIS.json`（在本目录，仓库根加入PYTHONPATH）。
- 主agent检查metadata序列化、有效参数、预算与已付费失败计数。首次计数mock错误地用None当calculator，初始化阶段即失败、0PES；改为合法但不执行的FullPairLJ后，注入失败/拒绝计数检查通过。没有放松预算或物理标准。
- runner审查修复了未执行前的预检JSON返回问题和报告异常可能把已付费调用记0的问题；实际8臂没有触发这两类错误。

## Height-only跟进完成：降低高度本身不足以修复该失败

源码af7867c，CPU1505596，dpn01，COMPLETED/0:0，6秒；4027搜索+8fresh。
四臂初态、首中心/方向、旋转/优化设置及RNG来源与原控制逐项匹配。
8/8独立force资格通过、无预算截断、无PAM width/weight clipping。

| 体系/seed尾号 | height_only搜索 | ΔE/eV | 完整簇 | 首W/eV | Gaussian数 |
|---|---:|---:|---|---:|---:|
|LJ38/501|1429|+4.1337|否|5.7356|14|
|LJ38/502|990|−11.7526|是|2.1553|2|
|LJ55/501|779|−6.6575|是|2.6076|2|
|LJ55/502|829|−1.0063|是|3.3780|2|

所有width保持0.6Å。故LJ38/501中，仅从forward规则换成曲率高度，虽然将首W从114.18
降为5.74，仍未避免碎裂；height_width的联动调整则在该次逃逸有效。不能把结论说成
“114eV本身就是碎裂原因”，也不能把height_width当成独立只改位移的实验。
这支持有限幅度与Gaussian形状的耦合是一个真实设计因素；没有证明通用最优规则。

**本面板按预定规则收口：不再加臂、种子、目标能量或高度上限，不改默认。**
全组两作业合计11828搜索+24fresh=11852次调用；此前高度来源诊断164次另计，
本轮总12016次E/F。两CPU作业17秒，未用GPU。原C60/Cu55/water15反证仍保留。
下一步限于原版moveds的有限位移比较量语义核查，不复制未解释的距离/类别阈值；
若要新增核心试位移阶段或改变持久化身份，先提出具体设计与用户讨论。

证据：[跟进协议](height-only-followup.md)、[跟进预检](height-only-preflight.json)、
[跟进读出](analysis-1505596.json)。原始`run-1505596/`保存有效配置、输入/初态/落点、
climb记录与冻结runner/生成器。复核命令为
`python readout_height_only.py --followup run-1505596 --controls run-1505559 --output NEW_READOUT.json`，
不增加PES。该脚本验证四组精确初始/首mode配对及原成本闭合。
