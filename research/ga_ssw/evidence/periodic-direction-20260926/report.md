# 周期固定胞局部方向：当前资格与未关闭项

已按批准设计接入周期c1/c4/c6/c9与未包裹位移历史、普通/受限入口及方向checkpoint
cell/PBC身份。新模式显式opt-in，旧非周期默认不变。没有加入Q库或变胞方向。

## 文献材料bulk端到端资格

固定OMAT-small/omat_pbe，seed26092631，3外步/臂，旋转/优化/偏置设置相同，
方法差异为全局方向vs局部方向与记忆。全部6臂completed、各3落点及75Gaussian。

|材料|全局搜索请求|局部搜索请求|局部实际方向分支|
|---|---:|---:|---|
|rutile12|1259|1032|50torsion、25pair|
|anatase12|1405|1067|50torsion、25pair|
|brookite48|1251|929|25torsion、43pair、1pair_fallback、6forbidden|

总6943真实搜索+24独立端点复核，另3028同oracle重放（不计为新的独立物理轨迹）。
24帧通过原力门0.03eV/A和能量复算；三条局部连续/分段查询与完整检查点状态一致。
所有local轨迹实际用了多Gaussian历史。未证明独立盆地覆盖或结构发现优势；
rutile/anatase best未下降，brookite差值约0.000392eV不足以单独作科学改进证据。
此批是论文关联材料资格和成本pilot，不能替代论文级搜索效果验收或泛化。

## 保留的失败与修复

GPU1498877搜索正常完成，但runner访问result.best.energy（实际best是Atoms）导致
汇总失败并跳过fresh/replay。原脚本和runs/summary.json保留。GPU1498955仅读取
已存完整checkpoint/原始请求补做24fresh及3028重放，全部通过，无重复搜索。
正式读出在verification/summary.json。原始大轨迹/检查点本地保存，未默认放Git。

入口RED1498804，综合1498859有149项通过；最终新增几何1498873为9项。
根审查另修两项：固定第二轴端点且空活动组时draw_count错记0（RED1498901）；
辅助评分参考误允许固定原子（RED1498982，大胞极限对照现有受限算法复现）。
前者仅诊断，后者只影响受限新路径；已完成bulk全活动结果均不受影响。
为新周期受限路径补实际route字段（RED1498970），不改旧非周期记录。
最终相关42项1498990全部通过。早先物理源快照保留，TYPE4用修正后版本另存。

## 结构覆盖读出

CPU1499004零PES分析：包括初态在内，rutile/anatase/brookite的全局近似组数3/4/2，
局部2/1/1；局部返回初态2/3、3/3、3/3，对照1/3、0/3、2/3。
两组既有匹配容差给出相同关系。近似几何组不是经Hessian认证的盆地。
较少调用没有表现为更广覆盖，本pilot不支持切换默认，也不据单seed调参。
详见[完整覆盖分析](coverage-report.md)、coverage-summary.json及analyze_coverage.py。

## 后续TYPE4组合资格

[完整514原子固定基底结果](type4-report.md)：两臂共1074搜索+6fresh合格，
局部578响应重放/恢复一致；本例未实际触发受限c6。局部更贵且终点更高，
与bulk覆盖读出均不支持默认推广。来源、原始产物及脚本失败完整保留。
