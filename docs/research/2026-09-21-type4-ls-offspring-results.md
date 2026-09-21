# 固定基底 TYPE4 GA + LS：真实子代接口验收

CPU1435269完成Cu7Ag/EMT、FixAtoms-only全流程：3个quick、12个真实offspring_quick、1个fine。共5741次物理能量/力请求，无拒付或计算异常；另3681次合成辅助势计算单独计账。底层Calculator.calculate未实测，不将请求数冒充实际后端计算数。

一个原生整批生成14个候选，碰撞过滤后12个进入LS-SSW：2交叉、2重载、2扰动/交换、6重建。重构算子在本几何没有上部片段，返回零候选；不能宣称五类算子均获得真实产物。16次LS预处理/响应更新均有记录。独立审计核对候选到walker输入、每原子父代archive索引与source_atom_indices全部对应，固定基底、cell、PBC及组分保持。

原20次独立复核通过，但31个最终archive代表未全部覆盖，故原summary仍保留not_met。独立子验证CPU1435388仅对预注册缺失的12个代表检查，不重新搜索或淬火。合并后32/32观察结构复核通过，覆盖31/31最终代表；最大active fmax=.00994228 eV/Å，最大能量差7.11e-15 eV（数值一致性检查，不是势模型物理精度）。新协议总物理请求5773=5741+20+12；原错误协议881=873+8另外保留。整个系列物理请求6654，辅助计算3681另列。

结论：该指定体系上真实proposal→offspring LS-SSW→fine及约束/来源记录链已验收。LS键参数是诊断设置，结构matcher仅坐标比较，不是置换不变盆地识别。因此不把archive数作为独立盆地数，不宣称GA或LS搜索效率优势、材料物理预测或生产级通用验证。该接口支线停止扩算；固定胞SSW的C60长预算搜索对照继续优先。

- [冻结协议及原始运行](../../../vc-qualification-audit/research/ga_ssw/evidence/type4-cuag-ls-offspring-20260921-v2/README.md)
- [独立补充复核与原始输出](../../../vc-qualification-audit/research/ga_ssw/evidence/type4-cuag-ls-offspring-20260921-v2-qualification/output/qualification-summary.json)
- [空批次根因与修复](2026-09-21-type4-empty-batch-diagnosis.md)

核心改动仅为原生TYPE4非法小批量提前拒绝，两源码/两测试文件已集成但未提交/发布；原分配、算子和默认策略不变。CPU1435258相关16项测试通过。
