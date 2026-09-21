# 完整方向外步断点恢复：实现与资格结果

目标是跨进程保留连续方向搜索状态，不改变方向算法或默认MC，不声明搜索效率收益。用户批准的最小范围已完成，池策略仍不支持checkpoint。

## 实际能力与变更

`pamssw/standalone/recovered_direction.py`增加有类型的方向边界状态，含settings、pair、group、group_marker及已有selection/refresh诊断。`paper_reference.py`以schema4存储并恢复它，沿用主RNG和nativeMC状态；恢复跳过初始淬火与方向initialize，旧schema1–3保留。无Calculator序列化，调用者重建同后端；steps仍是追加外步数。只恢复完成外步边界，不恢复Gaussian内部，不恢复终止错误快照。

范围继续限制于自由非周期团簇的完整方向模式。README补充用法，新增checkpoint回归并替换旧的“不支持”测试；没有改变默认MC、Gaussian、Safe-total或数值容差。研究worktree含未提交改动，不是发布版本。

## 验证

1. CPU1437382，指定mace_env且PYTHONNOUSERSITE=1，六个针对性模块 **51 passed**：direction checkpoint/driver、原SSW checkpoint、recovered rotation、nativeMC、starter selection。更强测试先暴露schema4/nativeMC拒绝及pickle状态验证缺口，已修复。测试替身改为scoped monkeypatch，避免污染其他测试。
2. CPU1437376，真实保存E/F回放：两个C60输入及Cu55，连续4步对比分别在0/1/3边界存盘加载，**9/9严格一致**，包括全部记录、落点/current/best、主RNG及累计请求。三个连续基准分别1521/2810/1004请求，回放不调用模型。来源和实际运行源码见[证据包](../../research/ga_ssw/evidence/direction-checkpoint-20260921/README.md)。
3. GPU1437389，2分20秒，C60/MH1-omol与Cu55/OMAT-small各连续2步，对比新进程1步保存及第三个进程恢复1步。**6段完成，8/8独立初态/落点能量力复核通过**；2729搜索请求+8复核，实际Calculator调用2280+8。CPU1437412确认所有成本闭合、预算满足、errors=[]。

| 体系 | 连续两步请求 | 分段两步累计请求 | 恢复末落点与连续末落点能差/eV | MC序列 |
|---|---:|---:|---:|---|
| C60 |839|834|+0.0003188|均为拒绝、接受|
| Cu55 |528|528|约−7.4e−13|均为接受、接受|

两个体系第二步的pair/group/marker一致。C60最大同标签坐标差0.115 Å，Cu55约2.2e−11 Å；保留差异，不声称GPU在线逐位一致，也不追逐小能差。严格控制流恢复由同E/F回放验证。终态均在既有阈值下连通；C60非目标富勒烯笼，不能把资格验证写成C60全局优化成功。

真实进程协议、原始每段账本、结果和读出：[运行与分析](../../../vc-qualification-audit/research/ga_ssw/evidence/direction-checkpoint-real-20260921/analysis-1437412.json)。该短程检查覆盖MC拒绝与接受边界，不证明通用搜索性能。LS与完整方向组合的跨进程恢复尚未单独真实验证。

## 决定

保留schema4能力与旧格式兼容；完成此次批准的持久化范围。池策略checkpoint保持显式拒绝，不能只靠SSW的minima/records重建PAM动态统计与selector随机序列。任何后续池持久化需按此前约定单独确定公开状态契约；不因本轮通过而推广池策略或增加奖励项。
