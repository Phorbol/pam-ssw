# 提案：GA 内部正在执行的 SSW 长轨迹恢复

状态（2026-09-25）：用户已批准最小方案；固定胞两项有界诊断结束后恢复实施。源码已在隔离分支实现，通过Cu13/EMT四阶段严格恢复及C60/MH1有界恢复资格；GPU轨迹不保证逐位一致。详见 [资格记录](../../research/ga_ssw/evidence/ga-active-walk-qualification-20260925/README.md)。

## 为什么现在需要决定

同预算对照完成，GA早期优势混合，不值得继续加变异/描述符参数。论文级C60搜索涉及
数千外步；当前每臂60000请求仅61–70次尝试。进入长程验证前，现有GA恢复能力不足：
只在quick_complete、generation_complete、cycle_complete三个完成边界保存。

CPU1464993读取本轮真实产物：

| 种子 | 搜索已消耗 | 最新可恢复GA边界 | 边界后fine调用 | fine含SSW checkpoint |
|---|---:|---:|---:|---|
| 3 | 60000 | 49360 | 10640 | 否 |
| 17 | 60000 | 44168 | 15832 | 否 |

结果结构与日志没有丢失，但不能从fine的最近完成外步继续原来的搜索状态。
`paper_ga.walk`没有请求SSW checkpoint；`paper_reference`仅在传入checkpoint或
checkpoint_path时创建它。仅另存best结构并重新run不是连续恢复。

依据：`pamssw/standalone/paper_ga.py`的walk/boundary、`ga_checkpoint.py`v1，
证据`research/ga_ssw/evidence/population-comparison-20260923/checkpoint-gap.json`。

## 推荐的最小方案

1. SSW新增显式、可选的外步完成checkpoint回调。复用已有SSWCheckpoint内容；
   回调可请求合作式暂停。保留checkpoint_path及旧默认行为；不开启时不新增每步复制开销。
   回调不调用势、不消费随机数、不在Gaussian/线搜索中间暂停。
2. GA checkpoint v2新增可选active_walk：当前阶段/循环/代、已选seed队列及游标、
   本次walk目标外步数和已完成数、已有亲子/算子元信息、嵌套SSWCheckpoint、阶段前
   成本与累计实际成本。档案、统计和RNG仍归各自已有控制器所有，复用现有SSW状态，
   不另造一套LS/方向/MC状态格式。
3. 初期覆盖共用walk路径的quick、generation_short、fine、offspring_ssw。
   walk进行中不把其全部累计落点反复导入GA archive；完成后按原顺序只导入一次。
   恢复继续原seed和剩余步数，不重新分区、抽父母或初始化方向/LS。
4. 原有checkpoint_callback默认仍只收到原来的完成边界；新增明确opt-in使GA
   将外步快照交给同一保存/暂停机制。旧v1按旧阶段语义读取；新写v2，旧读者明确拒绝v2。
5. `max_evaluations`及科学配置兼容检查保持严格。新能力不允许把已经耗尽60000预算的
   旧结果改成更大预算继续；未来长协议在开始时声明总预算，分段恢复共享同一账本。

确切新增参数命名可在实施中按仓库风格确定；公共语义和状态归属先固定。

## 备选及取舍

- 维持当前阶段级恢复：无改动，但长fine只能重算或从best结构另起新实验，不能称连续恢复。
- 在GA内反复调用单步run_ssw：不增加SSW回调，但反复进入driver初始化/验证路径，
  容易把恢复实现变成新的运行语义；需要更多证明才能确保与连续walk等价。
- 保存内层优化器/每次Gaussian的执行栈：状态和成本显著扩大，本问题不需要，暂不做。

推荐外步安全边界方案。主要成本是可选状态复制/I/O，不改变搜索公式、遗传操作或评分。
回调频率由运行器的存储/调度需求决定，不引入新的科学调参。

## 验证与边界

先用确定性EMT Cu13验证连续/分段的RNG、请求账本、父子关系、选点、落点顺序一致，
覆盖quick/generation/fine及已有offspring_ssw路径，检查中断后不重复导入archive。
旧v1恢复、未开启时行为、配置不一致的零PES拒绝、GA与嵌套SSW成本不重复计数均需回归。
复用既有LS/完整方向状态测试，并用C60/MH-1做有界真实Calculator跨进程恢复检查；
力与结构资格单独核验，不把GPU逐位重现或测试通过数当搜索收益。

合作式暂停只保证完成外步边界。硬杀发生在外步中间时，最多回到上个完成边界；
未提交区间的实际已付调用仍从外部实验账本保留并计入总实验成本，不能静默退还预算。
不承诺线搜索中间恢复，不新增预算上调功能，不在本提案里更换描述符、调GA或接VC/RC。

授权记录：用户随后明确选择“采用最小方案，继续实现”，覆盖本提案的GA活动walk归属和v2持久化。此前“尚未覆盖”的待讨论状态已被该决定替代，不重复申请。实现范围不包含内层执行栈或预算上调。
