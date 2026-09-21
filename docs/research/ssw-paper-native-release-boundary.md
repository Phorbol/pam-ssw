# SSW 论文、原版释放与独立实现的边界

2026-09-11。本轮优先核对本体状态机，不新增失败回退或放宽成功定义。

原始论文为 Cheng Shang、Zhi-Pan Liu，2013，*Stochastic Surface Walking
Method for Structure Prediction and Pathway Searching*，JCTC9,1838–1845，
DOI10.1021/ct301010b；题录核验 https://pubmed.ncbi.nlm.nih.gov/26587640/ 。
全文为已归档 `pam-ssw-research/ga-ssw-20260909/literature/74.pdf`，算法见
第1840页 Overall Algorithm steps3–6及式7、8。

论文定义：沿当前方向位移ds、在加入Gaussian后的势上局部优化；达到Gaussian
上限或相应低能条件后，移除全部偏置，再做真实势局部优化。该页使用同一ds
表达位移和Gaussian宽度，但没有规定实际碰撞修正、retry耗尽或优化器超限
应该如何恢复轨迹。不能用论文高层步骤补造发布二进制的失败分支。

静态证据已指出原版成功位移会以实际位移重写方向和width；retry耗尽则恢复
已有轨迹点并切到Allopt，详见native-moveds-scale.md末段。Luna新探针已验证
retry算术，正在继续核对实际宽度写回；synthetic谓词输入不等于完整几何判断。

本轮补充固定胞表0x53cc8c0核验：+0x1f8=allopt(0x5d46e0)，
+0x200=allopt_judge_converg(0x5cf6f0)，+0x110=noncrystal_opt。
因此旧文档中的“slot+0x200未知”不应作为未研究过的永久缺口。证据为
`evidence/native-moveds-retry/fixed-table-allopt.json`，由
`inspect_native_cssw_dispatch.inspect(...,table_bases=(0x53cc8c0,))` 提取。
这是表内容核验，不是所有调用实例的初始化/完整返回轨迹执行。

ssw_move的0x5bd4b0–0x5bd4e2可见Allopt setter后调用+0x1f8；周围条件
涉及初始优化，不能把该单点调用冒充所有retry失败后的完整消费路径。
下一步保留的问题是：失败选择的结构、能量与力如何同步进入下一真实PES
评估，再交给Allopt。只要这一边界未闭合，独立实现继续明确报告失败，
不将其改写为成功逃逸或把最后被拒trial送入archive。
