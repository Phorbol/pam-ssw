# 原版阶段位移：记录索引、小位移与释放边界

实现诊断，零 PES。承接已批准的共享 ASE 驱动/独立方向规则设计。

## 新原指令检查

探针 `probe_native_direction_update.py` 扩展为可提供多个轨迹记录，原来的
一个记录调用保持兼容。新增可重复入口
`research/ga_ssw/probe_native_direction_record_boundary.py`，输出
`research/ga_ssw/evidence/native-cbd-reentry-guard-20260917/record-boundary-v2.json`。

在 N=2/5 的受控无约束投影输入中：
- 三个不同记录、每个索引分别被选中，方向符合当前位置减去指定记录位置。
- 位移范数为 0、0.000999、0.001、0.001001 的输入覆盖归零边界。
- 共14/14通过；原始 n_normal 执行，只有同形状分配和 memset 被等价stub。
- 捕获点仍是 gen_randommode 入口，未执行该生成器或预旋转。

`n_normal` 在0x578f2a比较平方范数与ELF常量0x4a43838=1e-6；不大于该数
跳到归零分支0x579072，并返回false。对于这些Cartesian位移输入，对应总位移
范数0.001 Å边界。此函数也用于其他向量，不能把0.001 Å泛化成所有调用的单位。
它不是机器epsilon，也不是本项目新设计的阈值。

## 轨迹记录与Gaussian索引的静态联系

update_mode0用 object+0x1660 索引 object+0x1668 的0x690字节记录。
set_status在0x5c1a3b–0x5c1a45递增该计数，并在第一个阶段建立起点记录。
climb完成分支在0x5cbf49–0x5cbf5a计算计数+1，对应新记录；后续把work2
坐标写到这个记录。update_mode0本身继续使用当前计数记录，而非刚准备的+1记录。
这支持“相对当前Gaussian阶段的起点记录”的解释，不能笼统称任意全局archive点。
但本轮没有执行整个climb/optimizer，仍不声称所有退出路径的work2/当前坐标配对
均正确，也不声称周期穿越时的位移已得到最短像处理。

## 后续零方向如何退出：静态证据

gen_randommode尾部0x5d865a再次调用n_normal，0x5d865f检查返回标志。
非零方向直接返回；零方向经0x5d870c–0x5d8725调用set_status(Allopt)，
随后control+0x120写为1。常量0x4a45ed4已核查为Allopt。
所以最终方向归零的原版分支是进入后续全松弛状态，不是无限重试随机向量。
这项仍为静态边界；初始位移为零不保证最终方向也为零，因为生成器可能加入新分量。

## 下一步

预旋转结果到实际参考方向的复制也已独立执行：
`probe_native_anchor_copy.py` 从0x5c45fe运行到0x5c4887，在N=2/5/15、
两个符号的6组输入中，把object+0x1788完整复制到+0x17e8，原向量不变，
6/6通过。产物为同目录`anchor-copy.json`。仅替代同形状内存分配，
预旋转输出是外部注入，未执行预旋转求解。因此它闭合了“保存哪个向量”，
不能证明预旋转本身保持输入方向或满足某个角度界。

由独立probe核验gen_randommode的保留/局部混合算术及最终零方向分支，再决定
Python状态转换的最小实现。不得把归零输入直接记为Calculator失败；也不得
把进入Allopt当成已经得到合格真实极小值，仍须执行并验证真实面淬火。

运行环境：mace_env Python，PYTHONNOUSERSITE=1，
PYTHONPATH=/tmp/pam-ssw-unicorn-probe:.，单线程BLAS；
`python research/ga_ssw/probe_native_direction_record_boundary.py`。
既有只测exact-zero的record-and-zero.json保留为初步诊断，不覆盖。
