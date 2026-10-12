# Native VC：三个cell Gaussian阶段都会引入随机模式生成

2026-10-12。只用于原程序控制流程，不是LJ相序、效率、原文势面或Python性能验证。
上传ELF未修改。CPU1754601因原件mode0664没有执行权限，启动前失败；保留失败。
CPU1754624对同SHA可执行副本完成跟踪：native/debugger3.25秒、无残留进程，
Slurm分配12秒；第一失败作业14秒。未建立E/F请求计数器，不拿这些耗时作算法比较。

[协议](protocol.md)，[离线分析](readout-v1/analysis.json)。原始`run-1754624/`
保存输入、原程序输出、101条trace、脚本和process.json。派生修正记录在
`readout-v1/corrected-trace.json`；分析脚本不改写原始数据。运行命令：

```bash
python research/ga_ssw/evidence/native-vc-lifecycle-trace-20261012/analyze.py \
 research/ga_ssw/evidence/native-vc-lifecycle-trace-20261012/run-1754624 NEW_OUTPUT_DIRECTORY
```

## 本次直接支持的结论

程序回显24原子、Run_type15、seed20261012、原子/晶胞NG均3，实际ratio2。
输入SSWsteps2在此运行产生初始化step0及一个搜索外步nsswstep1，不能报两个完整搜索外步。
本外步lcellmove为真，顺序为：首次get_random_mode0→gen；第一stage完成→update→gen；
第二stage完成→update→gen；第三stage完成→Allopt。原输出另有stage标签，足以排除
把每次逆通信返回当新stage的误读：

|cell Gaussian阶段|随机生成事件|disp_step日志行|finish ng日志行|climb调用/内层迭代|
|---|---:|---:|---:|---:|
|1|14|154|221|5|
|2|28|291|488|15|
|3|66|567|582|1|

总计68次ssw_move入口、21次climb入口、3次cell pattern和3次gen；只有2次
post-stage update。21次climb不是21个cell cycle。轨迹包括Cell CBD、位移、
Gaussian爬坡和Allopt；native自己记录了初始minimum及最后尝试/替代淬火，
本诊断不独立认证它们的完整应力、结构身份或最小值稳定性。

## 修正旧字段解释，而不是掩盖异常

先前文档错误地把`run_type`归为CSSW对象+0x100；trace沿用了这一解释，入口读值
明显不是合法模式。DWARF `<41634>`的结构名是`ssw_para`，其成员`<41699>`才是
run_type偏移256。指令0x5f246c把rbx设为参数记录0x53ed7a0；0x5f3750读取
`[rbx+0x100]`，是参数run_type。

原始事件25/63的该值15有效；这两条事件的object/nsswstep/lcellmove因错误地把rbx
当CSSW对象而无效。其r13与紧随update入口的rdi相同，后者解引用到对象0x54774a0；
分派至入口之间无这些状态写入，因此派生记录显式用相邻入口恢复对象与状态。
其余入口的错误run_type全部标为未观察（null），没有用期望值15填充。
原始记录保持原样；已更新脚本采用参数基址及r13描述符，但未为此重新运行PES。

## 对实现的含义与边界

这比旧静态证据前进一步：本次原版在同一外步的每个cell Gaussian阶段都调用随机
生成，不是只在外步入口调用一次。它仍不能证明每次完全独立重抽方向；
update在生成前读取持久状态，是否与前步位移混合仍需明确的数据流证据。
也不能把这三个native Gaussian阶段直接等同2014论文的CBD-cell cycle。
因此不据此删除Python现有cell方向，也不把`fresh_random_each_cycle`改称native parity。
下一项只核对模式生成中持久位移的实际作用，不做参数扫描或重复TiO2预算释放实验。

## 同输入向量快照：历史项确实参与（1754853）

新增只读快照，不改输入/ELF/Python算法。CPU作业15秒，native/debugger3.471秒，
正常结束、无trace错误/截尾/残留。原始路径`run-1754853/`，
[派生代数检查](mixing-readout-v2/analysis.json)由`analyze_mixing.py`产生。

|阶段|模式+0x18|模式+0x28|历史+0x48|有效历史向量范数|
|---|---:|---:|---:|---:|
|1|0.8|0.2|0|不解释未使用入口存储|
|2|0|0.2|0.3|1|
|3|0|0.2|0.3|1|

本臂lrigid=0。stage2/3的历史缩放逐项等于0.3乘入口向量，后续局部项逐项
等于0.2乘scratch向量再相加；代数检查最大差均为0。stage1的两个加法分量
也分别满足0.8、0.2乘加。这里只核对执行语义，不把精确浮点相等当科学效果。
因此“调用generator就是完全独立重抽”已被本次实际运行否定；.8/.2不是
二选一的概率。首段和后续段的配方也不同，不能概括为每段固定.8/.2。

静态链路：update_mode0从持久轨迹构造差值、施加约束并归一化，然后向量送入
gen_randommode；历史系数由保留模式系数和乘1.5。本次保留和为.2，故实际.3。
局部cell分量的完整物理构造、原子/晶胞度量映射仍有边界，不能仅凭scratch
名字宣称已恢复完整native VC生成器。

第一次派生分析v1对未启用历史的首段入口存储计算范数，平方溢出为Infinity；
这是分析错误，不是native数值失败。v2对这种未消费存储显式不作解释并禁止
非有限JSON输出。原始快照/v1保留，不为修改分析再运行PES。

Python固定胞LocalDirectionState已经有Gaussian中心位移反馈（独立原子分支系数1.2）；
block_ssw的九维cell更新是每周期纯fresh随机输入，且没有native同型cell Gaussian状态。
vc_reference则在同一log-strain chart的多Gaussian阶段保留同一anchor。
三者不能互相混称。下一设计若借用历史机制，必须先说明在哪个状态空间、哪个
阶段边界取位移；不能把native系数机械套给block前后cell差，也不能把1.2/1.5
差异单独当成应当修复的错误。本轮未改任何公共VC接口或默认。
