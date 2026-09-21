# GA框架与接口：以SSW/VC内核为优先

用户纠正主线后，暂停GA/LS/RC新增开发。已有实验保留，不因投入过逆向就继续移植外层。
没有LASP Java GA、ASE-GA、USPEX在相同体系/物理后端/初态/总E-F预算下的直接结果，
因此不能给它们做性能排序；GA+SSW相对SSW的提升也不能隔离GA实现本身的优劣。

2026-09-10核验官方资料：

- ASE-GA已从ASE分离为独立包。提供可配置population、配对、mutation、comparators与停止条件；
  bulk教程包含晶胞交叉、strain、permustrain与soft mutation及局部变胞淬火。
  它是有论文和实际教程支持的框架，不等于一个无需配置且普遍最优的固定算法。
  https://dtu-energy.github.io/ase-ga/
  https://dtu-energy.github.io/ase-ga/tutorials/ga_bulk.html
- USPEX26（官方手册2026-06-23）明确列出abinitioCode=20 ASE接口，99用户外部程序。
  99协议读取geom.in，写geom.out(POSCAR)与energy.txt；脚本负责与外压等条件一致。
  这是可扩展接口证据，不是本机USPEX已经安装或已完成ASE接入的证据。
  https://uspex-team.org/static/file/uspex_manual_english_v26.pdf §2.4、3.11。

上传Java与LASP为结构文件/进程协议，不是Python ASE Calculator回调；上传示例同时存在
LAMMPS及external Gaussian路径，不能将其概括为只支持LASP NN pot。
详细调用位置由Java接口只读审查记录。以后若接ASE，应区分：
1. 在LASP external通道提供E/F，仍由LASP二进制执行SSW；
2. 独立SSW调用ASE E/F/stress，不依赖LASP。
本项目主目标是第二条。无需为兼容Java外层而推迟SSW/VC核心工作。

未来外层候选优先评估ASE-GA可复用部分；USPEX作为周期晶体搜索的成熟对照候选；
原Java专有算子/分区调度只有在隔离贡献后才考虑吸收。此为工程优先级判断，非性能结论。
