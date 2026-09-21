# 位移保留项的几何作用：条件推导与核验边界

这是对已恢复系数关系的数学解释，不是性能结论，也不是新增角度限制。
独立数学复核与根agent推导一致。局部原子组分支的实际混合核验如下；
所有模式、真实几何生成和约束投影的完整复现仍未完成。

## 可证命题

设s为单位位移方向，cj非负，C=Σcj>0，每个uj的范数不大于1。
若混合公式确为

    z = 1.2 C s + Σ cj uj,

则令v=Σcj uj，三角不等式给出||v||≤C，从而

    z·s ≥ 0.2 C > 0， ||z|| ≥ 0.2 C。

允许v遍历半径C球时，从原点到以1.2Cs为圆心的球的切线给出

    angle(z,s) ≤ asin(1/1.2) = 56.442690°。

维数至少2时，v/C的轴向分量为−5/6、横向长度为sqrt(11)/6可取等。
最小范数的取等情形则为v=−Cs；两个等号一般不同时达到。
这说明1.2可解释为大于总扰动幅度的保留系数，但数值1.2本身仍属经验选择，
不能从上述证明推导出它在搜索效率上最优。

## 原版适用范围必须逐项核对

- update_mode0普通modelevel=0、无压缩分支已动态核验c9=1.2(c4+c5+c6)。
- generator中c9乘原n0的循环静态可见于0x5d5eb6、0x5d5ef5、0x5d5fba–0x5d5fc4。
- 索引使用**从0开始**；[rbx+0x18]是c3，不是c4。c4为+0x20、c5为+0x28、c6为+0x30。
- c4的pair/group分支在加权前分别于0x5d7dd7/0x5d7a8d调用n_normal；
  group分支加完跳0x5d80aa，不能与另一pair分支重复计数。
- c6的归一化为0x5d828e；c5为0x5da271。完整向量流还须动态核验，
  不能用这些调用地址代替实际输出比对。
- 只有投影保持s时，正交约束投影才保留上述界；任意掩码或坐标变换不自动满足。
- native n_normal有平方范数1e-6阈值，而不是严格非零即单位化。
  因此数学非零不等于通过原版阈值；无投影时C>0.005才由最坏界充分保证
  ||z||>0.001。更小C仍可能成功，不能把充分条件当必要条件。
- c9自身也有1e-6启用门槛。小系数和零位移不能套用单位s的命题。
- 后续CBD预旋转、biased rotation会再次改变方向；56.44°即使对混合成立，
  也不是最终Gaussian方向相对轨迹位移的保证。

## 对主线的含义

### 新增原指令证据：局部原子组混合

`research/ga_ssw/probe_native_group_mixture.py`从gen_randommode入口运行，
c6=0.5、c9=0.6、其余系数为0。为隔离混合算术，替代rd_numb为0.5，
localatomgroup_mode为给定数组，setconstraints为恒等操作。原始数组清零、
分量归一化、加权累加、最终全1掩码和归一化均实际执行。

N=2/5分别检查同向、反向、垂直、零seed、零local、全零及圆锥切向极值，
14/14通过，最大误差1.11e-16。同时校验非零返回地址与全零进入Allopt前的
地址；没有执行Allopt后续淬火。证据为
`research/ga_ssw/evidence/native-cbd-reentry-guard-20260917/group-mixture.json`。

这闭合了该受控分支的`normalize(0.6*seed+0.5*normalize(local))`，
并直接验证**零seed加非零local仍返回非零方向**。不能将零位移等同于释放。
probe不验证真实localgroup的生成分布或刚体投影，也不证明完整walker的性能。

命令：`env PYTHONNOUSERSITE=1 PYTHONPATH=/tmp/pam-ssw-unicorn-probe:.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/gengjianrui/.conda/envs/mace_env/bin/python
research/ga_ssw/probe_native_group_mixture.py`（合为一行执行）。

在已核验的局部原子组混合分支，原版提供的是“保留已实现位移，再叠加受控局部方向”的
机制，而不是简单每步重抽方向或只跟随位移。这与固定初始anchor的论文参考
路径应保持独立语义。是否减少回头、提高有效盆地发现/力调用，必须通过同预算
多体系端到端实验判断；这一推导不能代替实验，也不支持现在修改默认算法。
