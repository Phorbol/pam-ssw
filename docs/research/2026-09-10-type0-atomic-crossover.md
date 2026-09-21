# TYPE0 原子/合金 crossover：独立实现与原 JAR 对照

2026-09-10。本轮只移植几何 crossover，不改 `paper_ga.py` 控制器，不声称完成 TYPE0 GA-SSW。

## 问题与来源

既有 TYPE3 按分子分组切割，不能用于没有单元拓扑的原子/合金父代。
本实现依据上传 `sgn.jar` 的 CFR 反编译 `ga_cluster_cell/Cut.java`、
`CutBasicAbstract.java`、`Cross.java` 和 `Compete.java`；原 JAR 直接执行用于核验。
没有引入新物理策略，保留发行版的经验分布和拼接尺度；不把它们称作最优设计。

## 已实现

`pamssw.standalone.atomic_ga` 新增：

- `cut_atoms`：几何中心归零；累计 x/y/z 随机旋转；按 `z+x/pl` 切割，直到原子数差小于2；
  对切面作原版 y 旋转，两半 z 分别加/减 0.3 Å；输出原子索引谱系。
- `build_atomic_pool`：所有父代同组成，允许元素顺序不同；原 Compete 权重抽样 n 个父代槽；
  每槽消费两个索引随机数但只使用第一项；默认每槽切 `10**(元素种数+1)` 次。
- `cross_atomic_pool`：从两池独立均匀抽样，直到元素计数精确匹配，按 son→daughter 拼接。
  允许同一父代供给两半；每个原子保留父代编号和原始索引。

所有函数用调用者提供的 RNG，不访问 calculator、LASP 或 Java。pool 显式记录父代槽、切割次数和
累计切割尝试数；child 记录配对尝试数和池索引。源规则没有额外 docking/碰撞修复，本模块也不增加。
caller 应随后做适用域/结构过滤和真实势面优化；几何候选不是合格极小值。

## 参数、边界与有意偏差

- 长度单位 Å；0.3 Å 分离和角度中的 3.14159 均为原程序常数。
- `max_cut_attempts`、`max_pair_attempts` 为必须显式给出的有限操作预算；耗尽抛出
  `SamplingExhausted`，不伪造子代。不复刻原程序无穷循环。
- `cuts_per_parent_slot=None` 保留原默认；显式缩小该值会改变池采样密度，必须在实验配置中报告。
- 父代输入不修改；私有副本保留重复中心化行为。非周期、无约束；至少2原子。
- 拒绝原 Compete 的 <=2 父代/零能量跨度退化，而不是复制 NaN。
- 拒绝奇异切面的 NaN 成员归属，而不是复刻 Java 的数组越界异常。
- child 只携带元素和坐标，calc=None；不自动拼接来自不同父代的磁矩、电荷等索引相关数组，
  也不保留原程序可视化包装 cell。这些需要由 caller 根据 calculator 明确设置。
- 尚无 TYPE0 mutation、批次配额、初始化生成器或主控制器接入。

## 可重现实验

`research/ga_ssw/java/Type0CutProbe.java` 编译的只是调用探针；被调用 Cut/Cross 来自原始 JAR。
设置 `Math.random` 的 Random 种子只用于重复随机几何算子测试，不改变 JAR，也不涉及 LASP 保护逻辑。

再生命令：

```bash
python research/ga_ssw/run_type0_reference.py --reference-root /home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909
python -m pytest tests/standalone/test_atomic_ga.py -q
```

fixture 保存 JAR SHA256、输入、种子、输出。三个 Cut 种子为17、71、555。
完整 Cross fixture 使用种子71、三个6原子 Cu3Ag3 几何父代、能量元数据0/1/2；
原默认共3000切割，输出3子代。测试专用 Java Random 重放与 JAR 输出的16个 draw 精确一致，
再比较完整池构建→匹配→三次子代生成。坐标使用绝对容差2e-12 Å、相对容差0。实测最大坐标差：Cut 4.44e-16 Å；
完整 Cross 1.33e-15 Å。完整 Cross 的3000次切割共消耗4558次切面尝试。

本轮新增6项测试通过，与既有 TYPE3 算子联合运行共23项通过：原 Cut 对照、原完整 Cross 对照、Cu/Ag 13原子候选组成与输入隔离、
确定性重放及原池规模、非法输入/切割耗尽、配对耗尽/奇异切割。
这些是实现一致性与几何流程证据，未运行真实势面端到端优化，未证明 crossover 提高搜索效率。
接下来需要接入 TYPE0 mutation/控制器，并用真实原子/合金体系完整验证；不能把这里的
Cu3Ag3 固定几何和虚拟能量元数据当作真实材料实验。
