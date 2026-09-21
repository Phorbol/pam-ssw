# 自动轴/原子组选择：函数恢复与调用链边界

2026-09-17 后续 root 审查：原始诊断的 ABI 和公式解释已纠正，下面保留错误记录。
独立几何函数已实现；完整方向控制器尚未闭合，不得称为已恢复的默认搜索策略。

## 新增、已核对的证据

`pamssw/standalone/native_local_group.py::select_native_local_group` 独立恢复
Run_type=5 的选择函数，不调用 LASP、不调用计算器，尚未接入 walker。
令当前坐标为 b、调用者保存坐标为 a，eligible 为每原子第一个自由度标记为真的集合：

1. 在 eligible 中找到 `||b-a||` 最小的 first。
2. 将距 b[first] 严格小于 2 Å 的 movement 条目清零，再取最小值得到 second。
   这是机器指令直接执行的操作；远处零位移也可参与并列，但当前证据不声称
   它与使用稳定索引次序的邻域选择一定产生不同结果。
3. `q[i]=||b[i]-b[first]||+||b[i]-b[second]||`；轴首原子取 eligible 中 q 最大者。
4. 组包含 eligible 中 `q > max(3 Å, max(q)-3 Å)` 且非轴首原子的条目。
5. 用一次均匀随机抽样选组中轴次原子。空组不抽样，原生索引零以 None 表达。

2/3 Å 是该二进制的经验规则，不是为本项目调出的参数，也不宣称通用最优。
mask 为每原子三个 int32，步长 12 字节；坐标步长 24 字节。
原生函数范围限定为 0x57e940–0x57f990；之后是另一个 get_atompair 函数。

针对性验证：`python -m pytest tests/standalone/test_native_group_selection.py
tests/standalone/test_native_local_group.py -q`，10 passed。原指令差分见
`research/ga_ssw/probe_native_axis_group_selection_v3.py` 及对应 JSON，root 复跑
22/22 pair、group 和实际 RNG 调用次数一致，含空组、边界及旋转输入。
旧 v2 中有效输入也由 root 逐项比对；旧探针错误结论不再作依据。

组合验证 `research/ga_ssw/probe_native_selected_group_geometry.py` 使用 C2H6、CH3OH、
C6H6 几何 × 三个抽样值，共 9 项。选择均一致；6 项有有效轴，原生局部方向→
刚体投影→0.6 seed+0.5 local 混合与 Python 最大差异 4.44e-16；3 项空轴跳过
generator，未伪造回退。输入和输出保存于 `native-selected-group-geometry-20260917.json`。
这是各独立函数的组合对照，不代表 allopt 真实调用链已保留同一个 pair。
没有能量计算，也没有新增科学效果结论。

## 尚需关闭的集成问题

allopt 是重复进入的状态函数。0x5d47d5 调用结束判断，0x5d47e0 在未结束时跳到
0x5d4a58 继续优化；只有结束分支调用选择函数。因此先前“在优化入口选择”的解释错误。
是否达到真实面力阈值，仍须按判断函数的具体退出原因区分。
随后 get_atompair 接收同一 pair 地址并可覆盖。后续实测与状态证据见
[调用链恢复](2026-09-17-native-axis-caller-lifecycle.md)。cart_copy 已确认由 NewStart
及后续周期重置的 copy_str 建立，不能默认等同于每个 Gaussian 中心。

## 最终 pair 刷新与退出语义：本轮关键发现

root 新建 `probe_native_get_atompair_v2.py`，实际执行 getpair、neighboringlist、
check_forbiden、get_dist、reci_latt 和 species_radius。只替代 RNG、分配/释放、
整数尺寸乘法、floor/acos 数学库；没有 stub 掉几何判定。显式设置 Run_type=5、
fixatom=0、全自由 Cartesian mask、30 Å 立方盒；这些是受控输入，不宣称原版默认。
先前 bounded probe 缺少 para/descriptor 初始化，其 blocked 不是算法卡点。
原生指令步数上限从 1e6 提升到 1e7 后九例均正常返回；这不是 PES 预算或算法阈值修改。

`probe_native_axis_pipeline.py` 将 findleast 的真实输出送入 getpair，再将改写后的
pair 和未改写 group 送入原生局部方向生成器。三个分子 × 三个 RNG 前缀，共9例：

- 9/9 pair 被改写，不能跳过 getpair 后声称完整原版局部方向。
- 9/9 最终局部方向/投影/混合与 Python 几何运算一致，最大差异 1.11e-16。
- 3/9 从 check_forbiden 接受分支退出；6/9 在 distance/fixatom 拒绝计数达到150后退出。
  150不是总尝试次数，forbidden/元素拒绝不增加该计数。
- 第二原子在检查前已写入输出，耗尽计数后未清空或恢复，故返回 pair 不构成几何检查
  通过的证据。这是原版控制流程，不据此认定原版搜索效果差或原子结构无效。

归档：`research/ga_ssw/evidence/native-axis-pipeline-20260917.json`，包含完整输入、
随机序列、函数路径、分支计数和输出。此前以固定 pair=(1,2) 做的9例探针有8例计数耗尽，
与此处按真实 findleast 输出初始化的6例不是同一实验，不混用分母。

决定：保留独立参考几何组件；当前搜索默认不变。完整移植需显式保留“检查接受”和
“计数结束”的差异，不能无声改成严格拒绝，也不能将后者标为检查通过。
getpair 尚未成为独立 Python 实现；Q 描述符与完整方向状态仍待闭合，不能报告整体完成。

## 可追溯的调用位置

## 本轮 getpair canonical 对照（2026-09-17）

`probe_native_get_atompair_v2.py` 的 Python 刷新实现已完成独立对照。canonical 输入集为
8 个体系 × 3 个 RNG 前缀，共 24 例；所有坐标整体平移到 30 Å 盒内，24/24
`completed`。edge 集为 C/Cu 混合体系的 heavy-first、空 second、重复 pair、
free/one-fixed/all-fixed、三种 RNG 及 90° 旋转，共 24 例；24/24 完成，pair、draw、
accepted 和三类拒绝计数全部一致。证据分别见
`research/ga_ssw/evidence/native-getpair-canonical-20260917.json` 与
`native-getpair-edges-20260917.json`。

旧 centered 坐标 pipeline 的 6/9 及固定 pair 的 8/9 统计解释撤回：
`get_dist` 对坐标先 wrap，而 `check_forbiden` 使用 wrapped neighbor 减原始端点，
使结果依赖坐标 chart。该限制由 `native-pair-coordinate-chart-20260917.json` 的
C6H6 shift 对照记录；旧产物保留用于追溯，不再解释原版正常行为。

getpair 首原子随机分支的条件是 `min(Z)<10 && max(Z)<10`，不是随机候选原子的 Z
条件；相关 literal `0x4a434bc` 的值为整数 1。该修正仅限现有 caller/指令证据。

上传ELF `ssw_fixlat_mp_allopt_` 在0x5d481f调用
`find_leastmoveatoms_` (0x57e940)，参数来自N、object+0x6d8的参考坐标、
object+0x170的当前坐标、object+0x8a8的mask、object+0x1ad8的pair和
object+0x1ae0的group。参考坐标字段名为cart_copy，不是整数记录。
另有0x5d48fc调用get_atompair_；覆盖条件和最终组策略需完整核对。

## 早期未通过审查的原因（历史，保留供追溯）

- 探针曾把参考坐标写成整数，后来修正；不能沿用那批解释。
- 当前脚本的mask写入步长为0x18，而文档曾宣称3个32-bit整数/atom；
  此矛盾尚未由调用者真实descriptor及callee逐项访问共同解决。
- 初期passed仅正常返回；后来才加入STOP断言，仍没有完整Python参考公式断言。
- 当前JSON缺少每例完整reference/current输入，需与脚本main共同解释；
  不够支持独立复现或对选择策略的通用断言。
- 第二临时数组曾被误解为位移、固定首原子距离，均应撤回。
  保持current不变而改变reference时，该数组实际发生变化。
- 最后三例报告的delta和pair恰好全部符合argmax(delta)，并非该规则的反例；
  第二数组与两倍距argmin(delta)的距离相符，但这仍是待判别假设。
  需要非退化候选、不同mask和原始读源指针共同验证，不能据函数名定论。

## 早期判别计划（ABI/几何部分已由上述新证据替代）

先恢复mask的元素宽度、维度与允许自由度语义，并保存全部输入；
然后固定current、独立改变reference，控制最小/最大位移的原子索引和并列情况，
记录两套临时数组及group写入。候选解释包括最小位移参考点、两个参考点距离和，
也保留ABI或scratch采样错误。核实真正控制group的距离公式与严格阈值后，
再处理get_atompair覆盖及Python移植。当前不影响独立CBD阶段求解器的有界验证，
但仍阻塞完整native方向生成策略的启用。

## Canonical combined chain after the coordinate-chart correction

Root reran `python -m research.ga_ssw.probe_native_axis_pipeline --canonical`.
All 9 selected-pair refreshes and final c6 outputs matched independent Python;
maximum output error 1.67e-16, with 7 counter-limit exits in these particular
fixtures. This does not estimate production rejection frequency. The earlier
centered-input artifact was not overwritten. Source/output:
`research/ga_ssw/probe_native_axis_pipeline.py`,
`research/ga_ssw/evidence/native-axis-pipeline-canonical-20260917.json`.
Both selection and refresh are now compared before projection/mixing. The
probe still composes isolated functions and does not execute allopt or CBD.
