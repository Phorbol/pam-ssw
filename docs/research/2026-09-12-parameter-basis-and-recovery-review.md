# 参数依据与失败恢复：按用户澄清修正研究定位

用户明确：旧PAM的Safe-LBFGS和adaptive-bias参数未经过细致测试。因此“继承自PAM”
只能表示实现来源，不构成参数有效性、最优性或通用性的证据。之前的短流程只证明
接口一致性；819 vs605调用不能评价完整PAM或经过合理校准的自适应策略。

## FD误差与位移规范

当前paper_dimer_direction和generalized_dimer使用单边力差，单位方向n满足整体
L2范数1，h=fd_step是整体配置位移而不是每原子位移。集体方向的每原子RMS为
h/sqrt(N_active)。在背景势光滑且每次力误差受epsilon_F约束时，误差界具有
O(h)+O(epsilon_F/h)形式；中心差分截断为O(h^2)，仍有噪声除h。没有力精度和
局部曲率变化尺度，无法指定统一的最佳h；机器精度也不能替代SCF/MLIP力精度。
例如epsilon_F=1e-6 eV/A、h=1e-4 A，双端误差上界量级2e-2 eV/A^2，已经等于
当前rotation_tol。这是量纲示例，不是对当前EMT/GFN2/MACE误差的测量。

1e-4 A目前只是诊断值，未完成多后端h收敛验证。后续用固定结构/方向检查
1e-2、3e-3、1e-3、3e-4、1e-4 A的Hn、Rayleigh曲率和旋转方向稳定区间，并区分
裸PES、加Hookean、加LS；再以真实多步成本/盆地发现评估可用区间。上述是预先
给定诊断网格，不是新默认。当前Cu/native-LS失败发生在HVP之前，与fd_step无关。

参考：SciPy官方数值微分文档指出过小步长可因减法消去/有限精度阻止精度改善：
https://docs.scipy.org/doc/scipy/reference/generated/scipy.differentiate.derivative.html
本项目误差表达是对单边力差公式的直接推导，不照搬其通用函数默认步长。

## 周期LS问题优先于优化器补丁

已存储末次line-search证据表明native-MIC在极小位移下发生镜像切换和有限LS力跳变。
这是当前“每对原子只使用最近image”的周期扩展不光滑，未发现指数势分支内的
力符号/导数错误。它应作为建模/实现适用域缺口处理，不能因叫native就认为已被
原版背书。已有periodic-images枚举并冻结所有选中image；两者不同数学势，需要
核验原版邻居/image语义后再决定本体实现，不能自动切换掩盖失败。

## Safe-total实际保护边界

独立flat Safe-total已有步长限制、Armijo回溯、正曲率secant筛选、有限性检查和
返回最后已接受点；但非下降方向或20次线搜索均失败后直接退出，没有清空历史
再用缩放最速下降重试的恢复分支。返回状态并保留合法点是失败保护，不是恢复。
旧PAM relax.py还有在已接受步的bias-image-signature变化后清历史的机制；当前
flat实现没有相同事件接口。该旧机制也不是线搜索失败后的fallback，更不保证
适用于这个LS的镜像记录。不得把两个实现笼统称为全部保护机制等价。

若势在尖点没有唯一普通梯度，即使清历史使用-g也不保证下降；因此不能用增加
重试或接受升能步骤修饰成功率。待势/梯度一致性通过后，对光滑目标上的异常
再检验“一次清历史+下降方向恢复”，与ASE LBFGSLineSearch/SciPy L-BFGS-B
作相同落点证书和完整成本比较。尚未实现或启用该恢复分支。

## 参数地位与顺序

- 0.03 outer fmax由研究runner在用户允许范围内选取；复用到LS预淬火是当前
  配置耦合，不是原版要求。原版有optsoftmax退出，Pythonconverged-only需单独核正。
- PAM0.6eV、0.05eV/A^2、width/weight界限，Safe-total history10/初始scale1/70，
  以及rotation_bias100均未被本组研究证明为合适的跨体系默认。
- 优先：周期LS目标/梯度和原版语义 → 预淬火停止/继续规则 → 后端FD稳定区间 →
  光滑目标上优化器baseline与恢复 → 自适应bias参数的受控跨体系评估。
- 不做所有参数全组合搜索，不围绕一个Cu111调参，不改最终评估资格掩盖失败。
