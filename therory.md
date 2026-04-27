下面把 **CBD / BP-CBD（你说的 CBD-bias）/ SSW / DESW / SSW-crystal / VC-DESW / SSW-RS / LS-SSW** 统一为一套形式化框架。核心结论先说：**这一系列方法可以被更优雅地理解为“带软模方向预言机的跨盆地 proposal 生成器”，而不是严格的热力学采样动力学；若把度量、偏置、软化、接受准则、端点约束都写成同一个流形优化问题，它确实可以被改造成更简单、更鲁棒的一族算法。**

## 1. 方法谱系：从“找 TS”到“全局 PES 图搜索”

复旦刘智攀组这条线大致是：2010 年 CBD 用 constrained Broyden minimization + dimer 做过渡态搜索；2012 年 BP-CBD 把 CBD 与 bias potential / basin filling 结合，用于从初态出发探索多步反应路径；2013 年 SSW 把这种“偏置驱动越过势垒”的思想推广成无预设终态的 PES 搜索；同年 DESW 用双端点的 surface walking 连接两个已知 minima 并定位 TS；2014 年 SSW-crystal 把原子和晶胞自由度结合；2015 年 SSW-RS 把 SSW 与 DESW 组合成反应采样；2015 年 VC-DESW 又把 DESW 推广到变胞固相相变；2024 年 LS-SSW 则通过 local softening 处理强共价键导致的 corrugated PES 问题。复旦主页的论文列表和 RSC / PubMed 摘要能对应到这些节点。([复旦大学教师个人主页](https://faculty.fudan.edu.cn/fdzpliu/zh_CN/zdylm/644118/list/index.htm))

更抽象地说，这不是一堆彼此独立的算法，而是同一个思想的不同边界条件：**CBD 是局部鞍点优化器；BP-CBD 是带人工势垒填充的单端路径推进器；SSW 是随机软模驱动的跨盆地 proposal；DESW 是给定两个端点后的双端 proposal；SSW-crystal / VC-DESW 是把坐标从原子笛卡尔空间扩展到“晶胞 + 原子”的广义坐标；LS-SSW 是对局部谱结构进行人工软化的预条件化版本。**

------

## 2. 统一数学对象：在流形上的“软模—偏置—淬火”算子

令体系构型为 (x\in\mathcal M)，势能或焓为

[
U(x)=
\begin{cases}
E(R), & \text{分子/团簇}\
E(s,L)+p\Omega(L), & \text{晶体， } R_i=L s_i .
\end{cases}
]

这里 (\mathcal M) 不应简单看作普通欧氏空间，而应是去除了整体平移、整体转动、等价原子置换、晶胞表示冗余后的构型流形。给定度量 (G_x)，梯度和 Hessian 应写成

[
\nabla_G U = G^{-1}\nabla U,\qquad
H_G = G^{-1/2}(\nabla^2 U)G^{-1/2}.
]

这一点很重要：SSW-crystal 原文已经把晶胞自由度纳入算法，使用晶胞矢量、应力张量和 enthalpy derivative 来构造 lattice mode，并指出晶胞与原子运动都需要参与固相相变搜索；VC-DESW 又明确使用“lattice vectors + scaled atomic coordinates”的 generalized coordinate 来描述晶体 PES。

统一的 SSW-family proposal 可以写成：

# [ x_{k+1}

\mathcal Q_U
\left[
\Phi^{H}_{U+B_H}(x_k;\xi_k)
\right].
]

其中：

[
\mathcal Q_U(z)=\arg\min_{y\in \text{basin}(z)} U(y)
]

是把偏置去掉后在真实 PES 上 fully relax / quench 到一个 minimum；(\Phi^{H}_{U+B_H}) 是在加了临时偏置 (B_H) 的 PES 上走 (H) 个小步；(\xi_k) 是随机方向或端点约束信息。SSW-crystal 文中也明确说：加 bias 的目的是驱动结构跨越 TS，抵达高能构型后会移除所有 bias 并在真实 PES 上完全弛豫；这与 metadynamics 中长期保留 bias、逐渐填 basin 的目的不同。

------

## 3. CBD 的本质：Rayleigh quotient 最小化 + 反号梯度流

Dimer / CBD 的核心是不用显式 Hessian，而用有限差分 Hessian-vector product：

[
H(x)n \approx
\frac{\nabla U(x+\varepsilon n)-\nabla U(x-\varepsilon n)}{2\varepsilon}.
]

软模方向 (n) 是球面上的 Rayleigh quotient 问题：

# [ n^\star

\arg\min_{|n|_G=1}
\rho_x(n),
\qquad
\rho_x(n)=
\frac{n^\top \nabla^2 U(x)n}{n^\top G_x n}.
]

在 index-1 saddle 附近，如果 (n) 对准唯一负曲率方向，则 modified force

# [ F_{\mathrm{dimer}}

-\nabla_G U
+
2, n,\langle n,\nabla_G U\rangle_G
]

等价于：沿 (n) 方向上坡，沿所有正曲率方向下坡。线性化后，若 Hessian 本征值为

[
\lambda_1<0<\lambda_2\le \cdots,
]

则在 (n=e_1) 时，流

[
\dot x=-(I-2nn^\top_G)\nabla_G U
]

沿 (e_1) 的线性系数是 (\lambda_1<0)，沿其他方向是 (-\lambda_i<0)，因此 index-1 saddle 变成吸引不动点。这就是 dimer/CBD 能找 TS 的数学核心。

CBD 相比普通 dimer 的算法设计，是把 dimer rotation 与 constrained Broyden / L-BFGS 式方向更新结合：文献摘要称其 rotation 只需一次能量和梯度计算来确定旋转角，translation 会持续执行直到满足终止准则，并对平行于 dimer 方向的 translational force 做 damping，从而提高效率和稳定性。([ResearchGate](https://www.researchgate.net/publication/263978975_Constrained_Broyden_Dimer_Method_with_Bias_Potential_for_Exploring_Potential_Energy_Surface_of_Multistep_Reaction_Process))

------

## 4. BP-CBD / CBD-bias：把“找一个 TS”变成“从一个 basin 走出去”

BP-CBD 在 CBD 上叠加 basin-filling bias。其思想可以写成：

[
U_h(x)=U(x)+B_h(x),
\qquad
B_h(x)=\sum_{\ell=0}^{h-1}
w_\ell K!\left(
\frac{\langle x-x_\ell,n_\ell\rangle_G}{\sigma_\ell}
\right),
]

常见核可抽象为 Gaussian：

[
K(s)=\exp(-s^2/2).
]

在 bias center 处，

# [ \frac{d^2}{ds^2}\left[w e^{-s^2/2\sigma^2}\right]_{s=0}

-\frac{w}{\sigma^2}.
]

所以一个正 Gaussian bias 在当前 minimum 附近会“抬高 basin 底部，同时沿指定方向减小曲率”。当

[
\lambda_n-\frac{w}{\sigma^2}<0
]

时，原本稳定的局部 minimum 在修改 PES 上会沿 (n) 方向变成不稳定，从而可以被推出 basin。

这就是 BP-CBD 与后续 SSW 的物理本质：**不是模拟真实动力学，而是在构造一个临时变形的 PES，让体系沿某个方向穿过势垒；之后再回到真实 PES 验证 minimum / TS。** BP-CBD 文献摘要明确说，该方法把 constrained Broyden dimer 与 bias potential 的 basin filling 结合，使体系能沿给定反应方向走出 energy trap；每个 elementary step 可指定如断键这样的反应方向，并通过 biased dimer rotation 细化和保留为 normal mode。([EurekaMag](https://eurekamag.com/research/057/495/057495675.php?srsltid=AfmBOoqlkQfPRZT8VnBF-_4u3iikdkYeERPZScZyon89bEPpL5l-cy2q))

------

## 5. SSW：把“给定反应方向”替换为“随机软模方向”

SSW 的关键创新，是把 BP-CBD 中需要人为指定的反应方向替换成随机方向经过 biased CBD rotation 后得到的“随机软模”。SSW-crystal 的方法部分总结得很清楚：SSW 使用自动 climbing mechanism，把一个 minimum 沿随机方向推到高能构型；移动时会逐个加入 Gaussian bias；方向 (N_n) 希望接近低本征值 Hessian 模式，但由于 biased CBD rotation，它不必严格收敛到最软本征矢，而是同时保留初始随机方向的特征并靠近某个 soft eigenvector。

所以 SSW 的单步可以抽象为：

[
\begin{aligned}
n_h
&=
\operatorname{SoftMode}
(x_h;n_{h-1},\xi),\
B_{h+1}
&=
B_h+
w_hK!\left(
\frac{\langle x-x_h,n_h\rangle_G}{\sigma_h}
\right),\
\tilde x_{h+1}
&=
x_h+\sigma_h n_h,\
x_{h+1}
&=
\operatorname{Relax}*{U+B*{h+1}}(\tilde x_{h+1}).
\end{aligned}
]

走完 (H) 个 bias-relax 小步后，移除 (B_H)，在真实 (U) 上 quench 到新 minimum。SSW-crystal 文中把这一过程拆成：更新方向、加入 Gaussian、沿方向位移、在 modified PES 上局部优化；到达高能构型后移除 bias 并 fully relax。

从优化理论看，SSW 是一种 **negative-curvature guided basin escape**；从图论看，它是在 minima graph 上生成边：

[
m_i \longrightarrow m_j,
]

并保留中间 pseudo-path 信息。相比 basin hopping 直接随机扰动再 quench，SSW 的优势是 proposal 有连续路径，因此之后可以提取候选 TS 或交给 DESW 精修。

------

## 6. DESW / VC-DESW：把 SSW proposal 条件化到两个端点

DESW 的形式化很自然。给定两个 minima (a,b)，定义两个 walker：

[
x_A^0=a,\qquad x_B^0=b.
]

每个 walker 都做 bias-relax surface walking，但方向 oracle 需要同时考虑软模与“朝向另一个端点”的约束。抽象成正则化 Rayleigh quotient：

# [ n_A^\star

## \arg\min_{|n|*G=1} \left[ \rho*{x_A}(n)

\eta \langle n,\log_{x_A}x_B\rangle_G^2
+
\mu d_G(n,n_{\mathrm{prev}})^2
\right].
]

这里 (\eta) 是端点吸引，(\mu) 是方向记忆，(\rho) 是软模性。DESW 的公开摘要也正是这个思路：两个 images 分别从 initial state 和 final state 出发，stepwise 向彼此 walking；surface walking 由重复添加 bias potential、local relaxation 和 CBD 方向修正组成；它不做完整路径的迭代优化，却能建立低能 pseudo-path 并方便定位 TS。([ResearchGate](https://www.researchgate.net/publication/283197116_Variable-Cell_Double-Ended_Surface_Walking_Method_for_Fast_Transition_State_Location_of_Solid_Phase_Transition))

VC-DESW 则是 DESW 的变胞版本：它把晶胞和 scaled atomic coordinates 合成 generalized coordinate，目标是固相相变中的快速 pseudo-pathway building 和 TS location，并强调不需要昂贵 Hessian 计算和迭代路径优化。([ResearchGate](https://www.researchgate.net/publication/283197116_Variable-Cell_Double-Ended_Surface_Walking_Method_for_Fast_Transition_State_Location_of_Solid_Phase_Transition))

------

## 7. LS-SSW：本质是对 Hessian 谱的局部软化预条件化

LS-SSW 面对的问题是：强共价键体系的 PES 很 corrugated，局部高频模式会把 SSW 困住。PubMed 摘要说 LS-SSW 通过加入 pairwise penalty potentials 和 self-adaption mechanism 来变换 vibrational mode space，使强局部模式 delocalize and soften，从而帮助 SSW 更有效捕捉通向邻近 minima 的局部原子运动，同时降低反应势垒、减少 local trapping time。([PubMed](https://pubmed.ncbi.nlm.nih.gov/39636281/))

统一形式中，LS-SSW 就是在 proposal 阶段使用

[
\tilde U_\theta(x)=U(x)+P_\theta(x),
]

其中

[
P_\theta(x)=
\sum_{(i,j)\in \mathcal P(x)}
\alpha_{ij},\phi_{ij}(r_{ij};r_{ij}^0,\sigma_{ij})
]

是自适应 pairwise penalty。若某个强局部键伸缩方向为单位矢量 (u_{ij})，那么在当前结构附近可以把软化写成 Hessian 修正：

## [ \nabla^2 \tilde U_\theta \approx \nabla^2 U

\sum_{ij}\alpha_{ij} u_{ij}u_{ij}^{\top}
+
\text{higher-order terms}.
]

于是 stiff eigenvalue

[
\lambda_{\mathrm{stiff}}
]

被变成

[
\tilde\lambda_{\mathrm{stiff}}
\approx
\lambda_{\mathrm{stiff}}-\alpha.
]

这正是“local softening”的线性代数解释：**它不是单纯加速器，而是对局部 Hessian 谱进行 proposal-only 预条件化。** 但这也意味着 LS-SSW 必须把最终 minima、TS、barrier 全部回到原始 (U) 上验证；否则得到的是软化 Hamiltonian 的路径，而不一定是真实 PES 的路径。

------

## 8. 统计力学视角：SSW 是 PES 图探索，不是自动正确的 canonical dynamics

把所有 minima 记为节点集合 (\mathcal V={m_i})，SSW-family proposal 给出转移概率

[
Q_\theta(i\to j).
]

若接受率写成普通 Metropolis：

[
A(i\to j)=\min\left(1,e^{-\beta(E_j-E_i)}\right),
]

只有在 proposal 近似对称，即 (Q(i\to j)\approx Q(j\to i)) 时，才对应 canonical distribution。更一般的正确接受率应是 Metropolis–Hastings：

[
A(i\to j)=
\min\left[
1,
e^{-\beta(F_j-F_i)}
\frac{Q(j\to i)}{Q(i\to j)}
\right].
]

这里 (F_i) 还应是 basin free energy，而不仅是 minimum energy：

# [ F_i

## E_i

k_BT\ln Z_i^{\mathrm{basin}}.
]

这一点是 SSW 框架最容易被误解的地方。文献中 global optimization 方法本来就常常放弃严格 detailed balance 与有限温度 free-energy kinetics，目标是更快跨越 minima 找 GM 或 reaction path；SSW-crystal 使用 Metropolis MC 作为 structure selection module，但这更适合作为优化/探索准则，而不是严格热力学采样证明。([Nature](https://www.nature.com/articles/s41524-022-00959-5))

因此，应区分三类目标：

**全局优化**：可以接受非物理 proposal，只要能快找低能 minima。

**反应网络发现**：SSW 负责找候选 IS/FS pair，DESW/CBD 负责真实 PES 上定位 TS；之后用 TST / microkinetics 处理速率。Nature 综述中也把 SSW-RS 描述为先由 SSW 生成 reaction pairs，再由 DESW 定位路径上的 TS。([Nature](https://www.nature.com/articles/s41524-022-00959-5))

**热力学/动力学采样**：必须引入 proposal-ratio 校正、nonequilibrium work correction、或后处理 reweighting；否则 SSW trajectory 不能直接解释为真实时间动力学。

------

## 9. 可以修正的设计点

### 9.1 坐标度量应从 ad hoc 欧氏距离改成自然流形度量

原子坐标、晶胞矢量、应变、分数坐标的单位和尺度不同。SSW-crystal 虽然已经把 lattice 和 atom 自由度结合，但算法参数如 (\Delta L)、(ds)、(H_{\mathrm{cell}}) 仍有经验性。更好的做法是统一使用广义度量：

# [ ds^2

\sum_i m_i |L,ds_i+dL,s_i|^2
+
\alpha |\operatorname{sym}(L^{-1}dL)|_F^2.
]

晶胞部分最好用 deformation gradient 或 logarithmic strain：

[
\varepsilon=\operatorname{sym}\log(L L_0^{-1})
]

而不是直接用 9 个 cell-vector 分量。这会减少旋转冗余、尺度不一致和各向异性晶胞下的数值病态。

### 9.2 biased CBD 应显式写成“带方向记忆的正则化软模问题”

SSW 文献说 biased CBD rotation 不一定收敛到 softest eigenvector，而是既保留初始随机方向，又接近 soft eigenvector。这个描述是合理的，但可以更形式化：

# [ n^\star

\arg\min_{|n|_G=1}
\left[
\rho_x(n)
+
\mu |n-n_0|_G^2
\right].
]

(\mu=0) 是纯 softest mode；(\mu\to\infty) 是原始随机方向；中间值是“随机但物理”的方向。这样可以把“biased rotation”从经验操作变成可调、可复现实验参数。

### 9.3 固定 (ds,H,w,\Delta L) 应改成 barrier-controlled trust-region

SSW-crystal 文中给出的常用参数如 (ds=0.6)、(H=10)、(\Delta L) 约 15% cell-vector criterion、(H_{\mathrm{cell}}=5) 对很多体系有效，但这些参数本质上仍是尺度参数。更稳健的设计是控制能量爬升而不是控制几何距离：

# [ \Delta U_h

U(x_{h+1})-U(x_h)
\approx
\Delta U_{\mathrm{target}}.
]

偏置强度可由局部曲率自适应：

# [ w_h

\sigma_h^2
\left[
\rho_x(n_h)+\lambda_{\mathrm{target}}
\right]_+.
]

这样保证 modified curvature 满足

[
\rho_x(n_h)-w_h/\sigma_h^2
\lesssim
-\lambda_{\mathrm{target}},
]

从而既能推出 basin，又不至于过度破坏结构。

### 9.4 LS-SSW 的 softening 应加“谱半径上限”和“退火回原 PES”

LS-SSW 的 pairwise penalty 非常有价值，但风险是过度软化强键，产生真实 PES 上不存在的伪路径。建议把软化强度限制为

[
0\le \alpha_{ij}\le c,\lambda_{ij}^{\mathrm{local}},
\qquad 0<c<1,
]

并设退火：

[
\tilde U_{\theta_t}=U+\gamma_t P_\theta,
\qquad
\gamma_t:1\to0.
]

最终所有 minima、TS、barrier 必须在 (\gamma=0) 的真实 PES 上重新优化。

### 9.5 DESW 应定位为“高效 path proposal”，而不是最终路径优化的替代品

DESW 的优点是快，不做完整 path iterative optimization；缺点是 pseudo-path 可能错过真正最低 MEP。更鲁棒的 pipeline 是：

[
\text{SSW/LS-SSW 找 pair}
\to
\text{DESW 快速筛选}
\to
\text{CBD 精修 TS}
\to
\text{top candidates 用 NEB/string 验证}.
]

这样保留 DESW 的高通量优势，同时避免在关键 barrier 上过度依赖 pseudo-path。

### 9.6 “unbiased” 应改名为“reaction-coordinate-free”

SSW 文献中的 unbiased 更准确地说是“不需要预先指定 reaction coordinate”。从统计力学角度，它不是严格无偏采样，因为 proposal kernel 受 soft-mode、bias、local minimization、Metropolis temperature、LS penalty 等影响。建议术语上区分：

[
\text{reaction-coordinate-free}
\neq
\text{statistically unbiased}.
]

这会让方法论更严谨，也更容易与增强采样、自由能计算、TST/kMC 接轨。

------

## 10. 一个更优雅的统一算法：Spectral Soft-Mode Walking, SSMW

可以把整套框架重写为一个算法：

# [ \boxed{ \text{SSMW}

\text{metric}
+
\text{soft-mode oracle}
+
\text{adaptive bias}
+
\text{optional local softening}
+
\text{original-PES validation}
+
\text{graph-level selection}
}
]

### 10.1 方向 oracle

在每个 minimum (m) 处，构造预条件 Hessian：

# [ \tilde H

G^{-1/2}
\left[
\nabla^2 U+\gamma P_\theta''
\right]
G^{-1/2}.
]

用 randomized Lanczos / block dimer 采样多个低曲率方向，而不是只取一个：

[
n_1,\dots,n_r
\sim
\pi(n|m),
\qquad
\pi(n|m)\propto
\exp[-\tau \rho_m(n)].
]

再加 diversity penalty：

[
\sum_{a<b}\kappa |\langle n_a,n_b\rangle_G|^2
]

避免所有 walker 都走同一个软模。

### 10.2 自适应 bias-walk

对每个方向 (n)，执行：

# [ B_{h+1}

B_h+
w_h
\exp\left[
-\frac{
\langle x-x_h,n_h\rangle_G^2
}{2\sigma_h^2}
\right],
]

其中

[
w_h=\sigma_h^2[\rho_x(n_h)+\lambda_\star]_+,
]

# [ \sigma_h

\min
\left(
\sigma_{\max},
\sqrt{
2\Delta U_{\mathrm{target}}/
|\rho_x(n_h)|
}
\right).
]

这比固定 (w,ds,H) 更稳健。

### 10.3 单端、双端、变胞都只是边界条件不同

单端 SSW：

[
a_h=\text{random memory direction}.
]

双端 DESW：

[
a_h=\log_{x_h}(x_{\mathrm{other}}).
]

晶体 SSW / VC-DESW：

[
x=(s,\varepsilon),\qquad
\varepsilon=\operatorname{sym}\log L.
]

局部软化 LS-SSW：

[
\gamma>0
\quad \text{during proposal},
\qquad
\gamma=0
\quad \text{during validation}.
]

### 10.4 graph-level selection

把搜索结果组织成图：

[
\mathcal G=(\mathcal V,\mathcal E),
]

节点是 minima，边是 DESW/CBD refined TS。边权为：

# [ \Delta E_{i\to j}^{\ddagger}

E^\ddagger_{ij}-E_i.
]

下一步从哪个 minimum 出发，不必只靠 Metropolis energy，可以用探索—利用平衡：

# [ S_i

- E_i

- 

## c_1,\mathrm{novelty}_i + c_2,\mathrm{uncertainty}_i

c_3,\mathrm{visited}_i.
]

这尤其适合 SSW-NN / LASP 场景，因为机器学习势存在模型不确定性；Nature 综述也强调 SSW-NN 通过 SSW 产生 global PES 数据，之后用 NN PES 进行大体系探索。([Nature](https://www.nature.com/articles/s41524-022-00959-5))

------

## 11. 最简洁的统一伪代码

```text
Input: energy U, metric G, optional endpoint b, optional local-softening Pθ
Initialize: minimum m0, graph Gmin

for cycle = 1 ...:
    choose start minimum m from graph policy

    build local operator:
        Uprop = U + γ Pθ          # γ=0 gives ordinary SSW
        Hvec via finite differences / ML Hessian-vector / dimer

    sample r soft directions:
        n1,...,nr from regularized Rayleigh quotient

    for each direction n:
        x = m
        B = 0

        for h = 1 ... Hmax:
            adapt σh, wh from curvature and target energy climb
            B += Gaussian bias centered at x along nh
            x  = relax on Uprop + B with trust region
            nh = update soft mode with memory / endpoint constraint

            if basin changed or walkers meet:
                break

        remove B and Pθ
        y = quench on original U

        if y is new or useful:
            add node y
            run DESW/CBD refinement between m and y
            add TS edge if validated on original U

    accept/select next minimum by annealing, MH-corrected rule, or graph policy
```

这段伪代码中：

- (\gamma=0)、无端点，就是普通 SSW；
- 有端点，就是 DESW；
- 坐标换成 ((s,\varepsilon))，就是 crystal / VC-DESW；
- (\gamma>0)，就是 LS-SSW；
- 只做局部 saddle refinement，就是 CBD；
- 加 Gaussian basin filling 并指定方向，就是 BP-CBD。

------

## 12. 总判断

这条算法线最有价值的思想是：**用 Hessian 的低曲率信息，而不是盲目随机扰动，生成跨越势垒的连续 proposal。** 这解释了为什么它能同时服务于 global minimum search、pathway sampling 和 reaction network discovery。

最需要修正的地方是：**把经验参数和坐标选择形式化。** 一旦使用自然流形度量、正则化软模 oracle、自适应 bias、proposal-only softening、原始 PES 验证和 graph-level selection，SSW-family 可以被重写成更简洁的一套算法。它会比原始表述更清楚地区分三件事：优化、反应网络发现、严格统计采样。

我认为最优雅的版本不是再发明一个完全不同的方法，而是把现有框架整理成：

[
\boxed{
\text{Preconditioned, Adaptive, Metric-aware SSW}
}
]

即：

# [ \boxed{ \text{PAM-SSW}

\text{soft-mode guided proposal}
+
\text{adaptive spectral bias}
+
\text{local-softening preconditioner}
+
\text{original-PES TS validation}
+
\text{minima-graph learning}.
}
]

这仍然完全在 Z.-P. Liu 组 CBD/SSW/DESW/LS-SSW 的思想框架内，但数学上更统一，参数更少，数值上也更鲁棒。