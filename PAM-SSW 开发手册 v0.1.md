可以。下面是一份可以直接交给 Codex 分阶段实现的 **PAM-SSW 开发手册 v0.1**。这里把前面形式化出的统一理论落实为工程规格：**Preconditioned, Adaptive, Metric-aware Stochastic Surface Walking**，简称 **PAM-SSW**。它兼容 CBD、BP-CBD、SSW、DESW、VC-DESW、LS-SSW 的核心思想，但实现上会更模块化、参数更少、验证更严格。

------

# PAM-SSW 开发手册 v0.1

## 0. 项目定位

项目名称建议：

```text
pam-ssw
```

完整含义：

```text
Preconditioned Adaptive Metric-aware Stochastic Surface Walking
```

目标是开发一个通用 PES 探索与反应路径发现框架：

# [ \text{PAM-SSW}

\text{metric-aware coordinates}
+
\text{soft-mode oracle}
+
\text{adaptive bias}
+
\text{optional local softening}
+
\text{true-PES validation}
+
\text{minima graph learning}.
]

它的核心用途包括：

1. **全局结构搜索**：从一个或多个初始结构出发寻找低能 minima。
2. **反应网络发现**：生成 minima graph，并用 TS refinement 验证边。
3. **单端路径搜索**：SSW-style 从一个 basin 自动走出。
4. **双端路径搜索**：DESW-style 连接两个已知 minima。
5. **晶体相变路径搜索**：VC-DESW / SSW-crystal-style 同时处理原子与晶胞自由度。
6. **强共价体系加速搜索**：LS-SSW-style 对局部强模式做 proposal-only softening。

文献背景上，SSW 被描述为一种基于 bias-potential-driven dynamics 和 Metropolis Monte Carlo 的 PES 搜索方法，核心能力是平滑扰动构型并越过高势垒；SSW-crystal 又把二阶信息扩展到 lattice 与 atomic degrees of freedom；LS-SSW 则通过 pairwise penalty potentials 和 self-adaptation 软化强局域振动模式。实现时我们保留这些思想，但把它们统一为一个可插拔框架。([PubMed](https://pubmed.ncbi.nlm.nih.gov/26587640/))

------

# 1. 总体工程原则

## 1.1 必须坚持的原则

### 原则 A：proposal 势与真实 PES 严格分离

算法可以在 proposal 阶段使用

# [ U_{\mathrm{prop}}(q)

U(q)
+
B(q)
+
\gamma P_\theta(q),
]

但所有最终输出必须在真实势能面

[
U(q)
]

上重新验证。包括：

- minima energy；
- force norm；
- TS energy；
- barrier；
- imaginary mode；
- endpoint connectivity。

SSW-crystal 文献中也明确区分了 SSW bias 与 metadynamics bias：SSW 的 bias 用于把结构推过 TS，之后会移除 bias 再 fully relax，而不是像 metadynamics 那样长期保留 bias 填 basin。

### 原则 B：不要把 SSW trajectory 直接解释成真实动力学

PAM-SSW 生成的是 **PES graph proposal trajectory**，不是物理时间动力学。若后续想做 canonical sampling 或 kinetics，需要额外做：

# [ A(i\to j)

\min\left[
1,
e^{-\beta(F_j-F_i)}
\frac{Q(j\to i)}{Q(i\to j)}
\right].
]

第一版实现只做优化与路径发现，不声称严格满足 detailed balance。

### 原则 C：所有自由度都必须经过统一坐标层

不要在 walker 中直接操作 Cartesian positions。应通过：

```text
State <-> GeneralizedCoordinates <-> TangentVector
```

统一处理：

- 非周期分子；
- slab；
- periodic crystal；
- fixed atoms；
- constraints；
- atom permutation；
- cell deformation。

### 原则 D：算法必须 energy-backend independent

核心库不绑定 DFT、ASE、MACE、CHGNet、VASP、LASP 或任何具体 calculator。所有能量后端都通过统一接口：

```python
energy, gradient, stress = calculator.evaluate(state)
```

实现。

------

# 2. 推荐仓库结构

```text
pam-ssw/
  AGENTS.md
  pyproject.toml
  README.md
  PLANS.md

  pamssw/
    __init__.py

    core/
      state.py
      coordinates.py
      metric.py
      random.py
      typing.py
      exceptions.py

    calculators/
      base.py
      ase_adapter.py
      analytic.py
      mlp_adapter.py

    geometry/
      pbc.py
      constraints.py
      alignment.py
      fingerprints.py
      neighbors.py

    bias/
      gaussian.py
      softening.py
      bias_set.py

    hessian/
      hvp.py
      finite_difference.py
      lanczos.py
      dimer.py

    optim/
      lbfgs.py
      trust_region.py
      line_search.py
      quench.py

    softmode/
      oracle.py
      objectives.py
      randomized.py

    walkers/
      ssw.py
      desw.py
      vc_desw.py
      ls_ssw.py
      common.py

    ts/
      cbd.py
      dimer_refiner.py
      validation.py
      irc.py

    graph/
      minima_graph.py
      node.py
      edge.py
      selection.py

    io/
      config.py
      trajectory.py
      checkpoint.py
      hdf5_store.py
      xyz.py
      ase_io.py

    cli/
      main.py
      run.py
      resume.py
      summarize.py
      refine.py

  tests/
    unit/
    integration/
    regression/
    property/
    benchmarks/

  examples/
    muller_brown/
    double_well/
    lj_cluster/
    ase_emt_cluster/
    toy_crystal/

  docs/
    theory.md
    api.md
    algorithms.md
    testing.md
    codex_workflow.md
```

Codex CLI 可以在本地终端中读取、修改并运行当前目录下的代码；官方文档也建议用 `AGENTS.md` 固化 repo layout、运行命令、测试命令、工程约束和验收标准。([OpenAI 开发者](https://developers.openai.com/codex/cli))

------

# 3. 最小依赖与开发环境

## 3.1 第一版核心依赖

```toml
[project]
name = "pamssw"
requires-python = ">=3.11"
dependencies = [
  "numpy>=1.26",
  "scipy>=1.11",
  "pydantic>=2.0",
  "pyyaml>=6.0",
  "h5py>=3.10",
  "networkx>=3.0",
  "ase>=3.22",
  "rich>=13.0",
]
```

## 3.2 开发依赖

```toml
[project.optional-dependencies]
dev = [
  "pytest",
  "pytest-cov",
  "hypothesis",
  "ruff",
  "mypy",
  "pre-commit",
]
```

## 3.3 可选依赖

```toml
[project.optional-dependencies]
mlp = [
  "torch",
]
parallel = [
  "ray",
  "dask",
]
docs = [
  "mkdocs",
  "mkdocs-material",
]
```

------

# 4. Codex 开发约束文件

在仓库根目录放置 `AGENTS.md`。Codex 官方文档说明，Codex 会在开始工作前读取 `AGENTS.md`，并按 global、project、nested directory 的顺序合并指令；靠近当前目录的指令优先级更高。([OpenAI 开发者](https://developers.openai.com/codex/guides/agents-md))

建议根目录 `AGENTS.md` 内容如下。

```markdown
# AGENTS.md — pam-ssw

## Project goal

Implement PAM-SSW: a preconditioned, adaptive, metric-aware stochastic surface walking framework for potential energy surface exploration, transition-path proposal, and transition-state refinement.

The implementation must separate:
1. true potential energy surface U(q),
2. proposal-only Gaussian bias B(q),
3. proposal-only local softening P_theta(q),
4. true-PES validation.

Never report biased energies as physical energies.

## Language and style

- Use Python 3.11+.
- Prefer small, testable modules.
- Use dataclasses or pydantic models for configs.
- Use NumPy arrays for core numerical data.
- Avoid hidden global state.
- All stochastic code must accept an explicit RNG or seed.
- Public APIs need docstrings.
- Do not introduce new runtime dependencies without updating pyproject.toml and explaining why.

## Force and gradient convention

The library uses gradients internally:

    grad = dU/dq

If a calculator returns physical forces F, convert using:

    grad = -F

Never mix force and gradient signs.

## Testing requirements

After changing numerical code, run:

    pytest tests/unit tests/property

After changing walkers or optimizers, run:

    pytest tests/integration

After changing public API, update docs and examples.

## Numerical verification

Every new analytic gradient must have a finite-difference test.

Every Hessian-vector product implementation must be checked against finite differences on at least one analytic potential.

Every transition-state validation routine must check:
- small true-PES gradient norm,
- one dominant negative curvature mode,
- downhill connectivity to expected endpoints when endpoints are known.

## Design constraints

- Core algorithm must not depend on ASE directly.
- ASE support must live in calculators/ase_adapter.py and io/ase_io.py.
- Walkers must use abstract Calculator and CoordinateSystem interfaces.
- Bias and local softening are proposal-only; all final minima and TS must be reoptimized on the original U.
- Do not claim strict canonical sampling unless proposal probabilities and MH correction are implemented.

## What done means

A task is done only when:
1. code is implemented,
2. tests are added,
3. tests pass,
4. docs or examples are updated if behavior changed,
5. the final response summarizes changed files and verification commands.
```

建议再放一个 `PLANS.md`，让 Codex 每次做大任务前先写执行计划。OpenAI 的 Codex best practices 也建议把反复有效的 prompting pattern 写进 `AGENTS.md`，并在复杂任务中使用计划模板、测试、lint、type checks 和 diff review。([OpenAI 开发者](https://developers.openai.com/codex/learn/best-practices))

------

# 5. 核心数学规格

## 5.1 状态与广义坐标

真实构型：

[
x = (R,L,Z,\mathrm{constraints}),
]

其中：

- (R\in\mathbb R^{N\times 3})：Cartesian positions；
- (L\in\mathbb R^{3\times 3})：cell matrix；
- (Z)：atomic numbers；
- constraints：fixed atoms、fixed cell、symmetry constraints 等。

内部广义坐标：

[
q = \operatorname{encode}(x).
]

对于非周期分子：

[
q = \mathrm{vec}(R)
]

并可投影掉整体平移、整体转动。

对于晶体，推荐使用：

[
q = (s_1,\ldots,s_N,\varepsilon),
]

其中：

[
R_i = Ls_i,
]

[
\varepsilon = \operatorname{sym}\log(LL_0^{-1})
]

是对数应变或近似对数应变。第一版可以先实现简化版本：

[
q = (\mathrm{vec}(s), \mathrm{vec}(L)).
]

但接口应预留 `CellCoordinateMode.LOG_STRAIN`。

------

## 5.2 度量

定义广义度量：

[
\langle u,v\rangle_G = u^\top G v.
]

非周期体系默认：

[
G = I.
]

mass-weighted 模式可用：

[
G = \operatorname{diag}(m_1,m_1,m_1,\ldots,m_N,m_N,m_N).
]

晶体模式建议：

# [ ds^2

\sum_i w_i|L,ds_i+dL,s_i|^2
+
\alpha |\operatorname{sym}(L^{-1}dL)|_F^2.
]

实现上先提供三个 metric：

```python
class MetricKind(str, Enum):
    EUCLIDEAN = "euclidean"
    MASS_WEIGHTED = "mass_weighted"
    ATOM_CELL_BLOCK = "atom_cell_block"
```

------

## 5.3 梯度、Hessian-vector 与曲率

内部统一使用：

[
g(q)=\nabla U(q).
]

Hessian-vector product：

# [ H(q)v

\nabla^2 U(q)v.
]

有限差分实现：

[
H(q)v
\approx
\frac{
g(q+\epsilon v)-g(q-\epsilon v)
}{2\epsilon}.
]

方向曲率：

# [ \rho_q(n)

\frac{n^\top H(q)n}{n^\top G n}.
]

所有 soft mode 均要求归一化：

[
|n|_G=1.
]

------

## 5.4 Gaussian bias

每个 bias term：

# [ B_k(q)

w_k
\exp\left[
-\frac{
\langle q-c_k,n_k\rangle_G^2
}{2\sigma_k^2}
\right].
]

令：

[
s_k = \langle q-c_k,n_k\rangle_G,
]

则：

[
B_k(q)=w_ke^{-s_k^2/(2\sigma_k^2)}.
]

在 (q=c_k) 处，沿 (n_k) 的曲率修正为：

[
-\frac{w_k}{\sigma_k^2}.
]

所以 adaptive bias 可令：

# [ w_k

\sigma_k^2
\max(\rho_q(n_k)+\lambda_\star,0),
]

使 proposal PES 在该方向上变成约 (-\lambda_\star) 的负曲率。

------

## 5.5 Local softening penalty

LS-SSW-style 局部软化项：

# [ P_\theta(q)

\sum_{(i,j)\in\mathcal P}
\alpha_{ij}
\exp\left[
-\frac{(r_{ij}-r_{ij}^0)^2}{2\tau_{ij}^2}
\right].
]

在 (r_{ij}=r_{ij}^0) 附近：

# [ \frac{d^2P_{ij}}{dr^2}

-\frac{\alpha_{ij}}{\tau_{ij}^2}.
]

所以它会降低该 pair-stretching 模式的局部刚性。工程上必须设置：

[
0\le \alpha_{ij}\le c,k_{ij}\tau_{ij}^2,
\qquad
0<c<1.
]

其中 (k_{ij}) 是估计 pair stiffness。这样避免过度软化导致非物理路径。

------

## 5.6 Soft-mode oracle

方向选择统一写成正则化 Rayleigh quotient：

# [ n^\star

## \arg\min_{|n|*G=1} \left[ \rho_q(n) + \mu*{\mathrm{mem}}|n-n_{\mathrm{prev}}|_G^2

\eta_{\mathrm{end}}\langle n,e_{\mathrm{end}}\rangle_G^2
+
\kappa_{\mathrm{div}}\sum_a|\langle n,n_a\rangle_G|^2
\right].
]

解释：

- (\rho_q(n))：偏向软模；
- memory term：保持路径连续；
- endpoint term：DESW 中朝向另一端；
- diversity term：多 walker 避免重复方向。

第一版实现三个 oracle：

```text
DimerSoftModeOracle
LanczosSoftModeOracle
RandomizedSoftModeOracle
```

------

# 6. 关键数据结构

## 6.1 State

```python
@dataclass(frozen=True)
class State:
    numbers: np.ndarray          # shape (N,)
    positions: np.ndarray        # shape (N, 3), Cartesian Angstrom
    cell: np.ndarray | None      # shape (3, 3), Angstrom
    pbc: np.ndarray              # shape (3,), bool
    constraints: ConstraintSet | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

要求：

- `State` 尽量不可变；
- 修改结构时返回新 `State`；
- `metadata` 不参与能量计算；
- 所有 positions 单位 Angstrom。

------

## 6.2 EnergyResult

```python
@dataclass(frozen=True)
class EnergyResult:
    energy: float                # eV
    gradient_q: np.ndarray       # dU/dq, internal generalized gradient
    forces: np.ndarray | None    # eV / Angstrom, optional
    stress: np.ndarray | None    # eV / Angstrom^3 or GPa, explicit convention required
    raw: dict[str, Any] = field(default_factory=dict)
```

------

## 6.3 Calculator

```python
class Calculator(Protocol):
    def evaluate(self, state: State) -> EnergyResult:
        ...
```

核心算法只依赖 `Calculator`。

ASE adapter：

```python
class ASECalculatorAdapter:
    def __init__(self, ase_calculator: Any, coordinate_system: CoordinateSystem):
        ...

    def evaluate(self, state: State) -> EnergyResult:
        ...
```

注意：

```python
gradient_positions = -forces
```

这是必须测试的 sign convention。

------

## 6.4 CoordinateSystem

```python
class CoordinateSystem(Protocol):
    def encode(self, state: State) -> np.ndarray:
        ...

    def decode(self, q: np.ndarray, reference: State) -> State:
        ...

    def tangent_to_cartesian(self, state: State, vq: np.ndarray) -> np.ndarray:
        ...

    def project_tangent(self, state: State, vq: np.ndarray) -> np.ndarray:
        ...

    def difference(self, a: State, b: State) -> np.ndarray:
        ...
```

`difference(a,b)` 表示从 `a` 指向 `b` 的 tangent vector，用于 DESW endpoint term。

------

## 6.5 Metric

```python
class Metric(Protocol):
    def dot(self, state: State, u: np.ndarray, v: np.ndarray) -> float:
        ...

    def norm(self, state: State, u: np.ndarray) -> float:
        ...

    def normalize(self, state: State, u: np.ndarray) -> np.ndarray:
        ...

    def apply(self, state: State, u: np.ndarray) -> np.ndarray:
        ...

    def inverse_apply(self, state: State, covector: np.ndarray) -> np.ndarray:
        ...
```

------

## 6.6 BiasTerm

```python
@dataclass(frozen=True)
class GaussianBias:
    center_q: np.ndarray
    direction_q: np.ndarray
    sigma: float
    weight: float

    def energy(self, q: np.ndarray, metric: Metric, state: State) -> float:
        ...

    def gradient(self, q: np.ndarray, metric: Metric, state: State) -> np.ndarray:
        ...

    def curvature_at_center(self) -> float:
        return -self.weight / (self.sigma ** 2)
```

------

## 6.7 BiasSet

```python
@dataclass
class BiasSet:
    terms: list[GaussianBias] = field(default_factory=list)

    def add(self, term: GaussianBias) -> None:
        ...

    def energy(self, q: np.ndarray, metric: Metric, state: State) -> float:
        ...

    def gradient(self, q: np.ndarray, metric: Metric, state: State) -> np.ndarray:
        ...

    def clear(self) -> None:
        ...
```

------

## 6.8 LocalSoftener

```python
@dataclass
class PairSofteningTerm:
    i: int
    j: int
    r0: float
    tau: float
    alpha: float

@dataclass
class LocalSoftener:
    terms: list[PairSofteningTerm]
    gamma: float = 1.0

    def energy(self, state: State) -> float:
        ...

    def gradient_q(self, state: State, coordinate_system: CoordinateSystem) -> np.ndarray:
        ...

    def anneal(self, step: int, total_steps: int) -> None:
        ...
```

退火建议：

[
\gamma_t = \gamma_0\left(1-\frac{t}{T}\right)^p.
]

------

# 7. 核心算法模块

## 7.1 Quench optimizer

真实 PES 上最小化：

```python
class QuenchOptimizer:
    def minimize(
        self,
        state: State,
        calculator: Calculator,
        coordinate_system: CoordinateSystem,
        metric: Metric,
        config: QuenchConfig,
    ) -> Minimum:
        ...
```

输出：

```python
@dataclass(frozen=True)
class Minimum:
    state: State
    energy: float
    grad_norm: float
    n_steps: int
    converged: bool
    fingerprint: str
```

验收：

```text
grad_norm < f_tol
max_steps not exceeded
energy finite
no invalid geometry
```

------

## 7.2 Modified PES evaluator

Proposal 阶段需要临时势能：

[
U_{\mathrm{mod}}(q)=U(q)+B(q)+\gamma P_\theta(q).
]

```python
class ModifiedEvaluator:
    def __init__(
        self,
        true_calculator: Calculator,
        coordinate_system: CoordinateSystem,
        metric: Metric,
        bias_set: BiasSet | None = None,
        softener: LocalSoftener | None = None,
    ):
        ...

    def evaluate(self, state: State) -> EnergyResult:
        ...
```

注意：`ModifiedEvaluator` 的结果不得写入最终 graph 的 physical energy 字段，只能写入 proposal diagnostics。

------

## 7.3 Hessian-vector product

```python
class HessianVectorProduct:
    def apply(
        self,
        state: State,
        vq: np.ndarray,
        evaluator: Calculator | ModifiedEvaluator,
    ) -> np.ndarray:
        ...
```

有限差分步长建议：

[
\epsilon =
\epsilon_0
\frac{1+|q|}{|v|},
]

其中：

```python
epsilon_0 = 1e-4  # first default for atomistic coordinates
```

并提供 config：

```yaml
hessian:
  method: finite_difference
  epsilon: 1.0e-4
  central_difference: true
```

------

## 7.4 DimerSoftModeOracle

```python
class DimerSoftModeOracle:
    def find_mode(
        self,
        state: State,
        evaluator: Calculator | ModifiedEvaluator,
        initial_direction: np.ndarray,
        previous_direction: np.ndarray | None,
        endpoint_direction: np.ndarray | None,
        config: SoftModeConfig,
    ) -> SoftModeResult:
        ...
```

输出：

```python
@dataclass(frozen=True)
class SoftModeResult:
    direction_q: np.ndarray
    curvature: float
    n_iterations: int
    converged: bool
    diagnostics: dict[str, Any]
```

第一版不必实现完整 constrained Broyden dimer，可以先实现：

1. randomized initial vector；
2. finite-difference HVP；
3. Lanczos 取最低本征方向；
4. memory / endpoint 修正；
5. metric normalize。

------

## 7.5 AdaptiveBiasPolicy

```python
class AdaptiveBiasPolicy:
    def propose(
        self,
        state: State,
        direction_q: np.ndarray,
        curvature: float,
        step_index: int,
        config: BiasConfig,
    ) -> GaussianBias:
        ...
```

公式：

```python
sigma = clip(
    sqrt(2 * target_energy_climb / max(abs(curvature), curvature_floor)),
    sigma_min,
    sigma_max,
)

weight = sigma**2 * max(curvature + lambda_target, 0.0)
```

对应数学：

# [ \sigma

\operatorname{clip}
\left(
\sqrt{
\frac{2\Delta U_{\mathrm{target}}}{\max(|\rho|,\rho_{\min})}
},
\sigma_{\min},
\sigma_{\max}
\right),
]

# [ w

\sigma^2\max(\rho+\lambda_\star,0).
]

------

# 8. Single-ended PAM-SSW

## 8.1 算法目标

从一个 minimum (m) 出发，生成一个新的 candidate minimum (m')。

形式：

# [ m'

\mathcal Q_U
\left[
\Phi_{U+B+\gamma P}^{H}(m;\xi)
\right].
]

------

## 8.2 伪代码

```python
def pam_ssw_step(start_minimum):
    state = start_minimum.state
    bias_set = BiasSet()
    previous_direction = None

    if config.local_softening.enabled:
        softener = LocalSoftener.build_from_state(state)
    else:
        softener = None

    for h in range(config.walk.max_bias_steps):
        evaluator = ModifiedEvaluator(
            true_calculator=calculator,
            coordinate_system=coords,
            metric=metric,
            bias_set=bias_set,
            softener=softener,
        )

        initial_direction = direction_initializer.sample(state)

        mode = softmode_oracle.find_mode(
            state=state,
            evaluator=evaluator,
            initial_direction=initial_direction,
            previous_direction=previous_direction,
            endpoint_direction=None,
            config=config.softmode,
        )

        term = adaptive_bias_policy.propose(
            state=state,
            direction_q=mode.direction_q,
            curvature=mode.curvature,
            step_index=h,
            config=config.bias,
        )

        bias_set.add(term)

        state = biased_relaxer.relax_one_surface_walk_step(
            state=state,
            evaluator=evaluator,
            direction_q=mode.direction_q,
            step_size=term.sigma,
            config=config.walk,
        )

        previous_direction = mode.direction_q

        if stop_criterion.should_stop(state, h):
            break

    # remove all proposal-only terms
    new_minimum = true_quench.minimize(
        state=state,
        calculator=calculator,
        coordinate_system=coords,
        metric=metric,
        config=config.quench,
    )

    return new_minimum
```

------

## 8.3 Biased relax step

每一步建议分成两段：

### Step 1：沿 soft mode 显式位移

[
q_{\mathrm{trial}} = q + \sigma n.
]

### Step 2：在 modified PES 上 trust-region relax

# [ q_{\mathrm{relaxed}}

\arg\min_{q'}
U_{\mathrm{mod}}(q')
\quad
\text{s.t.}
\quad
|q'-q_{\mathrm{trial}}|*G < r*{\mathrm{trust}}.
]

第一版可用 L-BFGS + 最大位移限制近似。

------

# 9. DESW / Double-ended PAM-SSW

## 9.1 输入

```python
initial_minimum: Minimum
final_minimum: Minimum
```

## 9.2 核心思想

从两个端点同时出发：

[
x_A^0=a,\qquad x_B^0=b.
]

每个 walker 的方向 objective 加 endpoint attraction：

[
e_A = \log_{x_A}(x_B),
]

[
e_B = \log_{x_B}(x_A).
]

方向：

# [ n_A^\star

## \arg\min_{|n|*G=1} \left[ \rho(n) + \mu|n-n*{\mathrm{prev}}|^2

\eta\langle n,e_A\rangle_G^2
\right].
]

DESW 原始思想就是两个 images 分别从 initial 和 final states 出发，stepwise 朝向彼此 walking，用 repeated bias potential addition、local relaxation 和 CBD 方向修正建立 pseudo-path 并定位 TS。([ACS Publications](https://pubs.acs.org/doi/10.1021/ct4008475?utm_source=chatgpt.com))

------

## 9.3 DESW 输出

```python
@dataclass(frozen=True)
class PathProposal:
    endpoint_a: Minimum
    endpoint_b: Minimum
    images: list[State]
    energies_true: list[float]
    energies_modified: list[float]
    max_energy_image_index: int
    connected: bool
    diagnostics: dict[str, Any]
```

------

## 9.4 Meet criteria

两个 walker 满足任一条件即可认为 pseudo-path 已连接：

```text
1. metric distance between current states < d_meet
2. true quench of both current states reaches same minimum
3. image A crosses image B along path ordering
4. maximum number of DESW steps reached, then return partial path
```

------

## 9.5 DESW 后处理

DESW 只负责生成 pseudo-path，不直接宣称 MEP。后处理流程：

```text
PathProposal
  -> choose highest true-energy image
  -> initialize dimer/CBD TS refinement
  -> validate TS on true PES
  -> optional NEB/string verification for important barriers
```

------

# 10. Transition-state refinement

## 10.1 TSRefiner interface

```python
class TransitionStateRefiner:
    def refine(
        self,
        path: PathProposal,
        calculator: Calculator,
        coordinate_system: CoordinateSystem,
        metric: Metric,
        config: TSConfig,
    ) -> TransitionStateResult:
        ...
```

------

## 10.2 初始化

从 path 中选择：

[
i^\star = \arg\max_i U(x_i).
]

初始 TS guess：

```python
ts_guess = path.images[i_star]
```

初始 dimer mode：

[
n_0 = \operatorname{normalize}*G(x*{i+1}-x_{i-1}).
]

------

## 10.3 Dimer / CBD force

若 (n) 是负曲率方向，modified force / gradient descent target 为：

# [ \dot q

-(I-2nn^\top_G)\nabla_G U.
]

即：

- 沿 (n) 上坡；
- 沿正交子空间下坡。

实现时可写成：

```python
grad_g = metric.inverse_apply(state, grad)
parallel = metric.dot(state, grad_g, n) * n
modified_grad_g = grad_g - 2.0 * parallel
```

注意根据 optimizer 使用的是 gradient 还是 force，符号要严格测试。

------

## 10.4 TS 验证标准

一个 TS candidate 必须满足：

```text
1. true-PES gradient norm < ts_force_tol
2. lowest Hessian eigenvalue < -lambda_min
3. second-lowest Hessian eigenvalue >= -lambda_noise_tol
4. downhill from +mode quenches to endpoint A or connected minimum
5. downhill from -mode quenches to endpoint B or connected minimum
6. TS energy is finite and above both endpoint minima
```

输出：

```python
@dataclass(frozen=True)
class TransitionStateResult:
    state: State
    energy: float
    grad_norm: float
    lowest_eigenvalue: float
    second_eigenvalue: float | None
    mode: np.ndarray
    endpoint_plus: Minimum | None
    endpoint_minus: Minimum | None
    validated: bool
    diagnostics: dict[str, Any]
```

------

# 11. Minima graph

## 11.1 节点

```python
@dataclass
class MinimaNode:
    node_id: str
    minimum: Minimum
    visits: int = 0
    discovery_step: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
```

## 11.2 边

```python
@dataclass
class ReactionEdge:
    edge_id: str
    node_a: str
    node_b: str
    ts: TransitionStateResult | None
    path: PathProposal | None
    barrier_a_to_b: float | None
    barrier_b_to_a: float | None
    validated: bool
    metadata: dict[str, Any] = field(default_factory=dict)
```

## 11.3 图选择策略

全局优化模式：

# [ S_i

## -E_i + c_{\mathrm{novel}}\mathrm{novelty}_i

c_{\mathrm{visit}}\log(1+\mathrm{visits}_i).
]

Metropolis-like 模式：

# [ P(i\to j)

\min(1,e^{-\beta(E_j-E_i)}).
]

但文档里要明确：这是 optimization selection，不是严格 canonical sampling。

------

# 12. Fingerprint 与去重

## 12.1 非周期分子 / cluster

第一版 fingerprint：

```text
sorted pairwise distance vector
```

增强版：

```text
Coulomb matrix eigenvalues
SOAP descriptor
bond graph hash
```

## 12.2 周期晶体

第一版 fingerprint：

```text
reduced cell parameters + sorted local environment distances
```

增强版：

```text
pymatgen StructureMatcher
spglib standardized cell
SOAP/crystal graph descriptor
```

为了避免依赖太重，第一版只实现轻量 fingerprint，pymatgen/spglib 作为 optional。

------

# 13. 配置文件格式

建议用 YAML。

```yaml
run:
  name: lj7_pam_ssw
  seed: 20260427
  mode: single_ended
  max_cycles: 1000
  checkpoint_every: 10

system:
  input: examples/lj_cluster/lj7.xyz
  pbc: [false, false, false]
  units:
    energy: eV
    length: Angstrom

calculator:
  kind: analytic
  name: lennard_jones
  parameters:
    epsilon: 1.0
    sigma: 1.0
    cutoff: 3.0

coordinates:
  kind: cartesian
  remove_translation: true
  remove_rotation: false

metric:
  kind: euclidean

quench:
  method: lbfgs
  f_tol: 1.0e-4
  max_steps: 500
  max_step: 0.1

softmode:
  method: lanczos
  n_lanczos: 12
  finite_difference_epsilon: 1.0e-4
  memory_weight: 0.1
  endpoint_weight: 0.0
  diversity_weight: 0.0
  curvature_floor: 1.0e-6

bias:
  adaptive: true
  target_energy_climb: 0.2
  lambda_target: 0.05
  sigma_min: 0.05
  sigma_max: 0.5
  weight_max: 10.0

walk:
  max_bias_steps: 12
  trust_radius: 0.5
  local_relax_steps: 20
  stop_when_energy_above: 5.0
  stop_when_distance_above: 3.0

local_softening:
  enabled: false
  gamma0: 1.0
  anneal_power: 1.0
  pair_selection: covalent
  alpha_fraction: 0.5
  tau_scale: 0.2

graph:
  duplicate_energy_tol: 1.0e-4
  duplicate_fingerprint_tol: 1.0e-3
  selection: novelty_metropolis
  temperature: 1000.0
  novelty_weight: 1.0
  visit_penalty: 0.1

ts:
  enabled: true
  refine_after_new_edge: true
  method: dimer
  f_tol: 5.0e-3
  max_steps: 200
  lowest_mode_tol: -1.0e-3
  validate_connectivity: true

io:
  output_dir: runs/lj7_pam_ssw
  save_trajectories: true
  save_bias_history: true
  hdf5: true
```

------

# 14. CLI 设计

## 14.1 运行

```bash
pamssw run config.yaml
```

## 14.2 恢复

```bash
pamssw resume runs/lj7_pam_ssw/checkpoint_latest.h5
```

## 14.3 总结 graph

```bash
pamssw summarize runs/lj7_pam_ssw
```

输出：

```text
Number of minima
Number of unique validated TS
Lowest energy minimum
Top 10 barriers
Acceptance statistics
Force evaluations
```

## 14.4 单独 refine TS

```bash
pamssw refine-ts runs/job/path_001.h5 --method dimer
```

## 14.5 导出

```bash
pamssw export runs/job --format xyz
pamssw export runs/job --format graphml
pamssw export runs/job --format json
```

------

# 15. 测试体系

## 15.1 Unit tests

必须覆盖：

```text
State immutability
Coordinate encode/decode roundtrip
Metric dot/norm/normalize
Gaussian bias energy/gradient finite difference
Local softening gradient finite difference
Calculator force-to-gradient sign
Finite-difference HVP
Lanczos lowest-mode extraction
LBFGS convergence on quadratic potential
Fingerprint duplicate detection
Checkpoint save/load
```

------

## 15.2 Analytic potentials

实现 `calculators/analytic.py`：

### Harmonic well

[
U(x)=\frac12 x^\top A x.
]

用途：

```text
test gradient
test HVP
test softmode eigenvector
```

### Double well

[
U(x,y)=(x^2-1)^2+\frac12 y^2.
]

用途：

```text
test dimer TS at x=0,y=0
test downhill connectivity to two minima
```

### Müller-Brown potential

用途：

```text
test SSW escapes minima
test DESW finds reasonable path
test TS refinement
```

### Lennard-Jones cluster

用途：

```text
test global optimization behavior
test duplicate minima
test graph growth
```

------

## 15.3 Property tests

用 Hypothesis 检查：

```text
normalize(v) has norm 1
dot(u,v) == dot(v,u)
Gaussian bias at center has zero gradient
Bias curvature at center is negative along direction
Encode/decode preserves atom count and species
```

------

## 15.4 Integration tests

第一批 integration tests：

```text
1. SSW on 2D double well finds both minima within 20 cycles.
2. Dimer refiner on double well converges to index-1 saddle.
3. DESW between double-well minima returns connected path.
4. LJ7 search discovers at least two unique minima.
5. ASE EMT adapter can quench a small Cu cluster.
```

------

## 15.5 Regression tests

保存小型 benchmark 的 expected statistics：

```text
seed
number of discovered minima
lowest energy
number of force evaluations
validated TS count
```

不要要求每条随机路径完全一致，但要求统计范围稳定。

------

# 16. Codex 分阶段实现路线

## M0：初始化项目骨架

给 Codex 的任务：

```text
Create the initial Python package scaffold for pamssw.

Requirements:
- pyproject.toml with dependencies and dev dependencies.
- package layout exactly as described in docs/development_manual.md.
- root AGENTS.md and PLANS.md.
- pytest configuration.
- ruff configuration.
- minimal README.
- no algorithm implementation yet.
- add a smoke test that imports pamssw.
Run pytest before finishing.
```

验收：

```bash
pytest
python -c "import pamssw"
```

------

## M1：State、coordinates、metric

Codex 任务：

```text
Implement core State, CoordinateSystem, CartesianCoordinateSystem, and EuclideanMetric.

Requirements:
- State is immutable.
- CartesianCoordinateSystem supports encode/decode/difference.
- EuclideanMetric supports dot/norm/normalize/apply/inverse_apply.
- Add tests for encode/decode roundtrip and metric normalization.
- Use explicit gradient convention docs.
Run pytest tests/unit.
```

验收：

```bash
pytest tests/unit/test_state.py tests/unit/test_coordinates.py tests/unit/test_metric.py
```

------

## M2：analytic calculators

Codex 任务：

```text
Implement analytic calculators:
1. HarmonicCalculator
2. DoubleWellCalculator
3. MullerBrownCalculator
4. LennardJonesCalculator

Each calculator must return energy and gradient_q using the internal gradient convention dU/dq.

Add finite-difference gradient tests for each calculator.
```

验收：

```bash
pytest tests/unit/test_analytic_calculators.py
```

------

## M3：Gaussian bias 与 ModifiedEvaluator

Codex 任务：

```text
Implement GaussianBias, BiasSet, and ModifiedEvaluator.

Requirements:
- Bias energy and gradient must support metric dot product.
- Gradient must pass finite-difference tests.
- ModifiedEvaluator must add true U + bias energy.
- ModifiedEvaluator must never mutate the underlying State.
- Add tests for zero gradient at bias center and negative curvature along direction.
```

验收：

```bash
pytest tests/unit/test_bias.py tests/unit/test_modified_evaluator.py
```

------

## M4：HVP、Lanczos soft mode

Codex 任务：

```text
Implement finite-difference Hessian-vector product and a LanczosSoftModeOracle.

Requirements:
- HVP uses central finite difference on gradients.
- Lanczos returns approximate lowest-curvature mode.
- Direction must be metric-normalized.
- Test on diagonal harmonic potential with known eigenvalues.
```

验收：

```bash
pytest tests/unit/test_hvp.py tests/unit/test_lanczos_softmode.py
```

------

## M5：Quench optimizer

Codex 任务：

```text
Implement an L-BFGS-based QuenchOptimizer using scipy.optimize.

Requirements:
- Works in generalized coordinates.
- Uses calculator gradients.
- Returns Minimum with energy, grad_norm, n_steps, converged.
- Test on harmonic and double-well potentials.
```

验收：

```bash
pytest tests/unit/test_quench.py
```

------

## M6：Single-ended SSW walker

Codex 任务：

```text
Implement SSWWalker for single-ended PAM-SSW.

Requirements:
- Starts from a Minimum.
- Repeatedly finds a soft mode, adds adaptive Gaussian bias, displaces, locally relaxes on modified PES.
- Removes bias and quenches on true PES at the end.
- Stores a WalkTrace with states, true energies, modified energies, bias terms, curvatures.
- Add integration test on double-well potential.
```

验收：

```bash
pytest tests/integration/test_ssw_double_well.py
```

------

## M7：Minima graph 与 selection

Codex 任务：

```text
Implement MinimaGraph, duplicate detection, and graph selection policies.

Requirements:
- Add minima nodes with fingerprints.
- Avoid duplicates within energy/fingerprint tolerance.
- Store proposal edges even before TS validation.
- Implement lowest_energy, random, novelty_metropolis selection.
- Add tests for duplicate detection and selection.
```

验收：

```bash
pytest tests/unit/test_minima_graph.py tests/unit/test_selection.py
```

------

## M8：DESW walker

Codex 任务：

```text
Implement DESWWalker using two SSW-style walkers with endpoint attraction.

Requirements:
- Accept two Minimum endpoints.
- Use CoordinateSystem.difference to build endpoint directions.
- Return PathProposal.
- Stop when walkers meet by metric distance or quench-to-same-minimum.
- Add integration test on double-well potential.
```

验收：

```bash
pytest tests/integration/test_desw_double_well.py
```

------

## M9：Dimer / TS refinement

Codex 任务：

```text
Implement a first version of TransitionStateRefiner using dimer-like modified gradient.

Requirements:
- Initialize from highest-energy path image.
- Estimate lowest mode using Lanczos.
- Optimize with inverted gradient along lowest mode.
- Validate index-1 saddle on analytic double-well.
- Validate downhill connectivity.
```

验收：

```bash
pytest tests/integration/test_ts_double_well.py
```

------

## M10：Local softening

Codex 任务：

```text
Implement LocalSoftener with Gaussian pair penalty.

Requirements:
- Select pairs by distance cutoff.
- Penalty energy and gradient must pass finite-difference tests.
- Support gamma annealing.
- Integrate into ModifiedEvaluator.
- Add a toy test showing that pair curvature is reduced but final quench uses true PES.
```

验收：

```bash
pytest tests/unit/test_local_softening.py tests/integration/test_ls_ssw_toy.py
```

------

## M11：ASE adapter

Codex 任务：

```text
Implement ASECalculatorAdapter and ASE IO utilities.

Requirements:
- Convert ASE forces to internal gradients.
- Support nonperiodic and periodic structures.
- Add smoke test with ASE EMT on a small Cu cluster.
- Core modules must not import ASE.
```

验收：

```bash
pytest tests/integration/test_ase_adapter.py
```

------

## M12：CLI、checkpoint、HDF5

Codex 任务：

```text
Implement CLI commands:
- pamssw run config.yaml
- pamssw resume checkpoint.h5
- pamssw summarize run_dir
- pamssw export run_dir --format xyz/json/graphml

Implement checkpointing with HDF5.
Add tests for config parsing and checkpoint roundtrip.
```

验收：

```bash
pytest tests/unit/test_config.py tests/unit/test_checkpoint.py
pamssw run examples/double_well/config.yaml
pamssw summarize runs/double_well
```

------

# 17. 推荐 Codex 使用方式

## 17.1 安装与启动

Codex CLI 官方文档给出的基本方式是安装 CLI 后在仓库目录中运行 `codex`，它可以检查仓库、编辑文件并运行命令。([OpenAI 开发者](https://developers.openai.com/codex/cli))

```bash
npm i -g @openai/codex
cd pam-ssw
codex
```

## 17.2 每次只给一个垂直切片

不要一次让 Codex “实现 PAM-SSW”。应按 M0–M12 逐个任务推进。每个任务包含：

```text
背景
范围
禁止事项
目标文件
测试要求
验收命令
```

## 17.3 推荐任务 prompt 模板

```text
You are working in the pam-ssw repository.

Read AGENTS.md first.

Task:
[one milestone only]

Scope:
- Files you may edit:
  [...]
- Files you should not edit:
  [...]

Requirements:
[...]

Tests:
Run:
[...]

Before finishing:
- summarize changed files
- summarize tests run
- mention any numerical assumptions
```

## 17.4 让 Codex 做 code review

每个 milestone 完成后，用第二个 Codex session 或 `/review` 做 review。官方 best practices 建议让 Codex 创建测试、运行检查、确认行为符合要求，并 review diff。([OpenAI 开发者](https://developers.openai.com/codex/learn/best-practices))

Review prompt：

```text
Review the uncommitted changes for numerical correctness.

Focus on:
1. gradient/force sign errors,
2. mutation of State objects,
3. metric normalization,
4. finite-difference step-size mistakes,
5. places where biased energies might be reported as true energies,
6. missing tests.

Do not implement changes yet. Return a prioritized issue list.
```

------

# 18. 数值稳定性手册

## 18.1 梯度符号

最常见 bug：

```python
grad = forces
```

正确：

```python
grad = -forces
```

所有测试必须覆盖。

------

## 18.2 HVP 步长

太小会被数值噪声污染，太大会失去局部性。默认：

```python
epsilon = 1e-4
```

但应允许配置：

```yaml
hessian:
  epsilon: 1.0e-4
```

并记录每次 HVP 的实际步长。

------

## 18.3 Bias 权重上限

必须设置：

```yaml
bias:
  weight_max: 10.0
```

防止局部 PES 被过度扭曲。

------

## 18.4 Local softening 上限

必须设置：

```yaml
local_softening:
  alpha_fraction: 0.5
```

含义：

[
\alpha_{ij}/\tau_{ij}^2
\le
0.5 k_{ij}.
]

也就是最多软化掉约一半局部刚性。

------

## 18.5 Geometry sanity check

每次 proposal 后检查：

```text
no NaN
no atom overlap below min_distance
cell determinant positive
cell volume above lower bound
energy finite
gradient finite
```

建议接口：

```python
class GeometryValidator:
    def validate(self, state: State) -> ValidationResult:
        ...
```

------

# 19. 输出文件规范

每个 run directory：

```text
runs/job_name/
  config.yaml
  resolved_config.yaml
  metadata.json
  graph.json
  graph.graphml
  minima/
    node_000001.xyz
    node_000002.xyz
  paths/
    path_000001.h5
  ts/
    ts_000001.xyz
  checkpoints/
    checkpoint_000010.h5
    checkpoint_latest.h5
  logs/
    run.log
    diagnostics.jsonl
```

`metadata.json`：

```json
{
  "pamssw_version": "0.1.0",
  "seed": 20260427,
  "calculator": "...",
  "start_time": "...",
  "python_version": "...",
  "dependencies": {...}
}
```

`diagnostics.jsonl` 每行记录一次 proposal：

```json
{
  "cycle": 12,
  "start_node": "node_000003",
  "end_node": "node_000008",
  "accepted": true,
  "true_energy_start": -10.23,
  "true_energy_end": -10.41,
  "n_bias": 8,
  "max_curvature": 0.5,
  "min_curvature": -0.02,
  "force_evaluations": 220
}
```

------

# 20. 文档结构

`docs/theory.md`：

```text
- PES graph view
- metric-aware coordinates
- adaptive Gaussian bias
- local softening
- true-PES validation
- why SSW trajectory is not physical dynamics
```

`docs/algorithms.md`：

```text
- SSWWalker
- DESWWalker
- LocalSoftener
- DimerRefiner
- Graph selection
```

`docs/api.md`：

```text
- State
- Calculator
- CoordinateSystem
- Metric
- BiasSet
- Walker APIs
```

`docs/testing.md`：

```text
- finite difference tests
- analytic potentials
- integration benchmarks
```

`docs/codex_workflow.md`：

```text
- how to use Codex
- milestone prompts
- review prompts
- AGENTS.md policy
```

------

# 21. 第一版最重要的验收标准

项目 v0.1 不要求 DFT 生产级效率，但必须满足：

```text
1. double-well SSW can discover both minima.
2. DESW can connect known double-well endpoints.
3. TS refiner can converge to analytic double-well saddle.
4. Gaussian bias gradient passes finite-difference tests.
5. Local softening gradient passes finite-difference tests.
6. ASE EMT adapter can quench a small cluster.
7. All final reported minima energies are true-PES energies.
8. All random results are seed-reproducible.
9. Checkpoint/resume works.
10. CLI can run an example end-to-end.
```

------

# 22. 推荐的最小可运行示例

`examples/double_well/config.yaml`：

```yaml
run:
  name: double_well_demo
  seed: 7
  mode: single_ended
  max_cycles: 30

system:
  kind: analytic
  initial_q: [-1.0, 0.0]

calculator:
  kind: analytic
  name: double_well

coordinates:
  kind: analytic_cartesian
  dimension: 2

metric:
  kind: euclidean

quench:
  method: lbfgs
  f_tol: 1.0e-8
  max_steps: 200

softmode:
  method: lanczos
  n_lanczos: 4
  finite_difference_epsilon: 1.0e-5
  memory_weight: 0.1

bias:
  adaptive: true
  target_energy_climb: 0.2
  lambda_target: 0.1
  sigma_min: 0.05
  sigma_max: 0.5
  weight_max: 5.0

walk:
  max_bias_steps: 8
  trust_radius: 0.3
  local_relax_steps: 10

graph:
  selection: novelty_metropolis
  temperature: 1000.0
  duplicate_energy_tol: 1.0e-8
  duplicate_fingerprint_tol: 1.0e-5

ts:
  enabled: true
  method: dimer
  f_tol: 1.0e-6
  validate_connectivity: true

io:
  output_dir: runs/double_well_demo
  save_trajectories: true
```

运行：

```bash
pamssw run examples/double_well/config.yaml
pamssw summarize runs/double_well_demo
```

期望：

```text
Found minima near x=-1 and x=+1
Found TS near x=0, y=0
Validated one negative mode
```

------

# 23. 后续高级功能路线

v0.2：

```text
- block Lanczos multiple soft modes
- diversity-aware multi-walker
- graph-level active learning selection
- optional pymatgen/spglib crystal fingerprint
- NEB validation plugin
```

v0.3：

```text
- log-strain cell coordinates
- variable-cell DESW
- stress-consistent gradient
- pressure enthalpy H = E + pV
```

v0.4：

```text
- MLP uncertainty integration
- active DFT labeling loop
- parallel walkers with Ray/Dask
- remote calculator execution
```

v0.5：

```text
- optional MH-corrected basin sampling
- free-energy basin correction
- kinetic network export
```

------

# 24. 最关键的实现判断

第一版不要追求一次性复刻所有 Liu group 原始实现细节。更好的路线是：

[
\boxed{
\text{先把数学对象做对，再逐步提高物理复杂度。}
}
]

优先级应是：

```text
1. State / coordinates / metric 正确
2. gradient sign 正确
3. bias gradient 正确
4. HVP 与 soft mode 正确
5. true-PES quench 正确
6. SSW proposal 可复现
7. graph 去重可靠
8. TS validation 严格
9. periodic / cell 自由度
10. local softening 与 MLP/DFT 生产级扩展
```

这套手册给 Codex 的实现方式是：**每次一个垂直切片，每个切片必须有测试，每个物理量必须有单位与 sign convention，每个 proposal-only 项都不能污染真实 PES 输出。** 这样开发出来的 PAM-SSW 会比直接仿写 SSW/DESW 更干净，也更容易扩展到 LS-SSW、VC-DESW、MLP active learning 和后续 reaction network 自动化。