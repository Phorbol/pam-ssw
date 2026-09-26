# 外部PES搜索实现审查：2026-09-27

## Review contract
- Question: 哪些现成算法/工程机制能解决已观察的碎裂、漏斗困陷和MLIP串行开销，值得加入ASE SSW后续计划？
- Reader: 项目用户及研发者；不是全面综述或新算法实现授权。
- Retrieval cutoff: 2026-09-27；实际检出的仓库HEAD/本地软件版本单独固定。
- Unit: 具体版本的代码路径及对应原始论文；覆盖用户列出的ASE GA/BH、GOCIA GCGA、VSSR-MC、EON、GOFEE、HASGO。
- Include: 约束/容器/断连处理、外层选点/温度、可批量的计算层、局域并发前提；补充MACE/TorchSim及副本交换/ITS原始来源。
- Exclude: 运行外部搜索、安装依赖、下载模型、改变PAM公共接口、未授权大预算、为某case拟合新参数。
- Comparison axes: 目标函数/状态空间；机制实际作用位置；理论前提；改变的物理目标；新增成本/状态；已有证据；最小判别实验；保留/暂缓决定。
- Evidence: reported=源码/论文明确；derived=公式可推导；inferred=针对本项目判断；not-disclosed=缺失。代码事实不等于性能优势。

## Search map
| Family | Evidence | Owner |
|---|---|---|
| ASE GA/BH/MH | installed ASE3.26 + official source | agent A |
| GOCIA/VSSR | pinned official git source + cited papers | agent B |
| EON/GOFEE/HASGO | pinned official source/docs | agent C |
| MACE/batching/replicas/ITS/locality | installed source + primary docs/papers | root |

## Planned synthesis
- Central question, not pre-decided finding: 与当前SSW相比，哪个机制能以最少改动产生可归因的改善？
- Separate search efficacy per E/F from hardware throughput; parallelism alone is not algorithmic gain.
- Separate fixed-composition optimization from grand canonical equilibrium or kinetics.
- Stop retrieval when each named family has a source-backed finding or explicit access gap; no unlimited repo survey.

## Deliverables
- Short Chinese decision review, source-ledger.csv and claim-evidence-matrix.csv, bounded next-test priorities.
- Source checkouts outside production repo at /home/gengjianrui/bin/pam-ssw-research/reference-code-20260927.
- Subreports here; MAINLINE links one final decision, no core changes or PES jobs for this review.
