# SSW / VC benchmark 原始文献与坐标补充

> 2026-09-10 更新：用户提供的 http://lasphub.com/publication/87.pdf 已成功取得并核验为12页正式排版全文（2,477,395 bytes），存于 `literature/benchmark-sources/vc2014/`。以下此前获取失败的记录保留作历史；正文缺口已解决，SI仍未取得。全文揭示2014为CBD-cell/atomic-SSW分块耦合，见 `vc2014-native-crosscheck.md`；当前joint log-strain方法须保留独立扩展标识。

2026-09-10。**已合法取得2017 SSW-NN全文、SI及24组TiO2结构坐标，也取得Cambridge
LJ38/LJ75的GM参考坐标。2014 VC-SSW全文/SI仍未下载成功，缺口不阻碍独立VC实现。**
所有工作仅下载公开材料、解析与格式转换，没有PES调用、优化或模型下载。

材料目录：本工作树 `literature/benchmark-sources/`。`manifest.json`记录文件大小与
摘要，`attempts-*.json`保存HTTP状态及失败，`coordinate-manifest.json`逐结构保存
名称、角色、原子数、晶胞和来源。不同格式明确区分，不将HTML验证页存成PDF。

## 2017 SSW-NN：全文与TiO2结构已到位

Huang, S.-D.; Shang, C.; Zhang, X.-J.; Liu, Z.-P. *Material discovery by combining
stochastic surface walking global optimization with a neural network*.
Chemical Science **2017**, 8, 6327–6337，DOI
[10.1039/c7sc01459g](https://doi.org/10.1039/c7sc01459g)。

RSC下载入口失败后，使用官方Europe PMC接口成功获取：

- [全文XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC5628601/fullTextXML)
  → `nn-fulltext.xml`，另有提取正文`nn-fulltext.txt`；是完整XML，不声称取得正文PDF。
- [Supplementary files](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC5628601/supplementaryFiles)
  → `nn-supplementary.zip`，约1.4 MB，包括文章图和唯一SI PDF。
- 提取 `SC-008-C7SC01459G-s001.pdf`（310,818 bytes）与`nn-si.txt`。
  PDF MD5 `e8e463594e595fa81b2cb2fec1b73997`，与全文XML的原附件标记相同。

SI§7明确给出晶胞PBC行及Cartesian原子坐标，不需要从图片猜测。转换产物同时保留
原数字ARC和ASE extxyz，位于 `coordinates/`：

| 结构 | 原子数 | 产物例子 |
|---|---:|---|
| TiO2-B、anatase、TiO2-II、rutile | 各12 | `tio2-b.arc`、`anatase.extxyz`、`rutile.extxyz` |
| Str-1至Str-8、lepidocrocite、TiO2-R、TiO2-H、baddeleyite、TiO2-OII | 各12 | 按原文名称命名的ARC/extxyz |
| TiO2-OI | 24 | `tio2-oi.extxyz` |
| brookite、phase-87、phase-139 | 各48 | `brookite.extxyz`等 |
| phase-87→anatase的IS、FS、TS | 各12 | 名称含is/fs/ts的文件；manifest保留角色 |

合计21个reported phase、2个endpoint、1个TS，共24组。**TS不能加入GM参考极小值集。**
所有转换都用ASE读取、核对Ti:O=1:2、原子数、有限坐标及正体积；extxyz回读坐标误差
不超过约5×10⁻⁹ Å，原ARC保留PDF文本的原精度。没有声称这些结构在新calculator上
仍是驻点或相同能量次序。SI列出的VASP能量只有两位小数，保留为rounded_source_energy，
不能直接作为严格能量比对阈值。

提取脚本：`research/ga_ssw/extract_benchmark_coordinates.py`。适合独立VC主线先使用
12原子anatase/rutile/TiO2-B作为有真实文献来源的输入，再由选定ASE后端进行独立
E/F/stress与驻点检查。这里没有下载或转换原NN权重，不把坐标可用性等同于原模型复现。

## Cambridge LJ38 / LJ75参考

来自Doye/Wales作者库的[原始GM表](https://www-wales.ch.cam.ac.uk/~jon/structures/LJ/tables.150.html)，
表中明确长度单位为σ、能量单位为pair well depth ε：

| N | 点群 | 原表GM能量/ε | 原始坐标 |
|---|---|---:|---|
| 38 | Oh | -173.928427 | [points/38](https://www-wales.ch.cam.ac.uk/~jon/structures/LJ/points/38) → `lj38.points` |
| 75 | D5h | -397.492331 | [points/75](https://www-wales.ch.cam.ac.uk/~jon/structures/LJ/points/75) → `lj75.points` |

两份仅2,394和4,725 bytes。只下载这两份坐标和小型索引页，没有下载全库tar或35k轨迹。
解析分别为38×3、75×3有限坐标；转换`coordinates/lj38-gm.extxyz`、`lj75-gm.extxyz`
使用占位元素X和pbc=False，不冒用真实Au。未计算能量；使用ASE LJ时必须显式核对
截断/shift等约定，不能期待ASE默认截断势精确给出原表未截断LJ能量。

## 2014 VC-SSW：明确缺口和已尝试渠道

Shang, C.; Zhang, X.-J.; Liu, Z.-P. *Stochastic surface walking method for crystal
structure and phase transition pathway prediction*. PCCP **2014**, 16,
17845–17856，DOI [10.1039/c4cp01485e](https://doi.org/10.1039/c4cp01485e)。

官方RSC可索引页面明确有author-version PDF和129 KB Supplementary information。
本轮仍无法取得其文件：

- RSC landing、articlehtml及
  [author version](https://pubs.rsc.org/en/content/getauthorversionpdf/C4CP01485E)
  返回403。搜索缓存能见作者稿摘要，不视作已取得全文。
- 主agent此前尝试 `/suppdata/cp/c4/...` 返回404；本轮另一常见路径
  `/suppdata/c4/cp/...`也404。这些是候选路径失败，**不代表SI不存在**；尚未获得
  可验证的实际attachment href。未无限枚举路径。
- Crossref元数据成功（`vc-crossref.json`），未提供可成功下载的公开全文副本。
  OpenAlex结果（`vc-openalex.json`）报告无OA仓储链接，仅RSC/PubMed位置。
- 作者LASP公开站HTTPS出现TLS EOF错误，HTTP publication.html返回404；学术检索
  只找到其他论文引用此文，未找到可验证的作者公开原文镜像。
- 2017 RSC SI候选请求曾返回429，本轮没有持续重试该主机；转用Europe PMC成功。

需要用户补充的精确材料（若后续要求原2014逐式/逐参数复现）：

1. DOI10.1039/c4cp01485e全文PDF，出版社版或34页accepted manuscript均可。
2. 同文129 KB的Supplementary information PDF。
3. 若SI没有给机器坐标，再补该文SiO2/SrTiO3/TiO2案例的原始坐标和晶胞文件；目前
   不预断言该SI是否包含这些内容。

这些是文献复现证据缺口，不是独立联合坐标与E/F/stress链式法则实现的阻塞条件。
2017年真实TiO2坐标及上传AlOH已提供独立VC输入来源，继续主线时应注明具体来源和
替换后的ASE势面。
