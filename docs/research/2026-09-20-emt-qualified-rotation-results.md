# Cu55 EMT rotation control: qualified landing follow-up

这组结果是 Cu55/ASE EMT 的辅助诊断，不与 MH1 总能量合并，也不构成 DFT 或全局最低点结论。

## B：两个固定落点的曲率资格

[emt-cu55-landing-qualification-20260920](../../research/ga_ssw/evidence/emt-cu55-landing-qualification-20260920) 在看到此前 octa 对照结果并发现其起点负曲率之后、重新运行前，固定 baseline seed29 前两个收敛落点，并做了 fresh 检查和 330 EF 内部 Hessian 检查。两个落点均通过 `fmax <= 0.03 eV/Å`，且去除 6 个刚体模式后没有负 eigenvalue：candidate 1 的最低内部曲率为 `+0.0565544 eV/Å²`，candidate 2 为 `+0.00305224 eV/Å²`。这两个结构因此可作为后续固定候选；资格检查本身不证明全局稳定性或独立 basin。

## C：同预算旋转结果

[emt-cu55-qualified-rotation-20260920](../../research/ga_ssw/evidence/emt-cu55-qualified-rotation-20260920) 对 candidate 1/2 的 baseline 与 recovered CBD 各使用 6000 search EF。candidate 1 的 baseline/CBD 分别完成 8/12 个落点，candidate 2 分别为 8/15；四臂最终均在预算边界结束。这些是算法事件计数，不能直接解释为独立 basin 数。

最低 fresh 合格能量如下：

| 候选组 | baseline | recovered CBD | 差异 |
|---|---:|---:|---:|
| candidate 1 | 26.9527997 eV | 26.9731664 eV | baseline 低 0.0203667 eV |
| candidate 2 | 27.0003597 eV | 26.8874739 eV | CBD 低 0.1128858 eV |

candidate 1 的原始约 `0.02 eV` 排名差异很小，不能视为稳健优势；candidate 2 的 CBD 优势较大，但仍只来自一个固定 EMT 起点组和一次有限预算轨迹。

## D：统一 `.01` 复核

原始 [emt-cu55-qualified-refinement-20260920](../../research/ga_ssw/evidence/emt-cu55-qualified-refinement-20260920) 因导出输入错误排除其数值结果，但保留的 148 EF 运行证据仍归档。修正输入来源后的 [emt-cu55-qualified-refinement-20260920-r2](../../research/ga_ssw/evidence/emt-cu55-qualified-refinement-20260920-r2) 使用同一四个固定落点、`fmax=.01`，实际为 61 search EF 加 4 fresh EF，共 65 EF。

复核后的能量为：

| 候选组 | baseline | recovered CBD | 解释 |
|---|---:|---:|---|
| candidate 1 | 26.9429431 eV | 26.9391354 eV | 排名反向，CBD 仅低 0.0038076 eV，近似相同 |
| candidate 2 | 26.9984487 eV | 26.8861039 eV | CBD 仍低约 0.1123448 eV |

因此 candidate 1 的原始 `.02 eV` 差异在统一 `.01` 终止精度后反向且消失为小差异；candidate 2 的约 `.112 eV` CBD 优势在该 posthoc 复核中保留。额外 refinement 成本不计入原始 6000 search EF，也不能把 refinement 后的能量变化归因于原始旋转算法。

最初的高对称 octa 起点不作为稳定 basin 证据：[initial-curvature.json](../../research/ga_ssw/evidence/emt-cu55-equal-budget-rotation-20260920/initial-curvature.json) 给出的实际共同起点最低内部曲率为 `-0.0913761 eV/Å²`；旧资格档案中的 octa 也有一个负 eigenvalue。后续结论只使用两个已检查为正内部曲率的落点，并仍需独立结构去重和更广泛起点验证。
