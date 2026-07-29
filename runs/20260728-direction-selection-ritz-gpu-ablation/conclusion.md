# Direction-selection representation evidence

每一对仅在 `direction_selection_mode` 上不同（`discrete` 对 `rayleigh_ritz`）；native candidate generation、central HVP 协议、oracle_candidates、starter selector、safe-LBFGS proposal、uphill policy、true quench、LS softening 与总 FE budget 均经配置比对。
Ritz true curvature 是复用 native central-FD true-HVP 的子空间投影；在非线性 PES 上，它与 direct mixed-direction central-FD stencil 相差 `O(hvp_epsilon^2)`。真实方向成本取purpose-resolved `direction_oracle` FE；legacy `candidate_count` 在 Ritz 成功时包含零-HVP synthetic candidate，不能据此推断 Ritz 的 HVP 成本。

| system | seed | delta best energy (eV) | delta energy drop (eV) | delta direction FE | delta trials | delta wall (s) | escape true-PES FE (D / RR) |
|---|---:|---:|---:|---:|---:|---:|---:|
| C60 | 42 | +3.684418 | -3.684937 | -96 | -1 | -1.078 | 85 / 83 |
| C60 | 43 | -4.893707 | +4.894348 | +94 | -3 | +2.264 | 74 / 78 |
| C60 | 44 | -1.670715 | +1.670715 | +48 | +0 | -6.703 | 82 / 82 |
| PDO | 42 | +0.316650 | -0.316589 | +48 | +0 | +0.671 | 53 / 54 |
| PDO | 43 | +1.036560 | -1.036682 | +32 | +0 | +2.018 | 49 / 53 |
| PDO | 44 | -1.897766 | +1.897827 | +96 | -1 | +13.847 | 58 / 61 |

claim ceiling：本实验只比较相同方向候选/HVP 协议、并使用 projected native true-HVP curvature 的选择表示；不证明平衡态无偏性，也不证明一般系统优越性。能量 AUC 仅在 energy trace 提供 cumulative total FE 时报告；缺失时明确标记 unsupported。
