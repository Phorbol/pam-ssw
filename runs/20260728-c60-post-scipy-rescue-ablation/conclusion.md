# C60 post-SciPy accepted-endpoint refinement/rescue ablation

This experiment re-relaxes all 143 accepted endpoints produced by the completed SciPy landing path. It is a post-SciPy accepted-endpoint refinement/rescue study, not a fair replacement comparison against the original landing process.

## Initial force certificates

SciPy-arm initial counts at 0.10/0.05/0.01 eV/Å: 142/138/20. Safe-arm re-evaluations: 142/138/20. These derived loose-threshold counts do not add a third optimizer arm; there is no third optimizer arm.

## Terminal outcomes

Safe L-BFGS reached 59/143 versus 32/143 for the SciPy restart, but used 8,473 versus 2,188 evaluator calls. Per-task evaluator calls (median/p90/max) were 32/139.2/473 for safe and 13/26.8/51 for SciPy; total wall times were 142.252 s and 41.0081 s, respectively. Terminal force (median/p90/max, eV/Å) was 0.0114599/0.0286353/0.0598749 for safe and 0.016268/0.0337176/0.0666634 for SciPy.

The strict 0.01 eV/Å paired contingency (safe-only/SciPy-only/both/neither) was 34/7/25/77. 84 line-search failures consumed 7,876 evaluator calls (median 59), with 521 accepted steps, 7,196 rejected steps, and 7,717 line-search evaluations.

Initial-to-final energy changes (total/median/p90/min/max, eV) were -0.0168762207031/-6.103515625e-05/0/-0.000701904296875/3.0517578125e-05 for safe and -0.00885009765625/0/3.0517578125e-05/-0.0013427734375/0.000152587890625 for SciPy.

## Numerical resolution evidence

The minimum non-zero absolute final-minus-initial energy step was 3.0517578125e-05 eV, equal to the float32 ULP at approximately -500 eV (3.0517578125e-05 eV). This numerical match supports the inference that energy quantization may contribute to Armijo stalling near minima; it does not prove the root cause.

Under the current archive semantics of |dE| <= 1e-3 eV plus Kabsch RMSD <= 0.15 Å, 142/143 terminal pairs are same-basin.

The basin label is limited to those current archive semantics. It is not a general chemical-equivalence certificate or evidence that either arm fairly replaces the original landing workflow.
