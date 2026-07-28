# Fixed raw-landing tiered true-quench ablation

Stage A captures are conditioned on the fixed seed-42, safe-LBFGS proposal, first-16-trial capture policy.

| Stage | System | Arm | Certificates | Evaluator calls | Wall s |
| --- | --- | --- | ---: | ---: | ---: |
| loose | c60 | scipy-lbfgsb | 16/16 | 1381 | 21.98 |
| loose | c60 | safe-lbfgs-total | 16/16 | 1141 | 19.09 |
| loose | c60 | ase-fire2 | 11/16 | 3626 | 61.73 |
| loose | pdo | scipy-lbfgsb | 16/16 | 759 | 16.20 |
| loose | pdo | safe-lbfgs-total | 16/16 | 653 | 13.83 |
| loose | pdo | ase-fire2 | 16/16 | 912 | 19.59 |
| refine | c60 | scipy-lbfgsb | 0/16 | 431 | 7.04 |
| refine | c60 | safe-lbfgs-total | 0/16 | 1019 | 17.29 |
| refine | c60 | ase-fire2 | 16/16 | 1139 | 19.48 |
| refine | pdo | scipy-lbfgsb | 5/16 | 806 | 17.14 |
| refine | pdo | safe-lbfgs-total | 11/16 | 1218 | 26.72 |
| refine | pdo | ase-fire2 | 15/16 | 864 | 19.24 |

Same-basin labels use only the current per-system archive energy and RMSD semantics.

Stage B is a fixed raw-landing local true-quench comparison. Stage C shares each task's SciPy loose endpoint, so it measures conditional refinement rather than an independent optimizer or global-search outcome.
