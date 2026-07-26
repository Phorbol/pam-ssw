# Thread-safe equal-budget multi-seed rerun

This is the final clean rerun of the 3-seed FIRE versus safe-L-BFGS matrix
after commit `61f32ef` serialized each independent calculator's first real
evaluation.

It preserves the previous matrix exactly:

- C60 and PdO
- seeds 42, 43, 44
- uniform policy
- ThreadPool workers 2
- action budget 1000
- total budget 3000
- same certified bootstrap state

The rerun is required because one pre-fix arm had a first-evaluation MACE/e3nn
race and was not benchmark-eligible. Results are not spliced across commits.
