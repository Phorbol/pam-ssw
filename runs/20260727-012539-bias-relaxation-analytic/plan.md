# Deterministic bias-relaxation ablation

Purpose: test correctness and raw objective-call behavior before any GPU
campaign. This run does not rank optimizers or tune controls from outcomes.

Execute the committed harness once on two frozen diagonal-quadratic PES cases
with one or two analytic Gaussian hills. Compare `ase-fire`, `ase-fire2`,
`safe-lbfgs-total`, and `bias-separated-lbfgs` using the same case-level
`fmax=1e-6` and `maxiter=400`.

Advance both custom modes to the GPU G1 screen only if both converge on both
cases, close their objective-call ledgers, emit finite results, and preserve
the preregistered secant invariants. Do not retune a failed arm.
