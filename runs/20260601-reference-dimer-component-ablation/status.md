# Status

Current phase: CUDA Phase 1 completed.

Approved scope executed:
- System: C60
- Variant: `paw_ucb_reference_dimer_bias_relax`
- Seed: 0
- Trials: 1
- Device: CPU

Result:
- Exit code: 0
- Best energy: -480.509155 eV
- Minima: 2
- Force evaluations: 1048
- Wall time: 124.662 s
- Reference-dimer steps: 8
- Reference-dimer convergence count: 0
- Reference-dimer mean rotations: 15.0

Diagnostic note:
- The implementation path is exercised because `direction_selected_reference_dimer=8`.
- The smoke is not a production-quality benchmark. `reference_dimer_converged_fraction=0.0` indicates the default dimer rotation tolerance or step budget needs attention before interpreting larger ablations.

Next proposed phase:
- Use CUDA only for formal tests.
- Start with C60 Phase 1: seeds `0,1,2`, 40 trials per seed, all six variants in `plan.md`.
- Do not mix CPU smoke rows into formal conclusions.
- Primary comparison fields: best energy, n_minima, duplicate_rate, force_evaluations, wall_time_s, reference_dimer_mean_rotations, reference_dimer_converged_fraction, direction-type productivity.

CUDA Phase 1 attempt:
- Started: 2026-06-01T12:09:21+08:00
- Scope: C60, seeds `0,1,2`, 40 trials, six variants, device `cuda`
- Status: failed before first case completed
- Failed case: `c60_reference_original_seed0_trials40_cuda`
- Failure: `torch.cuda.is_available() is False`
- Root cause: command was launched from a non-GPU shell. The reference script printed the required allocation command: `srun -p 4V100PX --gres=gpu:1 --qos=improper-gpu --pty bash`.
- Follow-up probe: sandboxed `nvidia-smi` failed with `GPU access blocked by the operating system`; non-sandbox CUDA probe succeeded on `NVIDIA GeForce RTX 3060` with `torch.cuda.is_available() == True`.
- Next action: restart the approved Phase 1 command outside the sandbox so CUDA is visible.

CUDA Phase 1 final result:
- Completed cases: 18/18
- Rows in `results.csv`: 18
- Best overall variant by mean best energy: `paw_current_bias_relax`
- Best single run: `paw_current_bias_relax`, seed 0, best=-504.182098 eV
- `reference_original` mean best energy: -486.519063 eV
- `paw_current_bias_relax` mean best energy: -499.846965 eV
- `paw_ucb_reference_dimer_bias_relax` mean best energy: -493.887777 eV
- `paw_ucb_reference_dimer_direct_qp_adaptive50` mean best energy: -474.669576 eV

Final diagnostic:
- `paw_ucb_reference_dimer_bias_relax` failed as a production replacement because mean duplicate rate was 0.561 and mean minima was 18.0.
- Its dimer rotation diagnostics were pathological: every seed had `reference_dimer_mean_rotations=15.0`, `reference_dimer_converged_fraction=0.0`, and `reference_dimer_mean_abs_dot_initial≈0.999`.
- The current scored direction pool plus UCB remains the C60 production baseline to beat.

Runner safety gate:
- `run_matrix.py` prints planned cases by default.
- Benchmark execution requires the explicit `--execute` flag.
