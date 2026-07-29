# Staged C60 direction-efficiency ablation

This package runs a fixed-starter, one-proposal ablation of four existing
controls. It does not add a selector, posterior model, direction source, or
production default.

The closed campaign has these ceilings:

- Stage M: momentum on/off, 24 proposals.
- Stage K: `K=4/8/12`, 36 proposals.
- Stage B: five/eight bias steps, 24 proposals.
- Stage L: 40/80 proposal-relax iterations, conditionally 24 proposals.

Each arm is evaluated twice for every locked starter and seed. The repeat
order is reversed, and a starter/seed outcome is repeat-stable only when both
exact repeats are meaningful. Wall time is descriptive; decisions use strict
terminal outcomes and force-evaluation ledgers.

For the native discrete direction oracle, ranking `K` candidates costs exactly
`2*K` force evaluations per selection because every curvature is a central
finite-difference HVP. This is only the direction-ranking cost. Total cost is
the closed sum over direction oracle, biased proposal relaxation, escape
checks, strict terminal quench, validation, and any other declared purpose.

Run from the repository root in the `mace_les` environment:

```bash
python runs/20260729-staged-direction-efficiency-ablation/run_stage.py \
  --stage momentum \
  --output-dir runs/20260729-direction-efficiency-momentum \
  --expected-git-commit "$(git rev-parse HEAD)"

python runs/20260729-staged-direction-efficiency-ablation/run_stage.py \
  --stage candidate_count \
  --output-dir runs/20260729-direction-efficiency-candidate-count \
  --prior-evidence runs/20260729-direction-efficiency-momentum/evidence.json \
  --expected-git-commit "$(git rev-parse HEAD)"

python runs/20260729-staged-direction-efficiency-ablation/run_stage.py \
  --stage bias_steps \
  --output-dir runs/20260729-direction-efficiency-bias-steps \
  --prior-evidence runs/20260729-direction-efficiency-candidate-count/evidence.json \
  --expected-git-commit "$(git rev-parse HEAD)"

python runs/20260729-staged-direction-efficiency-ablation/run_stage.py \
  --stage relax_cap \
  --output-dir runs/20260729-direction-efficiency-relax-cap \
  --prior-evidence runs/20260729-direction-efficiency-bias-steps/evidence.json \
  --expected-git-commit "$(git rev-parse HEAD)"
```

If the saved Stage-L gate is false, replace the final command by the same
command with `--record-not-entered`. This writes a deterministic zero-case
record and performs no MACE evaluation.

Re-running an identical completed stage is a resume audit: pinned source,
commit, config, artifacts, hashes, and ledgers must all revalidate before a
case is reused. No result from this C60 campaign changes a production default.
