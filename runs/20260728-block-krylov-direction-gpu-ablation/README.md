# Block-Krylov Stage-2 GPU ablation

This directory preregisters a fixed-total-force-evaluation direction-oracle
experiment.  It does not change the production default: that remains
`discrete`.

## Fixed C60 cohort

Run exactly one case at a time, from a clean checkout at the preregistered
commit.  The loop intentionally makes every output path explicit.

```bash
commit=$(git rev-parse HEAD)
for arm in discrete variational_breadth balanced_refinement deep_refinement; do
  for seed in 42 43 44; do
    python runs/20260728-block-krylov-direction-gpu-ablation/run_ablation.py \
      --system c60 \
      --arm "$arm" \
      --seed "$seed" \
      --force-budget 6000 \
      --device cuda \
      --output-dir "runs/20260728-block-krylov-direction-gpu-ablation/cases/c60-${arm}-seed${seed}" \
      --expected-git-commit "$commit"
  done
done
```

Then build the C60 evidence.  It accepts only the complete `4 arms x 3 seeds`
cohort, checks exact purpose-ledger closure and block diagnostics, and applies
the preregistered survivor rule plus amended tie-break.

```bash
python runs/20260728-block-krylov-direction-gpu-ablation/analyze_evidence.py \
  --system c60 \
  --input-dir runs/20260728-block-krylov-direction-gpu-ablation/cases \
  --output-dir runs/20260728-block-krylov-direction-gpu-ablation
```

## Conditional PdO transfer

Only when C60 evidence records `selected_pdo_transfer_arm`, run `discrete` and
that exact selected arm for all three seeds.  The analyzer reads the selected
arm from committed C60 evidence; it rejects a manually chosen or unexpected
PdO arm.

```bash
selected=$(python -c 'import json; print(json.load(open("runs/20260728-block-krylov-direction-gpu-ablation/evidence.json"))["selected_pdo_transfer_arm"] or "")')
test -n "$selected"
commit=$(git rev-parse HEAD)
for arm in discrete "$selected"; do
  for seed in 42 43 44; do
    python runs/20260728-block-krylov-direction-gpu-ablation/run_ablation.py \
      --system pdo \
      --arm "$arm" \
      --seed "$seed" \
      --force-budget 6000 \
      --device cuda \
      --output-dir "runs/20260728-block-krylov-direction-gpu-ablation/cases/pdo-${arm}-seed${seed}" \
      --expected-git-commit "$commit"
  done
done

python runs/20260728-block-krylov-direction-gpu-ablation/analyze_evidence.py \
  --system pdo \
  --input-dir runs/20260728-block-krylov-direction-gpu-ablation/cases \
  --output-dir runs/20260728-block-krylov-direction-gpu-ablation
```

The C60 three-seed criterion is a survivor gate, not a significance claim.
