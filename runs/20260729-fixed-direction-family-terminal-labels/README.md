# Fixed C60 direction-family terminal-label experiment

## Question

Does direction-family identity explain strict terminal outcomes after starter
identity and all measured computational controls are frozen?

This is the evidence gate before any direction-family UCB, Thompson sampler,
classifier, or regressor is considered.

## Frozen protocol

- exact C60 starters: `intermediate_accepted`, `plateau_accepted`;
- seeds: 42, 43, 44;
- two exact repeats with reversed arm order;
- one proposal per case;
- four candidates per bias step;
- eight Gaussian-bias steps;
- 80-step `safe-lbfgs-total` proposal relaxation;
- ASE-LBFGS true quench with ASE-FIRE fallback;
- strict terminal force certificate;
- complete purpose-labelled force-evaluation ledger.

The only changed control is `n_bond_pairs`:

- `random_only`: momentum disabled and `n_bond_pairs=0`, yielding four random
  candidates;
- `bond_only`: momentum disabled and `n_bond_pairs=4`, yielding four bond
  candidates.

Every direction selection therefore costs exactly four central HVPs, or eight
force evaluations. The runner fails if either arm contains a mixed candidate
family.

## Pre-registered posterior gate

A stable label is meaningful only when both exact repeats:

1. have a strict true-quench certificate;
2. land in a new basin;
3. lower the starter energy by at least 0.001 eV.

The next posterior stage is entered only if:

1. both direction families have at least five stable meaningful labels; and
2. a starter-plus-family Beta(1,1) Bernoulli posterior predictive model has a
   lower Brier score than a starter-only model in every leave-one-seed-out
   fold.

There is no tunable acquisition weight or significance threshold in this
gate. Failure stops the direction-family posterior line instead of adding a
selector.

## Execution

Commit the protocol first, then run:

```bash
python runs/20260729-fixed-direction-family-terminal-labels/run_experiment.py \
  --output-dir runs/20260729-fixed-direction-family-terminal-labels-output \
  --expected-git-commit "$(git rev-parse HEAD)"
```

The raw output directory is intentionally not part of the method commit.
`evidence.json` and `conclusion.md` are reviewed before any result commit.
