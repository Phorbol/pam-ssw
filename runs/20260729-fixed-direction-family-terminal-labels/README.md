# Fixed C60 direction-family terminal-label experiment

## Question

Does direction-family identity explain strict terminal outcomes after starter
identity, per-selection oracle cost, and all algorithmic controls are frozen?

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
- strict terminal force-certificate audit;
- complete purpose-labelled force-evaluation ledger.

The only changed control is `n_bond_pairs`:

- `random_only`: momentum disabled and `n_bond_pairs=0`, yielding four random
  candidates;
- `bond_only`: momentum disabled and `n_bond_pairs=4`, yielding four bond
  candidates.

The outcome-gated `stagnation_bond_pair_boost` is frozen to zero in both arms;
otherwise an unproductive relaxation could silently add bond candidates to
the `random_only` arm.

Every direction selection therefore costs exactly four central HVPs, or eight
force evaluations. The runner fails if either arm contains a mixed candidate
family. Total realized force evaluations may differ because an action can
change walk length, proposal-relaxation work, and true-quench difficulty; those
differences are retained as outcomes rather than normalized away.

## Pre-registered posterior gate

A stable label is meaningful only when both exact repeats:

1. have a strict true-quench certificate;
2. land in a new basin;
3. lower the starter energy by at least 0.001 eV.

An uncertified terminal is retained with
`terminal_failure=strict_quench_nonconvergence`, its complete cost remains in
the ledger, and its meaningful label is false. It is not silently dropped or
retried until success.

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
