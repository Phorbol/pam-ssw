# Fixed proposal-relaxation replay design

## Question

Does a proposal-relaxation backend reduce force evaluations on the same real
MACE biased-PES task, rather than merely changing the adaptive SSW trajectory?

## Experimental separation

The experiment has two strata that are never pooled:

1. **Primary, optimizer-neutral stratum**: the first proposal-relaxation task
   generated from the same certified bootstrap minimum under independent random
   action seeds. Each task contains one Gaussian bias.
2. **Secondary, mechanism stress stratum**: later multi-bias tasks captured from
   a frozen FIRE reference walk. This stratum is explicitly FIRE-conditioned
   and is not evidence of an unbiased task distribution.

The primary stratum is unbiased only with respect to the optimizer comparison:
task construction does not depend on which replay backend is tested. It is not
claimed to be a canonical or uniform sample of the full PES.

## Minimal code boundary

Extract the already-existing pre-relax data into one immutable
`ProposalRelaxationTask` and one protected execution method on `SurfaceWalker`.
The task contains the initial `State`, cumulative Gaussian biases, optional
local-softening model, convergence certificate, iteration limit, and trust
radius. It does not contain an optimizer.

Production behavior remains unchanged: the walker constructs the task and
immediately executes it with the configured backend. A benchmark-only subclass
may intercept the protected execution method to freeze the task before any
proposal-relaxation force evaluation.

No starter, direction, curvature, bias, archive, posterior, quench, or budget
policy is changed.

## Replay contract

For each frozen task and backend:

- construct a fresh calculator/accounting context;
- start from exactly the same state and biased objective;
- use the same `fmax`, `maxiter`, and trust radius;
- require a finite result and the same per-atom force certificate;
- record backend force evaluations, wall time after model initialization,
  termination reason, final biased energy, and final positions.

FIRE, FIRE2, and safe-total L-BFGS are replayed. Bias-separated L-BFGS is not
reopened because the previous real-GPU gate rejected its mechanism.

## Interpretation

Speed claims are made only within the same endpoint class. Endpoints are
classified by final biased energy and MIC-aware structural displacement; a
different endpoint is search behavior, not relaxation-speed evidence.

No weighted score is introduced. Results remain a vector:

- certificate coverage;
- paired backend force evaluations;
- wall time;
- endpoint class and final biased energy;
- failure reason.

Only a backend that preserves certificate coverage and shows consistent paired
force-evaluation improvement on both C60 and PdO is eligible for later
multi-seed full-search validation. Mixed evidence triggers diagnosis, not
parameter tuning.

## Stop rules

- Keep the task-boundary refactor only if the full existing test suite remains
  green and default production outputs are unchanged.
- Do not add dependencies.
- Do not tune backend parameters in the baseline replay.
- At most two subsequent optimizer mechanisms may be tested, each from an
  explicit mathematical hypothesis and each independently removable.
