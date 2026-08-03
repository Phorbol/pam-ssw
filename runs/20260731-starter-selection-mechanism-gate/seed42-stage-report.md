# Seed-42 starter-mechanism stage report

## Question and frozen mechanism

This gate asks one narrow question:

> With direction generation, local softening, cumulative Gaussian uphill
> propagation, proposal relaxation, and true-PES quench frozen, how should the
> next starter minimum be chosen?

Each arm has a 20,000 force-evaluation ceiling.  One force evaluation means
one calculator energy-and-force call at one geometry.  Trial counts are not a
budget because different starters can make the biased relaxation and true
quench require different numbers of evaluations.

The three starter mechanisms are:

1. `uniform_archive`: every known minimum has equal probability;
2. `archive_ucb`: the existing fixed-weight UCB-like score over the full
   archive;
3. `metropolis_chain`: continue from the current accepted minimum, accepting
   downhill landings and accepting uphill landings with
   `exp[-DeltaE / 0.26 eV]`.

All physical-action random draws use a stream separate from starter selection.
The corrected CuO gate also true-quenches the raw structure once and reuses the
exact same bootstrap minimum in every arm.

## Exact-budget results

| System | Starter | Initial E (eV) | Best E (eV) | Drop (eV) | Actions | Archive minima | Wall time (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| C60 | uniform | -474.669708 | -490.094025 | 15.424316 | 42 | 40 | 346.8 |
| C60 | UCB-like | -474.669739 | **-496.659882** | **21.990143** | 49 | 40 | 342.1 |
| C60 | Metropolis | -474.669769 | -496.340729 | 21.670959 | 40 | 38 | 338.7 |
| PdO | uniform | -568.396973 | -572.520508 | 4.123535 | 72 | 63 | 450.6 |
| PdO | UCB-like | -568.396973 | -573.758911 | 5.361938 | 79 | 74 | 464.0 |
| PdO | Metropolis | **-568.396973** | **-575.166016** | **6.769043** | 74 | 68 | 461.7 |
| CuO | uniform | -198.677673 | -201.872681 | 3.195007 | 17 | 18 | 605.9 |
| CuO | UCB-like | -198.677673 | -201.686523 | 3.008850 | 18 | 19 | 608.0 |
| CuO | Metropolis | -198.677673 | **-201.874359** | **3.196686** | 20 | 21 | 608.7 |

C60/PdO were produced before the shared-bootstrap correction.  Their
independent bootstrap energies agree within about `6e-5 eV`, so they are useful
seed-42 evidence, but future paired runs must use the corrected protocol.  CuO
uses one shared 108-FE bootstrap; every arm then uses exactly 19,892 search
evaluations, for an exact total of 20,000.

## Physical interpretation of the selector result

Uniform archive sampling is clearly worse in C60 and PdO.  In CuO it is tied
with Metropolis to within 0.00168 eV.  This supports the previously observed
mechanism: the current escape kernel usually reaches a different basin, but
starting from arbitrary high-energy archive members often expands coverage
without lowering the landscape floor.

The existing UCB-like selector is not proven better than the much simpler
Metropolis chain:

- C60: UCB-like is 0.319 eV lower than Metropolis;
- PdO: Metropolis is 1.407 eV lower than UCB-like;
- CuO: Metropolis is 0.188 eV lower than UCB-like, but one numerically
  sensitive run is insufficient for a selector claim.

Metropolis does not have the same physical behavior in every system:

| System | Accepted / attempted | Downhill accepts | Uphill accepts |
|---|---:|---:|---:|
| C60 | 15 / 40 | 15 | 0 |
| PdO | 23 / 74 | 11 | 12 |
| CuO | 4 / 20 | 3 | 1 |

In C60 the 0.26-eV chain is effectively a downhill continuation policy.  PdO
uses finite-temperature uphill moves materially.  CuO mostly follows downhill
continuation but makes one uphill detour.

The clean algorithmic conclusion is therefore not “replace UCB by Thompson
sampling.”  It is:

> Treat low-energy funnel continuation and full-support archive restart as two
> physical action families.  Do not model every ever-growing archive node as
> an independent bandit arm until a simpler fixed continuation/restart
> comparator fails.

This retains nonzero global support while avoiding a growing-arm posterior in
which most nodes receive only one observation.

## CuO exposed a larger optimizer bottleneck

CuO is a 54-atom Cu(110)-Cu10O8 slab evaluated with the packaged finetuned
model.  The historical `z <= quantile(z, 0.35)` mask fixes two complete,
degenerate Cu layers: 24 fixed atoms and 30 movable atoms.

Despite being smaller than the 115-atom PdO slab, CuO completes only 17--20
actions per 20,000 evaluations.  Its cost is dominated by biased-PES proposal
relaxation:

| CuO starter | Proposal FE share | Direction FE share | Landing-quench FE share |
|---|---:|---:|---:|
| uniform | 85.35% | 6.24% | 7.31% |
| UCB-like | 84.79% | 7.52% | 6.50% |
| Metropolis | 83.28% | 7.20% | 8.34% |

More importantly, Safe-LBFGS line search behaves qualitatively differently:

| System / arm range | Rejected line trials / all line evaluations |
|---|---:|
| C60 | 2.69%--3.33% |
| PdO | 1.76%--5.64% |
| CuO | **57.57%--61.68%** |

For CuO, 77/78 uniform, 88/93 UCB-like, and 83/89 Metropolis proposal
relaxations terminate with `line_search_failed`.  The optimizer still often
makes useful displacement before failure, so this is not equivalent to a
failed SSW action.  It does show that roughly half of all local line-search
evaluations are rejected trial geometries and that nearly every proposal ends
because Armijo can no longer certify a step.

This cross-system result invalidates the earlier universal reading that
Safe-LBFGS backtracking is negligible.  The earlier statement remains true for
the tested C60/PdO corpus only.

## Evidence classification

- **Verified engineering correction:** starter RNG no longer advances the
  physical-action RNG.
- **Verified scientific correction:** all compared arms must reuse one exact
  bootstrap minimum and be charged the same bootstrap cost.
- **Finite-budget evidence:** low-energy continuation is useful; uniform
  archive sampling is a weak global-optimization baseline.
- **Statistically unresolved:** UCB-like versus Metropolis across systems and
  long horizons.
- **Mechanism-level bottleneck:** CuO Safe-LBFGS line search rejects about
  58%--62% of its trial evaluations and terminates almost every proposal on
  line-search failure.
- **Not admitted:** Thompson sampling, MACE-feature top-k/FPS, node-level
  posterior learning, or a new exact-bias optimizer.

## Next bounded experiment

Do not spend the next budget on a larger selector matrix yet.  First capture a
small fixed set of CuO biased-PES proposal tasks and inspect the final failed
line-search direction:

1. evaluate the modified-PES energy along the exact descent direction on a
   geometric alpha grid;
2. compare finite-difference directional slopes with the reported
   `g dot p`;
3. repeat in float32 and float64 if the packaged model supports both;
4. keep the state, cumulative Gaussian terms, L-BFGS history, and direction
   fixed.

Interpretation:

- if small-alpha energy changes disagree with `g dot p`, the first problem is
  calculator precision or energy-force consistency;
- if a descent interval exists but Armijo misses it, line-search acceptance or
  scaling is the isolated mechanism;
- if the energy is nonquadratic but internally consistent, only then test a
  nonmonotone or analytic-bias-aware step model.

After this optimizer gate, rerun the shared-bootstrap selector comparison with
multiple independent seeds/repeats.  The next selector comparator should be a
minimal full-support continuation/restart policy before any posterior or
embedding-based selector.
