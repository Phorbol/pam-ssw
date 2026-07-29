# Fixed-starter productive-escape ablation

This experiment tests whether a direction that is variationally softer also
produces a better landing minimum under the existing uphill propagator.
It does not change the production default.

## Frozen factors

- system: C60 only;
- starters: the preregistered `bootstrap_quenched`,
  `intermediate_accepted`, and `plateau_accepted` states from the fixed-state
  direction audit;
- paired RNG seeds: 42, 43, and 44;
- uphill propagator: one existing SSW walk, with at most eight serial
  bias-relax steps;
- proposal relaxation: `safe-lbfgs-total`, `fmax=0.05`, 80 steps;
- true quench: `scipy-lbfgsb`, `fmax=0.01`, 400 iterations;
- local softening, bias, trust, and step-length settings: unchanged from the
  frozen C60 production configuration;
- one proposal and one true-PES landing quench per case;
- no starter selector, posterior selector, UCB/TS update, retry, or rescue.

## Direction arms

Every completed direction selection is allocated 12 central-difference HVPs,
or 24 force evaluations:

1. `discrete`: the existing 12-candidate direction portfolio;
2. `variational_breadth`: block Krylov 6x1;
3. `balanced_refinement`: block Krylov 2x3;
4. `deep_refinement`: block Krylov 1x6.

The complete cohort is 3 starters x 3 seeds x 4 arms = 36 cases. Arms within
one starter/seed pair use the same RNG seed.

## Recorded outcomes

Each case records:

- starter, escape-configuration, and quenched-landing energies;
- landing energy relative to the starter;
- whether the landing is a distinct archive minimum;
- structural descriptor displacement;
- direction-selection count and HVP/force-evaluation closure;
- purpose-resolved direction, proposal-relax, true-quench, validation, and
  total force evaluations;
- wall time, quench convergence, fragmentation, and failure type.

The analysis is descriptive rather than a new promotion heuristic. It reports
state-stratified paired landing-energy differences versus `discrete`,
new-basin/downhill-landing rates, and cost distributions. No arm becomes the
production default from this experiment alone.

