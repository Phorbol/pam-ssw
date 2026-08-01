# U-O1: uphill action observability gate

## Question

At fixed production settings, does the current cumulative-Gaussian uphill walk
fail mainly because it does not deliver its requested true-PES endpoint height,
or because an attained escape endpoint quenches back into an unproductive basin?

## Frozen cohort

- systems: C60, fixed-bottom PdO slab, CuO;
- seed: 49;
- total budget: 20,000 force evaluations per system, bootstrap included;
- starter: Metropolis chain;
- macro target: current archive-scaled controller;
- all direction, softening, proposal-relax and true-quench settings: unchanged
  from the admitted production profiles used by the fixed-target gate.

## Observable

For micro step `k`, the walker already evaluates the true PES immediately before
and after biased proposal relaxation. No additional calculator call is made.

`observed_max_height = max(E_before_k, E_after_k) - E_before_0`

`target_delivery_ratio = observed_max_height / requested_macro_target`

This is an endpoint observable. It is not a transition-state energy, barrier,
or maximum along the optimizer line search.

## Diagnostic boundary

1. `TARGET_NOT_DELIVERED`: in at least two systems, a strict majority of actions
   has delivery ratio below one.
2. `DELIVERED_BUT_UNPRODUCTIVE`: in at least two systems, delivered actions are
   a strict majority, while most delivered actions are duplicates/rejections or
   landing true quench is the largest non-proposal action cost.
3. `MIXED_SCALAR_TARGET_EVIDENCE`: neither cross-system condition closes.

The label chooses the next mechanism gate only. One seed cannot promote a new
controller or support posterior training.

## Analysis correction after the first raw cohort

The first 60,000-FE execution exposed two observability defects before a final
claim was accepted:

1. the original second label conflated an unproductive landing distribution
   with landing being the largest *non-proposal* cost, even when most landings
   were new basins;
2. completed-step sums omitted the cost of an interrupted final micro step.

The confirmatory schema therefore records exact walk-level purpose totals and
splits `DELIVERED_BUT_UNPRODUCTIVE` from
`TARGET_DELIVERED_PROPOSAL_RELAX_COST_DOMINANT`. The raw first execution remains
preserved in ignored `output/`; it is not used as a confirmatory replicate.
No search decision, calculator call, random draw, or production parameter is
changed by this correction.
