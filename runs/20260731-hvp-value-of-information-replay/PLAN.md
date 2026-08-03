# Initial K4 HVP value-of-information replay

## Question

Does paying four central finite-difference HVPs (8 force evaluations) to rank
the initial K4 candidate pool improve terminal quality enough to justify that
cost?

This is deliberately narrower than asking whether every HVP used later in the
serial uphill walk should be removed.

## Frozen evidence

- The two exact-repeat shared-pool campaigns in
  `runs/20260731-direction-candidate-counterfactual-gate`.
- The independent live D0/K4 pairs in
  `runs/20260731-action-family-transfer-gate`.
- No new force evaluations.

## Result-independent rules

1. `static_k4`: select the existing static rank-1 candidate and add the
   separately recorded 8-FE shared-pool cost.
2. `uniform_no_hvp`: equal-probability expectation over all four generated
   candidates; do not add the initial shared-pool HVP cost.
3. `family_rotation_no_hvp`: alternate bond/random from the
   `(state_id, seed)` parity, giving 3 bond and 3 random groups per system;
   use the equal-probability expectation over the two candidates in the
   selected family.
4. `D0_exact_anchor`: report the existing live paired D0/K4 observation as an
   external anchor. It is not pooled with the K4 counterfactual candidates.

Only candidates with strict certificates and valid geometry in both exact
repeats are eligible. No missing result is imputed.

## Gate

A live selected-only-HVP gate opens only if at least one zero-HVP rule has, on
both C60 and PdO:

- median terminal regret no worse than static K4; and
- lower median projected force-evaluation cost.

The result cannot justify deleting HVPs from later continuation steps because
those steps remain unchanged in the source trajectories.
