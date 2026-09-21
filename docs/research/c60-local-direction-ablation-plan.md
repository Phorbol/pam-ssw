# C60 local-direction ablation, declared before new PES calls

2026-09-11. One diagnostic ordinary-SSW step on the existing CCD1809asym C60
initial geometry, MACE-OMAT-small CPU float64, seed3. Existing ordinary paper
arm is the matched baseline:179 E/F, later strict quench supports same basin.
Keep its frozen source, input, max12 Gaussians, width.6, rotation_bias100,
T150K, fmax.01, relax400, Safe memory400, dimer100HVP/fd1e-4/tol.02 and
zero-margin early exit. No LS, terminal reconnection or altered MC temperature.

Two component controls, each at most2000 E/F including two reserved fresh
certificates and900 seconds, without retuning or retries:

1. Unit local: preserve the exact paper global vector, eligible-pair draw and
   lambda draw; independently normalize the raw two-atom local contribution.
2. Native cooperative unit local: preserve those same global, pair andlambda
   draws; replace only the local contribution with the independently reproduced
   native local-pair helper, then unit-normalize it before mixing.

For control2, use a cloned RNG state after the common lambda draw for the
helper's additional neighbor draws. The outer one-step MC RNG stream remains
common across controls. This is deliberate common-random-number bookkeeping
for this diagnostic, not a proposed multi-step production RNG policy or complete
native sampler. Save the chosen pair,lambda,global/local/mixed vectors, helper
draws, input/output source and complete force-request ledger.

Success measures: a certified intact distinct low-energy minimum, compared with
the starting basin and original paper endpoint, and full E/F cost. Also report
failed quench/rotation, fragmentation, coordination and energy, independent of
MC acceptance. A certificate or more completed Gaussian stages alone is not
success. If a near-identical landing occurs, independent stricter quench and
structural comparison require separately stated additional diagnostic cost.
No minimum is assumed to be the C60 global optimum; MACE transfer accuracy,
Hessian stability and DFT chemistry are unqualified. These existing seed/input
experiments diagnose a mechanism, not independent cross-system validation.
No default will change on these two outcomes alone.

## Second-stage diagnostic, after the zero-margin outcomes

Both new controls return the original basin: independent fmax.001 quenches cost
21 E/F combined, aligned RMSD<6e-5Angstrom and identical bond graphs against the
saved strict references. Thus the zero-margin energy-triggered exit masks the
intended escape comparison. Complete the existing3-by2 diagnostic matrix with
the already examined native0.1eV early-exit margin: raw-paper+margin is archived
(1315EF, fragmented C58+C2); unit-local+margin and cooperative+margin are new.
Use the same0.1 value, input, seed and all other settings, with2000EF/900s per
new arm. This is a research-only source patch, not a new production parameter.
It tests interaction between two recovered mechanisms, not retrospective tuning
of a new threshold. Preserve the unsuccessful zero-margin outcomes and their
costs. Do not claim independent validation or select a new default from this
single-input diagnostic matrix. Further sweeps of the margin, radii or mixture
coefficient are outside this plan.
