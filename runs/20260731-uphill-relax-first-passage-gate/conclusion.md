# G-UP1 C60 biased-relaxation first-passage conclusion

## Decision

The preregistered gate returns **`RETAIN_CURRENT_LENGTH_NO_PROMOTION`**.  A
single accepted-step cutoff of 45 preserves the final basin on all four
holdouts, but only two holdouts have trajectories longer than 45 steps.  The
frozen rule required strictly positive shortening headroom on all four, so it
does not open a full-action gate and no production default changes.

This result must not be simplified to “45 steps failed.”  Under the operational
semantics of a maximum-step cap, the two shorter trajectories would terminate
naturally at steps 43 and 44 and remain unchanged.  Thus the basin-fidelity
observation is 4/4, while the preregistered promotion decision is still a
failure.  The distinction is retained rather than changing the rule after
seeing the data.

## What was isolated

G-UP0 showed that biased-PES relaxation is causally active: an explicit
displacement alone returns to the starter in repeated contexts where the same
displacement followed by biased relaxation escapes.  G-UP1 does not rerun the
direction oracle or the biased relaxation.  It reads every accepted
Safe-LBFGS frame from four hashed source trajectories, independently quenches
each intermediate frame on the true PES, and asks when the trajectory enters a
terminal suffix whose every frame reaches the same minimum as the original
final frame.

Frame zero and the original final frame reuse previously hashed quenches.  All
230 intermediate frames receive a new true-PES quench.  Consequently the gate
measures basin attraction, not biased energy, path length, or a geometric proxy.

## Exact first-passage result

| Discovery trajectory | First certified escape | Stable final-basin step | Original final step |
|---|---:|---:|---:|
| plateau, seed 42, D0, h1 | 13 | **45** | 80 |
| plateau, seed 43, D0, h1 | 13 | **15** | 49 |
| plateau, seed 43, K4, h2 | 23 | **23** | 57 |
| plateau, seed 44, K4, h2 | 6 | **12** | 48 |

The fixed discovery cutoff is therefore

\[
k_{\max}=\max(45,15,23,12)=45.
\]

The large separation between first escape and stable final-basin arrival in
two trajectories is the central physical result.  In seed-42 D0, frames 13--44
have already left the starter but do not yet quench unambiguously to the final
minimum; only frames 45--80 form the stable final-basin suffix.  In seed-44 K4,
frames 6--11 first reach a different escaped minimum and frames 12--48 reach
the final one.  Biased relaxation is therefore a curved propagator through
basin attraction boundaries, not merely a scalar displacement amplifier.

The complete discovery contains 42 return frames, 183 certified escaped
frames and 13 matcher-ambiguous frames.  Among landing relations there are 8
explicitly different landings and 32 additional ambiguous relations before
the final-basin suffix.  These observations rule out a monotonic assumption
that the first escaped frame is automatically a sufficient stopping point.

## Untouched holdout

| Holdout trajectory | Original final step | Executed step | Same final basin | New FE |
|---|---:|---:|---:|---:|
| intermediate, seed 42, D0, h4 | 80 | 45 | yes | 84 |
| plateau, seed 42, K4, h1 | 43 | 43 (natural stop) | yes | 0 reused |
| plateau, seed 42, K4, h4 | 80 | 45 | yes | 41 |
| plateau, seed 44, K4, h4 | 44 | 44 (natural stop) | yes | 0 reused |

Both newly quenched step-45 landings agree with the original final minimum by
the matcher and descriptor checks.  Their energy differences from the source
final landings are only about 0.00018 and 0.00009 eV, respectively.  The other
two trajectories had already converged before the cutoff and reuse their final
landings.

Across discovery and holdout, applying 45 as a *maximum* accepted-step cap
would change 481 recorded accepted steps to 357, a descriptive reduction of
124 steps (25.8%).  This is not yet an FE saving: Safe-LBFGS line searches can
use different numbers of force calls per accepted step, and a shorter biased
relaxation can increase the following true-quench cost.

## Cost and numerical conditioning

The gate consumed 20,562/32,000 new force evaluations in 393.8 seconds:

- 20,098 landing true-quench evaluations;
- 232 true-PES escape checks;
- 232 post-relax validations;
- zero direction-oracle evaluations;
- zero biased-proposal-relaxation evaluations;
- zero unattributed evaluations.

For the 230 new discovery quenches, FE cost ranges from 26 to 594, with median
49.5 and mean 88.9.  Some earlier accepted frames preserve the final basin but
require more than 500 force calls to settle, whereas later frames often quench
in tens of calls.  A shorter biased relaxation can therefore move cost rather
than remove it: less work on the biased PES may leave the true-PES quench near
a flat or ill-conditioned basin boundary.

This is why neither accepted-step count nor earliest basin escape is a valid
standalone objective.  The next online comparison, if opened as a new
hypothesis, must measure the combined cost

\[
C_{\mathrm{action}} = C_{\mathrm{direction}} + C_{\mathrm{biased\ relax}}
                      + C_{\mathrm{true\ quench}}
\]

at fixed action inputs and must preserve landing-basin support.

## What is now closed

G-UP1 closes the following shortcuts:

- stopping biased relaxation at the first frame that merely escapes the
  starter;
- treating basin identity as monotonic along the biased optimizer path;
- using biased energy decrease or displacement magnitude instead of true
  quench to define completion;
- claiming an FE saving from fewer accepted optimizer steps;
- adding an adaptive confidence/trust controller before a fixed-cap online
  counterfactual is understood.

It does **not** show that 80 is optimal, nor that a 45-step cap is ineffective.
The 4/4 holdout basin preservation supports one narrowly defined follow-up:
a fresh, paired full-action comparison of the existing natural-convergence
reference against `max accepted steps = 45`, with identical starter,
direction, Gaussian parameters and random stream.  Because the strict G-UP1
promotion rule failed, that comparison must be treated as a new exploratory
hypothesis on unseen actions, not as a promoted setting.

No selector, direction family, Gaussian form, optimizer algorithm, matcher,
quench or production default is changed by G-UP1.

## Claim ceiling

This is a C60 accepted-frame counterfactual over four discovery and four
holdout trajectories selected from mechanism-informative G-UP0 cases.  It
establishes non-monotonic basin first passage and 4/4 holdout basin fidelity
under max-cap semantics.  It does not establish online FE savings, trajectory
distribution equivalence, transfer to PdO/CuO, or a production relaxation
cutoff.
