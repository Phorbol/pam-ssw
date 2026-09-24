# Energy-exit diagnostic, 2026-09-24

Question: do existing trajectories justify investigating premature `lower_true_energy` exits before changing bias policy? Competing explanations: small local energy improvements end climbing without meaningful escape; or this exit is rare / associated with substantial descent, so return-to-basin failures mainly concern the Gaussian-cap route.

Read the six completed C4H6/MH1 coverage arms and four periodic OMAT rotation arms, all records without filtering failures. Reconstruct current energy using accepted converged landings, reject pool routing. Report exit denominators and signed climb/landing energy drops, without defining a new cutoff or claiming basin identity. These are reused development trajectories, not new independent evidence. No parameter/algorithm changes. Maximum 5 CPU minutes, zero PES calls, no GPU. If tiny energy exits appear relevant, inspect saved structures next; otherwise close this branch rather than run an exit-policy parameter scan. Energy alone cannot establish whether a basin changed.

## Result and decision

CPU1477785 COMPLETED in16s, stderr empty; all10 inputs passed current-energy reconstruction checks. C4H6: native-LS800/800 and paper-LS800/800 end at Gaussian cap; SSW787/800 cap,12/800 lower-energy exits,1/800 nonpositive height. Those12 early exits have climb drops0.000040–0.001696eV; they are a small minority and no structural claim is made from these differences. Periodic four arms:78/84 cap,2/84 lower-energy (AlOH first event, ~0.236eV descent),4/84 budget-truncated evaluation_failed. These are not84 complete attempts.

Decision: no evidence that the lower-energy exit dominates these runs; do not add a tolerance or spend GPU time on this branch. Next useful contrast is saved LS-prequench-only true quench versus the existing full Gaussian+true-quench outcome. It isolates where the observed LS exploration comes from without extending a full search. No inference that Gaussian cap itself is defective.
