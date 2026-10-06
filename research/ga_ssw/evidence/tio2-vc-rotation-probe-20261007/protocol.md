# TiO2 saved-state VC rotation diagnosis

Both joint arms in GPU1663547 performed40 attempted outer moves; every move
ended at rotation_failed, before height/bias/quench/landing. Paid search was
1602/1607, including initial2/7. All cost sources agree. Model failures0.
This observation rules out biased relaxation as the cause of these joint-arm
failures; it does not show an intrinsic inability to find anatase.

Question: is failure mainly the limited two-vector plane rotation, or does the
existing retained-subspace central Ritz also fail to obtain a low-residual mode
at the same EFS budget? Other possible influences are the joint metric and
projected spectrum. This is a local diagnostic, not a global-performance test.

Use the saved12/48 initial endpoints, same OMAT-small/head/dtype as the panel,
and first two exact random anchors from the panel seeds. Failed rotation used
no subsequent RNG, so these anchors can be regenerated without rerunning an
initial quench. Both callbacks use the same symmetric log-strain chart,
translation projection, rank-one anchor rotation term, fd .001A, bias1eV/A^2,
strict HVP residual .02eV/A^2. Joint strain_length5A stays unchanged.

Only existing solvers: plane dimer39HVP+onecenter <=40EFS; central Ritz<=40EFS
including its direct residual. Four paired anchors, <=320search requests.
Central residual independently checked along each returned direction with two
new EFS requests, <=16fresh for the8 iterative modes. Before viewing Ritz
outcomes, add a full central-difference projected42-coordinate Hessian at the
12atom center only (84reference requests), with the two analytic rank-one
anchor terms solved independently. Two resulting full-reference directions
receive4further direct central EFS checks. Total<=320iterative+84reference+20fresh
=424requests. This separates inadequate iterative convergence from a target
residual that is inconsistent with direct directional differences. Reference
Hessian asymmetry is recorded, not silently discarded or declared a stability
certificate. No48 full Hessian, extra threshold or parameter fit.
Count both request and actual calculator calls,
including failures, never denials. Snapshot exact source/config/anchors/modes.
One V100, 5min allocation, 240sec bounded work; scheduled after the original
panel, no extra concurrency. No quench, bias, global trajectory or continuation.

Ritz passing the unchanged gate where dimer fails supports a later single-factor
walker trial via its already existing direction_solver callback. Both failing
suggests the retained subspace at40requests is insufficient or the chosen
metric/accuracy gate is unsuitable; distinguish those before any numerical
criterion change. Central/direct disagreement implicates directional finite
difference consistency rather than global selection. No outcome promotes a
new default or loosens physical force/stress/target acceptance.

The2014 cell force example .1eV/A at separation .005A corresponds in the
implemented cell formula Frot=2h||Hn-kappa*n|| to10eV/A^2, not .02. These
values concern different coordinate metrics. This probe does not silently
substitute the paper cell criterion into joint coordinates. Native CBD and
this plane dimer are also distinct algorithms; no native numerical parity claim.

## Execution registration

Source196d502; existing core treec572a1cc remains unchanged. Root review made
all solver/reference/direct-check callbacks project gradients exactly as the
joint walker does; closure checks inspect dictionary.closed rather than the
truth value of a nonempty dictionary. Unit-direction component norms are
dimensionless. Zero-real-PES preflight, compile and shell syntax passed.
GPU1663620 is afterany1663547; CPU1663621 is afterany1663620. Outcomes pending.
