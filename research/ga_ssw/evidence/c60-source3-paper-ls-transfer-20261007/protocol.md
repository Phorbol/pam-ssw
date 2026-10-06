# Published C60 #3: fixed local SSW / paper-form LS transfer

Purpose: test whether adding the published LS potential/controller to the
current independent fixed-cell SSW implementation improves discovery of Ih
from a new force-qualified defect cage at matched total E/F cost. Competing
explanations are useful local bond rearrangement versus paying for soft
deformation/prequench without a qualified target gain. This is a development
component transfer on MH-1, not reproduction of the paper's G-NN energies,
long random-start success rates, or LASP optimizer numerical trajectory.

One immutable starting geometry, two search RNG seeds26100791/26100792,
two arms per seed (`ssw`, `paper_ls`); seeds are repeated trajectories from
one source, not two independent structural inputs. Use exactly
`../c60-ls-source3-qualification-20261007/qualification-1664282/final-candidate.extxyz`.
Qualification used13 actual E/F; preserve its source member/CSV and qualified
reference. No new initial geometry, no favorable redraw, no old #1/#2 input.
The ordinary quench retained the published non-Ih cage, cold fmax.01321 and
energy1.84216eV above Ih; hence ordinary relaxation alone has not solved it.

Backend is the cached MH-1 model with SHA
`a522eb7f59c7879963d41586528f4980baf33e086c94aa92e3eafdeccad3be47`,
omol head, float64 CUDA, cueq/oeq off, deterministic seed0/one thread/TF32 off.
All atoms mobile;50Angstrom diagonal storage cell,pbc=False. No confinement,
descriptor direction, pool, GA, adaptive temperature, or optimizer change.

All non-LS settings match across arms and retain the current full_peratom
direction bundle: ratio_local50,local_probability.5,group_threshold.5,
pre_rotmax5,rotmax15,pre_ftol.2,ftol.02,Euclidean,max_force_calls40,
c1_radius_policy=per_atom,startup_order=legacy,geometry=nonperiodic.
Use the current C60 `protocol_config()` from the direction-transfer series:
width.6Angstrom,NG12,150K,real fmax.03,bias fmax.1,relax_steps1000,
FD.001,rotation_hvp39,rotation_tol.02,forward_force.1,global sampler,
Broyden Euclidean,direction_only,Safe-total/history500,force_or_budget;
native MC(.1eV,99999). The recovered direction overrides anchor generation
through the existing API. These choices isolate LS on our current baseline;
they are not labeled the original paper's exact direction/MC implementation.

Only the LS arm passes existing `LSSettings`:
`bond_energies={(6,6):3.61}`eV, `bond_lengths={(6,6):1.64}`Angstrom,
target_per_atom.02eV/atom,initial_fraction.03,xi.2,learning_rate1.8,
energy_filter=None,prequench=`LSPrequenchSettings(fmax=.05,steps=1000,
exit_policy='force')`. The soft prequench force tolerance is explicit: it
uses the paper SI's.05eV/Angstrom numerical scale and the user's requested
inner range; it is not inherited accidentally from the true quench .03.
No iteration-limit release, new strength clipping, or rescue is added.

LS2024 DOI[10.1021/acs.jctc.4c01081](https://doi.org/10.1021/acs.jctc.4c01081),
eq11–15 supplies3.61eV,3%,xi.2,lambda1.8; SIsection3/C60 inputsection7.3
supplies target.02eV/atom and numerical settings. Fulltexts215.txt and
ct4c01081_si_001.txt reside under the existing literature archive.
The1.64Angstrom neighbor cutoff is an existing project convention (recovered
native C-C length≈1.54 plus.1 tolerance), not specified by the paper.
No claim that3.61 is fitted to MH-1 or that this cutoff is universal.
The potential is frozen within each outer step:
`sum_(unordered pairs) Aij exp[-(rij-r0ij)/(xi*r0ij)]`.
After the soft prequench its true energy response is measured; after the
bare landing/MC decision, total strength updates by
`S_next=S_current-N*lambda*((E_after-E_before)/N-target)` and is redistributed
over next-step neighbors. Record failed soft quenches and negative-strength
domain exits without changing the rule. Recovered nativeLS is a different
controller and is not a third arm in this experiment.

Bounds per arm:100 outer attempts,20000 paid search requests,1200seconds,
whichever stops first;<=3 final independent E/F checks. Total<=80000 search
+12 final checks, four one-V100 allocations20min each, array concurrency2;
qualification13 requests remain separate overhead. Reserve/charge the
existing runner's cost ledger, with no automatic continuation, retry,
budget redistribution, or additional inputs. The outer-attempt cap bounds
low-cost loops; it does not replace the paid-request comparison. This scale
can resolve local repair or avoidable cost on a new cage, but cannot resolve
the paper's modest8.8% long-search mean gain or global success probabilities.

Readout uses all force-qualified landings, including MC-rejected discoveries.
At common request prefixes5000/10000/20000 report first Ih+reference-window
(.01eV total)+force(.03) target, lowest qualified connected energy, and actual
horizon reached. Require graph isomorphism to the frozen Ih reference at all
three cutoffs1.64/1.70/1.80 for the local Ih target, not merely12 pentagons.
Any target needs a fresh force/energy check and a3D cage review before
scientific qualification. Also report connected new graph classes, failed
landings/fragmentation, LS prequench cost, true responses/strength updates,
Gaussian counts, total requests, actual calculate calls, and wall time.
Do not rank from acceptance rate or archive size alone. Missing, truncated,
failed and not-started arms remain in the four-arm denominator.

Decision: a cold/physically qualified LS-only target or lower common-cost
connected endpoint is a scoped positive signal, with contrary seed outcomes
retained. No gain or LS failure retains LS as experimental and directs any
follow-up to the recorded failure layer; it does not justify parameter
tuning, default promotion, or refuting the long paper experiment. Do not
automatically extend this source or alter the ongoing new random C60 panel.
Research runner may parse `paper_ls` and an `outer_steps` cap using existing
core interfaces; no core/public checkpoint/API changes are needed.
