# Existing finite-budget atomic rotation exit: TiO2 saved-stage check

Observed gap: both original block panels reach phases via cell-only moves, but
all scheduled atomic moves (10/20 for12/48) fail before their first Gaussian.
The existing SSWConfig.force_or_budget policy was explicitly unsupported by
atomic_climb. Commit6129c5f connects that already approved input contract; it
adds no public parameter, persistence format, convergence threshold or default.
Only finite evaluated budget_exhausted axes may proceed, with converged=False
and rotation_budget_released=True recorded. Biased and physical quench gates
remain unchanged. Default force/legacy checkpoints preserve prior behavior.

Question: does this existing proposal policy produce usable, physically
qualified atomic-stage candidates, or does the bottleneck move to biased/true
quenching? A local two-size paired check can decide whether wiring the option
is useful before another long full trajectory. It does not measure global search
success, an exact native implementation or a general LS/VC efficiency advantage.

Inputs are each original block run's first scheduled atomic failure, outer1,
after its five cell/partial cycles. They are deliberately not qualified minima:
that is the position at which2014 hands from CBD-cell to atomic SSW. No cell
cycles are rerun. The original outer reference energy is derived from accepted
outer0 or initial if outer0 was rejected. Preserve source paths and raw inputs.
Use predeclared new seed26100771/72, paired across force/force_or_budget within
size. This is a development check on saved diagnostic states, not independent
validation of a new global default.

Only factor is rotation_exit_policy. Every numerical/model setting otherwise
comes unchanged from the prior panel: same OMAT-small/omat_pbe,float64 CUDA,
no cueq/oeq; .001A images,39HVP/.02eV/A^2 residual, rotation_bias1,
width.6A, max10Gaussians, Safe-total/history500,300quench iterations,
bias force.1, true force.05 and stress.001. Never change converged=False into
True or treat a budget marker as numerical convergence. The2014 six-EFS CBD-cell
example accepts approximate/hybrid modes; connecting an existing finite-budget
proposal option is inspired by that principle, not exact native CBD parity.

Each of4arms runs atomic_climb once; only its existing successful terminal
statuses (lower_true_energy/gaussian_limit) enter the ordinary joint true quench.
Failed biased quenches remain failures; no extra no-bias fallback. A valid
candidate requires cell_quench optimizer+physical certificate. Cold input and
candidate E/F/stress are separately paid; a large input force is diagnostic,
not exclusion or permission to alter inputs. Target identities and energy window
stay those frozen for anatase in the original panel; no novel-phase GM claim.

Perarm<=3000paid E/F/stress and120sec, total<=12000search+8cold, oneV100,
10min allocation with540sec work deadline. No retries, extension or tuning.
These bounds limit the stage rather than repeating the original12atom cell path,
which visited2.12A^3 volume, had a .0457 principal stretch and reached its wall
cap. Failed work/denials and actual calculator calls remain separately visible.
Save input/config/model/source provenance, Gaussian history, ledgers, endpoint,
all stopping/certificate flags. Current CPU regression verifies unchanged real
Cu/EMT climb and checkpoint behavior; pure stubs verify opt-in semantics only.

Scope clarified by a read-only source audit while the frozen experiment runs:
atomic_climb's rotation callback contains the physical PES and coordinate
projection; its solver adds its own rank-one rotation bias. Stored Gaussian
terms enter the biased quench, not later rotation callbacks. run_ssw similarly
uses physical + frozen LS (if enabled) for rotation, excluding stored Gaussians.
Native add_rotation_bias's rank-one force is recovered, but the native callback's
cumulative-Gaussian lifetime remains unresolved (see docs/research/
native-rotation-bias-field-trace.md). The2014 cumulative modified-PES quench
formula alone cannot resolve that callback. Both current arms share the same
callback, and first rotation has no Gaussian history; later stages do not
establish native Hessian parity. No component/config change follows this audit.

## Discriminating control after the frozen probe

GPU1663813 and CPU1663814 completed: strict exits before any Gaussian in both
sizes, while release completes10 terms and produces two cold-qualified minima;
12atom matches anatase,48atom does not. Cost1557search+6cold; no error/denial.
This shows the opt-in path can finish, but does not establish that Gaussian
escape rather than final cell relaxation creates the target basin.

New control: each same saved input gets exactly one direct cell_quench with
the identical true_quench settings; no atom displacement, Gaussian, LS or
extra fallback. Competing explanations are a new basin produced by climbing
versus a basin already selected by preceding cell cycles. Match both direct
and released certified endpoints under the same tight/broad matcher and report
the full paid costs. If direct quench reaches the same basin, do not attribute
the phase discovery or efficiency to ten Gaussian terms; retain only the
completed-path capability conclusion. If they differ, retain scoped basin
change evidence, still not an independent global-search success probability.

Bound:2inputs x1000search/100sec, at most1fresh certified candidate perinput,
oneV100/5min,240sec global deadline. Already-paid source input measurements are
reused explicitly, not charged again or relabeled as new independent evidence.
No replacement, retry, tolerance adjustment or automatic long trajectory.
