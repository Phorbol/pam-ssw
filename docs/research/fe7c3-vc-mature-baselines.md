# Fe7C3-80 complete VC optimizer comparison

Later correction: the zero-landing result is conditional on our strict
biased-convergence gate. All ten saved paid-attempt endpoints subsequently
passed no-bias quenching and independent physical certificates (835 additional
EFS). See `vc-schedule-and-release-correction.md`. Do not interpret the original
result as inability of these structures to yield physical stationary points,
or as failure of the published block VC schedule.

2026-09-11. The three tested numerical implementations did not produce a new
qualified landing in this bounded joint-VC experiment. Replacing Safe-total
with either mature library is therefore not an evidenced solution to this
case's biased-quench bottleneck. This does not establish universal optimizer
equivalence or prove that other settings cannot work.

## Frozen experiment

Directory: `research/ga_ssw/fe7c3-vc-mature-baselines/`. Input is byte-identical
to the qualified Fe7C3-80 input in the previous non-LS PQC/joint comparison.
The independent six-strain chart, MACE-OMAT-0-small model/hash, width 0.6,
rotation bias 100, NG 14, memory 10, maxiter 300, maxstep 0.2 for Safe, pressure
0, temperature 300 K, seeds 7/101, biased tolerance 0.001, final force 0.001
eV/Angstrom and full pressure-residual stress 0.0001 eV/Angstrom^3 are frozen.
Each arm requests two complete SSW attempts and reserves three of its maximum
2000 EFS requests for independent checks. No LS or parameter search was used.

SciPy L-BFGS-B and ASE LBFGSLineSearch are independent library runs on the
same chart and objective. Their complete native numerical rules differ:
SciPy relative-energy stopping is preserved, ASE retains its triplet step
restriction and initial Hessian, and Safe retains its joint block restriction.
The sufficient biased-norm conversion is documented separately in
`vc-mature-baseline-norm-contract.md`. This is not a line-search-only ablation.

The frozen package and runner were independently reviewed; a CPU Cu4 initial
quench/serialization preflight cost 33 EFS. No active-worktree fallback was
on the GPU PYTHONPATH. Runtime module paths and package versions are recorded.
Job 1257474 completed with exit 0 in 3m58s on one Tesla V100-SXM2-32GB
(`4v100n02`). Scheduler completion is separate from scientific success.

## Measured outcomes

| Solver | Seed | Charged EFS, including fresh | Paid attempts / requested | New qualified landings |
| --- | ---: | ---: | ---: | ---: |
| Safe-total | 7 | 1998 | 2 / 2 | 0 |
| Safe-total | 101 | 1998 | 1 / 2 | 0 |
| SciPy L-BFGS-B | 7 | 277 | 2 / 2 | 0 |
| SciPy L-BFGS-B | 101 | 129 | 2 / 2 | 0 |
| ASE LBFGSLineSearch | 7 | 1998 | 2 / 2 | 0 |
| ASE LBFGSLineSearch | 101 | 1998 | 1 / 2 | 0 |

Total 8398 EFS = 8392 search + 6 independent initial checks. All 12 requested
attempts remain in the denominator; 10 incurred search work, and two generated
only an uncharged budget-denial record. The fresh checks are all initial states,
not new candidates. Their E/F/stress differences are at roundoff scale and
all physical certificates pass. There is no new structure to qualify by Hessian
or identify as a different basin. Across the registered Fe7C3 experiments,
the cost now totals 47614 EFS (39216 previously + 8398 here).

Both Safe and ASE seed7 hit one 300-step biased-quench limit, then ran out
of request budget in the second attempt. Each seed101 exhausted its request
budget in the first attempt. These are censored search results; they cannot
support a relative time-to-discovery ranking.

SciPy's four failed biased quenches stopped on
`RELATIVE REDUCTION OF F <= FACTR*EPSMCH`. Their final common norms are
0.00536272/0.00601560 (seed7) and 0.00199124/0.00425187 (seed101), all above
0.001. Its 406 total requests therefore reflect earlier termination, not
successful low-cost discovery. The adapter correctly reports `native_stop`
rather than a common-certificate success.

The current Safe rerun matches the old negative outcome and total budget,
but seed7's per-attempt requests differ (1227/768 now versus 1207/788 before).
No exact trajectory identity across historical snapshots is asserted. The
three current solvers were run contemporaneously under one frozen snapshot.

## Consequence for the mainline

Do not promote a new optimizer, increase history or caps, or add new LS/GA
components based on this result. Keep the mature baselines and their explicit
failure reasons. Focus the next source recovery on the common SSW continuation,
state release and fresh E/F handoff; an Allopt setter is not a quench.

SciPy's relative-energy stop also depends on the additive energy reference:
its denominator contains the absolute objective, while gradients and the
physical optimization problem are unchanged by an energy shift. This follows
from the [documented stopping formula](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html).
If a future experiment removes that criterion, it must be a separately declared
gradient-controlled profile, not a relabeling of this failed native-default
baseline or a retroactive edit to these outputs. No such rerun was launched here.

Evidence: frozen `manifest.json` and `plan.json`, per-arm `result.json` and
`evaluations.jsonl`, `allocation-gpu.csv`, and the zero-PES `audit-summary.json`.
All six request ledgers reconcile. As elsewhere, this model evidence is neither
DFT validation nor magnetic phase qualification, and two seeds are not broad
statistical evidence for a general PES algorithm.
