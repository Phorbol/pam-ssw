# Paper-ordered LS mechanism gate conclusion

## Decision

Reconstructing the documented LS-SSW application order restores a real and
fully paired curvature-softening signal on C60. It rejects the current
active-neighbor, moving-reference implementation as a proxy for the published
mechanism.

The fixed initial-strength protocol does not yet beat no LS on landing energy.
It therefore survives only as a mechanism candidate, not as a production
default.

## Verified cohort

- exact execution commit:
  `4c459d208c09e48dc04ca0204be3742e7b063dea`;
- C60 fixed starters: bootstrap, middle, late;
- seeds: 42--44;
- arms: no LS, current active-neighbor LS, paper-ordered LS;
- 27/27 cases completed;
- 27/27 landing true quenches have force certificates;
- every force evaluation has a declared purpose;
- unattributed force evaluations: zero.

## Arm summaries

| Arm | Median landing delta | Median force evaluations | Median wall time | Median first inner curvature | Median first true curvature | Median first PR |
|---|---:|---:|---:|---:|---:|---:|
| no LS | -1.610168 eV | 441 | 8.535 s | 35.9569 | 35.9569 | 0.033333 |
| current active | -0.989044 eV | 537 | 9.264 s | 35.9576 | 35.9576 | 0.033333 |
| paper ordered | -1.507263 eV | 503 | 9.311 s | 34.6711 | 33.4196 | 0.033333 |

Arm medians must not be subtracted as if they were paired effects. The paired
results are:

| Comparison | Landing effect, median | Cases improved | FE effect, median | Median FE ratio |
|---|---:|---:|---:|---:|
| paper - no LS | -0.000153 eV | 6/9 | -14 | 0.973 |
| paper - current active | -0.387573 eV | 6/9 | -39 | 0.929 |

Lower is better for landing and force-evaluation effects.

The paper-minus-no-LS landing effects span from -2.844666 to +2.620056 eV.
The near-zero median is therefore not a uniformly small effect; it is a
sign-changing result across fixed tasks.

## Direction mechanism

The current active-neighbor arm and no-LS arm select the exact same first
direction hash in all 9 paired tasks.

The paper-ordered arm selects a different first direction hash in all 9 paired
tasks and lowers curvature consistently:

| Paired effect | Mean | Median | Lower-curvature cases |
|---|---:|---:|---:|
| first inner curvature, paper - no LS | -1.018892 | -1.020784 | 9/9 |
| first inner curvature, paper - current active | -1.071318 | -1.018813 | 9/9 |
| first true curvature, paper - no LS | -1.844804 | -2.035079 | 9/9 |
| first true curvature, paper - current active | -1.844581 | -2.033109 | 9/9 |

First-direction participation ratio does not materially change: its paired
median effect is zero. The current evidence therefore supports **softening and
direction perturbation**, but not **mode delocalization**.

Direction hashes establish nonidentity, not the angular magnitude of the
change. No stronger claim is made without storing or replaying the raw vectors.

## Pre-relaxation mechanism and cost

The paper-ordered arm builds one frozen C-C graph and performs one softened-PES
pre-relaxation per case.

- graph size: 85--86 C-C pairs;
- pre-relaxations: 9/9 converged;
- iterations: 10 in every case;
- force evaluations: 13 in every case;
- final softened-objective maximum force: 0.0407--0.0487 eV/A;
- `P_LS`: 0.002689--0.002759 eV/atom.

The initial strength is the documented `3%` of the standard C-C bond energy:
`A_CC = 0.1083 eV`. The measured real-PES response is only about 13.5% of the
paper's C60 target `Y = 0.02 eV/atom`.

This explains why the present gate must not be interpreted as a test of the
paper's converged self-adaptive LS strength. It tests the application topology,
reference lifecycle, pre-relaxation order, and initial-strength response.

## State-resolved behavior

- bootstrap: paper-ordered median landing delta `-4.368958 eV`, compared with
  `-3.696198 eV` for no LS;
- middle: paper ordered `-1.507263 eV`, compared with `-1.610168 eV`;
- late: paper ordered `-0.988831 eV`, compared with `-0.047546 eV`.

The late-state improvement is promising, but the middle-state result and the
large paired sign changes prevent a general landing-efficiency claim.

## Scientific conclusion

The experiment separates three statements:

1. **Current active-neighbor LS:** rejected as a representation of the
   published direction-softening mechanism. It changes cost and trajectories
   without changing the first selected direction.
2. **Paper-ordered initial-strength LS:** mechanism-positive. It consistently
   lowers selected-direction curvature and changes the selected vector at a
   small, explicit pre-relaxation cost.
3. **Production search advantage:** unproven. Landing energy is neutral against
   no LS at the paired median and participation ratio is unchanged.

The next justified LS experiment is not a strength sweep. It is one isolated
implementation of the paper's scalar feedback from measured `P_LS` toward the
fixed C60 target `Y = 0.02 eV/atom`, with the bond graph, optimizer, direction
portfolio, and all downstream policies frozen.

PdO remains out of scope. The paper does not provide a system-independent rule
that maps its tight-bond filter to a Pd-O surface, and the current automatic
PdO graph is dominated by Pd-Pd pairs.

## Claim ceiling

This is C60 fixed-starter mechanism evidence. It is not an equal-budget
long-search result, a system-general LS validation, or a reason to change the
user-facing production profile.
