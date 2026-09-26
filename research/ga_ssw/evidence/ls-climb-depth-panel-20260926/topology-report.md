# C4H6: distinguish index-level events from topology coverage

Decision: the LS coverage signal is not explained solely by repeating the same
bond rearrangement. Retain LS as a candidate for coverage on this MH-1/omol case;
do not promote a global optimization advantage or general default. This is an
offline reanalysis of the four existing trajectories, with zero new PES calls.

|Arm|Seed|Total archived E/F|Force-qualified connected landing events|New topology classes vs initial|New classes in MC-accepted subset|
|---|---:|---:|---:|---:|---:|
|SSW|61|263213|393|4|0|
|SSW|67|258156|390|4|0|
|NativeLS|61|260522|386|10|0|
|NativeLS|67|259049|386|8|0|

Classes use exact graph isomorphism with element labels under the previously
frozen H/H, H/C and C/C distance cutoffs. First-discovery costs are in
`topology-audit.json`. Failures and unqualified landings remain in total costs.
The historic force threshold remains0.03; it was not relabeled0.05 retrospectively.

Across both seeds, SSW found6 classes and LS10, with5 shared: LS added5 classes
but missed1 seen by SSW. Thus even this small panel does not support dominance.
The26 first-discovery representatives all passed species-preserving permutation
checks. None triggered the inspection flags H degree!=1 or C degree>4; shortest
pair distance was1.0739Å. These checks exclude those specific simple anomalies,
not all chemical defects, bond-cutoff artifacts, model errors or saddle points.
No bond orders, vibrations, independent fresh forces or DFT validation were added.

All26 first discoveries lie0.4034–3.1994eV above the archived initial energy.
The actual old effective configuration uses150K. Suppression of uphill transitions
by this low-temperature Metropolis policy is consistent with no accepted new
connectivity class; this is not evidence of a coding error or physical kinetics.
Coverage of discovered candidates and routing of the subsequent walker are
separate outcomes. This result motivates evaluating the already implemented pool
restart policy for coverage, not silently replacing an energy-minimization objective.

The previously reported accepted changed-edge event counts8/7 remain correct for
that different metric: they compare edges by fixed atom index. Independent review
checked all15 events: all satisfy the force/surface/convergence criteria, but their
connectivity is isomorphic to the initial graph. They must not be described as
8/7 accepted new chemical topologies. Same graph also does not mean same geometric
minimum or purely a coordinate permutation.

Evidence: CPU1500389 ran audit_topology.py, CPU1500417 ran inspect_topology.py;
both exited0. No PES calls. Source SHA and first-record indices are retained in
JSON. Source initial+all per-step E/F costs reconcile exactly with all4 run totals.
Independent source review checked counting/filtering and old-event semantics.
Commands (write new output paths; existing files cannot be overwritten):

```sh
python audit_topology.py --output NEW_AUDIT.json
python inspect_topology.py --audit NEW_AUDIT.json --output NEW_INSPECTION.json
```

This reanalysis strengthens a C4H6 model-PES coverage observation. It neither proves
LS improves C60 cage discovery nor identifies true reaction pathways/barriers.
