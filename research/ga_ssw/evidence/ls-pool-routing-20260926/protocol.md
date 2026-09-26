# Prospective LS candidate reuse screen (not submitted)

Question: with a fixed Native-LS escape kernel, does choosing from all qualified
landings improve subsequent discoveries relative to MC continuation? The rival
explanation is that higher-energy candidates merely consume restart/prequench
cost and do not produce useful new minima. This is a developer screen, not a
statistical global-optimization ranking or random-C60 acceptance test.

Prerequisites: approved ASE pool identity implementation and real-geometry
bookkeeping/resume qualification; offline rearchive of old trajectories to
identify material false-merge/false-split concerns before more GPU work.

Inputs: qualified C4H6 butadiene (prior MH-1/omol model qualification) and saved
C60 isomer-2 defect (prior MH-1/omol defect qualification). Keep each input's
published repository configuration and Native-LS settings frozen; only true
fmax becomes the user's common0.05 eV/Å for all new arms. Retain Safe-total,
history500 and bias_fmax0.1. These are not PBE reproductions. No new model or
periodic boundary change, no LS tuning, no descriptor or reward change.

Arms for each input: Native-LS+MC, Native-LS+uniform pool, Native-LS+PAM pool.
The latter two use the approved `ase_permute_v1`, rmsd_tol0.1Å and existing
energy_tol0.001eV diagnostic. Same initial atoms and main seed181; selector RNG
seed193 is separate. All preserve existing local-state restart semantics.
Different selected starts inevitably yield different later physical proposals;
this is an outer-policy comparison, not a common-trajectory causal decomposition.

Stage1 has6 arms, each100 outer steps maximum,80,000 search E/F maximum,
101 independent fresh checks maximum,30min process maximum; at most2 V100 GPUs.
Ceiling480,000 search +606 fresh E/F,3 GPU hours. Actual allocation/worker plan
must enforce these bounds before submission; a cap is not a consumption target.
One new seed per case is an execution/mechanism screen only. A second seed is
not automatic: decide from complete stage1 evidence, without tuning thresholds
or scores to its outcomes. No favorable-performance/generalization claim from
one seed; retain all failures and missing/censored arms.

Rationale for100 rather than3–10 steps: old C4H6 LS new topology first discoveries
include steps50,59,74; cost about650 E/F per outer step. C60 defect native-LS
old short runs likewise cost roughly640 E/F/outer step. A few steps cannot test
whether a pool first accumulates alternatives and then exploits them. The fixed
request ceiling, not completed-step count alone, controls matched-cost reading.

Report at common paid E/F prefixes: force-qualified and chemically interpretable
new candidates/topology classes and their first-discovery cost; best true energy;
for C60, Ih cage recovery and reference-energy criterion separately. C4H6 uses
existing frozen element-labelled graph isomorphism, including all rejected
qualified landings. Graph classes are not minima or kinetics. Report geometric
identity and topology as separate approximate views. Fresh-check every returned
landing and initial structure; no graph/energy gate is silently added to search.

Mechanism diagnosis: number of committed jumps to other entries, energy/topology
of those starts, LS reset/prequench cost, and subsequent discoveries. Selection
counts/acceptance/archive size alone are not success. All failure costs remain
in denominators. High-energy discovery is coverage evidence, not better global
minimum search. If uniform and PAM show no useful discovery-cost advantage,
do not tune weights or automatically expand the budget. If matching/qualification
fails, stop scientific comparison and resolve that specific prerequisite.

Status before submission: all prerequisites passed. CPU1500664 exercised the same
worker path with the established Cu13/EMT Native-LS fixture: MC/uniform/PAM each
completed two steps and three independent force certificates; search costs67/67/64,
fresh3 each. Worker SHA256123612e948f634d4c4019f0244a28061133a53074261e87a8774ae9b3355c76b.
The preserved CPU fixture failures were in input setup and JSON serialization, not
changes to the scientific protocol. Existing checkpoint serialization is reused;
JSON contains scientific results and a checkpoint-file reference.
CPU1500671 then passed three offline analyzer regressions and read all three actual
EMT outputs with closed costs and shared prefix64; no additional PES evaluation.
Analyzer SHA256d3ae179ed6c18eaa3c1828baf37deba54955fa03e9663a2d40b4e199421f2b6a.
The GPU run will use the frozen scientific plan, not the EMT fixture.
