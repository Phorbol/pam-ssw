# Existing Gaussian-policy single-escape probe

Development diagnostic; not independent global-search validation. Four frozen inputs from the
existing paper-cluster generator: N=38,55, seeds25092501/25092502. Two arms: existing
forward-force rule versus existing PAMCurvatureGaussian() height_width defaults. No changes
of parameters after seeing results. Match input, search child RNG, paper direction/recovered CBD,
Safe-total/history500, inner0.1/outer0.01 eV/Å, max14 Gaussians, kBT0.8 eV and initial quench.
Retain inherited outer0.01 solely to reproduce the existing baseline; this is not a proposed new tolerance.

Question: does the existing adaptive Gaussian bundle prevent extreme finite-step forcing while
still producing valid intact landings, and at what cost? Alternatives: it helps; it suppresses
useful escape and returns to the same basin; its local quadratic approximation or clipping fails.
The bundle changes height and width, so this is not a pure width ablation or native LASP test.

Each arm runs only one complete SSW outer attempt, including initial quench; max2500 search E/F,
max2 fresh E/F for initial/landing. Total20000+16; one CPU job,5min hard wall,240s internal deadline,
no GPU, no automatic continuation or parameter scan. Budget-truncated outcomes remain censored.
Save exact generated input, effective configuration/initial/RNG lineage, all climb records,
landing geometry, actual request cost, reason for failure/stop, fresh energy/force and connectivity
at1.3/1.5 LJ sigma. Initial qualification and first rotated mode must match between arms before
interpreting downstream differences. Preserve every failure; do not count an accepted step alone
as a new basin or compare only successful rows.

Readout: reproduce baseline LJ38 first outer scalar behavior from old stage probe; check final
force≤0.01 and connectivity separately; compare landing energy to its own initial and distinguish
same-basin returns with structural evidence. No GM-hit or generic speed claim from one escape.
If adaptive bundle fails/mixes, retain negative result without tuning. If valid improvement appears,
review existing molecule/material tests before one prospectively matched real-system check.
No new public API/guard/constraint or core changes are authorized by this diagnostic.
