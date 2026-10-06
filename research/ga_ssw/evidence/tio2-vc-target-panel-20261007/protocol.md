# Paper-sourced TiO2 anatase target: existing joint versus block VC walkers

Question: can the existing VC implementations produce a lower-energy,
force/stress-qualified **anatase geometry** from the paper's porous phase-87
coordinates, and where is cost lost if they fail? Prior VC panels qualified
force/stress but did not identify the resulting phase. This fixed development
panel adds that physical target; its four trajectories do not estimate the
published global success rate, barriers, equilibrium sampling or kinetics.

## Source, qualification and comparison boundary

Source is Huang et al., 2017 SSW-NN, DOI10.1039/c7sc01459g, SI section7,
phase-87 and 12-atom phase-87/anatase pathway coordinates. The paper uses a
trained TiO2 NN followed by DFT and VC-DESW, not MACE. The 2014 VC paper's
block design (10.1039/c4cp01485e, section2.3) informs one implementation.
This is a same-model paper-inspired structural task, not numerical paper parity.
No source NN/DFT energy or barrier is substituted for a current model result.

Qualification GPU1663408/source3f19072 used25 paid E/F/stress+6 independent
cold checks. Phase-87 IS12 and anatase FS12 remain distinct after all-DOF
quench; anatase is lower by .504902341eV per12atoms on OMAT. Their endpoints
meet fmax .05 and stress .001. Source/endpoint identity CPU1663379/1663414
passes both inherited matching tolerances: relabel, wrap, equivalent cell basis,
supercell positive controls and distinct-phase negatives. Source48 phase-87
matches IS12 under both. Its own initial quench is still needed and paid in
the new panel; no force certificate is inferred merely by repeating a cell.

Anatase source and relaxed target have space group141 at symprec .01/.1A.
The supplied phase-87 representative reads C2/m rather than the paper label87,
and supplied `rutile` reads31 rather than136. These source-label/symmetry
differences are retained, not cured by weakening tolerances. The panel uses
the traceable phase-87 *geometry* and verified anatase target; it does not
claim exact source phase symmetry. Reported-rutile input is excluded from this
search until independent canonical-phase comparison resolves its identity.
Force/stress-qualified stationary endpoints are not Hessian stability proofs.

## Treatments and fixed parameters

Two input sizes12/48, one predeclared paired RNG per size (26100751/52), two
existing whole-walker treatments: joint log-strain and paper-inspired cell/atom
block. This compares entire implemented pipelines, including their differing
direction, bias norm and schedule. It does **not** isolate coordinate choice,
line search, rotation policy or any single mechanism. The48 and12 runs are
related phase representatives, not independent structures or a universal
size-scaling validation.

`plan.json` gives all constructor fields. Common: same OMAT-small/omat_pbe,
float64 CUDA/no cueq/oeq, pressure0, Gaussian width .6A, max10Gaussians,
rotation bias1eV/A^2, fd .001A,39rotationHVPs, residual .02eV/A^2,
Safe-total/history500,300quench iterations, .2A step cap, force .05eV/A and
stress .001eV/A^3. Inner atomic fmax .1, joint biased generalized norm .1
are operational stopping controls; they are different norms, not identical
accuracy. Strict release stays enabled, without new fallback or early-stop law.

Block uses5cell cycles, atomic every second outer step, .15||L||F displacement,
.005A cell dimer image,6cell evaluations and .1eV/A rotational-force threshold,
25fixed-cell atom iterations capped at fmax .1. These cycle/schedule/step values
come from2014 section2.3 (5actual cycles is our explicit interpretation, not an
assertion about an ambiguous original index endpoint). Cell direction is the
existing plane-dimer approximation, not a recovered native full Broyden routine.
Atomic climbing currently uses its existing dimer entry, not the newest recovered
full fixed-cell direction. Joint uses the existing3N+6 log-strain chart with5A
strain length; this is an inherited metric, not size-invariant or paper-identical.

4000K is the reported2014 SiO2 search temperature, deliberately used as the
same exploratory MC setting in both treatments. It is not a reported TiO2 optimum
or a physical trajectory temperature. The2014/2017 pathway sampling returns to
the IS when another phase is found; this panel instead tests existing global
search MC and counts all valid observations, including rejected candidates.
Other numerical values are inherited development choices, not tuned here.

## Budget and success criteria

Each of4arms: at most40outer attempts,20000paid E/F/stress including initial
quench and failures,600seconds. Array concurrency≤2, oneV100 per arm,12minutes
per allocation; totalsearch≤80000 and cold checks≤13 (3per arm+one sharedtarget).
No continuation/restarts/added seeds/tuning after outcomes. Pure model/geometry
qualification cost31 is a single reused prerequisite, not counted once perarm.

Report geometric anatase arrival separately from joint acceptance. A joint
candidate must match the qualified anatase reference under BOTH inherited tight
and broad matchers, pass force/stress thresholds, and have energy/atom no higher
than that reference+.001eV/atom. This energy window concerns the same model
stationary structure, not MLIP accuracy against DFT. A target match alone is not
a lower-energy or numerically qualified result. Independent cold certificates
check initial,best,and firstjoint candidate (otherwise lastavailable minimum),
deduplicating identical geometries, with≤3paid checks. The firstcandidate counts
as confirmed only after its cold checks pass; the reported first-discovery cost
is paid cost up through that completed outer step. All later cost is still paid
and reported. No callback-driven early stopping or public API change is added.

## Expected signals and follow-through

Qualified different phases plus matched anatase arrival answer the capability
gap and justify later independent efficiency tests. If strict rotations/biased
quenches prevent valid landings, localize their paid cost and stopping reasons;
do not label it a phase-search failure until its numerical layer is distinguished.
If landings are qualified but do not leave the source geometry, inspect direction
and cell/atom displacement from saved stages. If they leave into other phases,
retain their geometries and energies, without calling them anatase or a GM.
Any case/arm outcome remains in the denominator. Do not raise the budget or tune
metric/temperature/bias from these same four outcomes.

Numerical source review, compile/wrapper checks and zero-real-PES ledger/config
preflight precede submission. Preserve exact source imports/snapshots, effective
configuration, raw scalar EFS ledgers, outer decisions/cells/minima, budget
denials and cold checks. Readout runs onCPU, separates execution, numerical and
phase identity, compares common paid horizons, and keeps related-size caveats.
No core algorithm or default changes in this panel.
