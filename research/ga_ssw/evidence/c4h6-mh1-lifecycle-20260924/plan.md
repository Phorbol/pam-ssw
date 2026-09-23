# C4H6 / MH-1 fixed protocol lifecycle and cost pilot

Purpose: test the existing SSW/paper-LS/native-LS execution chain on the newly
qualified molecular oracle and measure complete attempt cost before a 400-step
comparison is budgeted. This is one development seed, not a performance study,
independent validation, new system class or full A–J paper reproduction.

Start: the exact ASE G2 trans-butadiene input used in the successful MH-1
qualification (not the relaxed endpoint). All three arms include initial
quench costs, share seed59 and all SSW, quench and recovered-rotation settings.
Only LS mode changes. Run12 outer attempts per arm, at most12000 E/F requests
and840 seconds each. At most13 saved minima per arm get one independent fresh
E/F; no silent truncation. Totals:36000 search+39 fresh, one V10045min. Resource
and oracle failures stop the relevant arm; no retries or budget extensions.

Source rules: C4H6 paper width0.1 Å, NG25,150K, LS response target0.7 eV/atom.
Safe-total/history500, physical fmax0.03 and biased fmax0.1 eV/Å,400 relaxation
iterations, fd0.001 Å are existing project settings, not paper-equivalent
numerics. The same recovered CBD settings are used in all arms; its ftol must
not be interpreted as a direct HVP residual. LS soft preparation0.1 eV/Å,
50steps with the existing explicit normal-step-limit policy; numerical failures
are never accepted as successful preparation. Explicit H/C lookup tables and
all effective LS defaults are exported with the run.

Paper LS: target0.7 eV/atom, xi0.2, initial_fraction0.03, learning_rate1.8.
Native LS: target700 meV/atom, existing bounded update/cycle defaults unchanged.
The target matches, but adaptation rules and effective responses need not match.
No constraints, reconnection force, GA, pool selection, new descriptors or
per-arm optimizer changes. Three-arm order is fixed in plan.json.

Expected signals: all arms exercise complete attempts and independent landing
checks; costs and response trajectories allow a prospective longer budget.
If an arm fails soft preparation/true relaxation, diagnose the exact stage
from preserved telemetry before any larger run. If the initial model itself
fails, stop this lane rather than change the SSW parameters. Twelve steps cannot
establish LS adaptation convergence, reaction coverage advantage or rare-event
success probabilities. No method will be selected from pilot best energy.

Analysis must retain every attempted outer step and failed cost. Report fresh
numerical qualification, component formulas, element-graph classes, and CCCC
torsions for butadiene-like graphs; graph novelty alone misses A/B conformers.
Fragmented products are retained, not all labelled invalid: the LS paper
contains H2 products, but no electronic state/TS claim follows from connectivity.
Reference: Guan/Shang/Liu JCTC2024 DOI10.1021/acs.jctc.4c01081 §3.1/SI§7.1–7.2.
