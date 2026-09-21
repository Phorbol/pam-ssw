# Core SSW-family gap audit

Date: 2026-09-11. This is a read-only inventory of the current
`pamssw/standalone` code and the latest source-backed research records. The
ordinary SSW, CBD-cell/block, VC/VC-LS, LS, RC/RC-VC, and GA controllers are
implemented Python/ASE workflows in this checkout. They are often deliberately
independent substitutions, and their real-system evidence remains bounded; this
note lists the remaining *core* gaps rather than relisting completed families.

Status labels are intentionally separate: **implemented** means a callable
component exists; **independent substitution** means it is not native
LASP/Java execution parity; **absent** means the stated behavior has no current
implementation; **unvalidated** means available tests/evidence do not support a
scientific effectiveness or transfer claim. An item may have several labels.

## Updated execution priority after TYPE4 qualification

Later dispatch closure: `native-vc-consumer-followup.md` now resolves the
ordinary CSSW descriptor and `refresh -> stress2dedlatt` path from the actual
initialized type table. Item5's unknown consumer-slot claim is superseded for
that path; backend mixing/postprocessing and complete native coordinate
schedule remain distinct boundaries. Joint LS preparation is also implemented
as an explicit independent option, with the complete negative Fe7C3 comparison
in `fe7c3-80-joint-prequench-comparison.md`.

The numbered inventory below is retained as a dated audit, not an unchanged
execution queue. Subsequent evidence closes several of its narrow open items:

- `native-globalcompress-geometry.md`: C7/C8/C9 complete piecewise directions,
  exact3/6Angstrom boundaries, masks and caller normalization recovered;
  57 boundary cases exactly match isolated original instructions. Local pair
  geometry and C60 matched controls also completed. Empirical direction mixtures
  have no demonstrated cross-system advantage and are not new defaults.
- `native-moveds-scale.md`: failure restores saved trajectory coordinates,
  sets the Allopt/control flag and returns; accepted retries scale ds by0.95
  and save measured displacement. The flag is not a quench certificate.
- `native-lj-stress-producer.md`: ordinary CSSW stress transfer, native LJ pair
  virial, absolute-volume normalization and sign are closed for the inspected
  path. NN/external backend units and full native VC chart remain separate
  boundaries, not a reason to question the independently differentiated ASE chart.
- `constrained-direction-subspace.md`: physical fixed atoms and search-direction
  exclusions are now separate; native masks are oracle-verified and the public
  optional interface is implemented. Cu28 default traces are exactly unchanged.
- `type4-source-mask-results.md`: complete514atom MACE searches and two positive
  landing Hessians are available. The initial still has negative curvature at
  fmax0.001; a stricter ordinary-descent control and conditional certified-start
  repeat take priority over interpreting its raw descent as search efficiency.

The initialization control and four real multi-step state/cost audits are now
complete; both held-out best points have positive two-step Hessians. TYPE4
listed cost30997E/F. New external CuO64 input qualification also passes
representation invariance and195-dimensional two-step joint curvature (806EFS).
`native-ls-fixed-atom-audit.md` closes the ordinary fixatom parser and both
Nb/allbonds gates: zero-valued ordinary fixed atoms remain in LS neighbors and
normalization; only negative sentinel factors are excluded. This closes the
previous native counting uncertainty. The independent constrained-LS walker is
now implemented with explicit LSSettings and1D/2D/3D image-resolved fixed-cell
pairs; Cu28 complete flows, physical certificates and exact2D/3D representation
comparison are recorded in constrained-ls-results.md. Native constrained-LS
trajectory parity and TYPE4 LS efficacy remain unqualified.

The frozen CuO PQC/joint VC comparison is complete: eight attempts were requested,
six entered (two biased-quench `maxiter`, four budget-censored), two were not
started, and no search landing was produced. The four arms used 7,988 search
E/F/stress requests plus four fresh checks (7,992 total); the accounting is
recomputed in `cuo64-pqc-joint-comparison/comparison/summary.json`. This is a
negative, censored result for this CuO64/MACE comparison and does not establish
relative kernel effectiveness, phase identity, or native VC parity. The subsequent frozen-target history comparison and whole-SSW
history500 controls are now complete; see `cuo64-vc-comparison.md`. They do
not establish whole-search improvement. Full native CBD parity remains optional: the
recovered block-sum inner product is degenerate and not rotation invariant, so
completeness of decompilation is not a sufficient reason to replace a verified
independent solver. Likewise, adding GA scheduling or more empirical cluster
controls has lower priority than a demonstrated SSW/VC lifecycle failure.

The follow-up CuO frozen-quench history diagnosis used 997 E/F requests and
found history500 local quenches converged in both seeds while history10 reached
the 300-step limit. A separate two-seed whole-SSW history500 control consumed
3,996 E/F requests and exhausted its biased-quench budgets at stages 13/12 with
no landing. These are fixed-objective, budget-censored diagnostics and do not
close the SSW/VC effectiveness gap or support a general history-size claim.

## Priority gaps

1. **C60/native cluster controls — optional reconnection implemented; remaining caller/direction controls incomplete.**
   `run_ssw` and `run_ls_ssw` now expose `reconnect_distance=None`, retaining the baseline by default. For explicit isolated/unconstrained input, the recovered geometric component is applied only between successful climbing and unbiased true quenching. The original geometry and component translations are recorded separately. Its medoid/component attachment procedure matches 54 isolated original-instruction cases to 1.8e-15; this is geometry evidence, not search efficacy. The empirical native target is `0.7 * criterion`, and the carbon SI value 1.7 Angstrom is not a universal distance. A matched MACE terminal counterfactual reconnects the C58+C2 fragments but remains 4.95161 eV above the original minimum with one two-coordinate and one four-coordinate carbon, so it does not establish fullerene discovery. Full-step interface validation completed1462EF: the first1381 physical requests exactly reproduce the unmodified LS prefix, and the final connected but high-energy2/4-coordinated outcome is MC rejected. In-quench vapor stop, the complete `globalcompress` direction action, `Ratio_Local` downstream geometry, and exact native post-Allopt refresh/MC chain remain open. Parser selection is partly recovered: `globalcompress` is compared with a random draw; `Ratio_Local` enters `0.1 + 0.1 * Ratio_Local * u`, not a percentage. See `native-vapor-python-recovery.md` and `native-cluster-control-selection.md`.

2. **Native climbing displacement, saved width and release-state selection — partly recovered; complete state-machine parity absent.**
   `native-gaussian-caller.md` shows `moveds` stores an actually measured projection in `width[ng]`, has incompletely recovered retries, and some completion paths select `work2` coordinates / `work1` forces. The independent kernels use declared fixed displacement/width and their own stop conditions. The 2013 paper explicitly permits stopping after reaching lower true energy, so the ordinary C60 return to its initial basin does not by itself prove an implementation bug. Current priority is to compare the exact native release/termination gates and saved state, rather than invent a same-basin retry threshold. This gap was omitted from the first seven-item inventory and is now explicit.

3. **Native CBD/Broyden full optimizer and caller stream — absent; Ritz/dimer and Safe-total are independent substitutions; unvalidated for native parity.**
   `pamssw/standalone/native_broyden.py:1-7,24-70` implements only verified BRZERO4 arithmetic prefixes and explicitly has no search driver, matrix/history update, or default use. `paper_reference.py:1-11` identifies the current Ritz/dimer, ASE-LBFGS and plain-MC differences. The recovery record stops before matrix construction, history pruning and outgoing step (`docs/research/native-broyden-prefix-recovery.md:6-20,118-160`). Scientific impact: a converged independent direction or optimizer cannot be called native CBD. The uploaded binary/Unicorn path can recover a narrow matrix/history oracle; the minimal next step is one bounded caller-plus-history contract, with no production replacement until checked.

4. **Native LS policy beyond the recovered normal cycle — partially implemented; independent substitution; unvalidated.**
   `native_ls.py:69-171` and `NativeLSCycleState` implement recovered table initialization/update and normal save/zero/restore transitions; `paper_reference.py:207-226` and `ls_cycle.py:55-133` wire independent LS lifecycles. The remaining explicit branch limits are constrained-atom counting (`native_ls.py:45-47`) and `update_native_table(branch != "normal")` (`native_ls.py:145-156`); this does not mean all save/restore is absent. Native failed-prequench eligibility, iteration-limit response policy and MIC versus image-resolved periodic LS remain open (`docs/research/native-ls-cycle-state.md:87-93`; `docs/research/native-ls-python-component.md:37-73`; `ssw-family-current-status.md:150-162`). Scientific impact: component arithmetic cannot establish native LS trajectory, especially on failure or periodic paths. Minimum next step is a frozen caller oracle covering eligibility and one periodic image case.

5. **Native VC stress/backend-to-optimizer handoff — producer and orientation pieces are partly closed; final consumer contract remains absent/unvalidated.**
   The independent VC/VC-LS path is implemented with exact E+pV, atomic/nonaffine stress pullback and certificates (`pamssw/standalone/vc_reference.py:62-120`; `docs/research/vc-rc-core-consistency-review.md:26-51`). Native producer evidence establishes stored-cell inverse/orientation facts, but the backend result-to-object nine-stress-field transfer and final consumer remain open (`docs/research/native-stress-backend-boundary.md:1-12,53-81`; `docs/research/native-cell-orientation.md:33-62`). The current log-strain chart is an explicitly independent six-coordinate contract (`docs/research/vc-native-gap.md:55-88`). Scientific impact: finite-cell certificates support the independent chart only, not native VC metric, sign, schedule or phase behavior. Minimum next step is the initialized stress-field consumer oracle; do not infer it from `NG_cell`, `ds_cell` or names.

6. **Native RC force transmission and finite-angle map — independent forest/RC/RC-VC workflows are implemented; full native map is not closed; broad physical validation is unvalidated.**
   `run_rc_ssw`, `run_rc_forest_ssw`, and `run_rc_vc_ssw` provide exact Jacobian/pullback, reduced climb, unrestricted quench and MC (`pamssw/standalone/rc_reference.py:46-110`; `rc_forest_reference.py:1-18`; `rc_vc_reference.py:14-27`). The native transmit slice now verifies a particular force-buffer routing, but explicitly says the whole finite-angle map, Kabsch/inherited-twist behavior and scheduled coefficient are not established (`docs/research/rc-native-transmit-contract.md:28-38,94-164`). The principal SO(3) chart rejection is implemented before oracle work (`rc-optimization-domain.md:1-22`), while closed-loop constraints are an explicit supported-domain limitation; the inspected native release support for such loops is unknown, so this is not labeled a missing native feature. Scientific impact: current RC results are independent reduced-coordinate feasibility, not native force-transmission or molecular-crystal efficacy. Minimum next step is one finite-angle two-body whole-map oracle, then repeated real-system qualification.

7. **Unified restart/checkpoint semantics for SSW/LS/VC/RC — partially implemented; complete cross-family restart is absent and unvalidated.**
   `atomic_climb.py:17-47,71-107` saves completed Gaussian boundaries, pending stage and configuration, and can resume without resampling; `native_ls.NativeLSCycleState` preserves its own table transition state. The ordinary `paper_reference.run_ssw` returns failure work and costs but has no complete serialized outer checkpoint containing Gaussian/LS state, optimizer history, RNG and pending observer (`paper_reference.py:141-160,227-441`). Scientific impact: interrupted or budget-censored full searches cannot be resumed with demonstrated trajectory identity, and replay evidence must remain a separate zero-PES artifact. Minimum next step is a frozen schema and round-trip test for one full outer step; do not infer this from the atomic-climb checkpoint.

8. **GA exact paper/Java controller semantics and scientific effectiveness — controllers/proposal families implemented for declared subsets; native parity and broad validation unvalidated.**
   `paper_ga.run_ga_ssw` implements explicit quick/archive/fine orchestration (`pamssw/standalone/paper_ga.py:39-180`); TYPE0–TYPE4 and periodic/substrate proposal/controller entry points are present (`ga_operators.py:471-620`, `periodic_ga.py:246-273`, `molecular_periodic_ga.py:87-120`, `surface_ga.py:419-470`, `periodic_ga_reference.py:16-90`). Current documents explicitly separate these independent controllers from Java schedule/DCCD identity and note that proposal fixtures use zero PES (`research/ga_ssw/PAPER_GA_CONTROLLER.md`; `research/ga_ssw/GA_OPERATORS.md`; `docs/research/periodic-ga-controller.md:1-40`). Scientific impact: implemented routing or geometric JAR fixtures do not establish GA-SSW discovery, descriptor equivalence, or efficiency. Minimum next step is one bounded real-PES multi-generation controller with declared periodic identity and full cost/lineage ledger; exact Java schedule remains an optional parity task.

## Explicit code-marker audit

The repository search over `pamssw/standalone` found no `TODO`. Bare `pass` occurs only as exception-class bodies in `molecular_auxiliary.py` and `surface_ga.py`. Algorithmic `NotImplementedError` sites are explicit domain contracts: Safe-total with Eckart (`paper_reference.py:53`, `surface.py:112`), native-derived height with Eckart (`paper_reference.py:176`), constrained/partial-PBC fixed SSW (`paper_reference.py:182-185`), constrained native LS counting (`native_ls.py:47`), and the non-normal native LS table branch (`native_ls.py:156`). These restrictions are not silently counted as absent complete families.

The audit deliberately uses the latest records: native cell producer/orientation closure is not relabeled as wholly missing; RC closed loops are described as a current independent-domain limit without asserting that the original release supports them; and GA controllers are labeled implemented but unvalidated rather than incomplete.
