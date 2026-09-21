# AlOH26 / brookite48: proposed equal-cost four-arm validation

2026-09-10. **Prepared for review only. This document provides no authorization for long runs, GPU/HPC work, automatic submissions, or new PES calls.** No calculation was launched to prepare it. The concrete proposal is a bounded feasibility gate first, followed—only if separately approved and justified—by a paired efficiency campaign. Papers and native observations inform the algorithms; the comparison does not require literal reproduction of paper indexing or sign inconsistencies.

## Scientific question and actual inputs

Do cell/atom blocks or joint atomic/strain proposals deliver additional physically plausible, force/stress-qualified structure coverage per total oracle cost on complex material PESs, beyond atomic SSW and posterior cell relaxation?

- **AlOH26, Al8O14H4:** first frame of the uploaded `GA-SSW_examples_run/global_exploration/input-templates/TYPE1-AlOH/addition/add.arc`, rooted at `/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/`. Its periodicity is supported by the TYPE1 input, not inferred from an ARC box. It supplies a heterogeneous inorganic/proton network, not a known GM reference.
- **Brookite48, Ti16O32:** `literature/benchmark-sources/coordinates/brookite.extxyz`, extracted from SSW-NN SI §7; same-name ARC retains source precision. This is an actual reported-phase geometry and a provenance-bearing start. It is not a random initial structure, and beginning there cannot establish brookite discovery efficiency. Phase-87 and phase-139 48-atom coordinates are available for offline reference comparisons; their same-PES phase identities and stability have not been automatically qualified.

Both use the existing local MACE OMAT-small checkpoint `/home/gengjianrui/.cache/mace/mace-omat-0-small.model`, CPU float64, one thread, `PYTHONNOUSERSITE=1`, and zero pressure. Freeze model checksum, environment, exact coordinates, source snapshots and all scalar settings in a separate launch manifest. Replacing the original LASP NN with OMAT changes the PES; model-level search results are not DFT material stability claims.

## What the existing costs actually establish

All figures below are read from existing artifacts. They are development observations, not equal-cost arm comparisons.

| Existing artifact | Physical search requests | Search seconds | What completed |
|---|---:|---:|---|
| `block-aloh26-seed3/result.json` | 1,113 | 300.010 | Initial quench + one cell-only proposal; next combined proposal budget-censored |
| `block-brookite48-seed3/result.json` | 609 | 300.080 | Initial quench + one cell-only proposal; next combined proposal budget-censored |
| `block-aloh26-atomic-replay/result.json` | 465 additional; 1,578 accumulated | 135.190 additional | Continuation of interrupted development work; atomic substage 385 calls, Gaussian-limit status |
| `joint-vc-aloh26-l5-clean/result.json` | 1,748 | 391.792 | Recorded complete joint development run; two separate fresh checks |

The first two runs each have two additional fresh checks. They return top-level `completed`, but their second proposal explicitly reports budget exhaustion. That status must not be interpreted as completing both requested proposals. The replay remains attached to the original censored experiment; it is not an independent replicate or a new equal-cost run.

At inspection, `block-brookite48-atomic-replay/` contains plan/progress/source/script but **no final result.json**. It contributes no completed-run timing or success claim to this estimate. Recheck its eventual terminal result before freezing a launch manifest; do not extrapolate a success from a directory name.

Observed wall/request averages are about 0.27 s for AlOH block, 0.29 s for its replay, 0.224 s for AlOH joint, and 0.493 s for brookite block, including their respective setup/processing. Estimates below use **0.35 s/request for AlOH and 0.65 s/request for brookite** as explicit planning allowances, not measured guaranteed rates. Model-domain changes or harder structures can exceed them. The governing resource control is the declared request and wall ceiling, whichever arrives first.

The first valid block proposals were **rejected** and higher in energy: AlOH +0.30366 eV with +13.09% volume; brookite +6.86479 eV with +15.96% volume. Existing diagnostics show an AlOH proton-partner change and Ti coordination changes. These are real endpoint changes, not established new stable phases or an efficiency benefit. See [endpoint diagnosis](block-complex-material-initial-results.md).

## Four arms and a shared physical starting point

For each system prepare one common zero-pressure, same-calculator relaxed start, retaining its raw-source structure and residual certificate. All arms receive byte-identical prepared atoms/cell. Charge the preparation cost equally in end-to-end comparisons, even if physically computed once; report actual shared wall cost separately.

1. **Fixed-cell SSW:** atom-only escape and atom-only final quench, with the common cell fixed throughout the chain. Store stress, but do not require relaxed stress or call these variable-cell minima.
2. **Posterior cell quench:** atom-only escape at the current seed cell followed by the common all-DOF physical quench. The resulting cell becomes the next seed cell. This is the direct comparator isolating strain participation in escape.
3. **Block VC:** explicit cell-vector displacement/partial atomic relaxation cycles, scheduled atomic SSW, and the common final all-DOF physical quench. Intermediate partial relaxation is not itself a minimum certificate.
4. **Joint VC:** simultaneous atomic/log-strain softening, bias and quench in the established exact coordinate map, followed by the same physical stopping rule and final certificate.

Keep the existing independently implemented numerical optimizer and oracle conventions aligned; do not introduce arm-specific extra retries, rewards, scoring terms or tuned failure fallbacks. Atomic width, temperature, Gaussian cap, finite-difference precision and direction solver should be shared where they have the same definition. Cell-specific choices remain explicit, not forced into false equivalence.

Before the feasibility gate, freeze and explicitly label the **current per-cycle fresh-direction policy**. Closing the native direction lifecycle is not a prerequisite for this independent experiment. Native exact parity remains unknown; core coordinate/gradient consistency and a controlled experiment are the requirements. If a retained-direction alternative is later tested, give it a separate configuration and do not combine its results with the current policy. Preserve existing seed-3 results under their exact archived implementation. Similarly freeze joint L and cell-block step parameters. Existing experimental L=5 Å may be used as a stated development configuration; it is not a universal default or identified native ds_cell. A separate L/step sensitivity campaign is outside this budget and must not be hidden inside the equal-cost comparison.

## Concrete runner gaps before any executable campaign

The current `research/ga_ssw/compare_vc_arms.py` is a **Cu/EMT three-arm development runner**, not a ready AlOH/brookite four-arm runner. It repeatedly invokes a complete one-step walker. Naively adding a block call would restart its outer-step counter on each invocation: with `atomic_period=2`, this could repeatedly execute the cell-only branch and never test the scheduled combined cell+atomic branch. Thus successful existing three-arm parsing or Cu tests do not establish this campaign's execution readiness.

The minimal implementation work still required is:

1. Accept the actual source atoms and explicit MACE calculator factory with complete E/F/stress counting, instead of the runner's hardcoded Cu/EMT setup. Preserve the two real periodic structures, supplied cell and model provenance.
2. Maintain persistent outer-step indices, RNG state, current accepted atoms/cell and the block schedule across proposals. At least the first two block proposals must demonstrate cell-only then scheduled cell+atomic behavior under the frozen period; a numerical schedule test is necessary but not scientific efficacy evidence.
3. Compose reusable proposal stages with one shared initialization and one outer selection decision. Do not repeatedly call a whole walker in a way that silently requenches initialization, executes hidden inner MC, or resets bias/schedule lifecycle. If initialization is intentionally repeated by a defined algorithm, expose and charge it; do not introduce it accidentally through the comparison wrapper.
4. Preserve complete request accounting and interrupted-stage state for all four arms, including partial atom relaxation, HVP calls, failed quenches, final physical certificates and MC-rejected landings. Record initial quench cost separately and charge common preparation fairly; avoid both free initialization and double charging physical calls.
5. Adapt the frozen analysis to these materials: persistent step labels, known/unknown/ambiguous identity, full failure denominator and separate accepted-chain versus discovered-candidate metrics. Preserve full physical certificate data and the fixed-cell arm's different stress requirement.

These are implementation gaps, not authorization to modify core code or launch runs in this planning task. Once closed and checked, the proposed feasibility gate would exercise the actual four-arm path. Native literal parity, scheduler off-by-one recovery and paper diagram agreement are not launch gates for the explicitly independent implementation.

## Proposed feasibility gate: 16,000 requests maximum

Use development **seed 17**, one run for each system × four arms: eight runs. Set **2,000 total search E/F/stress requests per run**, including initialization, all HVP differences, failed steps, line searches and final in-run certificates. Cap each AlOH run at **1,200 seconds**, each brookite run at **1,800 seconds**, single CPU thread. Thus request ceiling is 16,000 and serial wall ceiling **12,000 seconds = 3 h 20 min**; the planning-rate estimate is about **2 h 13 min**. No automatic extension when a limit is hit.

The purpose is to establish callable four-arm plumbing, full accounting, and at least one finished appropriate proposal per arm. For the block arm require witnessing a **cell+atomic** proposal, not merely the first cell-only branch. If the cap arrives first, record the result as censored and report the exact incomplete stage. Do not raise caps or shorten the algorithm after seeing failures and retain the same protocol label.

Pass criteria: valid geometry/finite oracle responses, consistent physical force/stress certificates, actual intended branch completion, complete cost reconciliation, and no systematic pathological expansion or chemical collapse. A failed/censored gate produces a concrete implementation or cost question for review. It does not establish inefficiency or justify new heuristics. Eight short runs cannot establish search advantage.

## Proposed independent efficiency campaign: 160,000 search requests

Proceed only after the gate and **separate long-run approval**. Use prospective paired seeds **29, 43, 71, 101**, shared across two systems and four arms: **32 runs**. Neither seed 3 nor development seed 17 belongs to the evaluation set. Four repeats give only exploratory dispersion; they are not a paper-quality statistical guarantee or a generality claim.

Each run has **5,000 total search requests**, with completed-candidate reporting checkpoints at 1,000, 2,500 and 5,000. Stop by oracle cost, not equal outer-step count. Record partial proposals and failed quench costs at each checkpoint; only fully certified completed candidates count as hits. Proposed wall ceilings are **2,400 seconds per AlOH run** and **4,200 seconds per brookite run**, totaling **29 h 20 min serial wall ceiling**. Planning-rate estimate is approximately **22 h 13 min** for 160,000 requests. The hard whole-campaign request cap is 160,000 even if faster arms finish early; do not donate censored arm budget to favored arms.

Endpoint qualification is a separately itemized proposed reserve of **8,000 E/F/stress requests** (4,000 per system), at most **90 minutes serial CPU wall**. This is not an instruction to run it now. Before launch specify how the same reserve policy is applied across arms. Fresh endpoint certificates can cover all finished landings; any tighter relaxation/Hessian subset must use a frozen arm-blind selection rule and retain the unqualified denominator. If there are more candidates than the reserve can qualify, keep the remainder pending instead of counting them as stable minima.

Combined optional campaign ceiling is **168,000 physical requests and 30 h 50 min serial wall**, excluding the separately approved gate. Including both phases gives **184,000 requests and 34 h 10 min wall ceiling**. These are concrete upper bounds for review, not booked resources. GPU/HPC use and parallel resource allocation require a new explicit approval; none is assumed here.

## Metrics and scientific qualification

Primary comparisons use **completed, structurally distinct, numerically qualified and physically reviewed candidates versus all requests**, separately from the accepted chain. Keep valid MC-rejected landings: the prior first block steps already show why acceptance alone misses explored endpoints. Report per-seed counts, first arrival costs, energies/enthalpies, dispersion and all failures/censors; do not pool only successful trajectories.

- AlOH: species-preserving periodic structural identity; endpoint H–O partner changes and Al–O coordination changes; density/volume and proton/fragment plausibility. A changed nearest O is an endpoint observation, not a TS mechanism or rate.
- Brookite48: known-reference identity where supported, Ti–O coordination/topology, density/volume and energy relative to the common start. Starting brookite recovery is not a discovery. Unmatched candidates remain unknown/ambiguous, not newly named phases.
- Both: lowest observed valid energy/enthalpy and improvement beyond numerical uncertainty; distinct endpoint coverage with consistent matching; all-request cost to first changed candidate. Separate first candidate observation from first accepted move.

Matcher controls must allow species-preserving lattice bases, rotation, translations and equivalent repeated cells (`scale=False`, primitive/supercell equivalence as appropriate). Do not hide density differences by isotropic rescaling. Freeze a tolerance sweep from development raw/strict reference pairs; disagreements across calibrated levels are ambiguous. The existing Cu calibration proves the need for representation handling, but its geometric thresholds are not automatically valid for AlOH/brookite. Reference-coordinate matching alone cannot certify a material phase.

Physical final thresholds should remain the shared implemented values (current fmax 0.01 eV/Å, maximum absolute allowed stress component 0.001 eV/Å³ for variable-cell arms) unless a preregistered precision qualification changes them **before** the evaluation. Retain all nine physical stress entries. Fixed-cell residual stress is diagnostic rather than a rejection criterion; compare that arm's constrained-space coverage separately from the three variable-cell arms. Posterior-relaxing an auxiliary copy for cross-arm identity would cost extra and must not feed back into the fixed-cell chain.

Positive volume and small force/stress do not establish chemical validity, local stability or thermodynamic phase stability. Inspect coordination reduction, vacuum/fragment formation, severe density changes and model extrapolation. A Hessian, if subsequently approved, must include the allowed atomic/cell freedoms and distinguish translation zero modes and residual-gradient effects; finite-cell positivity does not prove larger-cell phonon stability. Report each qualification layer independently.

## Review decision requested later, not authorization now

The next actionable review can approve **only the 16k-request feasibility gate**, after the runner gaps above are closed, with the exact source/configuration and current fresh-direction policy frozen. Its result should determine whether a 168k-request campaign is scientifically useful and whether the proposed wall ceiling is realistic. Do not launch either phase from this document, broaden systems, silently revise parameters, or present the prepared protocol as completed validation.

## Subsequent implementation and replay status

The initial runner gaps listed above are now addressed by
`research/ga_ssw/compare_material_arms.py`: block/joint whole-chain calls retain
scheduling, and fixed/PQC use pure atomic climb followed by the relevant quench.
Real EMT lifecycle tests cover identical initial states/costs, rejected landings,
actual block atomic scheduling and budget censors. Full module suite: 182 passed,
1 skipped. This qualifies wiring, not material efficacy.

The exact eight-run, two-step gate is frozen at
`research/ga_ssw/prospective/complex-vc-feasibility/manifest.json`, with source,
config and coordinates. It remains unapproved and unlaunched. The 12,000-second
ceiling bounds search time; model startup adds separate wall overhead.

Brookite replay ended at 600.473 seconds with 1,092 additional requests,
1,701 accumulated search requests plus 2 prior fresh checks. Nine Gaussians
completed; the tenth biased quench was censored. No final combined landing or
fresh certificate exists. Do not treat it as a completed branch or use its
intermediate structure as a minimum.
