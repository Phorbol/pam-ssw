# SSW / joint-VC prospective benchmark protocol

> 2026-09-10 更新：用户提供的 http://lasphub.com/publication/87.pdf 已成功取得并核验为12页正式排版全文（2,477,395 bytes），存于 `literature/benchmark-sources/vc2014/`。以下此前获取失败的记录保留作历史；正文缺口已解决，SI仍未取得。全文揭示2014为CBD-cell/atomic-SSW分块耦合，见 `vc2014-native-crosscheck.md`；当前joint log-strain方法须保留独立扩展标识。

2026-09-10. Preparation only: no PES calls or long calculations were launched to prepare this document. The word production names the intended evaluation protocol, not a claim that the implementation or its scientific effectiveness is production-validated. Machine-readable cases: [benchmark_cases.json](../../research/ga_ssw/benchmark_cases.json). Do not alter MAINLINE or introduce LS/RC/GA mechanisms for this campaign.

## Question and available inputs

Does a consistent joint atomic/strain SSW proposal discover additional certified crystal basins per total oracle cost, beyond fixed-cell walking followed by cell relaxation? First establish the independent SSW kernel and fixed-cell PBC baseline. Merely enabling a local optimizer's cell relaxation answers a different question from including strain in the escape direction, bias, and Hessian-vector products.

| Case | Initial/reference roles | PES and boundary | Current evidence boundary |
|---|---|---|---|
| Cu13 | Existing common initial from Safe-total study; no supplied certified GM | ASE EMT, isolated | Development/regression system already used in optimizer decisions; not held-out evidence |
| Cu8 | Existing periodic SSW initial | ASE EMT, full PBC | Periodic development case; prepare same-PES cell-relaxed reference before VC comparison |
| LJ38 / LJ75 | Cambridge coordinates are GM references only | Untruncated pair LJ, reduced units, isolated | No non-GM discovery initial selected; starting at GM cannot measure GM discovery |
| AlOH26 | Uploaded TYPE1 first frame, H4Al8O14, force-bearing | MACE OMAT-small, full PBC | 13 E/F/stress preflight calls passed; no reference minimum or search validation implied |
| rutile / anatase / TiO2-B | Each SI 12-atom geometry is a separate initial and source phase reference | MACE OMAT-small, full PBC | Parsed actual coordinates; each requires model evaluation and same-PES reference relaxation |
| C60 | Existing ASE fullerene cage | GFN2-xTB, isolated; deferred | Near low-energy cage; suitable for internal escape coverage, not random-carbon GM discovery |
| PdO / CuO | Not obtained | Prospective oxide MACE cases | No provenance-bearing input coordinates; do not fabricate or infer from a formula |

The full uploaded AlOH archive and TiO2 SI coordinates retain their actual cells. An ARC box alone is not evidence of periodicity; AlOH TYPE1 input and TiO2 crystalline SI establish it. Source paths are in JSON. TiO2 source: [2017 paper/SI retrieval and extraction](ssw-benchmark-literature.md), DOI 10.1039/c7sc01459g. The SI's rounded original VASP energies are not OMAT energy targets. Its reported TS is excluded from the minimum library. A calculator substitution tests our algorithm on another PES; it does not numerically reproduce original LASP NN results.

## Freeze configuration before evaluation

Freeze code snapshots, environment, model checksum, exact input arrays, energy/free-energy convention, pressure, all kernel and quench settings, RNG implementation, and matching policy in the launch manifest. Preserve raw source structures separately from common relaxed starts. Use CPU float64 and one thread for the bounded protocol; this document does not authorize a new HPC/GPU campaign.

Use zero external pressure for this first crystal comparison, E+pV with p=0, six symmetric strain freedoms, and no imposed atomic constraints. Both cell-changing arms use exactly the same pressure, physical final quench and certificate. If a nonzero pressure or restricted cell is subsequently needed, define a new campaign. Never compare an enthalpy minimum to a fixed-cell energy minimum as if they share the same admissible space.

The existing Cu development settings are traceable to `compare_cu13_safe_total.py`. Do not silently copy its temperature, bias width or Gaussian count to oxides and label them universal. The prospective oxide kernel configuration remains a required preflight output: the independent VC implementation and original VC paper are still being closed. This preparation freezes the comparison structure, not missing physical parameters. No parameter may be chosen from evaluation-seed outcomes.

Development seeds 3 and 17 remain development only. Prospective evaluation seeds are 29, 43, 71, 101, 137, 173, 211, 257, shared between arms. Exact trajectories need not consume equal random streams after differing branch decisions. These seeds are a proposed evaluation set; do not represent the available short development trials as eight completed repeats.

## Comparable arms and complete cost

1. Fixed-cell SSW: atoms move with cell fixed at the shared initial reference cell. This checks PBC kernel behavior and establishes a constrained-space baseline.
2. Fixed-cell escape plus posterior cell quench (PQC): each escape uses atomic coordinates and the seed cell; only the final true-PES quench changes the cell. The relaxed cell becomes the next seed's fixed cell. All post-quench cell cost is charged. This is the direct comparator to joint VC.
3. Joint VC: atomic and six strain coordinates participate throughout direction softening, bias construction, modified-objective quench and final physical quench. No cell wrapping/reference reset may invalidate stored bias coordinates.

Start all three from the same physical cell-relaxed initial reference, prepared once per input/calculator. Preserve and charge this shared preparation in the reported per-arm end-to-end cost; also report a search-only subtotal. For the strict fixed-cell arm, evaluating an additional posterior cell-relaxed copy is an auxiliary diagnostic, separately charged and never fed back into its trajectory. Do not compare its fixed-cell minimum count directly to the variable-cell minimum count without identifying this difference.

Proposed per-arm cap is 20,000 physical oracle geometry requests, with cost checkpoints at 5,000, 10,000 and 20,000. These are resource ceilings, not optimal algorithm parameters. E/F or E/F/stress obtained at one geometry count as one logical request; additionally log actual calculator executions, requested properties, energy/force/stress totals, and wall time because model caching and stress cost can differ. Charge initialization, all rotation/HVP finite differences, rejected steps, failed line searches, biased quenches, and physical quenches. A failed request still consumes budget. Validation is reported separately and added to total cost, with the same validation policy in each arm.

Check the cap before launching an oracle request. An unfinished step at the cap is censored, not a valid basin and not a convergence success. Report its spent work and current status. Archive prefix comparisons must include only candidates whose quench and certificate completed by that checkpoint. Freeze a separate wall-time safety ceiling from the bounded cost preflight before launching; a wall-limit termination remains a censored run, never a dropped replicate. Equal outer-step counts alone are not equal cost.

## Pre-registered strain length sensitivity

The implemented chart uses `q=(vec(X), L*s)` and an orthonormal six-component symmetric logarithmic strain basis; L has units Å and sets an algorithmic atom/strain metric. See [exact coordinate/gradient contract](vc-logstrain-chart.md).

For each shared relaxed initial cell define `ell=(V0/N)^(1/3)` and `L0=sqrt(N)*ell`. Compare **L/L0 = 0.5, 1, 2**, keeping L fixed throughout each trajectory. This scale makes a strain increment's metric weight extensive in N, analogous in order of magnitude to N local displacements of scale ell. It is an explicit dimensional convention, not a derivation of the true Hessian metric, an exact affine displacement norm, a native LASP default, or a proven optimum. It does not depend on an arbitrary cluster origin. The factor-two grid is a prospective sensitivity probe, not a fitted improvement mechanism.

Report all three choices on every input and evaluation seed, with equal per-run cost. No best-of-three result may be compared to one baseline run without charging three runs. If one L is later selected on development data, freeze it before an independent evaluation; retain the full development cost. Large outcome variation is a metric-sensitivity finding and limits claims of a general default. L does not alter the physical force/stress acceptance certificate.

## Success, failure and phase identity

A completed landing first needs a fresh same-calculator physical E/F evaluation, plus stress for variable-cell cases, after removing all bias. Proposed common numerical reporting tolerances are maximum per-atom force 0.01 eV/Å and full Frobenius norm `||sigma+pI|| <= 0.001 eV/Å^3` for unconstrained six-strain quenches. The latter is a prospective accuracy criterion, not taken from the GA-SSW paper and not evidence of phase stability; freeze it and verify attainable accuracy before the evaluation set. Fixed-cell landings require atomic forces only and retain their residual stress as an observable.

A force/stress certificate alone is not a scientifically valid basin. Record positive cell volume, cell condition, distances, species-resolved coordination, connectivity, density and energy; inspect dissociation, slab/vacuum formation, collapsed overlaps and model extrapolation independently. Never discard an inconvenient force-certified structure silently. Distinguish numerical quench failure, calculator/domain failure, budget censoring, physical-invalid candidate, ambiguous structure identity and valid candidate. Keep every seed in the denominator. Extremely expanded cells or separated fragments require physical review even when stress and force are tiny.

For isolated Cu/C60 use species-preserving, rotation/translation/permutation-aware structure comparison and chemistry/fragment diagnostics; pair-distance screening alone is not a final identity certificate. For periodic solids compare structures allowing PBC translations, equivalent lattice bases, cell orientation and same-species assignment. Use coordination/topology to corroborate matches; space group or energy alone cannot identify a phase. Before evaluation, calibrate numerical matching tolerance from repeated tightened quenches of the *same reference* and verify separation of the supplied references; freeze the resulting tolerance and preserve near-threshold cases as ambiguous. No matching implementation/tolerance is certified by this document.

Prepare an independent same-PES reference library from the supplied reported phases. If two named TiO2 inputs relax to the same structure under OMAT, report that collapse rather than counting two reference phases. Conversely an unmatched landing is initially an **unassigned candidate**, not a new phase. A new minimum claim requires tighter fresh relaxation, negative-mode checks in the allowed atomic/cell subspace with translational zero modes treated separately, structural uniqueness, and model-domain/chemical review. A new *material phase* claim additionally needs appropriate size/stability and higher-fidelity checks beyond this short 12-atom model campaign. No phonon or thermodynamic stability claim follows from a single-cell Hessian.

Primary readouts are certified structurally distinct basins versus total cost, known *other*-phase arrival probability/cost from each TiO2 start, first valid energy/enthalpy improvement beyond numerical uncertainty, and physical/numerical failure fractions. Report per-seed trajectories and dispersion, not just pooled best energy. Starting-phase recovery is not phase discovery. Low initial energy can leave no improvement available, so coverage and cost remain separate metrics. LJ GM hit requires both geometry and same untruncated-potential energy agreement at the precision supported by source data; LJ is supplementary evidence only.

## Launch gates and current limits

Before any production run: obtain/verify source coordinates and model provenance; qualify same-PES reference quenches; complete fixed-cell PBC and joint coordinate/gradient tests; close the VC walker/objective implementation; freeze oxide kernel settings and matching tolerances; and do a bounded one-step cost/interface preflight. A failed gate produces a documented limitation, not a replacement toy success. The AlOH stress preflight qualifies one interface point only. The 2014 VC paper/SI retrieval gap remains documented in the literature note. This protocol supplies an executable experimental specification once those implementation and input gates close; it does not claim that they have already closed.
