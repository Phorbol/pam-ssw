# Hard C60 LS eighth stage: exact offline replay and limits of diagnosis

`research/ga_ssw/diagnose_hard_c60_ls_offline.py` reconstructs the **original stage start** as saved center+.6*direction,90 frozen LS pairs (reference lengths from the initial true minimum, recorded strengths, xi=.2), and all8 saved Gaussian centers/directions/heights/widths. `paper-seed3/offline-stage8/frozen-objective.json` is the reviewable frozen subproblem. It replays Safe-total against only original logged raw E/F, requests999–1421; no calculator is instantiated and no new PES request occurs. This is not a restart from the failed endpoint.

All423 replayed coordinates match original coordinates **exactly (maximum error0)**. Final modified energy−3477.832151868307 eV, maximum force .011354581020178715 eV/Å,400 accepted steps and failure status reproduce exactly. This validates objective/state reconstruction, including LS reference lengths; it does not independently validate the GFN2 derivatives. The retained full source snapshot is available alongside the experiment.

Instrumenting the existing curvature admission function shows400 s·y values, all positive and all admitted. Thus no negative-curvature pair rejection, history-reset cascade, or exhausted Armijo search explains this stage. There are422 trial requests after the initial evaluation for400 accepted steps, hence22 rejected line-search trials. `curvature-pairs.json` records accepted request IDs, step lengths, s·y, and admission. `evaluation-components.json` records raw, LS, Gaussian and total energies/forces for all423 requests with accepted flags.

| Accepted step | Modified max force, eV/Å | Modified energy, eV |
|---:|---:|---:|
|0|9.83409|−3461.775107|
|50|.43102|−3476.237986|
|100|.09164|−3476.504877|
|200|.12861|−3477.457544|
|300|.05922|−3477.813071|
|400|.01135|−3477.832152|

Energy continues decreasing while max force is nonmonotonic. Atomic maximum accepted step ranges .000786–.2 Å, median .02612 Å; last50 median .01053 Å. This does not look like steps uniformly collapsing to numerical zero. Directional secant Rayleigh quotients `(s·y)/(s·s)` range .00504–81.44 eV/Å² (median .12471), showing widely varying encountered directional curvature. **These are not eigenvalues of one Hessian and cannot establish a condition number or prove stiffness is the cause.**

At the final accepted iterate, raw energy−3486.73565036, LS contribution8.88378452, Gaussian contribution.01971397 eV. Raw/LS/Gaussian maximum atomic force magnitudes are .58736/.50401/.08960 eV/Å; their vector sum has maximum .01135. Thus this is a nearly stationary *modified* surface, not a nearly stationary true PES. Force cancellation is expected for biased optimization and is not itself an E/F inconsistency.

Euclidean force projection onto three translations and three instantaneous rigid rotations gives final rigid-force norm fraction .0560 (only .00314 of squared norm); initial fraction .000286. At step100 the norm fraction reaches .2508, so artificial orientational components are present, but the final residual is not dominated by rigid motion. The maximum atom accounts for .3307 of total force norm at the end; neither a pure rigid drift explanation nor a single-atom force monopoly is established. Projected Cartesian Gaussians are not globally rotation invariant, so such torque alone is not a new bug finding.

Conclusion: a reproducible, still-descending biased local optimization reaches the declared400 accepted-step limit slightly above tolerance. Current evidence rules out several simple bookkeeping/negative-pair explanations but does **not** uniquely identify a condition-number, curvature-strategy, or force-contract defect. No extension, new tuning, or extra force calculation was performed. The recovered original subproblem is suitable for a separately declared local-optimizer comparison if justified later.
