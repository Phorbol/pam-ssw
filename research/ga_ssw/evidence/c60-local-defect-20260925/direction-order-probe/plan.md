# C60 atom-order robustness probe

## Question and scope

The original complete-direction panel produced Ih hits in 2/4 trajectories on one force-qualified local defect input. The prepared check asks whether this result is sensitive to atom row order when the same physical geometry is presented with one fixed, predeclared permutation. It does not change the direction kernel, native LS, MC settings, seeds, budgets, or method arms.

The permutation is `numpy.random.default_rng(25092573).permutation(60)`, chosen without reference to a direction axis, energy, or observed trajectory. `permutation.json` records both index maps; the wrapper's zero-PES preflight proves the permuted coordinates are exactly the original coordinates in that new order, with species, cell, and PBC preserved. Original qualification files remain tied to their original row labels, and the mapping file links that qualification to the relabeled input.

## Frozen panel

Run the same four existing arms: `ssw_without_ls` and `native_ls`, each with seeds 1101 and 1102, using the same ten-attempt/12,000-request per-trajectory caps, fresh-validation protocol, and one-V100 resource ceiling recorded in `plan.json`. The only physical input file difference is its atom row order. The core implementation, full recovered-direction settings, LS, native MC, and SSW settings remain fixed.

Prepared checkout and `pamssw` tree IDs are recorded in `plan.json`. The selected tree includes the approved compact observer change. The input permutation does not modify any core code or public API.

## Interpretation limits

All four runs share one physical structure, so they do not provide independent structural validation. Reusing numeric RNG seeds after row permutation does not preserve the same physical random vectors because random components are attached to atom indices. Therefore, a changed outcome cannot be attributed solely to the tied-axis initialization or any one mechanism. Ih hit/miss tests only label-order robustness for this one input and these seeds; they do not justify a global default or further automatic permutations. An intact starting cage is not a discovery success, and no physical pathway or barrier claim follows.

## Execution boundary

`python direction_order_probe.py --preflight` runs the existing CPU-only qualification/configuration checks plus an exact physical-identity and permutation-map check. It creates only the new preflight record and makes zero calculator/PES requests. `--run` remains opt-in and uses the existing full panel's resource and output bounds. This preparation task does not submit a job, load MACE, or run PES calculations.
