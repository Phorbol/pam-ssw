# Legacy LJ38 connectivity: identify the failure layer before tuning

CPU1493015 completed in5s, zeroPES. Both distinct starting true quenches are connected38-atom clusters under both predeclared cutoffs1.3sigma and1.5sigma. Their energies are-166.830309 and-155.165025epsilon. Thus fragmentation of the initial quenched structures does not explain these particular negative discovery results; the unpublished paper initial ensemble still prevents exact statistical reproduction.

| Direction | Seed | Connected landings | Fragmented landings | Accepted fragmented landings |
|---|---:|---:|---:|---:|
| global |25092501|324/387|63/387|0/63|
| global |25092502|334/383|49/383|0/49|
| paper |25092501|191/400|209/400|0/209|
| paper |25092502|187/396|209/396|0/209|

Both cutoffs produce identical classifications. All418 fragmented paper-direction landings are37+1; the global group also has a few35+3/36+2 partitions. Every saved geometry is present, no analysis errors, scalar best and summary best energies match; every best structure is connected. This is geometric connectivity of force-qualified saved endpoints, not TS connectivity or basin identity. Denominators are completed outer boundaries with saved minima; initial and partial terminal costs remain charged separately.

Observed decision: none of the fragmented endpoints was accepted. Therefore the current walkers are not trapped because MC accepted separated fragments; however substantial proposal effort lands in rejected fragmented states. The paper-direction trajectories have fewer requests per step but a much larger fraction of fragmented proposals. These observations do not establish that fragmentation causes the overall target miss, nor that adding confinement/reconnection would help.

Keep at least two explanations open: local pair anchors plus approximate mode softening may over-localize motion; alternatively later Gaussian continuation may destroy a useful intermediate transition. No saved intermediate stage costs/geometries in this scalar ledger distinguish them. Do not tune the radius, add a restraint, or change Gaussian/rotation parameters from this result. Finish the already frozen compact full-budget comparison first, then choose a single bounded stage-level diagnostic only if the target-discovery gap remains.

Source checks: SSW2013 Eq1–2 specify normalized global plus unnormalized pair-displacement mixture; current paper sampler implements that expression. Recovered CBD is an explicit native-inspired numerical choice, not a claim of exact2013 iterations: settings override SSWConfig rotation_bias/rotation_hvp/rotation_tol; ftol0.02 and fd0.001 imply HVP residual threshold2eV/Angstrom² through10*fd*residual. The existing pre/rotation limits and quench thresholds remain frozen. Table1's average-step/success columns are retained as printed and not treated as medians or a two-seed success-rate prediction.

Artifacts: `lj38-connectivity.json`; decision/cost join reproduced with `summarize_lj38_fragments.py`, output `lj38-fragment-costs.json`.

## Competing low-energy endpoint identity

Zero-PES graph-isomorphism/proper-rotation matching against the independently qualified OPTIM second-lowest endpoint supports the same structure for global seed25092502 and both paper-direction seeds (RMS0.000280,0.000573,0.000674 Angstrom). Global seed25092501 has no graph mappings under the fixed cutoff and remains unidentified here; its energy alone is not an identity check. Parent reran the shared matcher and reproduced all four classifications. A first verification command demanded bitwise equality of SVD-derived RMS and failed at ~3e-16 Angstrom; this was an overstrict verification assertion, not a changed structure, and was replaced by checking the actual pre-existing geometric acceptance decision without modifying its threshold.

Doye, Miller and Wales, JCP110,6896(1999), Fig1 and SecI identify the second-lowest LJ38 minimum as an incomplete Mackay icosahedron and the global minimum as an fcc truncated octahedron ([original paper](https://www-wales.ch.cam.ac.uk/pdf/JCP.110.6896.1999.pdf)). The match supports arrival at this competing endpoint; it does not reconstruct this search's transition network or prove residence in a funnel. Existing reference coordinates/qualification come from the previously archived OPTIM example, with sigma scaling already applied. See `analyze_lj38_competitors.py` and `lj38-competitor-geometry.json`.
