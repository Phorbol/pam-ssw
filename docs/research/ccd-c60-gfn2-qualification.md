# CCD-derived C60 is a useful non-Ih starting candidate after GFN2 relaxation

Source: `research/ga_ssw/datasets/ccd-c60/c60-1809asym-source.extxyz` and its `c60-1809asym-provenance.json`, copied into this experiment. The parent extraction traces this geometry to the Kumeda/Wales 2003 CCD BLYP archive's 1809asym CPMD input; that source calculation is single-point, so the original geometry was not presumed to be a minimum.

Runner: `research/ga_ssw/qualify_ccd_c60_gfn2.py`. Evidence: `research/ga_ssw/evidence/ccd-c60-gfn2-qualification/`, including source snapshot, preregistered plan, model/package versions, all charged E/F requests, original and relaxed extxyz, optimizer results, fresh certificates, network diagnostics. No SSW, Hessian, GPU, retries, or public kernel changes.

Predeclared shared bound: 900 E/F, 480 seconds, one CPU thread, source first then ASE `molecule('C60')`. Both use GFN2-xTB/tblite 0.7 (accuracy .001), Safe-total fmax .01 eV/Å, maximum400 optimizer steps. Fresh certificates use a new calculator and count against the same budget. Actual total: **39 E/F in18.78 seconds**.

| Initial source | Quench + fresh E/F | Optimizer steps | Fresh E (eV) | Fresh maximum force (eV/Å) |
|---|---:|---:|---:|---:|
| CCD1809asym |30+1|27|-3494.059094221710|.0079207620|
| ASE Ih |7+1|5|-3495.670543924394|.0095934598|

The relaxed CCD candidate lies **1.61144970 eV above relaxed Ih under this GFN2 model**. Before relaxation its maximum force was1.8144 eV/Å, energy−3486.9428104 eV, and the1.8 Å carbon graph had56 degree-three and4 degree-two atoms. After relaxation all60 atoms have degree3 in one connected planar90-edge network, **not graph-isomorphic to the Ih reference**. Its shortest distance is1.35260 Å, longest graph bond1.46375 Å, nearest graph nonbond2.27355 Å. All graph edges and degrees are unchanged at diagnostic cutoffs1.7,1.8,1.9 Å. Thus this identity distinction is not an edge just straddling the chosen cutoff. Radius of gyration changes3.61182→3.52801 Å; post-relaxation radial extent is3.30483–3.75624 Å. No fragmentation is indicated by these diagnostics.

Ih stays a connected three-coordinated Ih graph, radius of gyration3.52422 Å. Both fresh energy differences from optimizer reports are below1e-12 eV. The CCD geometry changed chemical graph during quench: use the saved **`ccd1809asym-after.extxyz`** for the force-qualified non-Ih starting candidate, not the unrelaxed source file. Initialization costs remain part of subsequent source-to-result accounting even if a campaign reuses this prepared structure.

These are force-qualified GFN2 cage candidates, not Hessian-certified metastable minima, confirmed original-DFT isomers, or proofs of global optimality. The non-Ih candidate is an appropriate next difficult initial condition for an independently budgeted escape comparison; this precheck alone says nothing about SSW/LS efficacy.
