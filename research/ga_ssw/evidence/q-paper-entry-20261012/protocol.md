# Explicit S1/S2 Q-direction paper-case entry probe

## Question and scope

Does an explicitly supplied S1 or S2 descriptor gradient enter the existing
recovered-direction → CBD → multi-Gaussian → physical-quench lifecycle on one
saved C60 defect and one saved paper TiO₂ structure, while Q-off provides the
matched recovered-direction baseline? This is a bounded integration probe. It
does not estimate a success rate, establish native LASP parity, demonstrate
speed or superiority, or claim a physical phase transition or pathway.

The only arm change is the explicit descriptor library: empty/Q-off, one S1
mode, or one S2 mode, with descriptor selection probability 1 in S1/S2 arms.
The recovered pair/group controller, CBD limits, SSW settings, seed, model,
input, fixed cell, and numerical force threshold are shared within each
structure. Q-off retains the recovered pair/group and CBD controller; it turns
off only descriptor selection.

## Frozen inputs and model provenance

| Case | Input and source | Existing qualification | Model |
|---|---|---|---|
| C60 author isomer 2 | `research/ga_ssw/evidence/c60-local-defect-20260925/qualification/isomer-2/final.extxyz`; SHA-256 `ce9a47447535c400dc7d3c527ef891ed326563f357858916c2265232fbf980fb` | `.../result.json`; saved MH-1/omol force-qualified at 0.0284649076 eV/Å, cage graph/labelled edges retained at 1.64, 1.70, 1.80 Å; approximate curvature gate is archived separately in `.../curvature/results.json` | `/home/gengjianrui/.cache/mace/mace-mh-1.model`, SHA-256 `a522eb7f59c7879963d41586528f4980baf33e086c94aa92e3eafdeccad3be47`, head `omol` |
| TiO₂ paper phase 87 | `research/ga_ssw/evidence/tio2-phase-qualification-20261007/run-1663408/phase87_is/endpoint.extxyz`; SHA-256 `564aae0f5f7638b79edc6e23befe1e9ee12ecb6641fbf922e31a20ad8b23afc5`; the source is SI section 7, DOI `10.1039/c7sc01459g` | `.../phase87_is/result.json` and sibling `effective-plan.json`; the saved model-relaxed endpoint was independently recomputed at 0.0296822003 eV/Å and 0.0008728179 eV/Å³ maximum stress. Its already-relaxed cell is held fixed for this panel; this probe performs no cell optimization. | `/home/gengjianrui/.cache/mace/mace-omat-0-small.model`, SHA-256 `0abfde07862cf1e93b8b4d03cb702f29ce9c344ff2fc4de2ec0d7166d6c113a5`, head `omat_pbe` |

C60 coordinate provenance is Liu, Jin, Liu, *Mapping structure-property relationships
in fullerene systems: a computational study from C20 to C60*, npj Computational
Materials (2024), DOI10.1038/s41524-024-01410-7, SI archive
`41524_2024_1410_MOESM3_ESM.zip`, member `c60/c60-iso-2_opt.xyz`.
The coordinate dataset is not the LS paper itself; the adjacent-pentagon
rearrangement task is motivated separately by LS2024 section2.2. Its original
B3LYP-D3 geometry was already qualified/relaxed on MH-1 in the archived input
preparation. Neither original DFT energies nor the LS paper barrier are targets
of this fixed-cell integration panel.

The TiO₂ starting geometry is the saved qualified endpoint, rather than the
paper SI starting coordinates: the latter's recorded initial force is
0.1049938074 eV/Å. The endpoint is an existing artifact from the paper-case
model qualification, not geometry generated for this experiment. The runner
checks input and model hashes, composition, PBC, saved qualification status,
and paper/model metadata before loading the calculators. Neither case uses
xTB or a newly generated structure.

## Descriptor modes and parameter provenance

The radial primitive is

`phi(r) = r^n tanh(1-r/rc)^3`,

with pairs at or beyond `rc` excluded by the explicit guard. S1 sums this radial
weight over the selected center's neighbors. S2 uses the paper Q-value
addition-theorem form

`S = sqrt((2L+1)/(4 pi) sum_jk phi_j phi_k P_L(e_j dot e_k))`.

The mathematical S2 form is documented in
`docs/research/2026-09-26-q-s2-evidence.md` (Huang et al., *Chemical Science*
2018, DOI `10.1039/C8SC03427C`, Eq. 7); the bounded archived-binary audit
leaves native parameter-slot mapping, type partition, and exact cutoff
convention unresolved. The independent S1 and S2 primitives are
`research/ga_ssw/s1_reference.py` and `research/ga_ssw/s2_reference.py`.
This runner uses the explicit `DescriptorMode` adapter and its periodic-image
mapping. These formulas and parameters are not asserted to reproduce the
native LASP implementation.

Both cases use one fixed center: C60 atom 0 with carbon neighbors; TiO₂ atom 8
(the first Ti in the saved endpoint) with oxygen neighbors. Both use
`rc=3.0 Å`, `n=0`, `epsilon=0 Å`; S2 additionally uses Legendre degree `L=2`.
These are frozen diagnostic settings selected before the E/F campaign, not
fitted defaults. The 3 Å radius is a local first/second-neighbor scale for
these saved geometries; no evidence supports a universal cross-composition
default. `epsilon=0` avoids an extra near-cutoff truncation interval. Before
any PES call, CPU preflight records descriptor value, full gradient norm, and
periodic-image count for each case/family. A zero or nonfinite value/gradient
is a failed qualification; no center, parameter, or geometry is changed after
observing that result. The completed CPU precheck on these saved structures
returned nonzero values/gradients: C60 S1 value 0.40163784 and gradient norm
0.42188725 (10 selected images), C60 S2 value 0.09723730 and gradient norm
0.21748889 (10 images); TiO₂ S1 value 0.21375511 and gradient norm 0.20234660
(5 images), TiO₂ S2 value 0.00125792 and gradient norm 0.09619622 (5 images).
These are input qualification values, not PES/search results.

## Fixed settings and budgets

- One seed (`20261012`) per structure; six arms total (two structures × three
  arms). A single seed is a lifecycle demonstration only.
- Two outer attempts per arm, at most six Gaussians per attempt, 3,000 search
  E/F requests per arm, 90 seconds per arm, and a 30-minute campaign ceiling.
- One GPU. The maximum planned search workload is 18,000 requests. Independently
  fresh checks are capped at three returned minima per arm, for at most 18
  extra E/F requests. Model initialization, failed calls, denials, quench work,
  and fresh checks count toward the elapsed limits; they are recorded. Fresh
  checks have their own 30-second and three-request cap per arm.
- SSW: width 0.6 Å; 150 K Metropolis strategy temperature; physical force
  threshold 0.05 eV/Å; biased-stage threshold 0.1 eV/Å; 500 relaxation steps;
  Safe-total L-BFGS with memory 500; existing global proposal and recovered CBD
  controls are fixed as written in the runner. The fixed-cell coordinates and
  cell/PBC remain unchanged as the search domain.
- The runner writes a checkpoint after outer attempt 1, loads it explicitly,
  and requests exactly one additional attempt. It does not implicitly resume
  interrupted campaigns, retry a failed arm, extend a cap, or overwrite output.
  If the first segment ends without a valid completed outer boundary, the
  segment result, minima, checkpoint state and costs are retained and the arm
  is not resumed.
- Hard calculator/setup errors stop subsequent arms. Returned failures,
  rotation failures, and budget-censored algorithm outcomes remain in the
  records and do not trigger a retry.

## Records and interpretation

Each arm retains the frozen input, effective settings, request/failure ledger,
intermediate checkpoint, final checkpoint, complete SSW result, minima and
landing structures, wall time, request/denial counts, and independent fresh
energy/force certificates for every returned minimum (up to the cap). The
certificate records force stationarity, composition, cell, and PBC separately.
No stress pass/fail condition is imposed for fixed-cell exploration. For C60,
the saved cage identity and defect reference remain context; force convergence
or lower energy alone is not a cage/topology success. For TiO₂, this panel does
not classify phases or claim a transition. Offline input and returned-minimum
diagnostics record C60 distance-graph connectivity/degrees or TiO₂ periodic
minimum distance and coordination counts; these are geometry descriptors, not
phase or success labels.

The prepared command is CPU-only and must run before any scheduler submission:

```bash
python research/ga_ssw/probe_q_paper_cases.py /path/to/new-output
```

When the runner code is staged from a frozen source copy, pass the checkout
that contains the archived inputs and ledger explicitly with
`--data-root /home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification`
(or set `PAMSSW_RESEARCH_ROOT`). This data root does not control which runner
or `pamssw` source is imported.

Only after reviewing the prepared paths and descriptor precheck may the
bounded execution entry be used:

```bash
python research/ga_ssw/probe_q_paper_cases.py /path/to/new-output --execute
```

The runner and protocol are preparation artifacts. No GPU execution or job
submission is performed by this entry probe.
