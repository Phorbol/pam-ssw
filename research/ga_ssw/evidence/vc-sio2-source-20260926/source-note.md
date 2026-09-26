# SiO₂ source candidate for a held-out joint-VC numerical comparison

## Input provenance

`alpha-quartz-cod-1011097.cif` is the unchanged CIF downloaded from the
[Crystallography Open Database record 1011097](https://www.crystallography.net/cod/1011097.html)
and its [direct CIF endpoint](https://www.crystallography.net/cod/1011097.cif).
The CIF cites P. H. Wei, “Die Bindung im Quarz,” *Zeitschrift für
Kristallographie* **92** (1935), 355–362. It identifies low-temperature
alpha-quartz, SiO₂, space group P3₁21 (No. 152), with a=b=4.913 Å,
c=5.404 Å and γ=120°. COD states that its entries are public domain; the
original crystallographic source should still be acknowledged.

The retrieved file SHA-256 is
`e7014ed2bf991b90c21c0013f261e7f4112b9b51487da043024a96e7cfe62c2a`.
ASE's CIF reader expands it to 9 atoms (Si₃O₆), PBC in all directions,
volume 112.96385386559709 Å³. The reader emits a setting warning for this
trigonal space group; therefore the CIF remains the source of truth, and any
converted input must preserve the reported symmetry/cell and be independently
checked before use.

## Relationship to the 2014 VC-SSW paper

The primary article is Shang, Zhang, and Liu, PCCP **16** (2014), 17845–17856,
DOI [10.1039/C4CP01485E](https://doi.org/10.1039/C4CP01485E). The author text in
the local archive (section 3.1) states that its alpha-quartz and anatase-type
coordinates are in the ESI and uses BKS for the SiO₂ search; the article also
notes that BKS predicts the wrong ambient-pressure SiO₂ global minimum. The
official indexed SI is titled “Coordinates for the SiO₂ and TiO₂ Crystals” and
lists a 129 kB PDF at
`https://www.rsc.org/suppdata/cp/c4/c4cp01485e/c4cp01485e1.pdf`, but live retrieval
was unavailable during this check (web fetch timeout; direct request 404).

This COD structure is a traceable ambient alpha-quartz input, **not** the
paper's exact SI coordinates or a BKS reproduction. It can support only a
held-out numerical comparison on the project's OMAT-small PES, after initial
geometry qualification. This input has **not yet been PES-qualified**; the CIF
source and successful ASE parsing establish provenance and format only. It
cannot support SiO₂ phase stability, reproduce the paper's reported GM search,
or justify production ranking of optimizers.

## Minimal comparison recommendation

If root prioritizes another material after the current pool readout, reuse the
existing joint-VC optimizer-panel protocol unchanged: this COD alpha-quartz
input; OMAT-small `omat_pbe`; same Safe-total, ASE LBFGSLineSearch, and SciPy
L-BFGS-B methods; same paired seeds, three outer proposals, and existing request
and wall ceilings. Keep the existing physical endpoint certificate
(`fmax ≤ 0.05 eV/Å`, stress residual `≤ 0.001 eV/Å³`) and report paid search EFS
per attempted outer step plus qualified landings per attempted step. Retain
failed quenches and censored attempts in denominators and costs.

The discriminating question is whether Safe-total's cost advantage observed on
AlOH26/TiO₂ phase87 survives an ambient alpha-quartz network on the same model;
this does not test paper-block scheduling or establish search ranking. Stop if
the initial quartz structure cannot meet the frozen physical certificate within
the existing qualification policy, or if endpoint outcomes remain discordant;
do not retune on this case. A passing short comparison would extend method
transfer evidence to one more composition/topology, not establish stable or
general superiority or a production ranking.
