# Prepared TiO₂ fixed-cell holdout

Runner: `research/ga_ssw/prepare_tio2_fixed_holdout.py`. It is prepared but
has not been executed. Inputs are the provenance-bearing 12-atom rutile and
anatase SI cells in `research/ga_ssw/evidence/omat-small-tio2-stress/`, not
invented geometries. The local readable model is
`/home/gengjianrui/.cache/mace/mace-omat-0-small.model`; the runner records
its SHA256 in the manifest.

The two arms share CPU float64, one thread, full PBC, global sampling,
`translation_only`, Ritz rotation, width 0.1 Å, 25 Gaussian stages, inner
rotation tolerance 0.1 eV/Å², outer force tolerance 0.01 eV/Å, and
Safe-total 400 iterations. The fixed arm uses `a=100`; the staged arm uses
`pre_rotation_hvp=5`, total rotation HVP budget 100, and the helper's actual
`a=max(Ce,0)`. Seeds are 11 and 29. Search is capped at 1200 inclusive API
requests per arm; cold fresh certificates are charged and logged separately.
No cell relaxation is performed. Results support only a fixed-cell numerical
MACE-OMAT comparison, not phase stability, DFT agreement, or native CBD
parity.
