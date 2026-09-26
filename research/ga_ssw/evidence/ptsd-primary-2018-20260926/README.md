# PTSD primary formula source and independent S2 implementation check

Primary source: Huang et al., *Atomic structure of boron resolved using machine
learning and global sampling*, Chemical Science2018, DOI10.1039/C8SC03427C.
Source URLs, original equation-file mapping and retrieval are in manifest.json.
The article states CC BY-NC3.0; this subset is retained for research attribution.
The XML and original S1–S6 equation images are source records, not our derivation.

The private `research/ga_ssw/s2_reference.py` implements the S2 addition-theorem
value and full center/neighbor analytic gradient, using the already recovered
S1 tanh radial envelope with explicit cutoff/exponent/guard. This is an
independent paper-derived primitive, not a claim of native S2 parameter parity,
complete Q selection, periodic walker integration or search benefit. Zero-norm
and numerically negative squared norms are reported as domain errors, with no
regularization or hidden fallback.

CPU1499153 ran `check.sbatch`: S1/S2 targeted9tests passed. The independent value
check uses SciPy spherical harmonics; derivative probes perturb actual atomic
coordinates and explicitly accumulate periodic image contributions back to the
corresponding atom. Six checks on saved C60-defect and rutile12 environments
(L2/L4, one center per species) passed, including rutile center self-images.
Largest directional derivative discrepancy was6.64e-11; this establishes the
formula/image-accumulation implementation on these geometries, not numerical
precision required of a PES or physical search performance. Zero PES calls.

All parameters and source paths are in geometry-check.json. The fixture sums
all supplied species; it does not pretend to reproduce native species selection.
The source structures were existing development outputs; these are not new
independent efficacy inputs. Raw scheduler output is check-1499153.out. The
SciPy spherical-harmonic cross-check emits a deprecation warning on this
installed version, but passes; production code has no spherical-harmonic API
dependency. Follow-up requires explicit center/species/image/selection/state
contracts before a public walker path can be claimed.
