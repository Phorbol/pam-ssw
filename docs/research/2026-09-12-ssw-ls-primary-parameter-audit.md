# SSW/LS-SSW primary parameter audit

This is a local primary-source audit performed on 2026-09-12.  The source
files are the extracted text and PDFs under
`/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/literature`.
Line numbers below refer to the checked-in extracted text; printed page
numbers are given where the PDF has them.  These papers support definitions
and reported choices, not a universal parameter prescription.

The local file labels are: `215.txt`/`ct4c01081_si_001.txt` for LS-SSW
(4c01081), `74.txt` for the original SSW paper, and `BP-CBD-user.txt` for
BP-CBD (ct300250h).  `225.txt` is the later rigid-body-chain SSW article and
`244.pdf` is the GA-SSW article; they were checked for identity but are not
used below as primary support for LS parameter claims.

## Original SSW (74)

`74.pdf` is the 2013 SSW paper (JCTC 9, 1838–1845,
doi:10.1021/ct301010b).  The dimer consists of two images separated by a
fixed `Delta R`, with `0.005 Å` given as an example (printed p. 1839,
`74.txt:131-136`).  The Gaussian is written with height `w` and width `ds`.
The paper states that `ds` is typically `0.2–0.6 Å`, described as roughly
10–40% of a chemical bond length; larger values explore more rapidly at lower
pathway resolution.  It states that `H` is system-dependent and uses `H=14`
unless otherwise noted, corresponding to about `4–5 Å` total displacement per
MC step (printed p. 1840, `74.txt:200-211`).

The paper does not give one universal Gaussian-height constant.  Height is
chosen so the Gaussian force supplies the desired forward component in the
biased translation; the exact construction is referred to the BP-CBD method.
For the C4H6 example it reports `T=1000 K`, `ds=0.1 Å`, and `H=25`
(`74.txt:228-235`).  A minimum is identified by a maximum-force criterion,
but the criterion varies by model: the Morse-cluster passage gives
`0.04 epsilon/sigma` (`74.txt:328-337`).

## BP-CBD (BP-CBD-user)

`BP-CBD-user.pdf` is the 2012 BP-CBD paper (JCTC 8, 2215–2222,
doi:10.1021/ct300250h).  Its equations 5–7 define the quadratic rotational
bias and the perpendicular force; equations 12–18 define sequential Gaussian
translation and its force.  The printed p. 2218 passage states that
`ds=0.1 Å` is used in practice and that the Gaussian force is maximized at the
inflection point.  The height is selected by requiring the translated force
component to meet a target (`F_R0 · N_i = 0.1` in the procedure), rather than
being given as a universal energy height (`BP-CBD-user.txt:228-247`).

For the Baker reaction demonstration, the explicit rotation convergence
criterion is `tau1=0.1 eV/Å`; the translation relaxation criterion is
`tau2=0.15 eV/Å`, with `0.02 eV/Å` for two very flat rotation reactions.  The
paper reports these as this study's settings, not general defaults
(`BP-CBD-user.txt:316-334`, printed pp. 2217–2218).  It reports average
Gaussian counts of 6.24 forward and 5.24 reverse, and average energy/force
steps of 126.8 and 96.0; these are measured results, not stopping limits.

## LS-SSW (215 / ct4c01081_si_001)

`215.pdf` is the LS-SSW paper (doi:10.1021/acs.jctc.4c01081).  Its SSW review
defines the dimer separation as `Delta R`, with `0.005 Å` as an example, and
its eqs. 7–8 define the sequential Gaussian sum, width `ds`, and height
`w` (`215.txt:131-170`, printed p. B).  The LS penalty is added first, then
the starting structure is reoptimized with l-BFGS before climbing
(`215.txt:335-359`, printed p. D).  The main text does not state a universal
soft-prequench force tolerance or l-BFGS iteration cap.

The supporting input files provide run-specific input values.  For C4H6 both SSW
and LS-SSW use `SSW.ftol 0.01`, `SSW.MaxOptstep 3000`, `SSW.NG 25`, and
`SSW.ds_atom 0.1`; LS additionally enables `SSW.soft.LselfAdapt` and sets
`SSW.soft.SAbiasAtom 700.0` (`ct4c01081_si_001.txt:207-239`, S10–S11).
The SI gives `SSW.ftol 0.05` for its C60 and C70/C90 LS and SSW inputs;
C60 uses `NG=12`, while C70/C90 use `NG=9`, and these inputs use
`ds_atom=0.6` (`ct4c01081_si_001.txt:247-327`, S12-S15).  For the Fe7C3 LS run the SI gives `SSW.ftol 0.02`, `SSW.MaxOptstep 1000`,
`SSW.Rotftol_preRot 1.000`, `SSW.Rotftol 0.100`, `SSW.Rotftol_ini 1.00`,
`SSW.NG 8`, `SSW.ds_atom 0.6`, and `SSW.DimerdR 0.01`; it also shows
`E_maxlimit 999999` and `F_maxlimit 999999` for that input
(`ct4c01081_si_001.txt:343-377`, S15).  These are explicit input values for
those experiments, not evidence of generic defaults. `MaxOptstep` is an
input-level optimization cap and must not be identified directly with the
binary LS soft-prequench budget. The separately audited binary control
`SSW.LSoptsoftmax` has default 50; mapping `MaxOptstep` to that actual
soft-prequench exit limit is therefore implementation-specific.

The SI lists `Rotftol_preRot=1.000`, `Rotftol=0.100`, and `Rotftol_ini=1.00`
for Fe7C3, but does not state their units in the input table.  The BP-CBD
paper's related force criteria are reported in eV/Å; those criteria are not
automatically the same quantity as this repository's curvature or rotation
residual.  They must remain separately named and validated.

The SI confirms `DimerdR=0.005 Å` as the default used in the dimer-rotation
illustration (`ct4c01081_si_001.txt:94-96`, S3). The Fe7C3 input explicitly
sets `DimerdR=0.01 Å`. It reports the self-adaption
penalty reaching its target within typically 100 SSW steps and oscillating
around it for the C60 example, with target `0.02 eV/atom`
(`ct4c01081_si_001.txt:133-135`, S5).  The main text gives a wider target
range `0.01–1 eV/atom`, and says the target is an exploration choice
(`215.txt:311-335`, printed p. D).  The reported C4H6 search uses 150 K,
25 Gaussians, and `ds=0.1 Å` (`215.txt:393-395`, printed p. F).  The reported
maximum allowed search length is 50,000 SSW steps (`215.txt:415-417`, printed
p. F), while individual demonstrations use 400, 20, 7,000, or 20,000 steps;
these should not be conflated with a soft-prequench exit criterion.

## Parameter status for the shared driver

| Quantity | Explicit source statement | What remains unspecified |
|---|---|---|
| Dimer separation / finite difference | `0.005 Å` example/default illustration; `0.01 Å` in Fe7C3 input | No universal accuracy law or adaptive rule |
| Gaussian width | Original SSW `0.2–0.6 Å` typical; examples and LS C4H6 use `0.1 Å`; Fe7C3 uses `0.6 Å` | No universal best width |
| Gaussian height | Force-component construction / target condition in BP-CBD; LS paper defines `w` but does not provide one universal value | No universal energy-height default |
| Soft pre-quench force tolerance | Input files contain `ftol=0.01` (C4H6), `0.05` (C60/C70/C90), and `0.02` (Fe7C3) | These are input values; their mapping to soft-prequench stopping is implementation-specific |
| Input optimization cap | `MaxOptstep=3000` C4H6, `1000` C60/C70/C90/Fe7C3 | Not the binary LS soft-prequench exit; audited `LSoptsoftmax` default is 50 |
| Gaussian count cap | `H=14` original SSW unless noted; `NG=25` C4H6, `NG=8` Fe7C3 | System/run dependent |
| Search-step cap | LS SI reports 50,000 for fullerene runs | This caps outer search, not each pre-quench |

The papers therefore justify exposing these quantities as explicit numerical
settings with provenance and units.  They do not justify silently promoting
the C4H6/Fe7C3 values, or the PAM adaptive targets in this repository, to
general physical defaults.  In particular, a driver may use a policy to
derive width/height from measured curvature, but that is an implementation
choice requiring its own cross-system validation; the primary papers alone
do not establish such a policy.
