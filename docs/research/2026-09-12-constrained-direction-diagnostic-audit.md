# Constrained direction diagnostic audit

The completed diagnostic contains eight arms: Cu/Al, seed 3, and four
direction selectors. Every arm completed two outer steps with three records
and three fresh minima. Paid request counts agree exactly between the result
object, ledger length, and successful ledger entries:

| system | generalized dimer | Ritz | dimer | Euclidean Broyden |
|---|---:|---:|---:|---:|
| Cu | 443 | 417 | 443 | 454 |
| Al | 401 | 385 | 401 | 416 |

All 24 fresh minima have successful independent EMT E/F checks and exact fixed
coordinates and cells. The largest fresh active force is 0.009353 eV/Angstrom;
full raw forces are reported separately (largest 0.260092 eV/Angstrom) because
the substrate is fixed. No arm recorded a rotation or biased-quench failure. All four Al arms
rejected their second landing under MC; all four Cu arms accepted both.
Each arm nevertheless retains all three force-qualified observations, including
its initial minimum and any rejected landing. These counts describe this bounded EMT diagnostic and do not
establish a solver advantage or physical surface accuracy.

Adatom heights and minimum distances remain per-minimum diagnostics. In
particular, a height below the initial top substrate plane is recorded as a
geometric outcome and is not labeled physically invalid here.

Artifacts: `research/ga_ssw/evidence/constrained-direction-diagnostic-20260912/`.
The runner copied the full source before its child import and asserted the
snapshot package path. The model is ASE EMT; no DFT or materials conclusion is
intended.

An offline same-element MIC assignment diagnostic is saved as
`physical-audit.json`. It assigns only active atoms and leaves the fixed
substrate unmoved. The initial minimum has zero RMS to the input, while the
first outer landing has nonzero RMS (Cu about 1.468 Å; Al about 0.747 Å) and
the second landing has nonzero RMS to both. The record preserves explicit
assignments plus fixed-top and initial/current tag-1 top-plane heights; the
earlier one-label height summary was superseded. `records.accepted` shows all four
Cu arms with both outer landings accepted and all four Al arms with the second
landing rejected. These MC outcomes are retained separately from the returned
minima; the geometry differences are not attributed to label exchange or
called a new basin by this diagnostic.
