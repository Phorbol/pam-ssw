# Cu28 constrained-LS integration audit

This is a zero-new-PES audit of the archived ASE EMT integration run in
`research/ga_ssw/evidence/constrained-ls-integration/emt`. The system is the
same Cu28 Cu(111) adatom fixture, with full 3D PBC and a 16 Angstrom vacuum
direction. The three arms use seed 29 and two outer steps:

| arm | LS / mode | search E/F | fresh E/F | result |
| --- | --- | ---: | ---: | --- |
| plain | no LS, full active direction | 119 | 3 | 2 valid landings |
| ls | LS, full active direction | 1000 | 2 | 1 valid landing; second biased quench censored |
| ls_adatom_mode | LS, adatom-only rotation mask | 954 | 3 | 2 valid landings |

The LS table is the existing EMT interface test table: Cu--Cu bond energy
1 eV, pair distance 2.9 Angstrom, target response 0.001 eV/atom, initial
fraction 0.03, xi 0.2, and learning rate 1.8. These values are an explicit
composition test contract, not a Cu chemical calibration and not a TYPE4
parameter source. The default arm has `ls=None`; the masked arm changes only
the rotation subspace (coordinates 27--29, the adatom Cartesian coordinates),
while biased and true quenches retain the full active chart.

The independent checker `research/ga_ssw/evidence/constrained-ls-integration/emt/summarize_constrained_ls_emt.py`
produced `summary.json`. It verifies 138 frozen periodic pair records for each
LS preparation. Recomputing every stored image-resolved reference distance
from the event's `chart_reference` gives a maximum error of
`4.44e-16 Angstrom`. The stored preparation response equals
`(energy_after-energy_before)/28` exactly in the archived decimal values,
showing the all-N denominator used by the response measurement. For every
stored stage coordinate, `chart_reference` plus its ten active-atom
displacement blocks matches a raw calls ledger geometry at zero recorded
coordinate error. The final stage also matches `last_work` at zero recorded
error.

For each stage that records `true_energy`, the checker finds the identical raw
physical call geometry and energy in `calls.jsonl` (maximum recorded energy
difference zero). Thus the early comparison is physically evaluated E, rather
than the E+LS objective. The masked arm has zero stored direction component
outside coordinates 27--29. The LS second outer step has five converged
Gaussian stages followed by an `evaluation_failed` biased-quench stage; its
1000-call search cap is retained as censored cost rather than a landing.

The plain arm is a numerical default-preservation/interface control. The LS
and masked arms demonstrate the preparation, frozen-pair, full-N response,
physical-energy, constrained-chart, and failure-recording pathways on EMT.
They do not establish LS effectiveness, a new basin, or a physical Cu/TYPE4
policy. No TYPE4 LS run is justified without explicit bond-energy, bond-length,
and target provenance.


## Actual2D slab and final regression

The periodic pair component now supports1D/2D/3D PBC while requiring every
nonperiodic image component to be zero. Joint-cell LS retains its explicit
full3D boundary. The actual2D Cu28 LS/adatom-mode two-step trajectory completed
954 search requests plus3 independent physical endpoint checks. All954
coordinates, cells, energies and forces exactly match the previously archived
full3D-vacuum arm; the138-pair lists also match and all z images are zero.
This adds957EF, without providing another independent efficacy seed.
Evidence: `evidence/constrained-ls-partial-pbc/summary.json` and its comparator.

The ordinary ls=None Cu28 one-step default was evaluated before, during and
after integration:59EF each, with identical serialized results and calls.
Default-preservation cost177EF; three full3D contract arms2081EF; partial-PBC
control957EF. Listed constrained-LS integration cost3215EF, excluding unit
test evaluations. The final standalone/research regression passed369 tests,
with1 skip in the reference-compatible user-site environment. This verifies
code/interfaces, not molecular or material generalization.
