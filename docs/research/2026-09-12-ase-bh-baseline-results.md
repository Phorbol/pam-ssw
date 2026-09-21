# ASE BasinHopping versus staged Ritz SSW (2026-09-12)

Six bounded CPU BH arms completed their attempted runs: four metal arms hit
6000 search requests; both GFN2 molecular arms terminated on one SCF failure.
No retry or hidden recovery was added. The comparison does not show a material
best-energy advantage for SSW on these initial states. More BH local quenches
are not evidence of more distinct minima.

Evidence: `research/ga_ssw/evidence/ase-basin-hopping-baseline-20260912-v2/`;
`root-final-audit.json` checks sequential paid ledgers, each saved quench endpoint
against its final paid evaluation, every additional ASE endpoint-energy callback,
unfinished-local cost and fresh checks. v1 failed during import before PES;
its preparation directory is retained and explicitly marked.

The controller is the installed unmodified ASE3.26 `ase.optimize.basin.BasinHopping`;
its exact source is copied into `ase-source/`. Local optimization is the same
frozen Safe-total implementation, history10,400 steps,fmax .01 eV/Angstrom.
Inputs are frozen extxyz representations from the earlier three-system SSW
campaign, not new optimized guesses. Extxyz is a serialized coordinate
representation; bitwise identity to the original in-memory generator is not
claimed (initial energies agree within floating-point precision).
T150 K and the6000-request ceiling match the SSW controls. BH's dr=.5 Angstrom
is the [ASE documentation example](https://ase.gitlab.io/ase/ase/optimize.html),
not a fitted best choice or equivalent to SSW's .1-Angstrom Gaussian width.
Global NumPy RandomState and SSW default_rng are distinct random streams;
same integer seed does not couple individual proposals. BH keeps its native
pre-quench proposal coordinate as ro, while Eo is the locally quenched energy.
The post-quench get_value callback is paid and explicitly counted.

## Matched realized search prefix

Each comparison uses min(BH actual requests,6000), including all initialization,
failed work and callback costs. Every returned BH converged quench, including
initial and MC-rejected candidates, is independently cold evaluated. SSW rows
reuse the frozen two-stage Ritz result and its fresh certificates, sliced at
that exact request prefix. The old runner's fixed100/dimer configuration is
superseded by its recorded staged Ritz research wrapper; it is not a fixed100
rotation comparison.

| System | Seed | Common search requests | BH / SSW qualified landings | BH best change (eV) | SSW best change (eV) |
|---|---:|---:|---:|---:|---:|
| cu13 | 11 | 6000 | 90 / 10 | 0.00000000 | 0.00000000 |
| cu13 | 29 | 6000 | 87 / 10 | 0.00000000 | 0.00000000 |
| cu31_fixed | 11 | 6000 | 72 / 11 | -0.00001199 | -0.00001102 |
| cu31_fixed | 29 | 6000 | 69 / 11 | 0.00000000 | -0.00001344 |
| bicyclobutane | 11 | 3146 | 32 / 5 | -0.42272856 | -0.42272587 |
| bicyclobutane | 29 | 3390 | 38 / 5 | -0.56609143 | -0.56604958 |

BH totals **30536 search +388 fresh =30924**. All388 fresh checks pass. Four
metal arms each consume6000; molecule seed11/29 consume3146/3390 before
`SCF not converged in 250 cycles`. Each failed molecular local attempt spends
one E/F request; all earlier local results survive. These are real backend
failures, unlike the wall-cap labels in the LS campaign. Safe-total's completed
local result list contains no nonconverged return; that does not erase the
exceptional partial attempt. No calculator parameter or proposal retry was changed.

Cu13 began in a stable low-energy structure and neither method improved it;
Cu31 improvements are at the1e-5 eV scale. The molecular best differences
between methods are also small at these prefixes. These near-degenerate
energies must not be used to rank algorithms. The important retained evidence
is successful independent ASE-controller integration and its actual costs and
failures. These easy metal starts alone cannot resolve multifunnel efficiency;
no universal SSW or BH superiority is claimed from three inputs and two seeds.

Decision: keep the mature baseline available with its existing failure behavior.
Do not tune dr to this dataset or increase SSW complexity to win the comparison.
The immediate kernel follow-up targets the independently diagnosed Ritz
termination issue, with unchanged tolerances and budgets, then repeats paired
end-to-end tests across the metal/periodic/molecular inputs.
