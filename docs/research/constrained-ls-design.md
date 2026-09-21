# LS on a fixed-atom manifold

2026-09-11. Scope: an optional composition missing from the independent ASE
implementation, not a new escape strategy or a proposed default.

For active-coordinate injection J, R(q)=R0+Jq and fixed cell, the search
objective is E(R(q))+P_LS(R(q))+sum(B_j(q)). Its exact gradient is
J^T[grad E+grad P_LS]+sum(grad B_j). An active-fixed pair therefore contributes
to the active force even though its fixed endpoint cannot move. Removing fixed
neighbors would change this objective and the strength normalization. Native
parser/pair-gate evidence agrees that ordinary zero-valued fixed atoms remain
in the pair list; see native-ls-fixed-atom-audit.md. Negative native sentinel
masks are a separate unsupported contract, not inferred from ASE FixAtoms.

The existing paper pair function and response law are reused, with explicit
LSSettings tables/target. Source: Guan, Shang and Liu, JCTC 2024,
https://doi.org/10.1021/acs.jctc.4c01081, sections2.3–2.4, eqs11–15;
local full text literature/215.txt and author publication list
https://zpliu.fudan.edu.cn/publication/list.htm (entry215).
The fixed-manifold combination is our derivation, not claimed as the paper's
surface benchmark. Initial fraction0.03, xi0.2 and response rate1.8 retain
paper provenance, not evidence of universal optima. Target and pair tables
remain explicit. Response normalization uses all N atoms, as the existing
paper implementation; Nactive is not silently substituted.

Within each outer step, build/freeze the pair list at the selected true
minimum, prequench only E+P_LS on the same physical manifold, measure the true
energy change, then rotate and climb on the frozen soft surface. A separate
mode exclusion mask restricts only the direction problem. Biased relaxation
still uses every physical active coordinate. True-energy early release tests,
final quench, certificates and MC all use E without either artificial term.
After successful preparation, update the response for the next selected true
state even when a later proposal fails, matching paper_reference's existing
lifecycle. Negative strength, empty bonds and preparation errors stop explicitly;
no amplitude clipping, automatic retry or adjusted acceptance rule is added.

Any periodic direction now uses image-resolved FrozenPeriodicBondSoftening;
nonperiodic axes require exactly zero image shifts. The fixed-cell image sum
is valid for1D/2D/3D PBC with a finite positive-volume cell, without new cutoffs.
Isolated input uses FrozenBondSoftening. Joint-cell softening retains its own
full3D domain restriction. Every oracle receives raw constraint-free Atoms;
the reduced map owns fixed positions. Fixed force components remain visible
in evidence. The former partial-PBC restriction was removed after the same
Cu28 trajectory matched its full3D large-vacuum representation on every call.

Implementation reuses the reduced walker via a single internal LS runtime.
ls=None must preserve its numerical and random stream exactly. Validation:
pre-fix failing interface checks; exact Cu28/EMT before/after default calls;
whole explicit LS/mode-mask Cu surface trajectory and independent physical
endpoint checks; costs include initialization, prequench, failures and fresh
checks. Small Cu/EMT checks establish composition only. TYPE4 application waits
for explicit pair/target provenance; neither component tests nor native raw
lookup values establish a physically useful LS surface-search policy.
