# GA descriptor row ordering: measured defect and minimal design

## Problem and evidence

Goal: independent ASE GA-SSW whose geometry identity and routing do not depend
on relabeling identical atoms. Preserve the uploaded Java reconstruction for
explicit comparison; do not invent corrections to the ambiguous printed DCCD
Eq. 1 or substitute a new descriptor family during this fix.

CPU1447868 passed the three S1 checks and the strengthened LS rejection test
(4 passed). Its separate descriptor audit found three arbitrary atom-reversal
projection differences above the existing water15 tolerance 0.0001. Arbitrary
reversal changes the OHH input order, so this alone was not a valid TYPE3
lineage example.

CPU1447885 then checked two permutations preserving ordered species and the
contiguous OHH groups: reversing the order of whole molecules, and swapping
the equivalent hydrogens inside each molecule. For the same three saved
nonperiodic water15 structures:

| Permutation | Frame 0 drift | Frame 1 drift | Frame 2 drift |
|---|---:|---:|---:|
| Whole molecules | 0.001492500515 | 0.000725113311 | 0.001029011874 |
| Equivalent H atoms | 0.000735812087 | 0.000228284313 | 0.001214592105 |

All six exceed the unchanged 0.0001 identity threshold. With the experimental
complete fingerprint row ordering applied to candidates AND frozen references,
all six projection drifts are zero. These are offline representation checks
on real saved geometries, not six independent search runs or evidence of
global-search improvement. No PES evaluation was used in either permutation
audit; the LS regression separately uses bounded Cu13/EMT evaluations.

Inputs, explicit permutations and outputs:
`research/ga_ssw/evidence/s1-cutoff-contract-20260922/`
(`legal-permutation-manifest.json`, `legal-permutation-1447885.json` and
`permutation-1447868.json`). The generic audit command is
`python -m research.ga_ssw.audit_legacy_permutation_invariance --manifest MANIFEST --output RESULT`.
The existing archived bond lengths, range, weights and tolerance are unchanged.

## Root cause and limits

The active Java and Python legacy descriptor sort atom rows by counts only.
When these tie but continuous distances differ, stable sorting retains input
order and the subsequent rowwise comparison changes under relabeling.
Completing the key with all existing continuous components removes that
particular ordering dependence without changing the radial formula, neighbor
graph, weights or physical structure.

Complete ordering does not make the descriptor injective, certify distinct
chemical minima, recover paper DCCD, or prove faster GA search. Numerical
roundoff in underlying sums is not the research target. Keep the existing
structural matcher option and distinguish approximate projection identity.

## Approved minimal design (user agreement 2026-09-22)

1. Add explicit `descriptor_row_order="legacy_counts"` keyword to
   `run_ga_ssw`, accepting only `legacy_counts` and `full_fingerprint`.
   Existing calls/defaults remain unchanged. Do not add a generic descriptor
   framework or modify periodic/TYPE4 controllers in this change.
2. Use one private complete-row-order helper for both the preflight and every
   landing descriptor. For full mode, reorder copies of supplied references
   with the same key. Never mutate caller reference objects.
3. Keep the existing similarity, partition, weights and projection tolerance.
   This is an independent correction of Java row ordering, not paper-DCCD
   reproduction. It is optional until controlled end-to-end evaluation.
4. Record `full_fingerprint` explicitly in the existing checkpoint scientific
   contract. In legacy mode, leave the prior contract shape unchanged so old
   checkpoints still compare identically. An old legacy checkpoint resumed
   under full mode must fail before PES calls; no silent archive conversion.
   No checkpoint schema version bump or automatic migration is required.
5. Validation: same six legal permutation checks through the actual helper;
   old-default behavior and reference immutability; same-mode pause/resume;
   cross-mode resume rejection before evaluation. Then a bounded matched GA
   lifecycle check using already qualified ASE inputs, reporting archive
   identity separately from force qualification and search performance.

## Alternatives and decision boundary

- Retain legacy-only routing and require a structural matcher: smaller code
  change, but matcher affects identity only and leaves order-dependent region
  routing. It therefore does not fully remove this measured defect.
- Switch the default immediately and rebuild historical projections: fixes
  new runs immediately but changes existing experiments and persistence
  behavior; not recommended without the explicit compatibility transition.
- Replace with MACE features: a separate representation/normalization/model
  contract and inference cost; unnecessarily expands this ordering fix.

This affects public search configuration and persisted scientific identity.
Project AGENTS.md §8 requires discussion of important API/persistence designs.
User approved the minimal fix with the legacy default retained. Implementation and targeted CPU validation are complete. See
[execution and evidence](../../research/ga_ssw/evidence/ga-row-order-20260922/README.md).
The new mode remains optional; this does not establish search efficiency. Q gradient and LS contract checks already completed do
not depend on this decision.
