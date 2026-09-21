# Periodic LS implementation plan

Goal: complete fixed-cell periodic LS prequench, climbing, response update and
bias-free final quench with a well-defined extensive periodic pair potential.
Architecture: retain isolated/MIC legacy class unchanged; add an explicit frozen
periodic-image bond class and reuse the existing LS lifecycle. User authorized
implementation and continued core work. No changes to frozen material gate.

Scientific contract: at a true minimum enumerate (i,j,S) with ASE neighbor_list
and explicit element cutoff tables. Identify (i,j,S) and (j,i,-S), count each
undirected periodic bond once per cell. Include nonzero self images, exclude
zero self images. Freeze image shifts and reference lengths throughout each
climb; evaluate d=Rj+S L-Ri on unwrapped positions. Thus derivatives are smooth
away from zero bond distances, no dynamic MIC switching, and repeated cells
preserve energy per atom. Pair potential and true energy-response formula remain
existing LS expressions. This is an independent periodic extension; uploaded
native MIC behavior is a separate contract, not exact parity.

Scope: fixed full PBC, no constraints, no stress-biased VC-LS claim. Atom labels,
cell and PBC fixed per step; re-enumerate only at the next selected true minimum.
No new search parameter: use supplied bond cutoffs/energies, xi and response target.
Periodic bond topology is a frozen labeled-cover representation; wrapping an atom
without transforming image shifts changes that representation and is disallowed
inside a climb, matching the existing unwrapped Gaussian coordinate contract.

Files: new pamssw/standalone/periodic_softening.py; adapt ls_cycle validation and
LSResponseState reconstruction dispatch; select periodic class in paper_reference.
Tests: periodic Cu conventional cell must include image-multiple neighbors;
repeat-cell energy/force equivalence; all-coordinate force finite differences;
full Cu EMT LS pipeline with explicit synthetic coefficients, independent true
force checks, response/cost and unchanged cell. These are implementation/E2E
checks, not evidence of covalent-surface LS efficiency. Gate results unaffected.

- [ ] Write periodic bond/derivative and lifecycle tests; observe missing module failure.
- [ ] Implement frozen periodic-image pair class with exact E/F.
- [ ] Wire class through preparation, response reconstruction and full run_ls_ssw.
- [ ] Run targeted real-system and existing LS regressions; document limits.
