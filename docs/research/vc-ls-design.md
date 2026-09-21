# Joint VC-LS composition: independently derived contract

Goal: keep the full LS potential energy, atomic force and cell stress consistent
while a joint VC proposal changes the lattice. Not native VC-LS parity.

Use the same frozen periodic bond labels and reference lengths as fixed-cell LS.
For d=Rj+S L-Ri and P=A exp[-(r-r0)/(xi*r0)], atomic forces are existing LS forces
and tensile-positive affine stress is -sum(d outer outward)/V, including self
images. Feed the complete E/F/stress into the exact log-strain chain rule. Gaussian
terms and LS both disappear during final true quench and MC; no pressure is added
twice. Response uses physical E before/after a fixed-cell soft-only atomic
prequench, so it stays in eV/atom and does not conflate volume work with learning.

Keep the soft reference, image shifts and LS strengths fixed through one proposal;
rebuild only after final selected state, allowing its cell to change. Prequench is
fixed-cell by explicit design. This composition does not assert that it matches
an untraced native LS-cell schedule. No new empirical parameters are introduced.

Files: vc_softening.py (variable-cell frozen term plus stress), vc_reference.py
(optional ls argument and lifecycle). Tests: all generalized derivatives at finite
nonzero log strain; Cu EMT true certificates and full request ledger; no-LS VC
regressions unchanged. Full-covalent/material efficacy remains to be evaluated.
