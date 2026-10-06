# Qualification decision — 2026-10-07

The new author C60 isomer #3 is eligible for a later fixed local SSW/LS
comparison on MH-1/omol. This is a source/model qualification, not an LS
result, a Hessian certificate, or random-start global-search success.

Job1664282/source024486c completed in9 scheduler seconds. Ordinary initial
quench used11 E/F requests, plus one independent candidate and one reference
check:13 requests/13 actual Calculator evaluations. The root agent checked
all three ledgers for sequential IDs, matching starts/completions, no failed
or denied evaluations, and closed total cost. Actual core imports were
verified under the run's frozen source. Raw artifacts are
`qualification-1664282/` in this directory.

The endpoint energy is−62213.55121014446eV; its independent force norm is
.013206795839007137eV/Angstrom. Relative to the freshly checked unchanged Ih
reference it lies1.8421606461561169eV higher. It remains a connected fullerene
cage and graph-isomorphic to the source but not Ih at all three predeclared
bond cutoffs1.64/1.70/1.80Angstrom. Thus ordinary quench has not already solved
the intended repair target. Energy repetition within the same model checks
execution consistency; these digits are not a claim about physical accuracy.

Failed job1664262/source465ae47 is retained separately: missing ledger-parent
directory caused two attempted wrapper requests and zero actual evaluations.
It supplies no model-eligibility evidence. The focused correction was
reproduced with a dummy Calculator before the new run; no input selection or
algorithm parameter change occurred.

Next: define a paired local SSW/paper-LS protocol using this exact saved
endpoint, source-linked LS constants and identical remaining configuration.
Report qualified target discovery and all costs, including softened
prequench. A short negative result cannot refute the paper's long random-start
efficiency claim. Do not replace the current random C60 direction panel, tune
parameters from this input, or upgrade LS to a default. The current protocol
still allows no search; the later protocol must explicitly set its bound.
