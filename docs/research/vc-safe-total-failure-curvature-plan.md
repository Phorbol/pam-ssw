# Diagnose the complete VC biased objective while retaining Safe-total

2026-09-11. User explicitly restricted the optimizer mainline to Safe-total;
bias-separated and the new history-transport draft are not active work.
The current flat3N+6 implementation already uses total-gradient secants,
Armijo on the total objective and positive-curvature protection. All8 archived
Fe7C3 Safe-total histories end with the atomic maximum exceeding the cell-block
norm. Most accepted secants pass the safety filter; rejection alone cannot
explain those maxiter outcomes. The input physical Hessians have conditions
26.50 (Fe7C3) and147.72 (CuO); these are not Hessians of later biased stages.

Question: at the four frozen Fe7C3 history10 failed endpoints, is the complete
physical H + frozen LS + all Gaussian objective locally indefinite, flat or
strongly coupled? This is a diagnostic of the actual failed targets, not an
optimizer switch or a physical-minimum certificate.

Select all four already compared cases (all/filter,7/101), retaining the
history10 endpoint, chart reference, strain length, LS, Gaussian terms and
pressure. Compute its projected complete gradient once and verify against
archived final_fresh. Then evaluate full243-dimensional nontranslation
Hessians by central gradient differences at1e-4 and5e-5 in the original chart.
Each step costs486 EFS; two cost972, plus1 reconstruction check per case.
Total cap3892 EFS, one V100/20min,900s internal campaign deadline. Same
float64 MACE-OMAT-0-small model. No start/parameter selection from results.

Record raw/symmetric matrices, skew, spectral step difference, negative mode
counts with the measured step discrepancy as a numerical resolution reference,
atomic/cell participation of eigenmodes and full/atomic/cell block spectra.
Eigenmodes are coordinate-dependent diagnostics of this declared chart. An
indefinite biased Hessian is not itself a bug; actual gradients remain nonzero.
Do not diagnose a physical unstable material from a biased Hessian, or infer a
universal preconditioner from four selected failed targets. Charge failed EFS
and preserve incomplete cases. The next Safe-total change, if any, must follow
these measurements and retain the same energy/gradient/step contract.
