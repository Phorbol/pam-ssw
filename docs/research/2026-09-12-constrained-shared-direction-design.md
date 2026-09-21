# Share fixed-substrate direction solvers without changing the search policy

The existing fixed-substrate implementation already reconstructs full physical
structures from active Cartesian displacements. For the column selector S,
R(q)=R_ref+S q, S^T S=I, g_q=S^T g_R and H_q=S^T H_R S. Fixed coordinates
therefore remain unchanged at every oracle request. No global translation or
rotation projection is valid merely because part of this supported structure
can move.

The bounded change adds a direction callback to the existing reduced lifecycle.
Only the constrained caller supplies it. Its temporary atom-shaped object is
a numerical container for active coordinates in Angstrom; it has no physical
calculator. Every energy/force callback reconstructs the full substrate and
adsorbate through the existing chart, then returns the active gradient.
This reuses the public Ritz, plane dimer, Euclidean Broyden and staged solvers
without copying their algorithms or introducing new search parameters.

The existing generalized dimer remains the default. Gaussian deposition,
softening, Safe-total relaxation, true landing, MC and coordinate reconstruction
are unchanged. Direction-only exclusions restrict the Hessian problem before
solving and lift its vector afterward; they do not freeze extra atoms during
relaxation. A fixed rotation bias and an adaptive presweep cannot be silently
selected together. Both stages share the existing endpoint budget.

This is not yet a unified constrained SSW/LS/GA persistence or archive API.
Next evidence is an eight-arm Cu/Al(111) adatom diagnostic (one seed per case,
four solvers) with identical geometry, relaxation criteria and bounded costs.
Qualification requires active force tolerance and exact fixed coordinates/cell;
the full raw substrate force is reported independently. These are development
checks, not a global-search efficiency or materials prediction claim.
