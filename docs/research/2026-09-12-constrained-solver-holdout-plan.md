# Constrained direction solver diagnostic

This prepared diagnostic compares the existing generalized dimer with the
reviewed `ritz`, `dimer`, and `broyden-euclidean` selectors on Cu111 and Al111
EMT supported adatoms. Cu uses the existing ASE `fcc111(2,2,3,a=3.6 Angstrom)`
case; Al uses `a=4.05 Angstrom` from ASE `reference_states[13]['a']`. Both use
an fcc adatom at height 2 Angstrom, a constructed geometry rather than an
equilibrium claim, and the bottom two tagged layers fixed. The Al case is an
EMT model comparison, not a first-principles material claim.

Each arm uses seed 3 and 11, two outer steps, a 6000-request and 60-second CPU
cap, `gradient_tol=0.1` interpreted as the whole active Cartesian gradient L2
norm, outer `fmax=0.01`, 200 quench steps, 100 rotation HVP calls, 14
Gaussians, width 0.2, rotation bias 100, and temperature 300 K. The runner records
failed and incomplete arms, every surface request, fixed coordinates and cell,
fresh active/full force certificates, adatom height, and minimum distances. The
active force certificate is compared with the outer threshold; full raw force
is reported separately and may exceed it.

`prepare_constrained_solver_holdout.py` snapshots `pamssw` before importing it
and re-executes against that snapshot.  It currently refuses to run if the
shared constrained adapter has not added the explicit `rotation_solver` field;
this is intentional and prevents silently testing only the old generalized
dimer path.
