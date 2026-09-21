# LS pair-energy filtering: minimal independent extension

The Fe7C3 LS paper (Guan, Shang and Liu, JCTC 2024,
DOI 10.1021/acs.jctc.4c01081, section3.3 and SI7.7) applies penalties to
Fe–C/C–C, and the supplied input filters Fe–Fe. Static evidence is in
native-ls-bond-filter.md. Missing capability: the independent LSSettings
entrypoints could not express a zero-contribution species pair without deleting
it or inventing a positive bond-energy/cutoff value.

Use one sparse `energy_filter` mapping; omitted pairs have dimensionless factor1,
Fe–Fe receives0. Keep the explicit positive base energy/cutoff tables and every
geometrically selected image pair. A filter changes energy, force and stress
through the same analytic strength; it does not change the degrees of freedom,
neighbor distances, atom count, or physical calculator. Store an immutable
canonical mapping in the frozen potential and preserve it on response rebuild.

For the independent paper controller define effective weight Wp=Bp*fp.
Initialize Ap=initial_fraction*Wp. The existing response computes
Tnext=sum(Ap)-N*eta*((Eafter-Ebefore)/N-target), with all physical atoms in N,
and rebuilds Ap,next=Tnext*Wp,next/sum(Wp,next). An all-zero eligible weight sum
is an explicit unsupported LS initialization/rebuild, not a fallback. The
geometric pair list includes zero-weight entries. This controller has no explicit
Nb denominator: it is not the native multiplicative table controller, whose Nb
counts even zero-filter candidates. Combining a native-derived lookup with this
paper response is an explicitly independent experiment, not binary trace parity.

No search threshold, reward, clipping, or automatic parameter optimization is
added. Zero/one chemical selection has source evidence; arbitrary nonnegative
multipliers are user inputs, not recommended fitted defaults. Default absence
must preserve physical trajectories and costs. A mask that disappears on the
second step or changes cell-force consistency invalidates this implementation.

Validation: numeric/API checks cover immutable storage, zero contribution with
retained pairs, periodic images, response persistence and all three public
fixed/VC/constrained entrypoints. The pre/post Cu4/EMT default trace checks
regression only. Real Fe7C3-80/MACE filtered versus unfiltered controls use the
same published target and explicitly archived release lookup, seeds7/101,
2steps and2000 totalEFS perarm (including fresh checks). The previously frozen
plain joint arms provide the no-LS baseline. Report every failed/censored step;
no default promotion based on this small material-specific comparison.
