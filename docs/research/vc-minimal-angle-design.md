# Joint VC integration of the existing 87-degree height criterion

Problem: fixed-cell run_ssw exposes MinimalAngleHeightPolicy, but run_vc_ssw
only supports the independent fixed-forward-force height. This blocks an
isolated comparison of an existing native-derived core criterion on crystals.
It does not justify changing the default or releasing unconverged biased points.

Use the already implemented analytic policy, with no new fitted parameters.
In the explicit scaled symmetric-log-strain chart q, let F=-d(H+LS+oldGaussian)/dq,
n a unit direction in this same metric, Fparallel=F.n and
Fperp=F-Fparallel*n. At q=center+width*n the required positive Gaussian height
is W=(cot(87deg)*norm(Fperp)-Fparallel)*width*exp(1/2).
This is the same existing equation, including its cancellation and nonpositive
height failures. The angle depends on the chosen chart metric; this is not
native lattice-coordinate or full native height-history parity.

Implementation: optional height_policy accepts MinimalAngleHeightPolicy only;
None keeps the exact current code path. Reuse the already paid displaced
background evaluation, negate its joint gradient, and pass immutable history.
Save preparation inputs/result and freeze returned Gaussians through quench.
An already satisfied condition stops with nonpositive_height as in the fixed
walker; no automatic zero-height continuation or physical-landing bypass.
The numerical source-derived growth/history policy remains outside this change.

Verification: fail-first public entrypoint test, analytic height/result force
angle in a mixed atomic/strain direction, full biased E/g consistency using
existing conservative Gaussian formula, correct zero extra EFS count, default
trajectory equality, and a source-frozen Fe7C3-80 two-seed matched control under
the same 2000EFS/480s cap. LS and optimizer-history controls remain separate;
no simultaneous parameter changes and no default promotion from two seeds.
