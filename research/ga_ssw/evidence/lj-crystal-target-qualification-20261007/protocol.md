# Qualify the LJ crystal target before a VC search

Question: can the existing ASE truncated-shifted LJ bulk backend legitimately
use the 2014 VC-SSW paper's hcp target? This is model/reference qualification,
not an optimizer or global search experiment. A C60 direction transfer is
running independently; this cheap CPU qualification does not change it.

Primary sources: Shang et al., PCCP (2014), DOI
[10.1039/c4cp01485e](https://doi.org/10.1039/c4cp01485e), section3.2 and Fig.4;
author full text is archived at
`/home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity/literature/benchmark-sources/vc2014/vc2014-author.txt`.
It specifies epsilon=sigma=1, calls hcp lower than fcc by6.51e-4 epsilon/atom,
and compares atom/cell moves on LJ128/256. It does not specify a cutoff in
the available text. The vacancy-start description is LJ128; do not invent a
127-atom input by deleting a site from an unrelated 128-atom cube.

The [ASE LJ source](https://docs.ase-lib.org/_modules/ase/calculators/lj.html)
shifts energy at its cutoff and defaults to3 sigma. Existing bulk smoke uses
2.7 sigma, not established as the paper's potential. Pártay, Ortner, Bartók,
Pickard and Csányi, *Polytypism in the ground state structure of the
Lennard-Jonesium* (2017), DOI
[10.1039/C7CP02923C](https://doi.org/10.1039/C7CP02923C),
[author preprint](https://arxiv.org/abs/1705.01751), establishes that truncation
can change close-packed phase ordering and even stabilize other stacking
sequences. Thus merely labeling a calculator LJ does not qualify an hcp GM.

Competing explanations: hcp remains below fcc on the existing backend, so
the target's ordering is retained; or truncation changes that ordering, in
which case a missed hcp cannot be attributed to the walker. The cheapest
discriminating observation is independently relaxed fcc/hcp energy per atom
at zero pressure under fixed declared backend variants, before any VC search.

Fixed model variants: epsilon=1eV, sigma=1Angstrom, smooth=False,
rc/sigma=(2.7,3,6,10). The first two are existing/default implementations;
the latter two test the longer-range trend. These are model variants, not
search parameter tuning. Do not select a cutoff solely because it makes hcp
win or claim convergence to the untruncated model from four points.

ASE primitive fcc(1 atom) and hcp(2 atoms) are sufficient for this reference
question. Use initial nearest-neighbor length2**(1/6)sigma and ideal
hcp c/a=sqrt(8/3), then minimize within each crystal family atp=0.
FCC varies its isotropic scale; hcp varies a and c independently, preserving
hexagonal symmetry/fractional coordinates. SciPy BFGS with analytic stress
derivatives is a numerical tool, not the compared algorithm.
For logs of these scales, dE/d(log a)=V*(sigma_xx+sigma_yy) and
dE/d(log c)=V*sigma_zz for hcp; isotropic fcc derivative isV*trace(sigma).
Divide E and derivatives by atom count. No pressure term atp=0.

Qualification: finite E/F/stress, force<=.05eV/Angstrom and
max(abs(stress))<=1e-5eV/Angstrom^3, with a separate fresh calculator.
The tighter stress tolerance here resolves an approximately1e-3eV/atom
phase gap in tiny reference cells; it is not a proposed production SSW
threshold. Record optimizer exit separately from the physical certificate.
Finite differences of the analytic log-scale gradient at the declared
initial state, step1e-4, are an implementation check, not efficacy evidence.

Bound: eight phase/model slots, each at most100 E/F/stress requests including
three (fcc) or five (hcp) derivative-check calls plus terminal evaluation;
one independent cold request per numerically eligible slot. Aggregate<=800
search/reference requests+8cold; CPU-MISC/rush-cpu, one task,10minutes,
cooperative process deadline480sec and hard timeout520sec. No GPU, random seeds, global search, retries or
automatic extension. Save raw ledgers, structures, full E/F/stress,
parameters, versions, actual source paths and the executed script snapshot.

Decision: if ordering changes, freeze the model boundary before interpreting
any LJ crystal target search; do not rank VC variants against the paper's
GM. If ordering is retained, hcp/fcc references can support a clearly labeled
ASE-model target pilot, not paper-exact success-rate reproduction. Even
qualified hcp/fcc endpoints do not establish that either is the GM among all
polytypes. Vacancy source geometry and a suitable bulk search protocol remain
separate readiness gaps, not a reason to launch guessed inputs.
