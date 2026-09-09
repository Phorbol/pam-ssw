# SSW2013 paper contract and numerical substitution

Primary source: Cheng Shang and Zhi-Pan Liu, *Stochastic Surface Walking Method for Structure Prediction and Pathway Searching*, JCTC **9** (2013), 1838–1845, DOI [10.1021/ct301010b](https://doi.org/10.1021/ct301010b). Author [publication index](https://zpliu.fudan.edu.cn/publication/list.htm), [author-hosted PDF](http://www.lasphub.com/publication/74.pdf). Retrieved 2026-09-09 to `/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/literature/74.pdf`, extracted `74.txt`. Full text is retained in the external research archive, not redistributed in this repository.

Page 1839, equations 3–6: endpoint R1=R0+dr*n; curvature=(F0-F1)·n/dr; rotation potential is -a*((R1-R0)·N0)^2/2. a is a curvature, not Gaussian height. The paper describes dimer rotation as a numerical way of identifying a Hessian eigenvector. Page 1840 requires a sufficiently large to retain information about the initial random direction; no universal numeric a is specified.

Page 1840, equation 7 and Overall Algorithm:

1. At the accepted minimum choose the initial random N0; set n=1.
2. At each current modified-PES minimum refine the SAME original N0 by biased rotation.
3. Deposit Gaussian n at the current point, projected along the refined Nn, with width ds; displace by ds*Nn.
4. Minimize the modified surface by L-BFGS, producing Rn; increment n.
5. Stop if incremented n>H (therefore at most H Gaussian terms), or the energy of Rn is below the initial minimum; otherwise repeat 2–4.
6. Remove all biases, minimize the real surface, then apply ordinary Metropolis selection.

The text does not put a surface label on the step-5 energy. Interpreting it as real E(Rn) is explicit implementation interpretation; this is not a recovered ELF condition. The paper gives typical ds=0.2–0.6 Angstrom and used H=14 unless otherwise stated; neither is a universal optimum.

The initial direction distribution in equations 1–2 (page 1839) mixes a random global velocity direction and a non-neighbor atom-pair bond formation vector, lambda uniformly selected from 0.1–1.5 in this work; pairs require distance>3 Angstrom. A generic caller-provided anchor does not reproduce this random generator automatically.

Height rule is not explicitly parameterized in the 2013 text; it refers to BP-CBD for biased translation. Cheng Shang and Zhi-Pan Liu, JCTC **8** (2012),2215–2222, DOI [10.1021/ct300250h](https://doi.org/10.1021/ct300250h), [author PDF](http://www.lasphub.com/publication/65.pdf), archived alongside as `65.pdf`/`65.txt`: page 2218 section 2.2 chooses the new Gaussian weight so the total force projection at Rk+ds*Nk equals 0.1. This is distinct from the recovered later ELF's 87-degree angle schedule. Page 2217 section 2.1 states a should be at least the real curvature along the initial direction; its example a=Ce is not a universal robust bound for all subsequent directions.

`direction.paper_biased_direction` implements the quadratic biased surface with the original anchor, one-sided force differences and a bounded symmetric Ritz numerical solve. This substitutes the rotation solver; it does not reproduce native persistent Broyden history, native stopping norm, 40-degree angular control or retry schedule. Its `tol` is a curvature residual tolerance, and it returns biased curvature. It uses one center evaluation plus one endpoint per HVP, with one HVP reserved for direct final residual. Tests verify an analytic Hessian's shifted eigenvector and call accounting, not scientific search efficacy. Fixed-cell Cartesian coordinates only; callers must preserve the cell.
