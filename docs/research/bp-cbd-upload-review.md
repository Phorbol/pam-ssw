# Newly uploaded BP-CBD paper: contract and implementation comparison

Date: 2026-09-09. Read-only algorithm audit; no implementation change or experiment.

Source: Cheng Shang and Zhi-Pan Liu, *Constrained Broyden Dimer Method with Bias Potential for Exploring Potential Energy Surface of Multistep Reaction Process*, JCTC **8**, 2215–2222 (2012), DOI [10.1021/ct300250h](https://doi.org/10.1021/ct300250h).

User file: `/home/gengjianrui/bin/pam-ssw/BP-CBD.pdf`. Earlier author copy: `/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/literature/65.pdf`. Both are eight-page versions of the same article. Their PDF bytes/metadata differ (new copy includes download watermark); checked methods, equations and quoted parameter passages agree. No new article or revised algorithm was discovered. Text extracted using `pdftotext -layout`; equations and Scheme 1 visually checked on printed pages 2217–2218, PDF pages 3–4. Full extraction is `/tmp/bp-cbd-upload.txt`; temporary images `/tmp/bp-cbd-page-3.png`, `/tmp/bp-cbd-page-4.png`.

## 1. Rotation target versus numerical solver

Section 2.1, equations 4–9, adds a quadratic potential to the dimer endpoint, not an isotropic basin-filling Gaussian:

\[
V_N=-\frac a2[(R_1-R_0)\cdot N_{\rm init}]^2,
\quad F_N=a\Delta R(N_t\cdot N_{\rm init})N_{\rm init}.
\]

The force-difference rotation is

\[
\Delta F_\perp=2(I-N_tN_t^T)(F_1+F_N-F_0),
\]

and real versus rotation curvature are distinct:

\[
C_e=(F_0-F_1)\cdot N_t/\Delta R,
\quad C_{\rm rot}=C_e-a(N_t\cdot N_{\rm init})^2.
\]

Thus the local linearized rotation residual is the tangent gradient of the Rayleigh quotient of `H - a Ninit Ninit.T`. A converged residual is an eigenvector condition, not by itself a certificate of the lowest eigenvalue; the original Broyden root iteration can select a different stationary direction from a Ritz lowest-subspace-eigenvalue solve. Standard dimer rotation can address this operator without copying Broyden history. Standard *unbiased* dimer rotation solves a different target, and a full dimer TS translator solves a different outer problem. Replacing Broyden with another rotation solver preserves the stated operator but does not guarantee identical finite-difference trajectory, stopping behavior, or selected mode.

Current `direction.py:paper_biased_direction` implements this quadratic endpoint force and uses a one-sided secant with a Ritz solver. This is an explicit numerical substitution. Its returned `curvature` is the biased operator curvature, not `Ce`. A BP-CBD TS handoff cannot use that value as proof of negative curvature on the real PES.

## 2. Corrections and details that prior explanations omitted

1. **BP-CBD has an initial limited unbiased sweep.** Printed p2217 says the chemically specified raw `N0` is first refined by a few unbiased rotations into `Ninit`, with a loose force threshold, e.g. ten times the ordinary rotation threshold. Figure 1 uses two such steps. Current SSW `paper_reference.py` samples a random anchor and directly calls biased refinement; it does not implement this BP-CBD reaction-direction preparation. This is a difference from BP-CBD, not automatically a bug in the separate SSW2013 contract. Do not import the BP-CBD preliminary sweep into SSW without checking that paper's own procedure.

2. **The paper provides a scale argument for `a`.** It recommends at least the initial real curvature `Ce`, even writing `a=Ce`. Therefore saying there is no paper guidance at all is too strong. There is no universal numerical `a`, however. At the initial direction, the equation gives `Crot=Ce-a`: equality gives zero, not strictly negative curvature. The prose refers to subsequent rotations; `a=Ce` does not prove a strict negative eigenvalue or an angular bound. For physical Hessian positive definite, the exact rank-one negative-eigenvalue condition is `a > 1/(Ninit.T @ inv(H) @ Ninit)`; this is a project mathematical observation, not a parameter prescription in the paper. The reported typical angle below 30° is empirical, not a hard constraint.

3. **Rotation tolerances have different dimensions.** BP-CBD uses `|ΔF_perp| < τ1`, with τ1=0.1 eV/Å in the Baker tests. Current `rotation_tol` is a Hessian residual in eV/Å². In the locally linearized one-sided contract, `ΔF_perp = -2 ΔR (Hrot n - Crot n)`. Only with the same norm and finite-difference convention does `τ1/(2 ΔR)` translate between the two. Copying the numerical value 0.1 would not reproduce the same stopping criterion.

4. **Do not mix three lengths/forces.** The introductory dimer half-separation example is `ΔR=0.005 Å`; the biased-translation Gaussian width example is `ds=0.1 Å`; the forward force target is `FR0·Ni=0.1`, dimensionally eV/Å. They are independent parameters. Width is also the displacement to the Gaussian inflection point in this paper. None is a proven universal optimum.

5. **The 87° rule is not in this article.** It remains evidence from the uploaded later ELF. BP-CBD2012 specifies the forward-force target; they are different height policies and need separate attribution.

## 3. Gaussian height and accumulated-force consistency

Section 2.2, equations 12–18, uses

\[
V_G=\sum_i w_i\exp[-((R-R_i)\cdot N_i)^2/(2ds^2)].
\]

Each deposited term saves its own center and direction. The forces are derivatives of the entire sum, not just the newest term. At `Rk + ds Nk`, the newest term has forward force `wk exp(-1/2)/ds`. Therefore the paper's target implies

\[
w_k=[f_\star-F_{\rm background}\cdot N_k]ds\,e^{1/2},
\quad f_\star=0.1\ {\rm eV/Å},
\]

where background includes true PES plus all previous Gaussians. Current `paper_reference.py` computes this using the displaced modified calculator; `gaussian.py:ProjectedGaussian` saves each immutable center/direction/width/height. With LS enabled, the background also includes LS; that is the combined implementation choice, not a statement in this 2012 paper.

The paper does not specify the Python implementation's explicit failure on nonpositive resulting height. That is a declared supported-domain restriction, not faithfully recovered native behavior. BP-CBD instead checks whether the true force already points past the TS and can change phase/direction.

## 4. BP-CBD completion requires more than SSW escape

Section 2.3 and Scheme 1 are clear: detect `Ce<0` during biased rotation, or `Freal·Ni>0` during biased translation; verify a negative curvature using **unbiased** rotation; then use CBD on the real PES to locate TS. If an overshoot lands where the unbiased curvature is positive, reverse the current direction and climb back from the other basin. Finally displace from TS and L-BFGS-relax to the product; another elementary reaction requires another supplied reaction direction.

Current SSW does not implement these TS verification, reversal, CBD-to-TS and product-connection stages. Its end-of-escape real-PES quench plus Metropolis is an SSW global-search operation, not a BP-CBD pathway reproduction. Negative biased curvature, successful local optimization, or a Gaussian limit are not TS certificates.

## 5. Consequence for project plan

Keep three independently named claims: (a) paper operator implementation, (b) specific ELF behavior reproduction, and (c) physical search efficacy. A standard-dimer backend for the biased rotation is a justified implementation option and need not wait for full BRZERO4 recovery. Compare it with Ritz and original rotation under equal `R`, anchor, `a`, secant length, force accuracy and *dimensionally matched* stopping criterion. Then compare complete SSW escape on real systems under equal total force budgets. The newly inspected BP-CBD paper supplies no justification for treating full CBD TS translation as a prerequisite for GA-SSW, or for adding its reaction-specific handoff logic to global exploration by default.

The BP-CBD Supporting Information is potentially useful because the main text states it includes Fortran for reaction-direction generation. It is useful for a later explicit BP-CBD path API; it is not a blocker for the SSW direction solver.
