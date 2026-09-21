# 2014 SSW-crystal: paper/native crosscheck

2026-09-10. Independent reading of Cheng Shang, Xiao-Jie Zhang and Zhi-Pan Liu,
*Stochastic surface walking method for crystal structure and phase transition
pathway prediction*, PCCP **16**, 17845–17856 (2014),
[DOI 10.1039/c4cp01485e](https://doi.org/10.1039/c4cp01485e).
Source: [author-hosted PDF](http://lasphub.com/publication/87.pdf), saved as
`literature/benchmark-sources/vc2014/vc2014-author.pdf` with accompanying text.
PDF pages 2, 4 and 5 were also visually inspected to distinguish printed
formulas/figure labels from text-extraction errors. Page references below are
journal pages, with PDF page in parentheses. No kernel changes or calculations
were made in this review.

## Main conclusion: the published walker uses successive blocks

Fig. 1b, p. 17846 (PDF 2), explicitly depicts a sequence of blue lattice
perturbations with local atom relaxation, followed by a green **fixed-lattice
atomic SSW** segment. Section 2.2 and the eight steps of §2.3 specify:

1. At the current minimum, choose whether this outer move includes atomic SSW.
2. Run the CBD-cell block. Each cycle identifies a soft lattice direction,
   displaces the lattice, then relaxes atomic coordinates at that fixed lattice.
3. Repeat the cell cycle to its prescribed limit.
4. If scheduled, run atomic SSW with that lattice fixed.
5. Remove bias/constraints and relax **both** atomic and lattice degrees of freedom.
6. Apply minimum selection (Metropolis for structure search), then repeat.

This is a coupled cell/atom search, but the coupling does **not** mean one
combined atomic/strain Hessian direction or one Gaussian in a common 3N+6
coordinate metric. The final all-DOF quench is also distinct from the preceding
cell escape. Our independent symmetric log-strain joint walker is thus an
algorithmic extension/alternative, not a faithful implementation of the
published 2014 schedule. This conclusion concerns the paper; the later uploaded
LASP release's entire schedule has not yet been recovered.

## Coordinates and parameters are separate, not interchangeable

Section 2.2, pp. 17847–17848 (PDF 3–4):

| Symbol | Published meaning | Consequence for reproduction |
|---|---|---|
| L | Nine-component 3×3 lattice matrix | Eq. (3) updates L additively; three rotational directions are projected out during CBD rotation |
| N_cell | Normalized lattice direction obtained from a random nine-component vector by CBD rotation | Six non-rotational modes remain; this is not proof of six symmetric **log-strain** coordinates |
| ΔL | Actual lattice displacement magnitude in Eq. (3): L_new = L_old + ΔL N_cell | Paper uses ΔL = 0.15 sqrt(sum_ij L_ij²); it is a length scaled by lattice Frobenius norm |
| dL | Dimer image spacing in Eq. (4) | Example 0.005 Å; a numerical finite-difference length, distinct from ΔL |
| ds | Atomic Gaussian width and atomic initial displacement along N in §2.1, Eqs. (1)–(2) | Example 0.6; not a cell metric or lattice displacement |
| H | Maximum number of atomic Gaussians | Example 10 |
| H_Cell | Maximum CBD-cell cycles | Example 5; not defined here as a cell Gaussian count |
| λ | Outer-step partition between cell-only and cell+atomic-SSW proposals | Example 2, chosen after preliminary trials; not derived as universally optimal |

The cell direction acts on lattice vectors, not an indexed atom displacement.
The atomic SSW direction acts on the atomic Cartesian coordinate vector.
The prose motivates limiting lengths or volume; its printed norm formula does
not by itself impose a hard 15% bound on each lattice-vector length or on volume.
Local fixed-cell atom relaxation is limited to at most 25 steps in the reported
work (§2.2); it is not stated as an exact inner minimization defining a fully
relaxed-ion cell Hessian.

The paper does not explicitly specify the atom remapping convention for both
dimer images in enough implementation detail to claim native parity. Affine
movement at fixed fractional positions is the natural interpretation associated
with a stress-derived lattice gradient, but must be documented as an
implementation interpretation until source/SI or a native coordinate oracle
confirms it. Neither our user-supplied metric length nor native `ds_cell=1.6`
can be identified with the printed ΔL formula from naming alone.

## Cell gradient and an important printed sign ambiguity

On p. 17848 (PDF 4), Eq. (6) defines the dotted L as an **enthalpy derivative**:

```
dotL_ij = ∂H/∂L_ij = -Ω sum_k (σ_ik - p_ext) [(L^t)^(-1)]_kj .
```

The minus sign, inverse transpose, volume factor and multiplication order are
visible in the PDF. The printed scalar pressure term omits a Kronecker delta.
For physical isotropic pressure one would interpret it as p_ext δ_ik (p I),
not a scalar subtracted from all nine stress entries. The paper sets external
pressure to zero throughout this work, so its examples do not settle that
notation issue. Stress sign and row/column lattice conventions must be
translated explicitly before using ASE tensile-positive stress; the displayed
minus sign alone does not imply our ASE gradient has the wrong sign.

Equations (4), (5), and (7) are printed as:

```
L1 = L0 + dL N
Frot = 2(dotL0 - dotL1) - 2[(dotL0 - dotL1)·N] N
Ccell = (dotL0 - dotL1)·N / dL
```

If dotL is indeed ∂H/∂L as Eq. (6) says, a Taylor expansion gives
Ccell = -Nᵀ Hess(H) N + O(dL). It has the **opposite sign** to the conventional
Rayleigh curvature, whereas the paper describes positive curvatures for stable
quartz modes. This is a printed convention inconsistency or omitted force
convention, not an OCR artifact. The rotational-force expression itself is
compatible with minus the tangent derivative of the Rayleigh quotient (up to
scale); the problem is directly identifying Eq. (7) with a positive Hessian
eigenvalue while retaining the explicit derivative definition. A reproduction
must not silently copy all these signs and claim they are jointly verified.
The smallest decisive check is a stable crystal with an independently positive
finite-difference cell curvature and the native routine's reported value.

## Two schedule details also need explicit resolution

On p. 17849 (PDF 5), §2.3 step (1) says divisibility by λ selects combined
CBD-cell+SSW. **Scheme 1 labels the opposite branch:** its nonzero-remainder
branch enters SSW and its zero-remainder branch bypasses it. At λ=2 the
frequency is the same but the odd/even phase differs. For general λ the
frequency also differs; choose and record a convention rather than asserting
native phase parity.

Step (4) starts n_Cell at 1, performs a cycle, increments, then exits when it
reaches H_Cell. Taken literally this performs H_Cell−1 cycles; the surrounding
text calls H_Cell the maximum cycle count. Fig. 1b labels L0 through L0^4.
This leaves an indexing ambiguity for an exact five-cycle reproduction. It
does not alter the established ordering of cell block, optional atomic block,
and full quench.

## What the existing native evidence now means

[The native followup](vc-native-coordinate-followup.md) independently identifies:

```
control.maxstress = |trace(stress) + 3*para.externaltp| * 160.2176565 / 3
```

This is a scalar diagnostic with a GPa conversion. It is **not** Eq. (6), which
is a matrix-valued lattice derivative with volume and inverse-transpose
factors. Native `dedlatt` (+0x3d0), `frac` (+0x418), `scart` (+0x478), and `sfa`
(+0x4d8) are plausible dataflow anchors for testing the paper formula, but their
names do not establish that formula. The apparent paper/native pressure-sign
difference is unresolved because the stored stress and external-pressure
producers are not closed. At p=0 it cannot be diagnosed by the pressure term.

The previously observed additive native internal update is consistent with
Eq. (3), but consistency is not proof that this array is the physical lattice.
Unresolved dispatch slots +0x48 and +0xe8 still prevent that identification.
The paper resolves the **published design** (3×3 additive lattice directions
with rotation projection), while native finite update, metric, pressure
convention and stopping consumers remain independent parity questions.

## Research decision

Keep the independently derived joint log-strain implementation and its real
system results labeled accordingly. If the aim is faithful 2014 SSW-crystal
reproduction, add a separately named blocked CBD-cell → optional fixed-cell
SSW → all-DOF quench implementation, with parameter provenance and explicit
resolution of the sign/index ambiguities above. Existing successes of the joint
walker cannot substitute for validation of that schedule. Read the SI and
resolve one targeted native coordinate/curvature routine before asserting
bitwise or behavioral native equivalence. No new kernel was introduced here.
