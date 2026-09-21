# Uploaded CBD 2010: implications for independent SSW implementation

Source: user-uploaded `/home/gengjianrui/bin/pam-ssw/CBD.pdf`; Shang and Liu,
JCTC 6 (2010), 1136–1144, DOI 10.1021/ct9005147. Full text extracted with
pdftotext; PDF page3 (journal p1138), equations5–13 rendered and visually checked.
An external archival copy is `literature/CBD-user.pdf` in the upload research
workspace. The copyrighted full text is not copied into this code repository.

## What the new source resolves

1. Section2.1, p1137 explicitly distinguishes finding a zero rotational force
   from finding the globally lowest curvature. Every Hessian eigenvector is a
   stationary rotation direction; their chosen root solver can follow a local
   mode related to the initial guess. A lowest-mode dimer or Rayleigh/Ritz solver
   is a numerical/algorithmic alternative, not automatically identical CBD.
2. Section2.1, p1138 identifies Johnson modified Broyden (reference41) and gives
   equations5–11: history-dependent inverse-Jacobian action, U vectors, a history
   overlap matrix and regularized inverse beta, plus previous Z history. This
   explains why the executable BRZERO4 has several history matrices rather than
   a simple rank-one update. It does not by itself prove the specific ELF is an
   exact implementation of these printed equations.
3. Eq9 U=G_initial*delta_R+delta_x matches the shape of the recovered prefix;
   the executable also normalizes differences. Normalization, signs and weights
   must be reconciled with the complete residual convention before matching
   equations. The paper's R means a solver residual here, not atom coordinates.
4. Eq12 restores the fixed dimer radius after an unconstrained coordinate update.
   Its displayed RHS is center-relative: a Cartesian implementation must return
   R0+dr*(R1_trial-R0)/norm(R1_trial-R0), not forget the center translation.
5. Section2.2 is a separate TS translation algorithm: multiple translations with
   an approximate fixed mode; eq13/17 stop based on changes in parallel (and in
   positive-curvature regions perpendicular) force. Eq14–16 apply force-component
   damping. These TS rules do not replace SSW Gaussian climb/true-PES quench.
6. The 00/01/10/11 comparisons (p1140 onward) separately vary rotation and
   translation. They support testing these components independently, rather than
   attributing a complete-method speedup solely to a direction solver.

## Corrections to prior reasoning

The earlier statement that standard dimer and CBD simply solve the same lowest-
eigenvalue optimization was too strong. The common substrate is directional
curvature and tangent rotational force; root selection and stopping differ.
For a quadratic local surface the tangent residual vanishes at all eigenvectors.
Adding the BP-CBD rank-one bias retains a preferred direction but still does not
make every residual root the minimum-curvature eigenmode.

The paper supplies no support for treating the recovered XYZ block-sum bilinear
form as the physical Cartesian inner product. The independent paper path should
use a rotation-equivariant Euclidean geometry; preserving the ELF form remains
an explicitly labeled compatibility choice. The full-method consequence still
needs real-system testing.

The p1138 printed equation set does not fully specify implementation choices
such as history truncation/reset, all weights and numerical safeguards. In
particular, do not silently repair apparent notation issues in eq6/11 or infer
weights from unrelated SCF defaults. Consult the original Johnson source and
cross-check binary operations before claiming a complete paper-CBD solver.

## Literature follow-up and concrete development order

Reference41 verified at the primary publisher:
D. D. Johnson, Modified Broyden's method for accelerating convergence in
self-consistent calculations, Phys. Rev. B 38, 12807–12813 (1988),
https://doi.org/10.1103/PhysRevB.38.12807.
This lookup verifies the reference, not the full-text equations.

Proceed in two tracks, sharing ASE E/F accounting and explicit configuration:

- Independent SSW track: add an independently coded standard dimer rotation,
  with the same explicit BP rank-one bias and anchor as the existing Ritz
  alternative. Keep SSW climb/LS/GA development independent of unresolved native
  Broyden. Label mode selection and convergence semantics accurately.
- Reference recovery track: use CBD eq7–11 as a map for BRZERO4 history recovery;
  validate numerical operations with original instructions. Retain ELF quirks
  only in compatibility modules, not as default physics or new PAM knobs.

First compare direction solvers on fixed real geometries with matched E/F
precision, initial anchor, bias and call budgets: residual, curvature, anchor
angle, rigid-rotation equivariance and failures. Then compare complete escapes
and distinct, force-qualified minima per cost. Small matrix checks only verify
implementation. CBD2010 TS success counts cannot validate global-search efficacy.

This document records source review and a revised plan, not a newly implemented
dimer/CBD driver or a new real-system experiment. Existing native prefix and
control primitives remain partial, and existing paper_reference still uses Ritz.
