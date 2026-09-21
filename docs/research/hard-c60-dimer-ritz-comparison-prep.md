# Preparation: hard C60 dimer versus Ritz rotation

This is a future, paired diagnostic at the frozen failed-rotation boundary
supplied by the rotation-replay work. It must use the same C60 coordinates,
LS softening state, normalized anchor and direction-only projection for both
solvers. The frozen Gaussian terms explain how the boundary was reached, but
are excluded from the rotation oracle; `paper_reference.py`'s rotation surface
adds only the softening terms. No PES run is part of
this preparation.

The fixed numerical fields are `fd_step=1e-4` Angstrom,
`rotation_bias=100` eV/Angstrom^2, `tol=0.02` eV/Angstrom^2, and the existing
direction-only projection. In this mode `ClusterFrame(work)` supplies the
positions and `.project` removes translation and rotation; this is distinct
from `translation_only`, which uses `FixedCellTranslationFrame`. The dimer uses
`paper_dimer_direction`; the
comparator uses the existing `paper_biased_direction`/`reference_soft_mode`
Ritz implementation. No new algorithm or parameter is introduced.

The common budget is exactly 100 force requests, including the center force.
For dimer, `max_hvp=100` is retained: its stopping guard allows at most 99
HVP endpoint requests after the center, so the failed boundary has
`force_calls=100` (99 HVPs plus center). Its final `hvp(n)` is the direct
finite-secant check and is included in that count. For Ritz, use
`max_hvp=99`: the 98 Krylov HVPs plus the final direct returned-direction HVP,
along with the one forward-difference center force, total 100 requests.

Ritz first uses a subspace residual only for its internal early stopping, but
the returned `residual_norm` is recomputed by a direct finite-difference HVP
of the returned direction. Dimer likewise reports its directly evaluated
terminal secant residual. These residuals are comparable only as finite
separation numerical diagnostics, not as exact-Hessian or global-mode proofs.

Record per solver: all force requests and endpoint coordinates, HVP count,
force count, curvature, direct residual, projected antisymmetry, convergence,
and the exact frozen boundary fields. If either solver reaches `tol` early,
unused requests remain unused; do not spend them on an extra check. The later
paired outer escape, if authorized, must compare total E/F and valid landings;
same-numbered stages after memory10/400 divergence are not a causal local
speed comparison.
