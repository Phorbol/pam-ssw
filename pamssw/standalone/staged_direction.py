"""Two-stage fixed-cell rotation diagnostic.

This is a bounded numerical experiment inspired by the BP-CBD presweep.  It
does not implement CBD or claim native LASP parity.  The first dimer solve is
unbiased and supplies an anchor and measured curvature; the second solve uses
that anchor and ``a=max(Ce, 0)``.  Both solves share the caller's original
force budget.
"""
from dataclasses import dataclass

import numpy as np

from pamssw.standalone.dimer import paper_dimer_direction
from pamssw.standalone.broyden_direction import paper_broyden_direction
from pamssw.standalone.direction import SoftModeResult, paper_biased_direction


@dataclass(frozen=True)
class TwoStageDimerResult:
    """Main result plus explicit presweep accounting and curvature boundary."""

    direction: np.ndarray
    curvature: float
    residual_norm: float
    hvp_calls: int
    force_calls: int
    converged: bool
    projected_symmetry_error: float
    pre: SoftModeResult
    main: SoftModeResult
    pre_rotation_hvp: int
    main_hvp_budget: int
    bias_curvature: float
    curvature_scope: str
    main_solver: str
    stop_reason: str = 'unspecified'


def two_stage_dimer_direction(atoms, anchor, *, fd_step, max_hvp,
                              pre_rotation_hvp, tol, evaluate,
                              main_solver='dimer'):
    """Run an explicit unbiased presweep followed by anchored plane dimer.

    ``main_solver`` selects the explicitly independent numerical main solver:
    ``dimer`` (the default), ``ritz``, or ``broyden-euclidean``.  ``max_hvp`` retains the old
    single-stage force-request budget: total force requests are bounded by
    ``1 + max_hvp``.  The dimer main endpoint budget is
    ``max_hvp - pre.force_calls`` for both solvers. The paper Ritz wrapper
    explicitly selects forward differences, also one center plus one endpoint
    per HVP; it is not the generic reference solver's central-difference mode.  A positive
    ``Ce`` is used directly as the rank-one rotation bias.  For ``Ce <= 0``
    the main stage remains unbiased; this is an independent negative-curvature
    escape diagnostic, not a CBD transition or a native threshold rule.
    """
    for name, value in (('max_hvp', max_hvp),
                        ('pre_rotation_hvp', pre_rotation_hvp)):
        if (isinstance(value, (bool, np.bool_)) or
                not isinstance(value, (int, np.integer))):
            raise ValueError(f'{name} must be an integer')
    if main_solver not in ('dimer', 'ritz', 'broyden-euclidean'):
        raise ValueError("main_solver must be 'dimer', 'ritz' or 'broyden-euclidean'")
    main_min_hvp = 1 if main_solver in ('dimer', 'broyden-euclidean') else 2
    if max_hvp < 2 + main_min_hvp:
        raise ValueError('max_hvp must cover presweep and main solver minimum budgets')
    if pre_rotation_hvp < 1 or pre_rotation_hvp > max_hvp - main_min_hvp - 1:
        raise ValueError('pre_rotation_hvp leaves no HVP for the main stage')

    pre = paper_dimer_direction(atoms, anchor, rotation_bias=0.,
                                fd_step=fd_step, max_hvp=pre_rotation_hvp,
                                tol=tol, evaluate=evaluate)
    remaining = max_hvp - pre.force_calls
    if remaining < main_min_hvp:
        raise ValueError('presweep consumed the complete main-stage HVP budget')
    ce = float(pre.curvature)
    if not np.isfinite(ce):
        raise ValueError('presweep curvature must be finite')
    bias = max(ce, 0.)
    if main_solver == 'dimer':
        solver = paper_dimer_direction
    elif main_solver == 'ritz':
        solver = paper_biased_direction
    else:
        solver = paper_broyden_direction
    main = solver(atoms, pre.direction, rotation_bias=bias,
                  fd_step=fd_step, max_hvp=remaining,
                  tol=tol, evaluate=evaluate)
    return TwoStageDimerResult(
        direction=main.direction,
        curvature=main.curvature,
        residual_norm=main.residual_norm,
        hvp_calls=pre.hvp_calls + main.hvp_calls,
        force_calls=pre.force_calls + main.force_calls,
        converged=main.converged,
        projected_symmetry_error=max(pre.projected_symmetry_error,
                                     main.projected_symmetry_error),
        pre=pre,
        main=main,
        pre_rotation_hvp=pre_rotation_hvp,
        main_hvp_budget=remaining,
        bias_curvature=bias,
        curvature_scope=('rotation-surface curvature; LS terms, if present, '
                         'are included in evaluate'),
        main_solver=main_solver,
        stop_reason=main.stop_reason,
    )
