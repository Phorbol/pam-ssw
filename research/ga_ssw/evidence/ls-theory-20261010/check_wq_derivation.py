"""Equation check for LS paper Eqs.16–17; not an atomistic efficacy test.

Run with OPENBLAS_NUM_THREADS=1 python check_wq_derivation.py OUTPUT.json.
No Calculator, random search, tuning, or GPU is used. Output must be new.
"""
import json
import sys
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import root


def terms(z, a):
    x, y = z
    X, Y = x - 3.17, y - 3.0
    w = np.exp(-(x - 2.0) / 0.4)
    v = X**4 + Y**4 - 2*X**2 - 4*Y**2 + X*Y + 0.3*X + 0.1*Y
    gv = np.array([4*X**3 - 4*X + Y + 0.3, 4*Y**3 - 8*Y + X + 0.1])
    hv = np.array([[12*X**2 - 4, 1.0], [1.0, 12*Y**2 - 8]])
    gw = np.array([-w/0.4, 0.0])
    kw = np.diag([w/0.16, 0.0])
    return v, w, gv + a*gw, hv + a*kw, hv, kw, gw


def stationary(a, guess, index):
    sol = root(lambda z: terms(z, a)[2], guess,
               jac=lambda z: terms(z, a)[3], tol=1e-11)
    t = terms(sol.x, a)
    residual = float(np.linalg.norm(t[2]))
    eigenvalues = np.linalg.eigvalsh(t[3])
    if residual > 1e-9 or int(np.sum(eigenvalues < 0)) != index:
        raise RuntimeError((a, index, residual, eigenvalues.tolist(), sol.message))
    return sol.x, t, residual


def main():
    m0, t0, _ = stationary(0.0, [2.0, 4.48], 0)
    m = m0.copy()
    s, _, _ = stationary(0.0, [4.2, 3.1], 1)
    rows = []
    max_residual = 0.0
    max_slope_error = 0.0
    # Small fixed increments track these two branches; this is not a universal
    # continuation solver and makes no assertion about all WQ stationary points.
    for a in np.linspace(0.0, 5.0, 101):
        m, tm, rm = stationary(a, m, 0)
        s, ts, rs = stationary(a, s, 1)
        max_residual = max(max_residual, rm, rs)
        if not any(abs(a - b) < 1e-10 for b in (0.0, 0.1, 1.0, 2.0, 5.0)):
            continue
        barrier = ts[0] + a*ts[1] - tm[0] - a*tm[1]
        slope = ts[1] - tm[1]
        eps = 1e-4
        shifted = []
        for aa in (a-eps, a+eps):
            _, mm, _ = stationary(aa, m, 0)
            _, ss, _ = stationary(aa, s, 1)
            shifted.append(ss[0] + aa*ss[1] - mm[0] - aa*mm[1])
        fd_slope = (shifted[1] - shifted[0])/(2*eps)
        max_slope_error = max(max_slope_error, abs(fd_slope-slope))
        rows.append(dict(a=float(a), minimum=m.tolist(), saddle=s.tolist(),
                         lambda_min=float(np.linalg.eigvalsh(tm[3])[0]),
                         bare_hxx=float(tm[4][0, 0]), added_hxx=float(a*tm[5][0, 0]),
                         true_energy_rise=float(tm[0]-t0[0]),
                         response_slope=float(a*tm[6] @ np.linalg.solve(tm[3], tm[6])),
                         branch_barrier=float(barrier), barrier_slope=float(slope),
                         barrier_slope_fd=float(fd_slope)))
    if max_slope_error > 1e-7:
        raise RuntimeError(('barrier derivative check failed', max_slope_error))
    result = dict(purpose='analytic-equation illustration only; no search efficacy claim',
                  source='10.1021/acs.jctc.4c01081 Eqs.16–17; local PDF page E visually checked',
                  coordinate_units='paper model coordinates; not atomistic eV/Angstrom calibration',
                  numpy=np.__version__, scipy=scipy.__version__, rows=rows,
                  max_stationarity_residual=max_residual,
                  max_barrier_slope_error=max_slope_error,
                  note=('FD at a=0 uses infinitesimal negative a only for derivative verification; '
                        'branch_barrier is a specified stationary-point energy difference; '
                        'saddle connectivity at each a is not independently certified'))
    with Path(sys.argv[1]).open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(dict(max_stationarity_residual=max_residual,
                          max_barrier_slope_error=max_slope_error, rows=len(rows))))


if __name__ == '__main__':
    main()
