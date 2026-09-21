# Fixed-cell optimizer baseline interface

`SSWConfig.quench_optimizer` accepts the existing `ase-lbfgs` and
`safe-lbfgs-total` values plus two explicit baselines: `ase-lbfgs-linesearch`
and `scipy-lbfgsb`. The former uses ASE's `LBFGSLineSearch` class. The latter
calls SciPy `minimize(method="L-BFGS-B", jac=True)` directly from
`standalone.surface.quench`; it sets only `maxiter` and `gtol=fmax/sqrt(3)`.
SciPy's default `ftol`, `maxls`, and `maxcor` remain in force. `steps=0` only
evaluates the initial state and never invokes SciPy.

Both baselines retain iteration counts and the final force certificate for the
requested objective (modified when bias terms are present; true when absent).
SciPy additionally records its own termination message and success flag. An energy or optimizer-success message does not make a
quench converged when the returned maximum force exceeds `fmax`. SciPy's
`nfev/nit/message/success` are retained as solver diagnostics, while the
backend evaluation count remains the surface physical-request ledger. SciPy and
Safe-total reject Eckart/frame coordinates; no manifold optimizer is implied.

The GA controller preserves its historical non-Safe `ase-lbfgs` → ASE BFGS
choice. The two new names are explicit opt-ins and are passed through GA
initial/offspring quenches and the SSW walker stages. This is an independent
numerical baseline, not native trajectory parity.


References checked against local ASE3.26.0 and SciPy1.16.0:
- https://docs.ase-lib.org/_modules/ase/optimize/lbfgs.html (LBFGSLineSearch wrapper;
  live website may describe a newer ASE than this frozen experiment).
- https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html
  (component projected-gradient stopping and relative-energy reduction).
For unconstrained Cartesian vectors, component gtol=fmax/sqrt(3) is sufficient
for per-atom Euclidean force<=fmax if the gradient criterion terminates the solver.
It does not prevent earlier relative-energy termination; the final certificate
is checked independently. No free-energy or equilibrium sampling claim follows.
