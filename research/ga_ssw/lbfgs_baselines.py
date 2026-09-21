"""Research adapters for mature L-BFGS baselines.

The adapters deliberately preserve each library's native stopping and line
search rules.  They only translate a flat ``q -> (energy, dE/dq)`` oracle,
count oracle requests, and report a common post-run certificate.  They are
research-only and are not part of the PAM-SSW production optimizer paths.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class BaselineRelaxResult:
    q: np.ndarray
    energy: float | None
    gradient: np.ndarray | None
    status: str
    steps: int
    requests: int
    accepted_trace: tuple
    trial_trace: tuple
    certificate: dict
    native_success: bool
    native_message: str
    metadata: dict
    error: str | None = None

    @property
    def trace(self):
        """Alias matching the accepted-point trace terminology."""
        return self.accepted_trace

    @property
    def converged(self):
        return self.status == "converged"


class _RequestLimit(RuntimeError):
    pass


def _flat(q, name="q0"):
    q = np.asarray(q, dtype=float)
    if q.ndim != 1 or not q.size or not np.isfinite(q).all():
        raise ValueError(f"{name} must be a finite nonempty flat vector")
    return q.copy()


def _positive(value, name):
    if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be integer >= {minimum}")


def _validate_common(evaluate, gradient_norm, step_norm, convergence_norm,
                     gtol, maxiter, max_requests, max_step):
    if not all(callable(f) for f in (evaluate, gradient_norm, step_norm)):
        raise ValueError("evaluate, gradient_norm and step_norm callbacks required")
    if convergence_norm is not None and not callable(convergence_norm):
        raise ValueError("convergence_norm must be callable when supplied")
    _positive(gtol, "gtol")
    _integer(maxiter, "maxiter")
    if max_requests is not None:
        _integer(max_requests, "max_requests", 1)
    if max_step is not None:
        raise ValueError("mature L-BFGS baselines do not support max_step")


def _scalar(value, name):
    if np.ndim(value) != 0 or not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must return a finite nonnegative scalar")
    return float(value)


def _call_convergence(callback, q, gradient):
    return callback(q.copy(), gradient.copy())


def _oracle(evaluate, q, requests, trials, max_requests, *, cache=None, api_calls=None):
    if api_calls is not None:
        api_calls[0] += 1
    key = np.asarray(q).tobytes()
    if cache is not None and key in cache:
        energy, gradient = cache[key]
        trials.append({"q": np.array(q, copy=True), "energy": energy,
                       "gradient": gradient.copy(), "requests": requests[0],
                       "api_call": api_calls[0], "cached": True, "charged": False,
                       "accepted": False})
        return energy, gradient.copy()
    if max_requests is not None and requests[0] >= max_requests:
        trials.append({"q": np.array(q, copy=True), "energy": None,
                       "gradient": None, "requests": requests[0],
                       "api_call": None if api_calls is None else api_calls[0],
                       "cached": False, "charged": False, "accepted": False, "failed": True})
        raise _RequestLimit("evaluator request budget exhausted")
    requests[0] += 1
    try:
        energy, gradient = evaluate(np.array(q, copy=True))
    except Exception:
        trials.append({"q": np.array(q, copy=True), "energy": None,
                       "gradient": None, "requests": requests[0],
                       "api_call": None if api_calls is None else api_calls[0],
                       "cached": False, "charged": True, "accepted": False, "failed": True})
        raise
    gradient = np.asarray(gradient, dtype=float)
    if np.ndim(energy) != 0 or not np.isfinite(energy) or gradient.shape != q.shape or not np.isfinite(gradient).all():
        trials.append({"q": np.array(q, copy=True), "energy": None,
                       "gradient": None, "requests": requests[0],
                       "cached": False, "charged": True, "accepted": False,
                       "failed": True, "error": "invalid energy/gradient"})
        raise ValueError("evaluator returned invalid energy/gradient")
    row = {"q": np.array(q, copy=True), "energy": float(energy),
           "gradient": gradient.copy(), "requests": requests[0],
           "api_call": None if api_calls is None else api_calls[0],
           "cached": False, "charged": True, "accepted": False, "failed": False}
    trials.append(row)
    if cache is not None:
        cache[key] = (float(energy), gradient.copy())
    return float(energy), gradient.copy()


def _certificate(accepted, trials, gradient_norm, step_norm, convergence_norm,
                 requests, max_requests, native_success, native_message,
                 max_step_supported=False):
    last = accepted[-1] if accepted else None
    gnorm = None if last is None else _scalar(_call_convergence(convergence_norm, last["q"], last["gradient"]), "convergence_norm")
    snorm = 0.0 if len(accepted) < 2 else _scalar(step_norm(accepted[-1]["q"] - accepted[-2]["q"]), "step_norm")
    return {
        "gradient_norm": None if last is None else _scalar(gradient_norm(last["gradient"]), "gradient_norm"),
        "step_norm": snorm,
        "convergence_norm": gnorm,
        "requests": requests,
        "max_requests": max_requests,
        "native_success": bool(native_success),
        "native_message": str(native_message),
        "max_step_supported": max_step_supported,
        "accepted": last is not None,
        "last_trial": None if not trials else trials[-1]["q"].copy(),
        "last_accepted": None if last is None else last["q"].copy(),
    }


def _mark_accepted(trials, accepted):
    # A repeated library request at an already accepted point remains a
    # trial/evaluator request.  Consume one matching oracle record per
    # accepted event so request accounting does not erase that distinction.
    used = set()
    for point in accepted:
        for index, row in enumerate(trials):
            if index not in used and np.array_equal(row["q"], point["q"]):
                row["accepted"] = True
                used.add(index)
                break


def scipy_lbfgsb(q0, evaluate, *, gradient_norm, step_norm, gtol,
                 maxiter, max_requests=None, convergence_norm=None,
                 max_step=None, maxcor=None, maxls=20, native_gtol=None,
                 lbfgs_memory=None):
    """Run SciPy's native ``L-BFGS-B`` on a flat oracle.

    SciPy's native ``gtol`` is the projected-gradient infinity norm.  The
    returned certificate additionally reports the caller's requested norm;
    ``status == 'converged'`` requires that common certificate to pass.
    ``max_step`` is intentionally unsupported because SciPy has no equivalent.
    """
    q0 = _flat(q0)
    if maxcor is None:
        maxcor = 10 if lbfgs_memory is None else lbfgs_memory
    if native_gtol is None:
        native_gtol = gtol
    _validate_common(evaluate, gradient_norm, step_norm, convergence_norm,
                     gtol, maxiter, max_requests, max_step)
    if convergence_norm is None:
        convergence_norm = lambda q, g: gradient_norm(g)
    requests, api_calls, trials, accepted = [0], [0], [], []
    cache = {}
    current = {"q": q0.copy(), "energy": None, "gradient": None}

    def fun(q):
        e, g = _oracle(evaluate, q, requests, trials, max_requests,
                        cache=cache, api_calls=api_calls)
        current.update(q=np.array(q, copy=True), energy=e, gradient=g.copy())
        return e, g

    def callback(q):
        if current["gradient"] is None or not np.array_equal(q, current["q"]):
            e, g = fun(q)
        else:
            e, g = current["energy"], current["gradient"]
        accepted.append({"q": np.array(q, copy=True), "energy": float(e),
                         "gradient": g.copy(), "requests": requests[0],
                         "accepted": True})

    native_success, native_message, error = False, "", None
    native_result = None
    try:
        e, g = fun(q0)
        accepted.append({"q": q0.copy(), "energy": e, "gradient": g.copy(),
                         "requests": requests[0], "accepted": True})
        from scipy.optimize import minimize
        if maxiter == 0:
            native_message = "maximum iterations reached (maxiter=0)"
            native_result = None
        else:
            native_result = minimize(fun, q0, jac=True, method="L-BFGS-B",
                                 callback=callback,
                                 options={"gtol": native_gtol, "maxiter": maxiter,
                                          "maxcor": maxcor, "maxls": maxls,
                                          **({"maxfun": max_requests} if max_requests is not None else {})})
            native_success = bool(native_result.success)
            native_message = str(native_result.message)
    except _RequestLimit as exc:
        native_message, error = str(exc), str(exc)
    except Exception as exc:
        native_message, error = f"{type(exc).__name__}: {exc}", f"{type(exc).__name__}: {exc}"
    _mark_accepted(trials, accepted)
    final = accepted[-1] if accepted else None
    cert = _certificate(accepted, trials, gradient_norm, step_norm,
                        convergence_norm, requests[0], max_requests,
                        native_success, native_message)
    if error and isinstance(error, str) and error == "evaluator request budget exhausted":
        status = "request_limit"
    elif error:
        status = "evaluation_failed"
    elif final is not None and cert["convergence_norm"] <= gtol:
        status = "converged"
    elif native_success:
        status = "native_stop"
    elif native_result is None or maxiter == 0:
        status = "maxiter"
    elif (int(getattr(native_result, "status", -1)) == 1 and
          int(getattr(native_result, "nit", 0)) >= maxiter):
        status = "maxiter"
    else:
        status = "native_failed"
    return BaselineRelaxResult(final["q"].copy() if final else q0,
                               None if final is None else final["energy"],
                               None if final is None else final["gradient"].copy(),
                               status, max(0, len(accepted) - 1), requests[0],
                               tuple(accepted), tuple(trials), cert,
                               native_success, native_message,
                               {"api_calls": api_calls[0], "cache_hits": sum(bool(r.get("cached")) for r in trials),
                                "denied_requests": sum(bool(r.get("failed")) and not r.get("charged", False) for r in trials),
                                "distinct_rejected_trials": sum(bool(r.get("charged")) and not r.get("accepted", False) and not r.get("failed", False) for r in trials),
                                "scipy_options": {"gtol": native_gtol, "common_gtol": gtol, "ftol": "scipy_default",
                                                   "maxiter": maxiter, "maxcor": maxcor, "maxls": maxls,
                                                   "native_status": None if native_result is None else int(getattr(native_result, "status", -1)),
                                                   "native_nit": None if native_result is None else int(getattr(native_result, "nit", 0))}}, error)


def ase_lbfgs_linesearch(q0, evaluate, *, gradient_norm, step_norm, gtol,
                         maxiter, max_requests=None, convergence_norm=None,
                         max_step=None, memory=None, native_fmax=None,
                         lbfgs_memory=None, native_gtol=None):
    """Run ASE's native ``LBFGSLineSearch`` against a flat E/gradient oracle."""
    q0 = _flat(q0)
    _validate_common(evaluate, gradient_norm, step_norm, convergence_norm,
                     gtol, maxiter, max_requests, max_step)
    if memory is None:
        memory = 100 if lbfgs_memory is None else lbfgs_memory
    if q0.size % 3:
        raise ValueError("ASE LBFGSLineSearch requires q0 size divisible by 3")
    if convergence_norm is None:
        convergence_norm = lambda q, g: gradient_norm(g)
    if native_fmax is None:
        native_fmax = native_gtol if native_gtol is not None else gtol
    _positive(native_fmax, "native_fmax")
    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes
    from ase.optimize import LBFGSLineSearch
    requests, trials, accepted = [0], [], []
    native_maxstep = 0.2

    class FlatCalculator(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            e, g = _oracle(evaluate, atoms.get_positions().ravel(), requests,
                            trials, max_requests)
            self.results = {"energy": e, "forces": -g.reshape((-1, 3))}

    atoms = Atoms("H" * (q0.size // 3), positions=q0.reshape((-1, 3)))
    atoms.calc = FlatCalculator()
    native_success, native_message, error = False, "", None
    try:
        e, f = atoms.get_potential_energy(), atoms.get_forces()
        g = -f.ravel()
        accepted.append({"q": q0.copy(), "energy": e, "gradient": g.copy(),
                         "requests": requests[0], "accepted": True})
        optimizer = LBFGSLineSearch(atoms, memory=memory, maxstep=native_maxstep, logfile=None)
        def record():
            q = atoms.get_positions().ravel().copy()
            e, f = atoms.get_potential_energy(), atoms.get_forces()
            if accepted and np.array_equal(q, accepted[-1]["q"]):
                return
            accepted.append({"q": q, "energy": float(e), "gradient": -f.ravel().copy(),
                             "requests": requests[0], "accepted": True})
        optimizer.attach(record, interval=1)
        native_success = bool(optimizer.run(fmax=native_fmax, steps=maxiter))
        native_message = "ASE native fmax stop" if native_success else "ASE native step limit or stop"
    except _RequestLimit as exc:
        native_message, error = str(exc), str(exc)
    except Exception as exc:
        native_message, error = f"{type(exc).__name__}: {exc}", f"{type(exc).__name__}: {exc}"
    _mark_accepted(trials, accepted)
    final = accepted[-1] if accepted else None
    cert = _certificate(accepted, trials, gradient_norm, step_norm,
                        convergence_norm, requests[0], max_requests,
                        native_success, native_message)
    if error == "evaluator request budget exhausted":
        status = "request_limit"
    elif error:
        status = "evaluation_failed"
    elif final is not None and cert["convergence_norm"] <= gtol:
        status = "converged"
    else:
        status = "native_stop" if native_success else "maxiter"
    return BaselineRelaxResult(final["q"].copy() if final else q0,
                               None if final is None else final["energy"],
                               None if final is None else final["gradient"].copy(),
                               status, max(0, len(accepted) - 1), requests[0],
                               tuple(accepted), tuple(trials), cert,
                               native_success, native_message,
                               {"distinct_rejected_trials": sum(bool(r.get("charged")) and not r.get("accepted", False) and not r.get("failed", False) for r in trials),
                                "ase_parameters": {"fmax": native_fmax, "common_gtol": gtol, "maxstep": native_maxstep,
                                                    "memory": memory, "pseudo_symbols": "H per 3 q coordinates",
                                                    "pseudopbc": False}}, error)
