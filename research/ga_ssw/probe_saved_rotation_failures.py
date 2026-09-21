"""Probe saved fixed-cell rotation failures without replaying their presweep.

The input directory is an existing matched-materials evidence directory.  This
script only performs bounded direction solves on saved states; it does not run
SSW outer steps or native LASP.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np


def _json(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)): return value.item()
    if isinstance(value, Path): return str(value)
    if isinstance(value, dict): return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_json(v) for v in value]
    return value


def _atoms(payload):
    from ase import Atoms
    return Atoms(numbers=payload["numbers"], positions=payload["positions"],
                cell=payload["cell"], pbc=payload["pbc"])


def _certificate(atoms, direction, bias, anchor, evaluate, step, central=False):
    n = np.array(direction, dtype=float, copy=True)
    n /= np.linalg.norm(n)
    if central:
        plus, minus = atoms.copy(), atoms.copy()
        plus.positions += step * n
        minus.positions -= step * n
        _, fp = evaluate(plus); _, fm = evaluate(minus)
        hv = -(fp - fm) / (2 * step)
    else:
        _, f0 = evaluate(atoms)
        endpoint = atoms.copy(); endpoint.positions += step * n
        _, f1 = evaluate(endpoint)
        hv = (f0 - f1) / step
    # The solver's rotation operator is H - a |anchor><anchor|.
    a = np.array(anchor, dtype=float, copy=True); a /= np.linalg.norm(a)
    biased_hv = hv - bias * np.sum(a * n) * a
    curvature = float(np.sum(n * biased_hv))
    residual = float(np.linalg.norm(biased_hv - curvature * n))
    return {"step": step, "central": central, "curvature": curvature,
            "physical_curvature": float(np.sum(n * hv)),
            "residual": residual, "requests": 2}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True, help="existing matched evidence directory")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--tol", nargs="+", type=float, default=[.02, 2.])
    ap.add_argument("--max-states", type=int, default=1,
                    help="first failures per case/seed; default is at most four total states")
    args = ap.parse_args()
    evidence = args.input.expanduser().resolve()
    plan = json.loads((evidence / "plan.json").read_text())
    source = evidence / "source"
    if not source.is_dir(): raise SystemExit(f"missing frozen source: {source}")
    sys.path.insert(0, str(source))
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    from mace.calculators import MACECalculator
    from pamssw.standalone import ASESurface
    from pamssw.standalone.periodic_geometry import FixedCellTranslationFrame
    from pamssw.standalone.dimer import paper_dimer_direction
    from pamssw.standalone.direction import paper_biased_direction
    from pamssw.standalone.broyden_direction import paper_broyden_direction
    calc = MACECalculator(model_paths=plan["model"], device=plan["device"],
                          default_dtype="float64", enable_cueq=False, enable_oeq=False)
    surface = ASESurface(calc)
    report = {"scope": "saved Ritz rotation failures; no presweep replay",
              "input": str(evidence), "model": plan["model"], "device": plan["device"],
              "fd_step": .001, "max_hvp": 100, "tolerances": args.tol,
              "rows": [], "requests": 0}
    solvers = {"ritz": paper_biased_direction, "dimer": paper_dimer_direction,
               "broyden-euclidean": paper_broyden_direction}
    for case in plan["cases"]:
        for seed in plan["seeds"]:
            path = evidence / f"{case}-ritz-seed{seed}" / "result.json"
            if not path.exists(): continue
            data = json.loads(path.read_text())
            failures = [r for r in data["records"] if r["status"] == "rotation_failed"][:args.max_states]
            for state_no, record in enumerate(failures):
                climb = record["climb"][-1]
                atoms = _atoms(record["last_atoms"])
                frame = FixedCellTranslationFrame(atoms)
                anchor = np.asarray(climb["actual_anchor"], dtype=float)
                bias = float(climb["actual_rotation_bias"])
                # Saved failed main stages already paid this presweep cost.
                remaining = 100 - int(climb["pre_rotation"]["force_calls"])
                base = {"case": case, "seed": seed, "state": state_no,
                        "outer_index": record["index"], "source_result": str(path),
                        "geometry": record["last_atoms"],
                        "anchor": anchor, "bias": bias,
                        "presweep": climb["pre_rotation"], "remaining_hvp": remaining,
                        "original_failure": {"residual": climb["residual"],
                                              "force_requests": climb["force_requests"]}}
                def evaluate(candidate):
                    return frame.evaluate(candidate, surface.evaluate)
                for tol in args.tol:
                    for solver_name, solver in solvers.items():
                        before = surface.requests
                        row = dict(base, solver=solver_name, tol=tol)
                        try:
                            result = solver(atoms, anchor, rotation_bias=bias, fd_step=.001,
                                max_hvp=remaining, tol=tol, evaluate=evaluate)
                            row["result"] = asdict(result)
                            row["solver_requests"] = surface.requests - before
                            row["accounting_passed"] = result.force_calls == row["solver_requests"]
                            cert_before = surface.requests
                            row["certificate"] = _certificate(atoms, result.direction, bias, anchor,
                                                               evaluate, .001)
                            row["certificate"]["requests"] = surface.requests - cert_before
                            fine_before = surface.requests
                            row["fine_certificate"] = _certificate(atoms, result.direction, bias, anchor,
                                                                     evaluate, .0005)
                            row["fine_certificate"]["requests"] = surface.requests - fine_before
                            central_before = surface.requests
                            row["central_certificate"] = _certificate(atoms, result.direction, bias, anchor,
                                                                        evaluate, .001, central=True)
                            row["central_certificate"]["requests"] = surface.requests - central_before
                        except Exception as error:
                            row["error"] = repr(error)
                        row["total_requests"] = surface.requests - before
                        report["rows"].append(row)
                        report["requests"] = surface.requests
                        args.output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
                        args.output.expanduser().resolve().write_text(json.dumps(_json(report), indent=2) + "\n")
    report["requests"] = surface.requests
    args.output.expanduser().resolve().write_text(json.dumps(_json(report), indent=2) + "\n")


if __name__ == "__main__": main()
