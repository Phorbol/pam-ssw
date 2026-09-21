"""Prepare a fixed-geometry, four-way direction screen (no PES by default).

This is a numerical screen, not a search-efficiency experiment.  Each
case/seed shares one projected random anchor across the four solvers.  The
direction budget is 101 API evaluations (one centre plus at most 100 HVP
endpoints); the independent two-call certificate is accounted for separately.
Use ``--execute`` only after reviewing the generated protocol.
"""
import argparse
import hashlib
import importlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(v) for v in value]
    if hasattr(value, "__dict__"):
        return _json(vars(value))
    return value


def _write(path, value):
    path.write_text(json.dumps(_json(value), indent=2, allow_nan=False) + "\n")


def _atoms(payload):
    from ase import Atoms
    return Atoms(numbers=payload["numbers"], positions=payload["positions"],
                cell=payload["cell"], pbc=payload["pbc"])


def _projector(atoms):
    if atoms.pbc.all():
        from pamssw.standalone.periodic_geometry import FixedCellTranslationFrame
        frame = FixedCellTranslationFrame(atoms)
        return frame.project, "translation_only"
    from pamssw.standalone.cluster_frame import ClusterFrame
    frame = ClusterFrame(atoms)
    # ClusterFrame.project removes translations and rotations.  Keeping the
    # frame object alive also keeps its reference geometry immutable.
    return frame.project, "cluster_rigid6"


def _anchor(atoms, project, rng):
    value = project(rng.normal(size=atoms.positions.shape))
    norm = np.linalg.norm(value)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("random anchor has no internal component")
    return value / norm


class LedgerEvaluator:
    """Count one energy/force request, including failures and denials."""
    def __init__(self, calculator, ledger, project, start, cap=101, wall=60.):
        self.calculator, self.ledger, self.project = calculator, ledger, project
        self.start, self.cap, self.wall = start, cap, wall
        self.requests = 0
        self.denials = 0

    def __call__(self, atoms):
        if self.requests >= self.cap or time.monotonic() - self.start >= self.wall:
            self.denials += 1
            _append(self.ledger, {"kind": "search_denial", "request": self.requests,
                                  "reason": "request_cap" if self.requests >= self.cap else "wall_cap"})
            raise RuntimeError("direction screen budget exhausted")
        self.requests += 1
        work = atoms.copy(); work.calc = self.calculator
        try:
            energy = float(work.get_potential_energy())
            forces = np.asarray(work.get_forces(), dtype=float)
            if not np.isfinite(energy) or forces.shape != work.positions.shape or not np.isfinite(forces).all():
                raise ValueError("nonfinite evaluator result")
            projected = self.project(forces)
            _append(self.ledger, {"kind": "search", "request": self.requests,
                                  "atoms": {"numbers": work.numbers, "positions": work.positions,
                                            "cell": work.cell.array, "pbc": work.pbc},
                                  "energy": energy, "forces": forces,
                                  "projected_forces": projected,
                                  "fmax": float(np.linalg.norm(forces, axis=1).max())})
            return energy, projected
        except Exception as exc:
            _append(self.ledger, {"kind": "search_failure", "request": self.requests,
                                  "error": repr(exc)})
            raise


def _append(path, value):
    with path.open("a") as handle:
        handle.write(json.dumps(_json(value), allow_nan=False) + "\n")


def _calculator(case):
    if case == "bicyclobutane":
        from tblite.ase import TBLite
        return TBLite(method="GFN2-xTB", accuracy=0.001, verbosity=0)
    from ase.calculators.emt import EMT
    return EMT()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    # Snapshot before imports used by the protocol.  The snapshot is the only
    # source placed first on sys.path when --execute is requested.
    shutil.copytree(ROOT / "pamssw", out / "source" / "pamssw",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (out / "source" / "research" / "ga_ssw").mkdir(parents=True)
    for name in ("broyden_direction_reconstruction.py", "broyden_state_reconstruction.py",
                 "broyden_history_reconstruction.py"):
        shutil.copy2(ROOT / "research/ga_ssw" / name, out / "source" / "research/ga_ssw" / name)
    (out / "source" / "research" / "__init__.py").write_text("")
    (out / "source" / "research" / "ga_ssw" / "__init__.py").write_text("")
    shutil.copy2(__file__, out / "runner.py")
    sys.path.insert(0, str(out / "source"))
    import pamssw
    assert Path(pamssw.__file__).resolve().is_relative_to(out / "source")
    from pamssw.standalone.dimer import paper_dimer_direction
    from pamssw.standalone.direction import paper_biased_direction
    research_module = importlib.import_module("research.ga_ssw.broyden_direction_reconstruction")
    assert Path(research_module.__file__).resolve().is_relative_to(out / "source")
    broyden = research_module.broyden_direction

    input_sources = {case: ROOT / "research/ga_ssw/evidence/verified-ritz-multicase-20260912"
                     / f"{case}-verified-seed11/result.json"
                     for case in ("cu13", "cu31_fixed", "bicyclobutane")}
    inputs = {case: json.loads(path.read_text())["initial"]["atoms"]
              for case, path in input_sources.items()}
    input_path = out / "inputs.json"
    _write(out / "inputs.json", inputs)
    _write(out / "input-sources.json", {case: {"path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for case, path in input_sources.items()})
    config = dict(rotation_bias=0.0, fd_step=1e-4, max_hvp=100, tol=0.02,
                  initial_factor=1.0, metrics=["euclidean", "native_block_sum"],
                  direction_budget_requests=101, certificate_requests=2,
                  wall_seconds=60, seeds=[11, 29], methods=["paper_ritz", "paper_dimer", "broyden_euclidean", "broyden_native_block_sum"],
                  backends={"cu13": "ASE EMT", "cu31_fixed": "ASE EMT", "bicyclobutane": "GFN2-xTB accuracy=0.001"},
                  initial_factor_source="provisional comparison value1; not optimized or established native FACT default",
                  scope="fixed-geometry numerical direction screen; no search-efficiency claim")
    _write(out / "config.json", config)
    manifest = {}
    for path in sorted((out / "source").rglob("*.py")):
        manifest[str(path.relative_to(out / "source"))] = hashlib.sha256(path.read_bytes()).hexdigest()
    _write(out / "source-manifest.json", {"sha256": manifest, "input": str(input_path),
                                           "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest()})
    if not args.execute:
        return

    rows = []
    for case, payload in inputs.items():
        initial = _atoms(payload)
        project, projection = _projector(initial)
        for seed in (11, 29):
            # One draw only: all methods below receive the same anchor.
            anchor = _anchor(initial, project, np.random.default_rng(seed))
            _write(out / f"{case}-seed{seed}-anchor.json", {"case": case, "seed": seed,
                    "projection": projection, "anchor": anchor})
            for method in config["methods"]:
                folder = out / f"{case}-{method}-seed{seed}"; folder.mkdir()
                ledger = folder / "evaluations.jsonl"; start = time.monotonic()
                surface = LedgerEvaluator(_calculator(case), ledger, project, start)
                row = {"case": case, "seed": seed, "method": method,
                       "projection": projection, "status": "started"}
                cert_surface = None
                try:
                    if method == "paper_ritz":
                        result = paper_biased_direction(initial.copy(), anchor,
                            rotation_bias=0., fd_step=1e-4, max_hvp=100, tol=.02,
                            evaluate=surface)
                    elif method == "paper_dimer":
                        result = paper_dimer_direction(initial.copy(), anchor,
                            rotation_bias=0., fd_step=1e-4, max_hvp=100, tol=.02,
                            evaluate=surface)
                    else:
                        result = broyden(initial.copy(), anchor, rotation_bias=0.,
                            fd_step=1e-4, max_hvp=100, tol=.02, initial_factor=1.,
                            metric="euclidean" if method.endswith("euclidean") else "native_block_sum",
                            evaluate=surface)
                    row.update(status="completed", result=result, search_requests=surface.requests,
                               retries=sum(x.get("retries", 0) for x in getattr(result, "trace", ()) if isinstance(x, dict)))
                    _write(folder / "result.json", result)
                    direction_flat = np.asarray(result.direction).ravel()
                    row["anchor_norm"] = float(np.linalg.norm(anchor))
                    row["anchor_direction_cosine"] = float(np.dot(anchor.ravel(), direction_flat) /
                                                            np.linalg.norm(direction_flat))
                    # Independent, fresh two-call forward HVP certificate.
                    cert_surface = LedgerEvaluator(_calculator(case), folder / "certificate.jsonl",
                                                    project, time.monotonic(), cap=2)
                    centre = initial.copy(); e0, f0 = cert_surface(centre)
                    trial = initial.copy(); trial.positions = initial.positions + 1e-4 * result.direction
                    e1, f1 = cert_surface(trial)
                    hv = (f0 - f1) / 1e-4
                    n = result.direction / np.linalg.norm(result.direction)
                    cert_residual = float(np.linalg.norm(hv - (n.ravel() @ hv.ravel()) * n.ravel().reshape(hv.shape)))
                    row["certificate"] = {"requests": cert_surface.requests,
                        "curvature": float(n.ravel() @ hv.ravel()),
                        "residual_norm": cert_residual, "qualified": bool(cert_residual <= .02),
                        "energy_center": e0, "energy_endpoint": e1}
                    _write(folder / "certificate.json", row["certificate"])
                except Exception as exc:
                    row.update(status="certificate_failed" if cert_surface is not None else "direction_failed",
                               error=repr(exc))
                row.update(search_requests=surface.requests, denials=surface.denials,
                           certificate_requests=0 if cert_surface is None else cert_surface.requests,
                           elapsed_seconds=time.monotonic() - start)
                _write(folder / "summary.json", row); rows.append(row); _write(out / "summary.json", rows)
                print(case, seed, method, row["status"], row["search_requests"], row.get("certificate", {}).get("residual_norm"), flush=True)


if __name__ == "__main__":
    main()
