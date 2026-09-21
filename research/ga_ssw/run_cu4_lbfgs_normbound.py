"""One-step Cu4 EMT norm-bound check for VC L-BFGS baseline adapters."""
import json
import hashlib
import importlib.metadata
import sys
from pathlib import Path
from dataclasses import asdict
import numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT
from pamssw.standalone.vc_geometry import ASEStressSurface
from pamssw.standalone.vc_reference import VCSSWConfig, run_vc_ssw
from .vc_lbfgs_baseline_runner import run_vc_baseline


def _case():
    atoms = bulk("Cu", "fcc", a=3.65, cubic=True)
    atoms.positions[0] += [.03, -.02, .01]
    return atoms


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    if hasattr(value, "get_positions") and hasattr(value, "get_cell"):
        return _structure(value)
    if all(hasattr(value, name) for name in ("objective", "energy", "forces", "stress", "volume", "atoms")):
        return {"objective": float(value.objective), "energy": float(value.energy),
                "forces": _json(value.forces), "stress": _json(value.stress),
                "volume": float(value.volume), "atoms": _structure(value.atoms)}
    raise TypeError(f"unserializable evidence value: {type(value).__name__}")


def _structure(atoms):
    return {"numbers": atoms.numbers.tolist(), "positions": atoms.positions.tolist(),
            "cell": atoms.cell.array.tolist(), "pbc": atoms.pbc.tolist()}


def _fresh(minima, surface, config):
    before = surface.requests
    rows = []
    for minimum in minima:
        energy, forces, stress = surface.evaluate(minimum.atoms)
        residual = stress + config.pressure * np.eye(3)
        fmax = float(np.linalg.norm(forces, axis=1).max())
        smax = float(np.abs(residual).max())
        rows.append({"energy": float(energy), "fmax": fmax, "stress_residual_max": smax,
                     "atoms": _structure(minimum.atoms), "forces": forces,
                     "stress": stress, "stress_residual": residual,
                     "certified": fmax <= config.fmax and smax <= config.stress_tol})
    return rows, surface.requests - before


def _summary(result, requests, input_atoms, config, fresh, fresh_requests):
    def minimum(x):
        return {"objective": float(x.objective), "energy": float(x.energy),
                "volume": float(x.atoms.get_volume()), "atoms": _structure(x.atoms),
                "forces": x.forces, "stress": x.stress}
    return {"status": result.status, "requests": requests,
            "input": _structure(input_atoms), "config": asdict(config),
            "records": result.records,
            "initial": None if result.initial is None else minimum(result.initial),
            "minima": [minimum(x) for x in result.minima],
            "fresh": fresh, "fresh_requests": fresh_requests,
            "initial_certified": bool(fresh[0]["certified"]) if fresh else False,
            "valid_landing_count": sum(row["certified"] for row in fresh[1:]),
            "distinct_structures": None,
            "valid_candidate_count": sum(row["certified"] for row in fresh)}


def main():
    config = VCSSWConfig(strain_length=3.6, width=.2, rotation_bias=.5,
                         max_gaussians=1, relax_steps=150)
    arms = []
    for kind in ("safe_total", "scipy", "ase"):
        atoms, surface = _case(), ASEStressSurface(EMT())
        fresh_surface = ASEStressSurface(EMT())
        if kind == "safe_total":
            result = run_vc_ssw(atoms, surface, steps=1, config=config,
                                rng=np.random.default_rng(7))
            fresh, fresh_requests = _fresh(result.minima, fresh_surface, config)
            arms.append({"kind": kind, **_summary(result, surface.requests,
                                                   atoms, config, fresh, fresh_requests),
                         "efs_requests": surface.requests,
                         "total_requests": surface.requests + fresh_requests})
        else:
            out = run_vc_baseline(atoms, surface, steps=1, config=config,
                                  rng=np.random.default_rng(7), kind=kind,
                                  fresh_surface=fresh_surface)
            arms.append({"kind": kind, "run": _summary(out["result"], out["efs_requests"],
                                                         atoms, config, out["fresh"], out["fresh_requests"]),
                         "adapter_calls": out["adapter_calls"],
                         "phases": out["phases"], "fresh": out["fresh"],
                         "fresh_requests": out["fresh_requests"],
                         "total_requests": out["total_requests"]})
    source_files = [Path(__file__), Path(__file__).with_name("vc_lbfgs_baseline_runner.py"),
                    Path(__file__).with_name("lbfgs_baselines.py")]
    provenance = {"python": sys.version,
                  "numpy": np.__version__, "ase": importlib.metadata.version("ase"),
                  "scipy": importlib.metadata.version("scipy"),
                  "source_sha256": {str(p.name): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in source_files}}
    payload = {"provenance": provenance, "arms": arms,
               "scope": "one Cu4 EMT VC-SSW step; research comparison only"}
    path = Path(__file__).parent / "evidence/cu4-emt-lbfgs-normbound-v2.json"
    path.write_text(json.dumps(_json(payload), indent=2, allow_nan=False) + "\n")
    print(path)


if __name__ == "__main__":
    main()
