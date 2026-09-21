#!/usr/bin/env python3
"""Matched one-Gaussian MH1 rotation/CBD diagnostic.

This is a frozen-state development probe, not a complete SSW search.  It
compares the current Euclidean Broyden rotation with the recovered CBD stage
controller on the same saved Python minima, then applies the same existing
forward-force Gaussian rule and unrestricted Safe-total quenches.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PYTHON_RUN = ROOT / "research/ga_ssw/evidence/c60-mh1-python-20260919"
BACKEND_PLAN = ROOT / "research/ga_ssw/evidence/c60-mh1-qualification-20260919/plan.json"


class BudgetExceeded(RuntimeError):
    pass


class BudgetSurface:
    """Count calls through an existing ASESurface and stop at a hard total."""

    def __init__(self, surface, cap):
        self.surface = surface
        self.cap = int(cap)
        self.start = surface.requests

    @property
    def requests(self):
        return self.surface.requests - self.start

    def evaluate(self, atoms):
        if self.requests >= self.cap:
            raise BudgetExceeded(f"matched-arm request cap {self.cap} reached")
        return self.surface.evaluate(atoms)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def serial(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    if dataclasses.is_dataclass(value):
        return serial(dataclasses.asdict(value))
    if hasattr(value, "__dict__"):
        return serial(vars(value))
    raise TypeError(f"unsupported result type: {type(value).__name__}")


def quench_payload(result):
    return {
        "energy": float(result.energy), "max_force": float(result.max_force),
        "converged": bool(result.converged), "optimizer_steps": int(result.optimizer_steps),
        "evaluation_requests": int(result.evaluation_requests), "surface": result.surface,
        "optimizer_telemetry": serial(result.optimizer_telemetry),
        "positions": np.asarray(result.atoms.positions, dtype=float).tolist(),
    }


def prepare(out: Path):
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    sys.path.insert(0, str(ROOT))
    plan = json.loads((PYTHON_RUN / "plan.json").read_text())
    backend = plan["backend"]
    if backend["model_sha256"] != sha256(Path(backend["model"])):
        raise RuntimeError("MH1 model hash mismatch")
    inputs = {}
    for case in plan["cases"]:
        checkpoint_path = PYTHON_RUN / case / "checkpoint.pkl"
        with checkpoint_path.open("rb") as handle:
            checkpoint = pickle.load(handle)
        for label, minimum in (("first", checkpoint.minima[0]), ("best", checkpoint.best)):
            target = out / "inputs" / f"{case}-{label}.traj"
            target.parent.mkdir(exist_ok=True)
            from ase.io import write
            write(target, minimum.atoms)
            inputs[f"{case}-{label}"] = {
                "source_checkpoint": str(checkpoint_path),
                "source_checkpoint_sha256": sha256(checkpoint_path),
                "source_energy_eV": float(minimum.energy),
                "prepared": str(target),
                "prepared_sha256": sha256(target),
            }
    frozen = {
        "scope": "four frozen MH1 Python minima; one Gaussian rotation/CBD diagnostic",
        "python_plan": str(PYTHON_RUN / "plan.json"),
        "python_plan_sha256": sha256(PYTHON_RUN / "plan.json"),
        "qualification_plan": str(BACKEND_PLAN),
        "qualification_plan_sha256": sha256(BACKEND_PLAN),
        "backend": {k: backend[k] for k in ("model", "model_sha256", "head", "device", "dtype")},
        "inputs": inputs,
        "direction": {"sampling": "global", "seed_by_case": plan["seeds"]},
        "common_rotation": {
            "fd_step_A": 0.001, "max_force_calls": 40,
            "frame": "ClusterFrame(direction_only)", "metric": "euclidean",
            "baseline": {"solver": "broyden-euclidean", "rotation_bias": 1.0,
                          "max_hvp": 39, "tol_eV_per_A2": 0.02},
            "recovered_cbd": {
                "pre_rotmax": 5, "rotmax": 15,
                "pre_ftol_native_eV_per_A": 1.0,
                "ftol_native_eV_per_A": 0.1,
                "normalized_pre_hvp_tol_eV_per_A2": 20.0,
                "normalized_hvp_tol_eV_per_A2": 2.0,
                "parameter_source": "c60-mh1-native-20260919/seed17093/allkeys.log; ftol/(10*fd_step)",
            },
        },
        "common_gaussian": {
            "one_gaussian": True, "width_A": float(plan["config"]["width"]),
            "height_rule": "existing paper_reference forward-force formula",
            "forward_force_eV_per_A": float(plan["config"]["forward_force"]),
            "weight": "computed per state from forward-force rule",
        },
        "quench": {"optimizer": "safe-lbfgs-total", "lbfgs_memory": 500,
                   "biased_fmax": 0.1, "true_fmax": 0.03,
                   "maxiter_each": 1000, "arm_total_cap": 1500},
        "runtime": plan.get("runtime", {}),
    }
    source_root = out / "source"
    shutil.copytree(ROOT / "pamssw", source_root / "pamssw",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    source_files = [str(path.relative_to(source_root)) for path in source_root.rglob("*.py")]
    frozen["source_files"] = {relative: sha256(source_root / relative) for relative in source_files}
    probe_copy = out / "probe_mh1_matched_cbd.py"
    shutil.copy2(Path(__file__), probe_copy)
    frozen["probe_sha256"] = sha256(probe_copy)
    (out / "plan.json").write_text(json.dumps(frozen, indent=2) + "\n")
    print(json.dumps({"prepared": str(out), "states": len(inputs)}, indent=2))


def execute(out: Path):
    plan = json.loads((out / "plan.json").read_text())
    if (out / "trace.json").exists() or (out / "summary.json").exists():
        raise FileExistsError("refusing to overwrite existing matched-CBD results")
    for relative, expected in plan["source_files"].items():
        if sha256(out / "source" / relative) != expected:
            raise RuntimeError(f"frozen source hash mismatch: {relative}")
    for state, meta in plan["inputs"].items():
        if sha256(Path(meta["prepared"])) != meta["prepared_sha256"]:
            raise RuntimeError(f"frozen input hash mismatch: {state}")
    runtime = plan["runtime"]
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", runtime.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8"))
    sys.path.insert(0, str(out / "source"))
    import torch
    from ase.io import read
    from mace.calculators import MACECalculator
    from pamssw.standalone import ASESurface
    from pamssw.standalone.broyden_direction import paper_broyden_direction
    from pamssw.standalone.cluster_frame import ClusterFrame
    from pamssw.standalone.gaussian import ProjectedGaussian
    from pamssw.standalone.paper_reference import sample_initial_direction
    from pamssw.standalone.recovered_cbd import recovered_cbd_direction
    from pamssw.standalone.surface import SurfaceCalculator, quench

    torch.set_num_threads(runtime.get("torch_num_threads", 1))
    torch.set_num_interop_threads(runtime.get("torch_num_threads", 1))
    torch.manual_seed(runtime.get("torch_manual_seed", 0))
    torch.use_deterministic_algorithms(runtime.get("torch_deterministic_algorithms", True))
    torch.backends.cuda.matmul.allow_tf32 = runtime.get("tf32", False)
    torch.backends.cudnn.allow_tf32 = runtime.get("tf32", False)
    backend = plan["backend"]
    if sha256(Path(backend["model"])) != backend["model_sha256"]:
        raise RuntimeError("MH1 model hash mismatch at execution")
    calc_kwargs = dict(model_paths=backend["model"], device=backend["device"],
                       default_dtype=backend["dtype"], enable_cueq=False, enable_oeq=False,
                       head=backend["head"])
    calc = MACECalculator(**calc_kwargs)
    surface = ASESurface(calc)
    rows = []
    started = time.monotonic()
    for state_name, state_meta in plan["inputs"].items():
        atoms = read(state_meta["prepared"])
        frame = ClusterFrame(atoms)
        case = state_name.split("-", 1)[0]
        seed = int(plan["direction"]["seed_by_case"][case])
        raw_anchor = sample_initial_direction(atoms, np.random.default_rng(seed), mode="global")
        anchor = frame.project(raw_anchor)
        if np.linalg.norm(anchor) <= np.finfo(float).eps * anchor.size:
            raise ValueError(f"projected global direction is numerically zero: {state_name}")
        anchor /= np.linalg.norm(anchor)

        for arm in ("baseline_broyden", "recovered_cbd"):
            before = surface.requests
            arm_surface = BudgetSurface(surface, plan["quench"]["arm_total_cap"])
            calc.reset()

            def rotation_surface(candidate):
                mapped = candidate.copy()
                mapped.positions = frame.positions(mapped.positions)
                energy, forces = arm_surface.evaluate(mapped)
                return energy, frame.project(forces)

            row = {"state": state_name, "seed": seed, "arm": arm,
                   "source_energy_eV": state_meta["source_energy_eV"],
                   "raw_anchor": raw_anchor.tolist(), "projected_anchor": anchor.tolist(),
                   "rotation": {}, "requests_before": before}
            try:
                if arm == "baseline_broyden":
                    cfg = plan["common_rotation"]["baseline"]
                    mode = paper_broyden_direction(
                        atoms, anchor, rotation_bias=cfg["rotation_bias"],
                        fd_step=plan["common_rotation"]["fd_step_A"],
                        max_hvp=cfg["max_hvp"], tol=cfg["tol_eV_per_A2"],
                        evaluate=rotation_surface)
                    direction = mode.direction
                    row["rotation"] = serial(mode)
                else:
                    cfg = plan["common_rotation"]["recovered_cbd"]
                    mode = recovered_cbd_direction(
                        atoms, anchor, fd_step=plan["common_rotation"]["fd_step_A"],
                        max_force_calls=plan["common_rotation"]["max_force_calls"],
                        pre_rotmax=cfg["pre_rotmax"], rotmax=cfg["rotmax"],
                        pre_ftol=cfg["normalized_pre_hvp_tol_eV_per_A2"] * 10 * plan["common_rotation"]["fd_step_A"],
                        ftol=cfg["normalized_hvp_tol_eV_per_A2"] * 10 * plan["common_rotation"]["fd_step_A"],
                        metric=plan["common_rotation"]["metric"], evaluate=rotation_surface,
                        project=frame.project)
                    direction = mode.direction
                    row["rotation"] = serial(mode)
                center = atoms.positions.copy()
                displaced = atoms.copy()
                displaced.positions += plan["common_gaussian"]["width_A"] * direction
                background = SurfaceCalculator(arm_surface, terms=())
                displaced.calc = background
                background_force = displaced.get_forces()
                force_parallel = float(np.sum(background_force * direction))
                width = plan["common_gaussian"]["width_A"]
                weight = (plan["common_gaussian"]["forward_force_eV_per_A"] - force_parallel) * width * np.exp(.5)
                row["gaussian"] = {"center": center.tolist(), "direction": direction.tolist(),
                                    "width_A": width, "weight_eV": float(weight),
                                    "background_force_parallel": force_parallel}
                if not np.isfinite(weight) or weight <= 0:
                    raise RuntimeError(f"nonpositive Gaussian height {weight}")
                term = ProjectedGaussian(center, direction, width, weight)
                qcfg = plan["quench"]
                modified = quench(displaced, arm_surface, fmax=qcfg["biased_fmax"],
                                  steps=qcfg["maxiter_each"], terms=(term,),
                                  optimizer=qcfg["optimizer"], lbfgs_memory=qcfg["lbfgs_memory"])
                row["biased_quench"] = quench_payload(modified)
                if not modified.converged:
                    row["status"] = "biased_quench_failed"
                else:
                    bare = quench(modified.atoms, arm_surface, fmax=qcfg["true_fmax"],
                                  steps=qcfg["maxiter_each"], terms=(),
                                  optimizer=qcfg["optimizer"], lbfgs_memory=qcfg["lbfgs_memory"])
                    row["true_quench"] = quench_payload(bare)
                    row["status"] = "completed" if bare.converged else "true_quench_failed"
                    try:
                        calc.reset()
                        fresh_energy, fresh_forces = arm_surface.evaluate(bare.atoms)
                        fresh_fmax = float(np.linalg.norm(fresh_forces, axis=1).max())
                        row["fresh_bare"] = {
                            "energy": float(fresh_energy), "max_force": fresh_fmax,
                            "qualified": bool(np.isfinite(fresh_energy) and np.isfinite(fresh_fmax)
                                               and fresh_fmax <= qcfg["true_fmax"]),
                        }
                    except Exception as fresh_error:
                        row["fresh_bare"] = {"qualified": False, "error": repr(fresh_error)}
            except Exception as error:
                row["status"] = "exception"
                row["error"] = repr(error)
            row["requests_after"] = surface.requests
            row["arm_requests"] = surface.requests - before
            rows.append(row)
            (out / "trace.json").write_text(json.dumps(serial(rows), indent=2) + "\n")
            print({k: row.get(k) for k in ("state", "arm", "status", "arm_requests")}, flush=True)
    (out / "summary.json").write_text(json.dumps({"rows": serial(rows), "seconds": time.monotonic() - started,
                                                   "total_requests": surface.requests}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.prepare == args.execute:
        parser.error("choose exactly one of --prepare or --execute")
    if args.prepare:
        prepare(args.output.resolve())
    else:
        execute(args.output.resolve())


if __name__ == "__main__":
    main()
