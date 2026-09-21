"""Diagnose two frozen CuO64 joint-VC failed climbs with Safe-LBFGS history.

This script replays no SSW policy: it reconstructs ``chart_reference``, the
last Gaussian and the failed climb from frozen JSON, then compares only local
quench histories 10 and 500.  It is deliberately not a search or an algorithm
effectiveness experiment.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import math
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
from ase import Atoms

from pamssw.standalone.generalized_numerics import safe_lbfgs
from pamssw.standalone.vc_geometry import ASEStressSurface, SymmetricLogStrainChart
from research.ga_ssw.compare_vc_arms import BudgetExhausted, serial


class CountedSurface(ASEStressSurface):
    def __init__(self, calculator, *, cap, deadline, path):
        super().__init__(calculator)
        self.cap, self.deadline, self.path = int(cap), float(deadline), Path(path)
        self.stage = "unknown"
        self.exhausted = False

    def evaluate(self, atoms):
        if self.requests >= self.cap:
            self.exhausted = True
            self._write(dict(request=self.requests, stage=self.stage, charged=False,
                             error="BudgetExhausted: 330-request cap", atoms=atoms.copy()))
            raise BudgetExhausted("330-request cap")
        if time.monotonic() >= self.deadline:
            self.exhausted = True
            self._write(dict(request=self.requests, stage=self.stage, charged=False,
                             error="BudgetExhausted: 300-second diagnostic cap",
                             atoms=atoms.copy()))
            raise BudgetExhausted("300-second diagnostic cap")
        before = self.requests
        try:
            e, f, s = super().evaluate(atoms)
        except Exception as error:
            self._write(dict(request=self.requests, stage=self.stage,
                             charged=self.requests > before, error=repr(error),
                             atoms=atoms.copy()))
            raise
        self._write(dict(request=self.requests, stage=self.stage, charged=True,
                         energy=e, forces=f, stress=s, atoms=atoms.copy()))
        return e, f, s

    def _write(self, row):
        with self.path.open("a") as stream:
            stream.write(json.dumps(serial(row), allow_nan=False) + "\n")


def _atoms(data):
    return Atoms(numbers=data["numbers"], positions=data["positions"],
                 cell=data["cell"], pbc=data["pbc"])


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _norm(g, natoms):
    """The exact VC norm used by run_vc_ssw, including the cell block."""
    g = np.asarray(g)
    return max(float(np.linalg.norm(g[:-6].reshape(natoms, 3), axis=1).max()),
               float(np.linalg.norm(g[-6:])))


def _frozen_bias(objective, gradient, q, gaussians):
    """Apply the exact frozen Gaussian sum used by ``vc_reference.biased``."""
    objective = float(objective)
    gradient = np.asarray(gradient, dtype=float).copy()
    q = np.asarray(q, dtype=float)
    for gaussian in gaussians:
        center = np.asarray(gaussian["center"], dtype=float)
        direction = np.asarray(gaussian["direction"], dtype=float)
        width = float(gaussian["width"])
        projection = float((q - center) @ direction)
        bias = float(gaussian["weight"]) * math.exp(-.5 * (projection / width) ** 2)
        objective += bias
        gradient -= bias * projection / width ** 2 * direction
    return objective, gradient


def _source_path(root, plan, seed):
    sources = plan.get("sources", {})
    value = sources.get(str(seed), sources.get(f"seed{seed}")) if isinstance(sources, dict) else None
    candidates = [value] if value else [f"joint-seed{seed}.json", f"seed{seed}.json"]
    for candidate in candidates:
        path = Path(candidate)
        if not path.is_absolute():
            path = root / path
        if path.is_file():
            return path
    raise FileNotFoundError(f"frozen source JSON for seed {seed} not found")


def _plan_configs(plan):
    from pamssw.standalone.paper_reference import SSWConfig
    from pamssw.standalone.block_ssw import BlockSSWConfig
    from pamssw.standalone.vc_reference import VCSSWConfig
    cfg = plan.get("joint_config", plan.get("joint", {}))
    joint = VCSSWConfig(**cfg)
    atomic_cfg = dict(width=joint.width, rotation_bias=joint.rotation_bias,
                      max_gaussians=joint.max_gaussians, temperature_K=joint.temperature_K,
                      fmax=joint.fmax, relax_steps=joint.relax_steps, fd_step=joint.fd_step,
                      rotation_hvp=joint.rotation_hvp, rotation_tol=joint.rotation_tol,
                      forward_force=joint.forward_force, direction_sampling="global",
                      rotation_solver="dimer", cluster_frame="translation_only",
                      quench_optimizer="safe-lbfgs-total", lbfgs_memory=10)
    atomic = SSWConfig(**atomic_cfg)
    block = BlockSSWConfig(atomic=atomic, quench_length=joint.strain_length,
                           pressure=joint.pressure, stress_tol=joint.stress_tol,
                           max_step=joint.max_step)
    return atomic, block, joint


def _run_one(source, record, joint, model, out, history, deadline):
    chart = SymmetricLogStrainChart(_atoms(record["chart_reference"]),
                                    strain_length=joint.strain_length)
    gaussians = record["frozen_gaussians"]
    last = gaussians[-1]
    q_start = np.asarray(last["center"], dtype=float) + float(last["width"]) * np.asarray(last["direction"], dtype=float)
    q_failed = np.asarray(record["climb"][-1]["q"], dtype=float)
    if q_start.shape != q_failed.shape or q_start.size != chart.ndof:
        raise ValueError("frozen q dimensions do not match chart")
    source_dir = out / f"history{history}"
    source_dir.mkdir(parents=True, exist_ok=False)
    calc = model()
    surface = CountedSurface(calc, cap=330, deadline=deadline,
                              path=source_dir / "evaluations.jsonl")
    surface.stage = f"safe-lbfgs-history-{history}"
    def evaluate(q):
        ev = chart.evaluate(q, surface.evaluate, pressure=joint.pressure)
        return _frozen_bias(ev.objective, chart.project(ev.gradient), q, gaussians)
    result = safe_lbfgs(q_start, evaluate,
                        gradient_norm=lambda g: _norm(g, chart.natoms),
                        step_norm=lambda g: _norm(g, chart.natoms),
                        gtol=joint.gradient_tol, max_step=joint.max_step,
                        maxiter=300, max_requests=329, lbfgs_memory=history)
    surface.stage = "fresh-final"
    surface.calculator.reset()
    final = None
    try:
        final_ev = chart.evaluate(result.q, surface.evaluate, pressure=joint.pressure)
        final_objective, final_gradient = _frozen_bias(
            final_ev.objective, chart.project(final_ev.gradient), result.q, gaussians)
        final = dict(energy=final_ev.energy, objective=final_objective,
                     projected_gradient=final_gradient,
                     projected_gradient_norm=_norm(final_gradient, chart.natoms),
                     gradient_l2=float(np.linalg.norm(final_gradient)),
                     atoms=final_ev.atoms)
    except Exception as error:
        final = dict(status="fresh_failed", error=repr(error))
    payload = dict(seed=source["seed"], history=history, cap=330,
                   source_failed_record=record["index"], chart_reference=record["chart_reference"],
                   q_start=q_start, q_failed=q_failed, frozen_gaussians=gaussians,
                   safe_lbfgs=result, final_fresh=final, requests=surface.requests,
                   censored=surface.exhausted or result.status == "request_limit",
                   norm_definition="max(max atom-force-coordinate norm, six-cell-block L2)",
                   baseline_attempted_requests=source["baseline_attempted_requests"],
                   purpose="fixed-parameter local diagnosis; not independent material evidence")
    (source_dir / "result.json").write_text(json.dumps(serial(payload), indent=2, allow_nan=False) + "\n")


def _failed_check(record, joint, model, out, seed, deadline):
    chart = SymmetricLogStrainChart(_atoms(record["chart_reference"]),
                                    strain_length=joint.strain_length)
    q_failed = np.asarray(record["climb"][-1]["q"], dtype=float)
    gaussians = record["frozen_gaussians"]
    surface = CountedSurface(model(), cap=1, deadline=deadline,
                             path=out / f"failed-check-seed{seed}.jsonl")
    surface.stage = "fresh-failed-point"
    try:
        ev = chart.evaluate(q_failed, surface.evaluate, pressure=joint.pressure)
        projected = chart.project(ev.gradient)
        objective, biased_gradient = _frozen_bias(ev.objective, projected, q_failed, gaussians)
        result = dict(status="checked", physical_energy=ev.energy,
                      physical_objective=ev.objective, biased_objective=objective,
                      physical_projected_gradient=projected,
                      biased_projected_gradient=biased_gradient,
                      physical_gradient_l2=float(np.linalg.norm(projected)),
                      biased_gradient_l2=float(np.linalg.norm(biased_gradient)),
                      recorded_relaxation_gradient_norm=record["climb"][-1]["relaxation"].get("gradient_norm"),
                      atoms=ev.atoms, requests=surface.requests)
    except Exception as error:
        result = dict(status="failed", error=repr(error), requests=surface.requests)
    (out / f"failed-check-seed{seed}.json").write_text(
        json.dumps(serial(result), indent=2, allow_nan=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root
    plan = json.loads((root / "plan.json").read_text())
    model_value = plan.get("model", plan.get("model_path"))
    if isinstance(model_value, dict):
        model_value = model_value["path"]
    model = Path(model_value)
    if not model.is_file():
        raise FileNotFoundError(model)
    expected = plan.get("model_sha256")
    if expected and _sha256(model) != expected:
        raise ValueError("model SHA256 mismatch")
    atomic, block, joint = _plan_configs(plan)
    out = root / "diagnosis"
    out.mkdir(parents=True, exist_ok=False)
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from mace.calculators import MACECalculator
    def calculator():
        return MACECalculator(model_paths=str(model), device="cuda",
                              default_dtype="float64", enable_cueq=False)
    started = time.monotonic()
    manifest = dict(status="prepared", cap_per_arm=330, total_cap=1322,
                    wall_seconds=300, histories=[10, 500], seeds=[7, 101],
                    atomic_config=asdict(atomic), block_config=asdict(block),
                    joint_config=asdict(joint), model_sha256=_sha256(model),
                    note="history500 is an existing local optimizer parameter, not a new search setting")
    (out / "plan-used.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    for seed in (7, 101):
        source_path = _source_path(root, plan, seed)
        frozen = json.loads(source_path.read_text())
        record = frozen["records"][1]
        baseline = record["climb"][-1]["relaxation"]["attempted_requests"]
        source = dict(seed=seed, baseline_attempted_requests=baseline)
        _failed_check(record, joint, calculator, out, seed, started + 300.)
        for history in (10, 500):
            _run_one(source, record, joint, calculator, out / f"seed{seed}", history,
                     started + 300.)
    (out / "status.json").write_text(json.dumps(dict(status="completed", seconds=time.monotonic()-started), indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
