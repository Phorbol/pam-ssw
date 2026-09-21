"""Bounded local diagnosis of four frozen Fe7C3 LS failed quenches.

For each completed source arm (ls_all/ls_filter, seeds 7/101), select the
first outer attempt whose final biased relaxation is a 300-step ``maxiter``
failure.  Reconstruct its chart, every Gaussian, and the serialized frozen
periodic-cell LS potential, then compare Safe-LBFGS histories 10 and 500.
This is a local objective diagnosis, not an SSW rerun or a search result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
from ase import Atoms

from pamssw.standalone.generalized_numerics import safe_lbfgs
from pamssw.standalone.vc_geometry import ASEStressSurface, SymmetricLogStrainChart
from pamssw.standalone.vc_softening import FrozenPeriodicCellSoftening
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
                             error=f"BudgetExhausted: {self.cap}-request cap", atoms=atoms.copy()))
            raise BudgetExhausted(f"{self.cap}-request cap")
        if time.monotonic() >= self.deadline:
            self.exhausted = True
            self._write(dict(request=self.requests, stage=self.stage, charged=False,
                             error="BudgetExhausted: 300-second diagnostic cap", atoms=atoms.copy()))
            raise BudgetExhausted("300-second diagnostic cap")
        before = self.requests
        try:
            e, f, s = super().evaluate(atoms)
        except Exception as error:
            self._write(dict(request=self.requests, stage=self.stage,
                             charged=self.requests > before, error=repr(error), atoms=atoms.copy()))
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
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _norm(g, natoms):
    g = np.asarray(g)
    return max(float(np.linalg.norm(g[:-6].reshape(natoms, 3), axis=1).max()),
               float(np.linalg.norm(g[-6:])))


def _frozen_bias(objective, gradient, q, gaussians):
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


def _softening(data):
    if not isinstance(data, dict):
        raise ValueError("source record lacks serialized frozen_softening")
    required = ("numbers", "cell", "pbc", "pairs", "image_shifts",
                "reference_distances", "strengths", "xi")
    if any(key not in data for key in required):
        raise ValueError("incomplete frozen_softening record")
    return FrozenPeriodicCellSoftening(**{key: data[key] for key in required},
                                       energy_filter=data.get("energy_filter", ()))


def _total(chart, physical, softening, q, pressure):
    ev = chart.evaluate(q, lambda atoms: _combined(physical, softening, atoms), pressure=pressure)
    return ev, ev.objective


def _combined(physical, softening, atoms):
    pe, pf, ps = physical.evaluate(atoms)
    le, lf, ls = softening.evaluate_stress(atoms)
    return float(pe + le), np.asarray(pf) + np.asarray(lf), np.asarray(ps) + np.asarray(ls)


def _pick_failed(source):
    """Pick first outer maxiter failure; budget-only failures are ignored."""
    for record in source.get("records", [])[1:]:
        for climb in record.get("climb", []):
            relaxation = climb.get("relaxation", {})
            if climb.get("status") == "maxiter" and relaxation.get("steps") == 300:
                return record, climb
    raise ValueError("no completed 300-step maxiter failure found")


def _failed_check(source, record, climb, joint, model, out, deadline):
    chart = SymmetricLogStrainChart(_atoms(record["chart_reference"]),
                                    strain_length=joint.strain_length)
    softening = _softening(record["frozen_softening"])
    gaussians = record["frozen_gaussians"]
    q_failed = np.asarray(climb["q"], dtype=float)
    surface = CountedSurface(model(), cap=1, deadline=deadline,
                             path=out / f"failed-check-seed{source['seed']}-{source['arm']}.jsonl")
    surface.stage = "fresh-failed-point"
    try:
        ev = chart.evaluate(q_failed, lambda atoms: _combined(surface, softening, atoms),
                            pressure=joint.pressure)
        softened_gradient = chart.project(ev.gradient)
        ls_ev = chart.evaluate(q_failed, lambda atoms: softening.evaluate_stress(atoms),
                               pressure=0.)
        physical_gradient = softened_gradient - ls_ev.gradient
        physical_objective = ev.objective - ls_ev.objective
        objective, biased_gradient = _frozen_bias(
            ev.objective, softened_gradient, q_failed, gaussians)
        source_gradient_l2 = float(climb["relaxation"]["gradient_norm"])
        result = dict(status="checked", q_failed=q_failed,
                      physical_objective=physical_objective,
                      softened_objective=ev.objective,
                      biased_objective=objective,
                      physical_projected_gradient=physical_gradient,
                      softened_projected_gradient=softened_gradient,
                      biased_projected_gradient=biased_gradient,
                      softened_norm=_norm(softened_gradient, chart.natoms),
                      biased_norm=_norm(biased_gradient, chart.natoms),
                      source_gradient_l2=source_gradient_l2,
                      biased_gradient_l2=float(np.linalg.norm(biased_gradient)),
                      gradient_l2_abs_diff=abs(float(np.linalg.norm(biased_gradient)) - source_gradient_l2),
                      atoms=ev.atoms, requests=surface.requests)
    except Exception as error:
        result = dict(status="failed", error=repr(error), requests=surface.requests)
    (out / f"failed-check-seed{source['seed']}-{source['arm']}.json").write_text(
        json.dumps(serial(result), indent=2, allow_nan=False) + "\n")
    return result


def _run_one(source, record, climb, joint, model, out, history, deadline):
    chart = SymmetricLogStrainChart(_atoms(record["chart_reference"]),
                                    strain_length=joint.strain_length)
    softening = _softening(record["frozen_softening"])
    gaussians = record["frozen_gaussians"]
    last = gaussians[-1]
    q_start = np.asarray(last["center"], dtype=float) + float(last["width"]) * np.asarray(last["direction"], dtype=float)
    q_failed = np.asarray(climb["q"], dtype=float)
    if q_start.shape != q_failed.shape or q_start.size != chart.ndof:
        raise ValueError("frozen q dimensions do not match chart")
    source_dir = out / f"seed{source['seed']}" / source["arm"] / f"history{history}"
    source_dir.mkdir(parents=True, exist_ok=False)
    surface = CountedSurface(model(), cap=330, deadline=deadline,
                             path=source_dir / "evaluations.jsonl")
    surface.stage = f"safe-lbfgs-history-{history}"
    def evaluate(q):
        ev = chart.evaluate(q, lambda atoms: _combined(surface, softening, atoms),
                            pressure=joint.pressure)
        return _frozen_bias(ev.objective, chart.project(ev.gradient), q, gaussians)
    try:
        result = safe_lbfgs(q_start, evaluate,
                            gradient_norm=lambda g: _norm(g, chart.natoms),
                            step_norm=lambda g: _norm(g, chart.natoms),
                            gtol=joint.gradient_tol, max_step=joint.max_step,
                            maxiter=300, max_requests=329, lbfgs_memory=history)
    except Exception as error:
        result = dict(status="exception", error=repr(error))
    surface.stage = "fresh-final"
    surface.calculator.reset()
    final = None
    if not isinstance(result, dict):
        try:
            ev = chart.evaluate(result.q, lambda atoms: _combined(surface, softening, atoms),
                                pressure=joint.pressure)
            objective, gradient = _frozen_bias(
                ev.objective, chart.project(ev.gradient), result.q, gaussians)
            final = dict(status="checked", objective=objective, softened_objective=ev.objective,
                         physical_objective=ev.objective-softening.evaluate_stress(ev.atoms)[0],
                         projected_gradient=gradient, norm=_norm(gradient, chart.natoms),
                         gradient_l2=float(np.linalg.norm(gradient)), atoms=ev.atoms)
        except Exception as error:
            final = dict(status="fresh_failed", error=repr(error))
    payload = dict(status="prepared_result", seed=source["seed"], arm=source["arm"],
                   history=history, cap=330, source_record=record["index"],
                   source_climb_index=climb["index"], chart_reference=record["chart_reference"],
                   q_start=q_start, q_failed=q_failed, frozen_gaussians=gaussians,
                   frozen_softening=record["frozen_softening"], safe_lbfgs=result,
                   final_fresh=final, requests=surface.requests,
                   censored=surface.exhausted or getattr(result, "status", None) == "request_limit",
                   norm_definition="max(max atom-force norm, six-cell-block L2)",
                   source_path=source.get("path"),
                   baseline_attempted_requests=climb["relaxation"].get("attempted_requests"),
                   purpose="fixed-parameter local diagnosis; not independent material evidence")
    (source_dir / "result.json").write_text(json.dumps(serial(payload), indent=2, allow_nan=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root
    plan = json.loads((root / "plan.json").read_text())
    model = Path(plan.get("model", plan.get("model_path")))
    if not model.is_file():
        raise FileNotFoundError(model)
    if plan.get("model_sha256") and _sha256(model) != plan["model_sha256"]:
        raise ValueError("model SHA256 mismatch")
    from pamssw.standalone.vc_reference import VCSSWConfig
    joint = VCSSWConfig(**json.loads(json.dumps(plan.get("joint", plan.get("joint_config")))))
    out = root / "frozen-quench-diagnosis"
    out.mkdir(parents=True, exist_ok=False)
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    from mace.calculators import MACECalculator
    def model_factory():
        return MACECalculator(model_paths=str(model), device="cuda",
                              default_dtype="float64", enable_cueq=False)
    started = time.monotonic(); deadline = started + 300.
    source_entries = []
    for arm in ("ls_all", "ls_filter"):
        for seed in (7, 101):
            source_path = root / "comparison" / f"{arm}-seed{seed}" / "result.json"
            source_json = json.loads(source_path.read_text())
            source_entries.append(dict(arm=arm, seed=seed, path=str(source_path),
                                       sha256=_sha256(source_path),
                                       joint_config=source_json.get("joint_config"),
                                       model=source_json.get("runtime", {}).get("model"),
                                       model_sha256=source_json.get("runtime", {}).get("model_sha256")))
    manifest = dict(status="prepared", cap_per_history_arm=330, shared_failed_checks=4,
                    total_cap=2644, wall_seconds=300, histories=[10, 500],
                    seeds=[7, 101], arms=["ls_all", "ls_filter"],
                    joint_config=dict(plan.get("joint", plan.get("joint_config"))),
                    source_entries=source_entries,
                    note="history500 is an existing local optimizer parameter; LS and all Gaussians are frozen from each source record")
    (out / "plan-used.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    ran_arms = 0
    for arm in ("ls_all", "ls_filter"):
        for seed in (7, 101):
            source_path = root / "comparison" / f"{arm}-seed{seed}" / "result.json"
            source_json = json.loads(source_path.read_text())
            record, climb = _pick_failed(source_json)
            source = dict(seed=seed, arm=arm, path=str(source_path))
            check = _failed_check(source, record, climb, joint, model_factory, out, deadline)
            if check.get("status") != "checked" or check.get("gradient_l2_abs_diff", float("inf")) > 1e-8:
                for history in (10, 500):
                    skipped = out / f"seed{seed}" / arm / f"history{history}"
                    skipped.mkdir(parents=True, exist_ok=False)
                    (skipped / "result.json").write_text(json.dumps(dict(
                        status="skipped_source_gradient_mismatch", source_check=check,
                        source_path=str(source_path), history=history), indent=2) + "\n")
                continue
            for history in (10, 500):
                _run_one(source, record, climb, joint, model_factory, out, history, deadline)
                ran_arms += 1
    final_status = "completed" if ran_arms == 8 else "stopped_source_gradient_mismatch"
    (out / "status.json").write_text(json.dumps(dict(status=final_status, ran_arms=ran_arms,
                                                     seconds=time.monotonic()-started), indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
