"""Frozen-start, variable-cell L-BFGS implementation panel (research only).

The three optimizer implementations come from ``lbfgs_baselines`` and the
existing Safe-total routine. This runner only supplies the shared VC oracle,
per-paid-request telemetry, and independent endpoint certificates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _atoms(data):
    from ase import Atoms
    if "numbers" in data:
        return Atoms(numbers=data["numbers"], positions=data["positions"],
                     cell=data["cell"], pbc=data.get("pbc", True))
    return Atoms(symbols=data["symbols"], positions=data["positions"],
                 cell=data["cell"], pbc=data.get("pbc", True))


def _read_case(spec):
    source = json.loads(Path(spec["source_result"]).read_text())
    source = source.get("result", source)
    records = [r for r in source.get("records", [])
               if r.get("index") == spec["record_index"] and r.get("climb")]
    if len(records) != 1:
        raise ValueError(f"{spec['name']}: expected one selected search record")
    record = records[0]
    climbs = {int(c["index"]): c for c in record["climb"]
              if c.get("index") is not None}
    selected = climbs.get(spec["climb_index"])
    if not isinstance(selected, dict) or not isinstance(selected.get("q"), list):
        raise ValueError(f"{spec['name']}: selected climb endpoint lacks saved q")
    if record.get("chart_reference"):
        reference = record["chart_reference"]
        atoms = _atoms(reference)
    elif spec.get("source_chart_input"):
        from ase.io import read
        atoms = read(str(spec["source_chart_input"]), index=0)
    else:
        raise ValueError(f"{spec['name']}: explicit chart_reference or source_chart_input required")
    from pamssw.standalone.vc_geometry import SymmetricLogStrainChart
    chart = SymmetricLogStrainChart(atoms, strain_length=spec["strain_length_A"])
    q_saved = np.asarray(selected["q"], float)
    if q_saved.shape != (chart.ndof,):
        raise ValueError(f"{spec['name']}: selected q dimension differs from chart")
    if not record.get("chart_reference") and spec.get("source_chart_input"):
        # The chart is anchored to the original input, while result.initial is
        # the post-initial-quench state. Validate that saved state round-trips
        # in the original chart before reconstructing any frozen Gaussian.
        initial_state = source.get("initial", {})
        initial_atoms = _atoms(initial_state)
        initial_rebuilt = chart.unpack(chart.pack(initial_atoms))
        if (not np.allclose(initial_rebuilt.positions, initial_atoms.positions,
                            atol=1e-9, rtol=0) or
                not np.allclose(initial_rebuilt.cell.array, initial_atoms.cell.array,
                                atol=1e-9, rtol=0)):
            raise ValueError(f"{spec['name']}: result.initial does not round-trip in original input chart")
        if "cell" not in selected or not np.allclose(chart.unpack(q_saved).cell.array,
                np.asarray(selected["cell"], float), atol=1e-8, rtol=0):
            raise ValueError(f"{spec['name']}: saved q cell differs from selected climb cell")

    # Keep only Gaussians active at the selected climb. For the older AlOH
    # schema, centers are prior accepted climb q values, starting at q(initial).
    if record.get("frozen_gaussians"):
        terms = record["frozen_gaussians"][:spec["climb_index"] + 1]
    else:
        terms = []
        center = chart.pack(_atoms(source["initial"]))
        for index in range(spec["climb_index"] + 1):
            event = climbs.get(index)
            if not isinstance(event, dict) or not all(
                    k in event for k in ("direction", "weight", "q")):
                raise ValueError(f"{spec['name']}: incomplete climb {index}")
            terms.append({"center": center.tolist(), "direction": event["direction"],
                          "weight": event["weight"], "width": spec["width_A"]})
            center = np.asarray(event["q"], float)
    if len(terms) != spec["climb_index"] + 1:
        raise ValueError(f"{spec['name']}: Gaussian history is not truncated to selected climb")
    q0 = (np.asarray(terms[-1]["center"], float) +
          float(terms[-1]["width"]) * np.asarray(terms[-1]["direction"], float))
    if q0.shape != q_saved.shape or not np.isfinite(q0).all():
        raise ValueError(f"{spec['name']}: reconstructed Gaussian start is invalid")
    # The saved source objective is the unbiased physical enthalpy at the climb
    # endpoint, not the frozen biased objective. Preserve it as an independent
    # provenance value; do not pretend it validates the reconstructed bias.
    return atoms, chart, q0, q_saved, terms, record, selected


def _qnorm(g, natoms):
    g = np.asarray(g, float)
    return max(float(np.linalg.norm(g[:-6].reshape(natoms, 3), axis=1).max()),
               float(np.linalg.norm(g[-6:])))


def _biased_value_gradient(q, energy, gradient, terms):
    """Add frozen projected-Gaussian terms to an energy and its q gradient."""
    value = float(energy)
    grad = np.asarray(gradient, float).copy()
    q = np.asarray(q, float)
    for term in terms:
        center = np.asarray(term["center"], float)
        direction = np.asarray(term["direction"], float)
        width, weight = float(term["width"]), float(term["weight"])
        projection = float((q-center) @ direction)
        bias = weight * math.exp(-.5 * (projection/width)**2)
        value += bias
        grad -= bias * projection / width**2 * direction
    return value, grad


class Counted:
    """Shared objective; cache is keyed by the exact chart vector."""
    def __init__(self, calculator, cap, deadline, ledger_path, chart, spec, terms, phase):
        from pamssw.standalone.vc_geometry import ASEStressSurface
        self.surface = ASEStressSurface(calculator)
        self.cap, self.deadline = cap, deadline
        self.path, self.chart, self.spec = Path(ledger_path), chart, spec
        self.terms, self.phase, self.cache = terms, phase, {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=False)

    @property
    def requests(self):
        return self.surface.requests

    def evaluate(self, q):
        q = np.asarray(q, float).reshape(-1)
        key = q.tobytes()
        if key in self.cache:
            return self.cache[key]
        if self.requests >= self.cap:
            raise RuntimeError("stage_request_cap")
        if time.monotonic() >= self.deadline:
            raise RuntimeError("stage_wall_cap")
        before = self.requests
        try:
            ev = self.chart.evaluate(q, self.surface.evaluate,
                                     pressure=self.spec["pressure_eV_A3"])
            objective, gradient = _biased_value_gradient(q, ev.objective,
                self.chart.project(ev.gradient), self.terms)
            row = {"request": self.requests, "phase": self.phase, "q": q.tolist(),
                   "objective_eV": float(objective),
                   "biased_common_gradient_norm_eV_A": _qnorm(gradient, self.chart.natoms)
                   if self.terms else None,
                   "energy_eV": float(ev.energy), "volume_A3": float(ev.volume),
                   "fmax_eV_A": float(np.linalg.norm(ev.forces, axis=1).max()),
                   "pressure_residual_stress_max_eV_A3": float(np.abs(
                       ev.stress + self.spec["pressure_eV_A3"]*np.eye(3)).max()),
                   "gradient": gradient.tolist()}
            value = (float(objective), gradient.copy(), ev, row)
            self.cache[key] = value
            with self.path.open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            return value
        except Exception as exc:
            charged = self.requests > before
            with self.path.open("a") as stream:
                stream.write(json.dumps({"request": self.requests, "phase": self.phase,
                                         "charged": charged, "error": repr(exc)}) + "\n")
            raise

    def cached_row(self, q):
        """Read telemetry only when the exact point was already evaluated."""
        item = self.cache.get(np.asarray(q, float).reshape(-1).tobytes())
        if item is None:
            raise RuntimeError("accepted-point telemetry requested for uncached q")
        return item[3]


def _physical_certificate(row, spec):
    return max(row["fmax_eV_A"] / spec["fmax_eV_A"],
               row["pressure_residual_stress_max_eV_A3"] /
               spec["stress_tol_eV_A3"])


def _trace_rows(result, counted, phase, spec):
    rows, seen = [], set()
    for point in result.accepted_trace:
        q = np.asarray(point["q"], float)
        key = q.tobytes()
        if key in seen:
            continue
        seen.add(key)
        raw = counted.cached_row(q)  # cache-only; never triggers an evaluation
        row = {k: raw[k] for k in ("request", "q", "volume_A3", "fmax_eV_A",
              "pressure_residual_stress_max_eV_A3", "biased_common_gradient_norm_eV_A")}
        row["common_physical_certificate"] = _physical_certificate(raw, spec)
        row["accepted"] = True
        rows.append(row)
    return rows


def run_stage(method, phase, q0, spec, chart, terms, calculator_factory, outdir, deadline):
    from pamssw.standalone.generalized_numerics import safe_lbfgs
    from research.ga_ssw.lbfgs_baselines import scipy_lbfgsb, ase_lbfgs_linesearch

    outdir.mkdir(parents=True, exist_ok=False)
    ledger = outdir / "requests.jsonl"
    counted = Counted(calculator_factory(), spec["request_cap_per_stage"], deadline,
                      ledger, chart, spec, terms, phase)
    tol = spec["gradient_tol_eV_A"]
    def evaluate(q):
        energy, grad, _, _ = counted.evaluate(q)
        return energy, grad
    def gnorm(g):
        return _qnorm(g, chart.natoms)
    def stepnorm(dx):
        return _qnorm(dx, chart.natoms)
    def common(q, gradient):
        row = counted.cached_row(q)
        return gnorm(gradient) if phase == "biased" else _physical_certificate(row, spec)

    if method == "safe-lbfgs-total":
        result = safe_lbfgs(q0, evaluate, gradient_norm=gnorm, step_norm=stepnorm,
            gtol=tol if phase == "biased" else 1.0,
            max_step=spec["max_step_A"], maxiter=spec["maxiter_per_stage"],
            max_requests=spec["request_cap_per_stage"], convergence_norm=common,
            lbfgs_memory=spec["lbfgs_history_pairs"])
        accepted_trace = result.trace
        terminal_q = result.q
        native = {"status": result.status, "steps": result.steps,
                  "oracle_requests": result.requests, "rejected_trials": result.rejected_trials,
                  "accepted_secants": result.accepted_secants,
                  "rejected_secants": result.rejected_secants, "error": result.error}
    elif method == "scipy-lbfgsb":
        # The biased common norm is an L2-block maximum; SciPy stops on its
        # native projected infinity norm. This sufficient conversion is the
        # predeclared VC contract. For the unbiased task, retain the same q
        # gradient native scale and report physical first passage separately.
        native_gtol = tol / math.sqrt(6.0)
        result = scipy_lbfgsb(q0, evaluate, gradient_norm=gnorm, step_norm=stepnorm,
            gtol=tol if phase == "biased" else 1.0,
            maxiter=spec["maxiter_per_stage"],
            max_requests=spec["request_cap_per_stage"], convergence_norm=common,
            maxcor=spec["lbfgs_history_pairs"], native_gtol=native_gtol)
        accepted_trace = result.accepted_trace
        terminal_q = result.q
        native = {"status": result.status, "native_success": result.native_success,
                  "native_message": result.native_message, "steps": result.steps,
                  "oracle_requests": result.requests, **result.metadata}
    elif method == "ase-lbfgs-linesearch":
        native_fmax = tol / math.sqrt(2.0)
        result = ase_lbfgs_linesearch(q0, evaluate, gradient_norm=gnorm, step_norm=stepnorm,
            gtol=tol if phase == "biased" else 1.0,
            maxiter=spec["maxiter_per_stage"],
            max_requests=spec["request_cap_per_stage"], convergence_norm=common,
            memory=spec["lbfgs_history_pairs"], native_fmax=native_fmax)
        accepted_trace = result.accepted_trace
        terminal_q = result.q
        native = {"status": result.status, "native_success": result.native_success,
                  "native_message": result.native_message, "steps": result.steps,
                  "oracle_requests": result.requests, **result.metadata}
    else:
        raise ValueError(f"unknown optimizer: {method}")

    rows = _trace_rows(type("Accepted", (), {"accepted_trace": accepted_trace})(), counted,
                       phase, spec)
    try:
        terminal = counted.cached_row(terminal_q)
    except RuntimeError:
        terminal = None
    if phase == "biased":
        first = lambda threshold: next((r["request"] for r in rows
            if r["biased_common_gradient_norm_eV_A"] is not None and
            r["biased_common_gradient_norm_eV_A"] <= threshold), None)
        passage = {str(v): first(v) for v in (0.1, 0.05)}
        qualified = passage[str(tol)]
    else:
        passage = {}
        qualified = next((r["request"] for r in rows
                          if r["common_physical_certificate"] <= 1.0), None)
    return {"method": method, "phase": phase, "status": result.status,
            "native_termination": native, "search_requests": counted.requests,
            "accepted_iterates": rows, "first_common_qualified_request": qualified,
            "gradient_first_passage_requests": passage,
            "terminal_q": np.asarray(terminal_q).tolist(), "terminal": terminal,
            "request_ledger": str(ledger),
            "native_step_cap_note": "Safe joint chart cap; ASE triplet cap 0.2; SciPy has no equivalent"}


def _atom_dict(atoms):
    return {"numbers": atoms.numbers.tolist(), "positions": atoms.positions.tolist(),
            "cell": atoms.cell.array.tolist(), "pbc": atoms.pbc.tolist()}


def fresh_endpoint(q, spec, chart, terms, calculator_factory):
    from pamssw.standalone.vc_geometry import ASEStressSurface
    surface = ASEStressSurface(calculator_factory())
    try:
        ev = chart.evaluate(q, surface.evaluate, pressure=spec["pressure_eV_A3"])
    except Exception as exc:
        return {"requests": surface.requests,
                "error": f"{type(exc).__name__}: {exc}", "physical_certificate": False}
    gradient = chart.project(ev.gradient)
    objective, gradient = _biased_value_gradient(q, ev.objective, gradient, terms)
    row = {"requests": surface.requests, "energy_eV": float(ev.energy),
           "objective_eV": float(objective), "volume_A3": float(ev.volume),
           "fmax_eV_A": float(np.linalg.norm(ev.forces, axis=1).max()),
           "pressure_residual_stress_max_eV_A3": float(np.abs(
               ev.stress + spec["pressure_eV_A3"]*np.eye(3)).max()),
           "biased_common_gradient_norm_eV_A": _qnorm(gradient, chart.natoms) if terms else None,
           "physical_certificate": False, "atoms": _atom_dict(ev.atoms)}
    row["physical_certificate"] = (row["fmax_eV_A"] <= spec["fmax_eV_A"] and
                                    row["pressure_residual_stress_max_eV_A3"] <= spec["stress_tol_eV_A3"])
    return row


def validate_saved_physical_objective(q_saved, selected, spec, chart, calculator_factory):
    """One explicit E/F/stress call to check archived physical E+pV at q_saved."""
    from pamssw.standalone.vc_geometry import ASEStressSurface
    surface = ASEStressSurface(calculator_factory())
    try:
        ev = chart.evaluate(q_saved, surface.evaluate, pressure=spec["pressure_eV_A3"])
    except Exception as exc:
        return {"requests": surface.requests, "status": "validation_failed",
                "error": f"{type(exc).__name__}: {exc}"}
    source_value = selected.get("objective")
    if source_value is None or not np.isfinite(float(source_value)):
        return {"requests": surface.requests, "status": "missing_source_physical_objective",
                "recomputed_objective_eV": float(ev.objective)}
    difference = float(ev.objective) - float(source_value)
    tolerance = 1e-5
    return {"requests": surface.requests, "status": "pass" if abs(difference) <= tolerance else "mismatch",
            "source_physical_objective_eV": float(source_value),
            "recomputed_physical_objective_eV": float(ev.objective),
            "difference_eV": difference, "absolute_tolerance_eV": tolerance}


def _failed_stage(method, phase, q0, stage_dir, exc):
    ledger = stage_dir / "requests.jsonl"
    requests = set()
    if ledger.is_file():
        for line in ledger.open():
            try:
                row = json.loads(line)
                if int(row.get("request", 0) or 0) > 0:
                    requests.add(int(row["request"]))
            except (ValueError, TypeError, json.JSONDecodeError):
                continue
    return {"method": method, "phase": phase, "status": "runner_exception",
            "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
            "search_requests": len(requests), "accepted_iterates": [],
            "first_common_qualified_request": None, "terminal_q": np.asarray(q0).tolist(),
            "terminal": None, "request_ledger": str(ledger)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    if not args.execute:
        checks = []
        for spec in plan["cases"]:
            atoms, chart, q0, q_saved, terms, record, selected = _read_case(spec)
            checks.append({"case": spec["name"], "natoms": len(atoms), "ndof": chart.ndof,
                "terms_retained": len(terms), "selected_climb_index": selected["index"],
                "q_start_size": len(q0), "q_saved_size": len(q_saved),
                "start_volume_A3": float(atoms.get_volume()),
                "source_chart_reference_present": bool(record.get("chart_reference")),
                "source_frozen_objective_gradient_reconstruction_check":
                    "not_available: selected source stores physical climb objective, not frozen biased objective/gradient"})
        (out / "preflight.json").write_text(json.dumps(
            {"status": "preflight_ok_no_PES", "cases": checks}, indent=2) + "\n")
        return

    import ase, scipy, torch
    from mace.calculators import MACECalculator
    model = Path(plan["model"])
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    if digest != plan["model_sha256"]:
        raise ValueError("model hash does not match frozen plan")
    torch.set_num_threads(plan["calculator"]["torch_num_threads"])
    torch.set_num_interop_threads(plan["calculator"]["torch_num_interop_threads"])
    def calculator_factory():
        return MACECalculator(model_paths=str(model), device=plan["device"],
            default_dtype=plan["dtype"], head=plan["calculator"]["head"],
            enable_cueq=plan["calculator"]["enable_cueq"],
            enable_oeq=plan["calculator"]["enable_oeq"])
    (out / "environment.json").write_text(json.dumps({"model": str(model), "sha256": digest,
        "device": plan["device"], "torch": torch.__version__, "ase": ase.__version__,
        "scipy": scipy.__version__, "calculator": plan["calculator"]}, indent=2) + "\n")
    summary, arms = [], []
    for source_spec in plan["cases"]:
        spec = {**source_spec,
            "maxiter_per_stage": plan["budget"]["maxiter_per_stage"],
            "request_cap_per_stage": plan["budget"]["request_cap_per_stage"],
            "wall_seconds_per_arm": plan["budget"]["wall_seconds_per_arm"],
            "max_step_A": plan["common_contract"]["max_step_A"],
            "lbfgs_history_pairs": plan["lbfgs_history_pairs"]}
        try:
            atoms, chart, q0, q_saved, terms, record, selected = _read_case(spec)
        except Exception as exc:
            for method in plan["optimizers"]:
                failure = {"case": spec["name"], "method": method,
                           "status": "case_preparation_failed",
                           "error": f"{type(exc).__name__}: {exc}"}
                summary.append(failure)
                arms.append(failure)
            continue
        case_dir = out / spec["name"]
        case_dir.mkdir()
        case_calc_factory = calculator_factory
        source_validation = validate_saved_physical_objective(
            q_saved, selected, spec, chart, case_calc_factory)
        (case_dir / "source-objective-validation.json").write_text(
            json.dumps(source_validation, indent=2, allow_nan=False) + "\n")
        (case_dir / "frozen-objective.json").write_text(json.dumps({"case": spec,
            "chart_atoms": _atom_dict(atoms), "q_start": q0.tolist(),
            "source_q_saved": q_saved.tolist(), "gaussians_through_selected_climb": terms,
            "source_record_index": record.get("index"), "source_climb_index": selected.get("index"),
            "source_physical_objective_validation": source_validation,
            "objective_reconstruction_validation": "source does not archive biased objective/gradient at q_saved; Gaussian target formula is independently checked by CPU finite-difference test"},
            indent=2) + "\n")
        for method in plan["optimizers"]:
            started = time.monotonic()
            deadline = started + spec["wall_seconds_per_arm"]
            calc_factory = calculator_factory
            method_dir = case_dir / method
            method_dir.mkdir()
            try:
                biased = run_stage(method, "biased", q0, spec, chart, terms,
                                   calc_factory, method_dir / "biased", deadline)
            except Exception as exc:
                biased = _failed_stage(method, "biased", q0, method_dir / "biased", exc)
            # The unbiased cell-relax task begins at the identical archived
            # q_saved for every backend, independent of biased outcomes.
            try:
                unbiased = run_stage(method, "unbiased", q_saved, spec, chart, [],
                                     calc_factory, method_dir / "unbiased", deadline)
            except Exception as exc:
                unbiased = _failed_stage(method, "unbiased", q_saved, method_dir / "unbiased", exc)
            for stage, stage_terms in ((biased, terms), (unbiased, [])):
                if stage.get("terminal") is None:
                    stage["fresh_endpoint"] = {"requests": 0,
                        "error": "no_paid_terminal_evaluation", "physical_certificate": False}
                    continue
                try:
                    stage["fresh_endpoint"] = fresh_endpoint(
                        np.asarray(stage["terminal_q"]), spec, chart, stage_terms, calc_factory)
                except Exception as exc:
                    stage["fresh_endpoint"] = {"requests": 0, "error":
                        f"{type(exc).__name__}: {exc}", "physical_certificate": False}
            arm = {"case": spec["name"], "method": method, "biased_stage": biased,
                   "unbiased_cell_relax_stage": unbiased, "unbiased_start_q": q_saved.tolist(),
                   "wall_seconds": time.monotonic() - started}
            path = method_dir / "result.json"
            path.write_text(json.dumps(arm, indent=2, allow_nan=False) + "\n")
            arms.append(arm)
            summary.append({"case": spec["name"], "method": method,
                "biased_requests": biased["search_requests"],
                "biased_first_passage": biased["gradient_first_passage_requests"],
                "unbiased_requests": unbiased["search_requests"],
                "unbiased_first_certificate_request": unbiased["first_common_qualified_request"],
                "biased_fresh_certificate": biased["fresh_endpoint"]["physical_certificate"],
                "unbiased_fresh_certificate": unbiased["fresh_endpoint"]["physical_certificate"],
                "wall_seconds": arm["wall_seconds"]})
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
