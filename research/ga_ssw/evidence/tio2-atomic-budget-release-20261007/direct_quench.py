#!/usr/bin/env python3
"""Bounded direct cell-quench controls from the two saved TiO2 stage inputs."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PANEL_RUNNER = HERE / "run.py"
PLAN_PATH = HERE / "plan.json"
RELEASE_RUN_ID = "1663813"
RELEASE_SLOT = {"phase87_12": 1, "phase87_48": 3}
SEARCH_CAP = 1000
SEARCH_SECONDS = 100
TOTAL_SECONDS = 240
FRESH_CAP = 1
TRUE_QUENCH = {"strain_length": 5.0, "pressure": 0.0, "fmax": 0.05,
               "stress_tol": 0.001, "max_step": 0.2, "maxiter": 300,
               "lbfgs_memory": 500}


def load_panel_runner():
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("tio2_atomic_release_runner", PANEL_RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate(plan, panel):
    panel.validate_plan(plan)
    if plan["true_quench"] != TRUE_QUENCH:
        raise ValueError("direct quench must use the frozen true_quench plan settings")
    from ase.io import read
    sources = []
    for case in plan["cases"]:
        release_slot = RELEASE_SLOT[case["id"]]
        source = HERE / f"run-{RELEASE_RUN_ID}-{release_slot}"
        result_path = source / "result.json"
        input_path = source / "input.extxyz"
        if not result_path.is_file() or not input_path.is_file():
            raise FileNotFoundError(f"completed budget-release input evidence missing: {source}")
        result = json.loads(result_path.read_text())
        if result.get("case") != case["id"] or result.get("rotation_exit_policy") != "force_or_budget":
            raise ValueError(f"unexpected saved-stage release run: {source}")
        if result.get("status") != "complete" or not result.get("true_quench", {}).get("candidate_minimum"):
            raise ValueError(f"saved-stage release run lacks its certified minimum: {source}")
        cold = result.get("cold_checks", [])
        if not cold or cold[0].get("role") != "input_measurement":
            raise ValueError(f"existing cold input measurement missing: {source}")
        if panel.sha256(input_path) != case["input_sha256"]:
            raise ValueError(f"saved-stage input bytes differ from the frozen plan: {input_path}")
        atoms = read(input_path)
        if len(atoms) not in (12, 48) or not atoms.pbc.all() or atoms.constraints:
            raise ValueError(f"invalid direct-quench input: {input_path}")
        sources.append({"case": case, "path": input_path, "result_path": result_path,
                        "provenance_path": source / "provenance.json",
                        "result": result, "input_cold": cold[0], "source_run": source})
    return sources


def snapshot(out, panel, provenance):
    dest = out / "source-snapshot"
    dest.mkdir()
    shutil.copy2(__file__, dest / "direct_quench.py")
    shutil.copy2(PANEL_RUNNER, dest / "atomic-release-run.py")
    shutil.copy2(panel.QUAL_RUN, dest / "qualification-run.py")
    for name, path in provenance["imports"].items():
        shutil.copy2(path, dest / Path(path).name)


def stripped_write(path, atoms, **info):
    from ase.io import write
    value = atoms.copy()
    value.calc = None
    value.info.update({k: jsonable(v) for k, v in info.items()})
    write(path, value, format="extxyz")


def jsonable(value):
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def one_case(slot, source, plan, panel, provenance, slot_out, global_deadline):
    case = source["case"]
    out = slot_out
    out.mkdir(parents=True, exist_ok=False)
    helper = panel.load_helper()
    started = time.monotonic()
    search_budget = helper.CaseBudget(case["id"], out / "requests.jsonl", SEARCH_CAP,
                                      SEARCH_SECONDS, global_deadline)
    shutil.copy2(source["path"], out / "input.extxyz")
    shutil.copy2(source["result_path"], out / "source-release-result.json")
    shutil.copy2(source["provenance_path"], out / "source-release-provenance.json")
    panel.dump(out / "reused-input-cold-measurement.json", {
        "source_run": str(source["source_run"]),
        "source_role": "force_or_budget input_measurement; reused without new EFS",
        "measurement": source["input_cold"]})
    panel.dump(out / "provenance.json", {**provenance,
        "direct_input": {"path": str(source["path"]), "sha256": panel.sha256(source["path"]),
                         "case": case["id"], "source_release_run": str(source["source_run"]),
                         "source_release_slot": RELEASE_SLOT[case["id"]]},
        "reused_input_measurement": str(out / "reused-input-cold-measurement.json")})
    snapshot(out, panel, provenance)
    panel.dump(out / "effective-config.json", {"operation": "single direct cell_quench",
        "true_quench": TRUE_QUENCH, "calculator": plan["calculator"],
        "direct_budget": {"search_requests": SEARCH_CAP, "search_seconds": SEARCH_SECONDS,
                          "fresh_candidate_requests": FRESH_CAP, "shared_seconds": TOTAL_SECONDS}})

    from ase.io import read
    from mace.calculators import MACECalculator
    from pamssw.standalone.cell_relax import cell_quench
    from pamssw.standalone.vc_geometry import SymmetricLogStrainChart

    atoms = read(out / "input.extxyz")
    atoms.calc = None
    calc_spec = plan["calculator"]
    calc_kw = {"model_paths": plan["model"], **calc_spec}
    calc_kw["default_dtype"] = calc_kw.pop("dtype")
    fresh_budget = None
    search_calc = fresh_calc = surface = fresh_surface = None
    row = {"slot": slot, "case": case["id"], "natoms": len(atoms), "status": "running",
           "input_sha256": panel.sha256(out / "input.extxyz"),
           "reused_input_cold_measurement": True,
           "search_ledger": "requests.jsonl", "fresh_ledger": "fresh-requests.jsonl",
           "true_quench": None, "candidate_cold_check": None}
    quench_result = None
    quench_started = False
    quench_begin = 0
    try:
        if time.monotonic() >= global_deadline:
            raise TimeoutError("shared direct-quench deadline reached before model setup")
        search_calc = MACECalculator(**calc_kw)
        param_types = sorted({str(p.dtype) for model in search_calc.models for p in model.parameters()})
        if param_types != ["torch.float64"]:
            raise RuntimeError(f"loaded model parameter dtypes differ: {param_types}")
        row["model_parameter_dtypes"] = param_types
        helper.instrument_calculate(search_calc, search_budget)
        search_calc.reset()
        surface = helper.CountedStressSurface(search_calc, search_budget, "direct_cell_quench")
        before = search_budget.requests
        quench_begin = before
        quench_started = True
        quench_result = cell_quench(atoms, surface, **TRUE_QUENCH)
        end = search_budget.requests
        if quench_result.requests != end - before:
            raise RuntimeError("cell_quench request count did not close to the counted surface")
        ev = quench_result.evaluation
        candidate = bool(quench_result.converged and ev is not None and
            quench_result.certificate.get("certified", False) and
            float(quench_result.certificate.get("fmax", math.inf)) <= TRUE_QUENCH["fmax"] and
            float(quench_result.certificate.get("stress_max", math.inf)) <= TRUE_QUENCH["stress_tol"])
        qrow = {"status": "certified_minimum" if candidate else "uncertified_endpoint",
            "optimizer_converged": bool(quench_result.optimizer.converged),
            "optimizer_status": str(quench_result.optimizer.status),
            "certificate": jsonable(quench_result.certificate),
            "reported_requests": int(quench_result.requests),
            "ledger_request_delta": end-before,
            "ledger_request_range_1based_inclusive": [before+1, end],
            "candidate_minimum": candidate}
        endpoint = ev.atoms if ev is not None else None
        if endpoint is not None:
            metrics = {"energy_eV": float(ev.energy),
                       "fmax_eV_A": float(np.linalg.norm(ev.forces, axis=1).max()),
                       "stress_max_eV_A3": float(np.abs(ev.stress).max())}
            qrow.update(metrics)
            qrow["forces_eV_A"] = np.asarray(ev.forces, dtype=float).tolist()
            qrow["stress_eV_A3"] = np.asarray(ev.stress, dtype=float).tolist()
            qrow.update(panel.evaluate_target(endpoint, ev.energy, plan))
            stripped_write(out / ("certified-true-quench-minimum.extxyz" if candidate
                                  else "uncertified-true-quench-endpoint.extxyz"), endpoint,
                role=("certified_true_quench_minimum" if candidate else "uncertified_endpoint"),
                **metrics)
        else:
            qrow["evaluation_unavailable"] = True
            qrow["endpoint_geometry_saved"] = False
            q = getattr(quench_result.optimizer, "q", None)
            if q is not None and np.isfinite(q).all():
                chart = SymmetricLogStrainChart(atoms, strain_length=TRUE_QUENCH["strain_length"])
                endpoint = chart.unpack(q)
                stripped_write(out / "uncertified-optimizer-endpoint.extxyz", endpoint,
                               role="diagnostic_only_no_final_surface_evaluation")
                qrow["endpoint_geometry_saved"] = True
                qrow["endpoint_geometry_path"] = "uncertified-optimizer-endpoint.extxyz"
        row["true_quench"] = qrow

        if candidate:
            del surface, search_calc
            surface = search_calc = None
            row["candidate_cold_check"] = {"status": "not_started"}
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if time.monotonic() >= global_deadline:
                    raise TimeoutError("shared direct-quench deadline reached before candidate cold check")
                fresh_calc = MACECalculator(**calc_kw)
                fresh_budget = helper.CaseBudget(case["id"], out / "fresh-requests.jsonl", FRESH_CAP,
                    max(1., global_deadline-time.monotonic()), global_deadline)
                helper.instrument_calculate(fresh_calc, fresh_budget)
                fresh_surface = helper.CountedStressSurface(fresh_calc, fresh_budget, "cold_candidate")
                fresh_calc.reset()
                energy, forces, stress = fresh_surface.evaluate(endpoint)
                fmax = float(np.linalg.norm(forces, axis=1).max())
                stress_max = float(np.abs(stress).max())
                target = panel.evaluate_target(endpoint, energy, plan)
                physical = bool(np.isfinite([energy, fmax, stress_max]).all() and
                    fmax <= plan["target"]["fmax_eV_A"] and
                    stress_max <= plan["target"]["stress_tol_eV_A3"])
                reproduced = bool(abs(float(energy-ev.energy)) <= 1e-6)
                row["candidate_cold_check"] = {"status": "completed", "energy_eV": float(energy),
                    "energy_difference_from_search_eV": float(energy-ev.energy),
                    "energy_reproduced_within_1e-6_eV": reproduced,
                    "fmax_eV_A": fmax, "stress_max_eV_A3": stress_max,
                    "forces_eV_A": np.asarray(forces, dtype=float).tolist(),
                    "stress_eV_A3": np.asarray(stress, dtype=float).tolist(),
                    "physical_force_stress_passed": physical, "target": target,
                    "cold_confirmed_minimum": bool(physical and reproduced),
                    "cold_confirmed_target_candidate": bool(physical and reproduced and
                        target["both_identity_matchers_passed"] and target["energy_window_passed"])}
            except TimeoutError as exc:
                row["candidate_cold_check"] = {"status": "not_run_shared_deadline",
                                                "error": str(exc)}
            except Exception as exc:
                row["candidate_cold_check"] = {"status": "error", "error": repr(exc),
                                                "traceback": traceback.format_exc()}
        else:
            row["candidate_cold_check"] = {"status": "not_run_uncertified",
                                             "paid_requests": 0}
        row["status"] = "complete"
    except Exception as exc:
        row.update(status="exception", error=repr(exc), traceback=traceback.format_exc())
        if quench_started and row.get("true_quench") is None:
            row["true_quench"] = {"status": "exception", "error": repr(exc),
                "ledger_request_delta": search_budget.requests-quench_begin,
                "reported_requests": None}
    finally:
        if surface is not None:
            del surface
        if fresh_surface is not None:
            del fresh_surface
        if search_calc is not None:
            del search_calc
        if fresh_calc is not None:
            del fresh_calc
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        row["search_cost"] = {"requests": search_budget.requests,
            "calculator_calls": search_budget.calculator_calls, "failures": search_budget.failures,
            "denials": search_budget.denials, "boundary": search_budget.boundary,
            "elapsed_seconds": time.monotonic()-search_budget.started}
        row["fresh_cost"] = {"requests": 0 if fresh_budget is None else fresh_budget.requests,
            "calculator_calls": 0 if fresh_budget is None else fresh_budget.calculator_calls,
            "failures": 0 if fresh_budget is None else fresh_budget.failures,
            "denials": 0 if fresh_budget is None else fresh_budget.denials,
            "boundary": None if fresh_budget is None else fresh_budget.boundary,
            "elapsed_seconds": 0 if fresh_budget is None else time.monotonic()-fresh_budget.started}
        row["search_ledger_closure"] = panel.ledger_closure(out / "requests.jsonl", search_budget)
        row["fresh_ledger_closure"] = (panel.ledger_closure(out / "fresh-requests.jsonl", fresh_budget)
            if fresh_budget is not None else {"paid_ledger_rows": 0, "denial_rows": 0,
                                              "raw_calculator_calls": 0, "closed": True})
        row["search_reported_cost_closed"] = (
            (not quench_started and search_budget.requests == 0) or
            (quench_result is not None and quench_result.requests ==
             (row.get("true_quench") or {}).get("ledger_request_delta", 0)))
        row["elapsed_seconds"] = time.monotonic()-started
        panel.dump(out / "result.json", row)
    return 0 if row["status"] == "complete" else 1


def execute_all(plan, panel, sources, prov, out_root, run_id, deadline):
    if not run_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for ch in run_id):
        raise ValueError("run_id must be a nonempty filename-safe identifier")
    out_root.mkdir(parents=True, exist_ok=True)
    slot_paths = [out_root / f"direct-{run_id}-{i}" for i in range(len(sources))]
    summary_path = out_root / f"direct-{run_id}-summary.json"
    collisions = [p for p in [*slot_paths, summary_path] if p.exists()]
    if collisions:
        raise FileExistsError(f"refusing to overwrite existing direct-quench outputs: {collisions}")
    started = deadline - TOTAL_SECONDS
    rows = []
    for slot, (source, out) in enumerate(zip(sources, slot_paths)):
        try:
            code = one_case(slot, source, plan, panel, prov, out, deadline)
            result_path = out / "result.json"
            result = json.loads(result_path.read_text()) if result_path.is_file() else {}
            rows.append({"slot": slot, "output": str(out), "return_code": code,
                "status": result.get("status", "result_missing"), "case": result.get("case"),
                "search_cost": result.get("search_cost", {}),
                "fresh_cost": result.get("fresh_cost", {}),
                "candidate_minimum": (result.get("true_quench") or {}).get("candidate_minimum"),
                "target_candidate": ((result.get("true_quench") or {}).get("target_phase_energy_gate"))})
        except Exception as exc:
            row = {"slot": slot, "output": str(out), "status": "runner_exception",
                   "error": repr(exc), "traceback": traceback.format_exc()}
            if out.is_dir() and not (out / "runner-failure.json").exists():
                panel.dump(out / "runner-failure.json", row)
            rows.append(row)
    all_complete = len(rows) == 2 and all(r["status"] == "complete" for r in rows)
    cost_fields = ("requests", "calculator_calls", "failures", "denials")
    totals = {kind: {field: sum(int((r.get(f"{kind}_cost") or {}).get(field, 0) or 0)
                                for r in rows) for field in cost_fields}
              for kind in ("search", "fresh")}
    summary = {"status": "all_inputs_recorded" if len(rows) == 2 else "incomplete",
        "all_runner_slots_complete": all_complete,
        "run_id": run_id, "started_monotonic": started, "deadline_monotonic": deadline,
        "elapsed_seconds": time.monotonic()-started, "global_seconds": TOTAL_SECONDS,
        "per_input_search_request_cap": SEARCH_CAP, "per_input_search_seconds": SEARCH_SECONDS,
        "fresh_candidate_cap_per_input": FRESH_CAP,
        "totals": totals,
        "slots": rows}
    panel.dump(summary_path, summary)
    return 0 if all(r["status"] == "complete" for r in rows) else 1


def preflight(plan, panel, sources):
    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes
    from pamssw.standalone.cell_relax import cell_quench

    class Dummy(Calculator):
        implemented_properties = ["energy", "forces", "stress"]
        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.results = {"energy": 0., "forces": np.zeros_like(atoms.positions),
                            "stress": np.zeros(6)}

    with __import__("tempfile").TemporaryDirectory(prefix="tio2-direct-preflight-") as tmp:
        helper = panel.load_helper()
        atoms = Atoms("H3", positions=[[1., 1., 1.], [2.2, 1., 1.],
            [1., 2.3, 1.]], cell=np.eye(3)*8., pbc=True)
        budget = helper.CaseBudget("dummy-direct", Path(tmp)/"requests.jsonl", 2, 20,
                                   time.monotonic()+20)
        calc = Dummy()
        helper.instrument_calculate(calc, budget)
        surface = helper.CountedStressSurface(calc, budget, "dummy_direct_quench")
        result = cell_quench(atoms, surface, **TRUE_QUENCH)
        if not result.converged or result.requests != budget.requests:
            raise RuntimeError("dummy direct cell-quench/accounting check failed")
        try:
            surface.evaluate(atoms.copy())
        except RuntimeError:
            pass
        else:
            raise RuntimeError("dummy direct request cap did not deny an over-cap call")
        closure = panel.ledger_closure(Path(tmp)/"requests.jsonl", budget)
        if budget.requests != result.requests or budget.denials != 1 or not closure["closed"]:
            raise RuntimeError("dummy direct-quench ledger/cap did not close")
    return {"status": "preflight_passed", "real_ef_requests": 0,
        "saved_inputs": [{"case": x["case"]["id"], "sha256": panel.sha256(x["path"]),
                          "reused_input_cold_role": x["input_cold"]["role"]} for x in sources],
        "dummy_quench": {"converged": result.converged, "requests": result.requests,
                         "calculator_calls": budget.calculator_calls,
                         "denials": budget.denials, "ledger_closed": closure["closed"]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute-all", action="store_true")
    parser.add_argument("--out-root", type=Path)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    deadline = time.monotonic() + TOTAL_SECONDS
    panel = load_panel_runner()
    plan = json.loads(PLAN_PATH.read_text())
    sources = validate(plan, panel)
    if args.preflight:
        print(json.dumps(preflight(plan, panel, sources), indent=2, allow_nan=False))
        return 0
    if args.out_root is None or args.run_id is None:
        parser.error("--execute-all requires --out-root PATH and --run-id IDENTIFIER")
    panel.validate_plan(plan)
    provenance = panel.provenance(plan)
    provenance["direct_runner"] = str(Path(__file__).resolve())
    provenance["direct_runner_sha256"] = panel.sha256(__file__)
    provenance["panel_runner"] = str(PANEL_RUNNER)
    provenance["panel_runner_sha256"] = panel.sha256(PANEL_RUNNER)
    provenance["direct_quench_contract"] = {"search_cap": SEARCH_CAP,
        "search_seconds": SEARCH_SECONDS, "fresh_cap": FRESH_CAP,
        "global_seconds": TOTAL_SECONDS, "true_quench": TRUE_QUENCH}
    return execute_all(plan, panel, sources, provenance, args.out_root.resolve(),
                       args.run_id, deadline)


if __name__ == "__main__":
    raise SystemExit(main())
