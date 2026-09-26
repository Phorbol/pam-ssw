"""One isolated full joint-VC optimizer arm with bounded E/F/stress accounting."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
from ase import Atoms

from research.ga_ssw.compare_vc_arms import serial as _json
from pamssw.standalone.vc_geometry import ASEStressSurface
from pamssw.standalone.vc_reference import VCSSWConfig, run_vc_ssw
from research.ga_ssw.vc_lbfgs_baseline_runner import temporary_lbfgs_baseline


def _git_provenance():
    def git(*args):
        result = subprocess.run(["git", *args], text=True, capture_output=True, check=True)
        return result.stdout.strip()
    head = git("rev-parse", "HEAD")
    status = git("status", "--short")
    tracked = [Path("research/ga_ssw/run_vc_e2e_optimizer_panel.py"),
               Path("research/ga_ssw/vc_lbfgs_baseline_runner.py"),
               Path("research/ga_ssw/lbfgs_baselines.py"),
               Path("pamssw/standalone/vc_reference.py"),
               Path("pamssw/standalone/cell_relax.py")]
    return {"git_head": head, "git_status_short": status,
            "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in tracked}}


def _environment():
    packages = {}
    for name in ("numpy", "scipy", "ase", "mace-torch", "torch"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        import torch
        runtime = {"torch": torch.__version__, "torch_num_threads": torch.get_num_threads(),
                   "torch_num_interop_threads": torch.get_num_interop_threads()}
    except ImportError:
        runtime = {}
    return {"python": __import__("sys").version, "packages": packages, **runtime}


class CappedSurface(ASEStressSurface):
    """Counted search E/F/stress surface; denials and failed calls are explicit."""
    def __init__(self, calculator, *, request_cap, deadline, ledger_path):
        super().__init__(calculator)
        self.request_cap = int(request_cap)
        self.deadline = float(deadline)
        self.started = time.monotonic()
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.ledger_path.touch(exist_ok=False)
        self.attempts = 0
        self.budget_censor = False
        self.censor_reason = None

    def _append(self, row):
        with self.ledger_path.open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")

    def evaluate(self, atoms):
        self.attempts += 1
        attempt = self.attempts
        before = self.requests
        started = time.monotonic()
        start_row = {"event": "attempt_started", "attempt": attempt,
                     "request_candidate": before + 1,
                     "elapsed_seconds": started - self.started}
        self._append(start_row)
        reason = None
        if before >= self.request_cap:
            reason = "request_cap"
        elif started >= self.deadline:
            reason = "wall_deadline"
        if reason is not None:
            self.budget_censor = True
            self.censor_reason = self.censor_reason or reason
            self._append({"event": "budget_censor", "attempt": attempt,
                          "request": None, "charged": False, "reason": reason,
                          "elapsed_seconds": started - self.started})
            raise RuntimeError(f"vc_e2e_budget_{reason}")
        try:
            energy, forces, stress = super().evaluate(atoms)
        except Exception as exc:
            self._append({"event": "attempt_error", "attempt": attempt,
                          "request": self.requests if self.requests > before else None,
                          "charged": self.requests > before,
                          "error": f"{type(exc).__name__}: {exc}",
                          "elapsed_seconds": time.monotonic() - self.started})
            raise
        self._append({"event": "attempt_completed", "attempt": attempt,
            "request": self.requests, "charged": self.requests > before,
            "energy_eV": energy, "volume_A3": float(atoms.get_volume()),
            "fmax_eV_A": float(np.linalg.norm(forces, axis=1).max()),
            "stress_max_eV_A3": float(np.abs(stress).max()),
            "elapsed_seconds": time.monotonic() - self.started})
        return energy, forces, stress


def _physical_certificate(forces, stress, config):
    fmax = float(np.linalg.norm(forces, axis=1).max())
    residual = np.asarray(stress) + config.pressure * np.eye(3)
    stress_max = float(np.abs(residual).max())
    return {"fmax_eV_A": fmax, "stress_residual_max_eV_A3": stress_max,
            "force_pass": fmax <= config.fmax,
            "stress_pass": stress_max <= config.stress_tol,
            "certified": fmax <= config.fmax and stress_max <= config.stress_tol}


def _landing_candidates(result):
    candidates = []
    initial = getattr(result, "initial", None)
    if initial is not None:
        candidates.append({"source": "initial", "record_index": -1,
                           "accepted": True, "source_certificate": None,
                           "evaluation": initial})
    for row in getattr(result, "records", ()):
        if not isinstance(row, dict) or row.get("index") is None:
            continue
        landing = row.get("landing")
        if landing is None:
            continue
        candidates.append({"source": "landing", "record_index": int(row["index"]),
                           "record_status": row.get("status"),
                           "accepted": row.get("accepted"),
                           "source_certificate": row.get("certificate"),
                           "evaluation": landing})
    return candidates


def _fresh_checks(candidates, calculator_factory, config, path):
    surface = ASEStressSurface(calculator_factory())
    checks = []
    ledger = Path(path)
    ledger.touch(exist_ok=False)
    for index, candidate in enumerate(candidates):
        ev = candidate["evaluation"]
        before = surface.requests
        with ledger.open("a") as stream:
            stream.write(json.dumps({"event": "fresh_started", "fresh_index": index,
                "request_candidate": before + 1, "charged": None}, allow_nan=False) + "\n")
        row = {k: v for k, v in candidate.items() if k != "evaluation"}
        row["fresh_index"] = index
        try:
            reset = getattr(surface.calculator, "reset", None)
            if callable(reset):
                reset()
            energy, forces, stress = surface.evaluate(ev.atoms)
            row.update(status="checked", requests=surface.requests - before,
                       energy_eV=float(energy), objective_eV=float(energy + config.pressure * ev.atoms.get_volume()),
                       energy_error_eV=float(energy - ev.energy),
                       physical_certificate=_physical_certificate(forces, stress, config),
                       atoms=_json(ev.atoms))
            with ledger.open("a") as stream:
                stream.write(json.dumps({"event": "fresh_completed", "fresh_index": index,
                    "request": surface.requests, "charged": True,
                    "status": row["status"], "physical_certificate": row["physical_certificate"]},
                    allow_nan=False) + "\n")
        except Exception as exc:
            row.update(status="fresh_error", requests=surface.requests - before,
                       error=f"{type(exc).__name__}: {exc}")
            with ledger.open("a") as stream:
                stream.write(json.dumps({"event": "fresh_error", "fresh_index": index,
                    "request": surface.requests if surface.requests > before else None,
                    "charged": surface.requests > before, "error": row["error"]}) + "\n")
        checks.append(row)
    return {"checks": checks, "requests": surface.requests,
            "candidate_count": len(candidates), "qualified_count": sum(
                row.get("physical_certificate", {}).get("certified", False) for row in checks),
            "ledger": str(ledger)}


def run_arm(atoms, calculator_factory, config, *, method, seed, steps,
            request_cap, search_seconds, output_dir, case_name):
    """Run one full VC search arm and separately certify recorded endpoints."""
    if method not in {"safe_total", "ase", "scipy"}:
        raise ValueError("method must be safe_total, ase, or scipy")
    if request_cap < 1 or search_seconds <= 0 or steps < 0:
        raise ValueError("request_cap/search_seconds must be positive; steps nonnegative")
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    atoms = atoms.copy()
    config_data = asdict(config)
    task = {"case": case_name, "method": method, "seed": int(seed), "steps": int(steps),
        "request_cap": int(request_cap), "search_seconds": float(search_seconds),
        "config": config_data, "input": _json(atoms), "environment": _environment(),
        "provenance": _git_provenance(),
        "optimizer_adapter": "Safe direct" if method == "safe_total" else
            {"kind": method, "cell_native_gradient_tol": config.gradient_tol /
             math.sqrt(6.0 if method == "scipy" else 2.0),
             "biased_native_tol": "temporary bridge applies norm-contract conversion"}}
    (out / "task.json").write_text(json.dumps(task, indent=2, allow_nan=False) + "\n")
    (out / "input.json").write_text(json.dumps(task["input"], indent=2) + "\n")
    (out / "config.json").write_text(json.dumps(config_data, indent=2) + "\n")

    started = time.monotonic()
    surface = CappedSurface(calculator_factory(), request_cap=request_cap,
        deadline=started + search_seconds, ledger_path=out / "search-ledger.jsonl")
    calls = []
    result = None
    run_error = None
    try:
        kwargs = {"steps": steps, "config": config,
                  "rng": np.random.default_rng(seed)}
        if method == "safe_total":
            result = run_vc_ssw(atoms, surface, **kwargs)
        else:
            native_tol = config.gradient_tol / math.sqrt(6.0 if method == "scipy" else 2.0)
            with temporary_lbfgs_baseline(method, calls,
                                          native_gradient_tol=native_tol):
                result = run_vc_ssw(atoms, surface, **kwargs)
    except Exception as exc:
        run_error = f"{type(exc).__name__}: {exc}"

    algorithm_status = None if result is None else result.status
    if surface.budget_censor:
        status = "budget_censored"
    elif run_error is not None:
        status = "exception"
    else:
        status = algorithm_status
    result_payload = {"status": status, "algorithm_status": algorithm_status,
        "budget_censor": surface.budget_censor, "censor_reason": surface.censor_reason,
        "run_error": run_error, "search_requests": surface.requests,
        "adapter_calls": _json(calls),
        "result": None if result is None else _json(result)}
    # Persist all search results before any independent calculator construction.
    (out / "search-result.json").write_text(
        json.dumps(result_payload, indent=2, allow_nan=False) + "\n")
    candidates = [] if result is None else _landing_candidates(result)
    if len(candidates) > 1 + steps:
        raise RuntimeError("candidate fresh-check count exceeds initial plus one landing per outer step")
    fresh = _fresh_checks(candidates, calculator_factory, config, out / "fresh-ledger.jsonl")
    (out / "fresh-checks.json").write_text(json.dumps(fresh, indent=2, allow_nan=False) + "\n")
    elapsed = time.monotonic() - started
    summary = {"case": case_name, "method": method, "seed": int(seed),
        "status": status, "algorithm_status": algorithm_status,
        "budget_censor": surface.budget_censor, "censor_reason": surface.censor_reason,
        "search_requests": surface.requests, "fresh_requests": fresh["requests"],
        "total_requests": surface.requests + fresh["requests"],
        "outer_attempt_records": 0 if result is None else sum(
            isinstance(row, dict) and row.get("index") is not None for row in result.records),
        "successful_minima_in_core_result": 0 if result is None else len(result.minima),
        "fresh_candidate_count": fresh["candidate_count"],
        "fresh_qualified_count": fresh["qualified_count"],
        "adapter_call_count": len(calls), "wall_seconds": elapsed,
        "run_error": run_error, "search_result": str(out / "search-result.json"),
        "search_ledger": str(out / "search-ledger.jsonl"),
        "fresh_checks": str(out / "fresh-checks.json"),
        "fresh_ledger": str(out / "fresh-ledger.jsonl")}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--case-index", type=int, required=True)
    parser.add_argument("--method", choices=("safe_total", "ase", "scipy"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text())
    case = plan["cases"][args.case_index]
    if args.method not in plan["methods"]:
        raise ValueError("method is not declared in frozen plan")
    if args.seed not in plan["seeds"]:
        raise ValueError("seed is not declared in frozen plan")
    import torch
    torch.set_num_threads(plan["calculator"]["torch_num_threads"])
    torch.set_num_interop_threads(plan["calculator"]["torch_num_interop_threads"])
    model = Path(plan["model"])
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    if digest != plan["model_sha256"]:
        raise ValueError("MACE model hash does not match frozen plan")
    from mace.calculators import MACECalculator
    def factory():
        return MACECalculator(model_paths=str(model), device=plan["device"],
            default_dtype=plan["dtype"], head=plan["calculator"]["head"],
            enable_cueq=plan["calculator"]["enable_cueq"],
            enable_oeq=plan["calculator"]["enable_oeq"])
    atoms = Atoms(numbers=case["input"]["numbers"], positions=case["input"]["positions"],
        cell=case["input"]["cell"], pbc=case["input"].get("pbc", True)) if "numbers" in case["input"] else \
        Atoms(symbols=case["input"]["symbols"], positions=case["input"]["positions"],
              cell=case["input"]["cell"], pbc=True)
    config = VCSSWConfig(**case["config"])
    summary = run_arm(atoms, factory, config, method=args.method, seed=args.seed,
        steps=plan["steps"], request_cap=plan["budget"]["search_requests_per_arm"],
        search_seconds=plan["budget"]["search_seconds_per_arm"],
        output_dir=args.out, case_name=case["name"])
    print(json.dumps(summary, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
