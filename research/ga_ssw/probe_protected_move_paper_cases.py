#!/usr/bin/env python3
"""Bounded paired protected-move S2 paper-case probe.

Preparation is the default. ``--execute`` runs the frozen MACE panel and
``--smoke`` exercises the same one-step path with a synthetic CPU calculator.
Neither mode submits a scheduler job.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_DATA_ROOT = Path("/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification")
DATA_ROOT = Path(os.environ.get("PAMSSW_RESEARCH_ROOT", DEFAULT_DATA_ROOT)).expanduser().resolve()
OUT_ROOT = DATA_ROOT / "research/ga_ssw/evidence/protected-move-20261012"
PROTOCOL = DATA_ROOT / "research/ga_ssw/evidence/protected-move-20261012/protocol.md"
BASE_RUNNER = ROOT / "research/ga_ssw/probe_q_paper_cases.py"
SEED = 20261012
SEARCH_CAP = 3000
SMOKE_SEARCH_CAP = 500
SMOKE_ARM_WALL_SECONDS = 120
FRESH_CAP = 1
ARM_WALL_SECONDS = 7 * 60
CAMPAIGN_WALL_SECONDS = 30 * 60
MAX_GAUSSIANS = 6
ARMS = ("protected-off", "protected-on")


def load_base(data_root: Path = DATA_ROOT):
    global DATA_ROOT, PROTOCOL
    DATA_ROOT = Path(data_root).expanduser().resolve()
    PROTOCOL = DATA_ROOT / "research/ga_ssw/evidence/protected-move-20261012/protocol.md"
    base_runner = BASE_RUNNER
    spec = importlib.util.spec_from_file_location("protected_move_base_q_probe", base_runner)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen Q paper-case probe: {base_runner}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.configure_data_root(data_root)
    # Keep only the two frozen cases required by this panel.
    module.CASES = {name: module.CASES[name] for name in ("c60-isomer2", "tio2-phase87")}
    module._protected_move_source_path = base_runner.resolve()
    # Delegate the established serializers and hashing implementation.
    globals().update(sha256=module.sha256, dump=module.dump, serial=module.serial)
    return module


def api_preflight(base):
    import inspect
    from pamssw.standalone import run_ssw
    from pamssw.standalone.protected_move import ScalarMovePolicy
    parameters = set(inspect.signature(run_ssw).parameters)
    missing = {"move_policy"} - parameters
    if missing:
        raise RuntimeError(f"run_ssw API is missing protected-move fields: {sorted(missing)}")
    policy = ScalarMovePolicy()
    prepared = base.prepare()
    return {"run_ssw_move_policy_supported": True,
            "policy_type": f"{type(policy).__module__}.{type(policy).__qualname__}",
            "policy": serial(policy), "base_probe_preflight": prepared}


def current_revision():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True, stderr=subprocess.STDOUT).strip()
    except Exception as error:
        return f"unavailable: {error!r}"


class HarmonicSmokeCalculator:
    """ASE calculator factory target used only for plumbing smoke."""
    def __new__(cls, *, reference_positions, **kwargs):
        from ase.calculators.calculator import Calculator, all_changes
        reference = np.asarray(reference_positions, dtype=float).copy()

        class Harmonic(Calculator):
            implemented_properties = ["energy", "forces"]

            def __init__(self):
                super().__init__()
                self.reference_positions = reference

            def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
                super().calculate(atoms, properties, system_changes)
                displacement = np.asarray(self.atoms.positions, dtype=float) - self.reference_positions
                self.results["energy"] = float(0.5 * np.sum(displacement * displacement))
                self.results["forces"] = -displacement

        return Harmonic()


def _record_move_diagnostics(result):
    rows = []
    for record in result.records:
        for stage_index, stage in enumerate(record.climb):
            diagnostic_key = next((key for key in ("protected_move", "move_diagnostics")
                                   if stage.get(key) is not None), None)
            if diagnostic_key is not None:
                rows.append({"outer_index": record.index, "stage_index": stage_index,
                             "gaussian_index": stage.get("index"), "status": stage.get("status"),
                             "source_key": diagnostic_key,
                             "diagnostics": serial(stage[diagnostic_key])})
    return rows


def _execute_impl(out: Path, data_root: Path = DATA_ROOT, *, smoke=False):
    import ase
    import torch
    from ase.io import write
    from pamssw.standalone import SSWConfig, run_ssw
    from pamssw.standalone.protected_move import ScalarMovePolicy
    from pamssw.standalone.recovered_direction import RecoveredDirectionSettings

    base = load_base(data_root)
    if out.exists():
        raise FileExistsError(f"output must be a new directory: {out}")
    inputs = base.validate_inputs()
    prepared = api_preflight(base)
    ledger = base.load_module("protected_move_paper_ledger", base.LEDGER)
    if smoke:
        # This wiring smoke checks one outer-step plumbing, not the recovered
        # CBD solver. The quadratic oracle can make native acos roundoff
        # degenerate, so isolate only that solver in this explicitly synthetic
        # mode; the MACE execution branch never patches it.
        import pamssw.standalone.recovered_cbd as cbd

        def identity_cbd(atoms, anchor, **kwargs):
            return cbd.RecoveredCBDResult(
                anchor.copy(), 0.0, 0.0, 0.0, 0, "CBD_unbiasedRot",
                True, True, "force_converged", 0.0, (), anchor.copy())

        cbd.recovered_cbd_direction = identity_cbd
        calculator_factory = HarmonicSmokeCalculator
        calculator_args = "per-case-reference"
        backend = "synthetic CPU harmonic calculator (wiring only)"
    else:
        import mace
        from mace.calculators import MACECalculator
        calculator_factory = MACECalculator
        calculator_args = None
        backend = "frozen MACE models on cuda; one V100 intended"

    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cap = SMOKE_SEARCH_CAP if smoke else SEARCH_CAP
    wall = SMOKE_ARM_WALL_SECONDS if smoke else ARM_WALL_SECONDS
    out.mkdir(parents=True)
    shutil.copy2(Path(__file__).resolve(), out / Path(__file__).name)
    started = time.monotonic()
    record = {"status": "running", "scientific_run": not smoke,
              "scope": "paired one-outer-step fixed-cell S2 protected-move development comparison",
              "protocol": str(PROTOCOL), "seed": SEED,
              "limits": {"cases": list(base.CASES), "arms": ARMS, "outer_steps_per_arm": 1,
                         "max_gaussians": MAX_GAUSSIANS, "search_requests_per_arm": cap,
                         "fresh_requests_per_arm": FRESH_CAP, "arm_wall_seconds": wall,
                         "campaign_wall_seconds": CAMPAIGN_WALL_SECONDS,
                         "gpu_count": 0 if smoke else 1},
              "backend": backend, "checkout_head": current_revision(),
              "runner": str(Path(__file__).resolve()), "runner_sha256": sha256(Path(__file__).resolve()),
              "base_probe": str(base._protected_move_source_path),
              "base_probe_sha256": sha256(base._protected_move_source_path),
              "protocol_sha256": sha256(PROTOCOL), "sources": {}, "preflight": prepared,
              "runs": [], "search_requests_total": 0, "fresh_requests_total": 0}
    from pamssw.standalone import paper_reference
    from pamssw.standalone import protected_move as protected_move_module
    from pamssw.standalone import recovered_cbd as recovered_cbd_module
    frozen_sources = [Path(paper_reference.__file__).resolve(),
                      Path(protected_move_module.__file__).resolve(),
                      Path(recovered_cbd_module.__file__).resolve()]
    record["source_files"] = {str(path): {"sha256": sha256(path)} for path in frozen_sources}
    record["source_files"][str(Path(__file__).resolve())] = {"sha256": sha256(Path(__file__).resolve())}
    record["source_files"][str(base._protected_move_source_path)] = {
        "sha256": sha256(base._protected_move_source_path)}
    if smoke:
        record["smoke_direction_stub"] = {
            "identity": "identity_cbd_stub_v1", "scope": "--smoke only",
            "signature": "identity_cbd(atoms, anchor, **kwargs)",
            "result": "RecoveredCBDResult(direction=anchor, stage=CBD_unbiasedRot, converged=True)",
            "scientific_evidence": False}
    if not smoke:
        import mace
        import scipy
        record["runtime"] = {"python": sys.version, "platform": __import__("platform").platform(),
                             "torch": torch.__version__, "mace": mace.__version__,
                             "ase": ase.__version__, "numpy": np.__version__, "scipy": scipy.__version__}
    else:
        record["runtime"] = {"python": sys.version, "platform": __import__("platform").platform(),
                             "torch": torch.__version__, "ase": ase.__version__, "numpy": np.__version__}
    for name, spec in base.CASES.items():
        record["sources"][name] = {key: str(spec[key]) for key in ("input", "qualification", "model")}
        record["sources"][name].update(input_sha256=sha256(spec["input"]),
                                        model_sha256=sha256(spec["model"]), head=spec["head"])
        if spec.get("qualification_sha256"):
            record["sources"][name]["qualification_sha256"] = sha256(spec["qualification"])
        if spec.get("manifest"):
            record["sources"][name]["manifest"] = str(spec["manifest"])
            record["sources"][name]["manifest_sha256"] = sha256(spec["manifest"])
    dump(out / "provenance.json", record)

    for case_name, atoms in inputs.items():
        spec = base.CASES[case_name]
        case_preflight = prepared["base_probe_preflight"][case_name]
        if smoke:
            calc_args = {"reference_positions": atoms.positions.copy()}
        elif calculator_args is None:
            calc_args = dict(model_paths=str(spec["model"]), head=spec["head"], device="cuda",
                             default_dtype="float64", enable_cueq=False, enable_oeq=False)
        else:
            calc_args = dict(calculator_args)
        config = SSWConfig(**case_preflight["api_preflight"]["ssw_config"])
        mode = base.descriptor_mode(atoms, "S2", spec)
        settings = RecoveredDirectionSettings(
            ratio_local=50, local_probability=0.5, group_threshold=0.5,
            pre_rotmax=5, rotmax=15, pre_ftol=0.2, ftol=0.02,
            metric="euclidean", max_force_calls=40, c1_radius_policy="restricted",
            geometry=("periodic_local" if spec["periodic"] else "nonperiodic"),
            descriptor_modes=(mode,), descriptor_probability=1.0)
        record.setdefault("effective_case_configs", {})[case_name] = {
            "input": str(spec["input"]), "input_sha256": sha256(spec["input"]),
            "qualification": str(spec["qualification"]),
            "qualification_manifest": str(spec.get("manifest", spec["qualification"])),
            "model": str(spec["model"]), "model_sha256": sha256(spec["model"]),
            "head": spec["head"], "ssw_config": asdict(config),
            "recovered_direction": serial(settings),
            "geometry_diagnostics": base.geometry_diagnostics(atoms)}
        dump(out / "provenance.json", record)
        for arm in ARMS:
            if time.monotonic() - started >= CAMPAIGN_WALL_SECONDS:
                record["status"] = "campaign_wall_limit"
                dump(out / "provenance.json", record)
                if smoke:
                    raise RuntimeError("CPU smoke hit the campaign wall before all four arms completed")
                return record
            folder = out / f"{case_name}-{arm}"
            folder.mkdir()
            write(folder / "source-input.extxyz", atoms)
            policy = None if arm == "protected-off" else ScalarMovePolicy()
            row = {"case": case_name, "arm": arm, "seed": SEED,
                   "move_policy": "None" if policy is None else serial(policy),
                   "config": asdict(config), "recovered_direction": serial(settings),
                   "cell_fixed": True, "status": "started", "smoke_only": smoke}
            dump(folder / "summary.json", row)
            arm_started = time.monotonic()
            result = None
            partial = None
            surface = None
            counter = {"calls": 0}
            search_ledger = folder / "requests.jsonl"
            search_ledger.touch()
            try:
                calculator = calculator_factory(**calc_args)
                counter = ledger.instrument_calculate(calculator)
                arm_remaining = max(0.0, wall - (time.monotonic() - arm_started))
                campaign_remaining = max(0.0, CAMPAIGN_WALL_SECONDS - (time.monotonic() - started))
                search_wall = min(arm_remaining, campaign_remaining)
                surface = ledger.CountedSurface(calculator, search_ledger, cap, search_wall)
                result = run_ssw(atoms.copy(), surface, steps=1, config=config,
                                 rng=np.random.default_rng(SEED), recovered_direction=settings,
                                 move_policy=policy)
                row["status"] = result.status
                dump(folder / "result.json", result)
                if result.minima:
                    write(folder / "minima.extxyz", [item.atoms for item in result.minima])
                row.update(outer_records=len(result.records),
                           search_requests_reported=result.evaluation_requests,
                           records=[{"index": rec.index, "status": rec.status,
                                     "accepted": rec.accepted, "gaussians": len(rec.climb),
                                     "requests": rec.evaluation_requests,
                                     "landing_surface": None if rec.landing is None else rec.landing.surface,
                                     "landing_energy_eV": None if rec.landing is None else rec.landing.energy,
                                     "landing_fmax_eV_A": None if rec.landing is None else rec.landing.max_force}
                                    for rec in result.records],
                           move_diagnostics=_record_move_diagnostics(result))
            except Exception as error:
                row.update(status="exception", error=repr(error), traceback=traceback.format_exc())
                partial = getattr(error, "result", None)
                if partial is not None:
                    dump(folder / "partial-result.json", partial)
                    row["partial_result_available"] = True
                    result = partial
                row["failure_class"] = ("algorithm_or_budget" if partial is not None or
                                         (surface is not None and surface.boundary)
                                         else "hard_calculator_or_setup_failure")

            search_finished = time.monotonic()
            search_requests = 0 if surface is None else surface.requests
            search_denials = 0 if surface is None else surface.denials
            search_boundary = None if surface is None else surface.boundary
            try:
                with search_ledger.open() as stream:
                    row["search_failures"] = sum(
                        json.loads(line).get("kind") == "search_failure" for line in stream)
            except FileNotFoundError:
                row["search_failures"] = 0
            row.update(search_requests=search_requests, search_denials=search_denials,
                       search_boundary=search_boundary, search_calculate_calls=counter["calls"],
                       search_elapsed_seconds=search_finished - arm_started)
            if row["search_failures"]:
                row["failure_class"] = "hard_calculator_failure_during_search"

            endpoint = None
            endpoint_label = "unavailable"
            endpoint_kind = "unavailable"
            if result is not None and result.records:
                last = result.records[-1]
                if last.landing is not None:
                    endpoint, endpoint_label, endpoint_kind = last.landing.atoms, "last-outer-landing", "landing"
                elif last.last_atoms is not None:
                    endpoint, endpoint_label, endpoint_kind = last.last_atoms, "last-failed-state", "diagnostic_only"
            if endpoint is None:
                row["fresh_endpoint"] = {"status": "unavailable_no_landing_or_failed_state",
                                          "qualified": False, "requests": 0}
                row["fresh_requests"] = 0
                row["fresh_calculate_calls"] = 0
                fresh_finished = time.monotonic()
            else:
                row["endpoint_label"] = endpoint_label
                row["endpoint_kind"] = endpoint_kind
                fresh_ledger = folder / "fresh-requests.jsonl"
                fresh_ledger.touch()
                fresh_counter = {"calls": 0}
                fresh_surface = None
                fresh_started = time.monotonic()
                try:
                    write(folder / "endpoint-before-fresh.extxyz", endpoint)
                    fresh_calc = calculator_factory(**calc_args)
                    fresh_counter = ledger.instrument_calculate(fresh_calc)
                    campaign_remaining = max(0.0, CAMPAIGN_WALL_SECONDS - (time.monotonic() - started))
                    arm_remaining = max(0.0, wall - (fresh_started - arm_started))
                    fresh_surface = ledger.CountedSurface(
                        fresh_calc, fresh_ledger, FRESH_CAP, min(campaign_remaining, arm_remaining))
                    energy, forces = fresh_surface.evaluate(endpoint)
                    fmax = float(np.linalg.norm(forces, axis=1).max())
                    row["fresh_endpoint"] = {
                        "target": endpoint_label, "endpoint_kind": endpoint_kind,
                        "energy_eV": energy, "fmax_eV_A": fmax,
                        "force_qualified": bool(fmax <= config.fmax) if endpoint_kind == "landing" else None,
                        "same_numbers": bool(np.array_equal(endpoint.numbers, atoms.numbers)),
                        "same_cell": bool(np.array_equal(endpoint.cell.array, atoms.cell.array)),
                        "same_pbc": bool(np.array_equal(endpoint.pbc, atoms.pbc)),
                        "geometry_diagnostics": base.geometry_diagnostics(endpoint)}
                    write(folder / "endpoint-fresh.extxyz", endpoint)
                except Exception as fresh_error:
                    row["fresh_endpoint"] = {"target": endpoint_label, "endpoint_kind": endpoint_kind,
                                              "error": repr(fresh_error), "qualified": False}
                    row["fresh_failure_class"] = (
                        "budget_censored" if fresh_surface is not None and fresh_surface.boundary
                        else "hard_calculator_failure")
                fresh_finished = time.monotonic()
                try:
                    with fresh_ledger.open() as stream:
                        row["fresh_failures"] = sum(
                            json.loads(line).get("kind") == "search_failure" for line in stream)
                except FileNotFoundError:
                    row["fresh_failures"] = 0
                row["fresh_requests"] = 0 if fresh_surface is None else fresh_surface.requests
                row["fresh_denials"] = 0 if fresh_surface is None else fresh_surface.denials
                row["fresh_boundary"] = None if fresh_surface is None else fresh_surface.boundary
                row["fresh_calculate_calls"] = fresh_counter["calls"]
            row["fresh_elapsed_seconds"] = fresh_finished - (search_finished if endpoint is None else fresh_started)
            row["arm_elapsed_seconds"] = time.monotonic() - arm_started
            dump(folder / "summary.json", row)
            record["runs"].append(row)
            record["search_requests_total"] += search_requests
            record["fresh_requests_total"] += int(row.get("fresh_requests", 0))
            record["actual_calculate_calls_total"] = sum(
                item.get("search_calculate_calls", 0) + item.get("fresh_calculate_calls", 0)
                for item in record["runs"])
            record["elapsed_seconds"] = time.monotonic() - started
            dump(out / "provenance.json", record)
            hard_failure = (str(row.get("failure_class", "")).startswith("hard_") or
                            row.get("failure_class") == "hard_calculator_failure_during_search" or
                            row.get("fresh_failure_class") == "hard_calculator_failure")
            if hard_failure:
                record["status"] = "aborted_hard_calculator_failure"
                record["abort_run_id"] = folder.name
                dump(out / "provenance.json", record)
                if smoke:
                    raise RuntimeError(f"CPU smoke hard failure in {folder.name}")
                return record
            if smoke and (search_boundary or row.get("fresh_boundary")):
                row["smoke_boundary"] = True
                dump(folder / "summary.json", row)
                record["status"] = "smoke_censored"
                dump(out / "provenance.json", record)
                raise RuntimeError(f"CPU smoke hit a request or wall boundary in {folder.name}")
    record["status"] = "all_4_arms_recorded"
    if smoke:
        expected = {(case, arm) for case in base.CASES for arm in ARMS}
        observed = {(run["case"], run["arm"]) for run in record["runs"]}
        invalid = [run for run in record["runs"] if (
            str(run.get("failure_class", "")).startswith("hard_") or
            run.get("fresh_failure_class") == "hard_calculator_failure" or
            run.get("search_boundary") or run.get("fresh_boundary") or
            run.get("search_failures", 0) or run.get("fresh_failures", 0) or
            run.get("status") != "completed" or run.get("outer_records") != 1)]
        if observed != expected or len(record["runs"]) != 4 or invalid:
            record["status"] = "smoke_failed_validation"
            record["smoke_validation"] = {"expected_arms": sorted(expected),
                                           "observed_arms": sorted(observed),
                                           "invalid_runs": [f"{run['case']}/{run['arm']}" for run in invalid]}
        else:
            record["status"] = "smoke_complete"
    record["elapsed_seconds"] = time.monotonic() - started
    dump(out / "provenance.json", record)
    if smoke and record["status"] != "smoke_complete":
        raise RuntimeError(f"CPU smoke validation failed: {record['smoke_validation']}")
    return record


def execute(out: Path, data_root: Path = DATA_ROOT, *, smoke=False):
    # Keep the synthetic stub scoped to this call even when the runner is used
    # programmatically and a later scientific execution shares the process.
    import pamssw.standalone.recovered_cbd as cbd
    original_cbd = cbd.recovered_cbd_direction
    try:
        return _execute_impl(out, data_root, smoke=smoke)
    finally:
        cbd.recovered_cbd_direction = original_cbd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path, help="new output directory")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT,
                        help="checkout holding frozen inputs, models, and shared probe helpers")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--execute", action="store_true", help="execute bounded frozen MACE comparison")
    group.add_argument("--smoke", action="store_true", help="CPU-only synthetic-calculator wiring smoke")
    args = parser.parse_args()
    data_root = args.data_root.expanduser().resolve()
    base = load_base(data_root)
    out = args.out.expanduser().resolve()
    if not args.execute and not args.smoke:
        prepared = api_preflight(base)
        print(json.dumps({"status": "prepared_not_executed", "cases": list(base.CASES),
                          "arms": ARMS, "max_search_requests": 4 * SEARCH_CAP,
                          "max_fresh_requests": 4 * FRESH_CAP,
                          "preflight": prepared, "out": str(out)}, indent=2))
        return
    result = execute(out, data_root, smoke=args.smoke)
    print(json.dumps({"status": result["status"], "runs": len(result["runs"]),
                      "search_requests": result["search_requests_total"],
                      "fresh_requests": result["fresh_requests_total"],
                      "scientific_run": result["scientific_run"], "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
