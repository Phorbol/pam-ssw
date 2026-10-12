#!/usr/bin/env python3
"""Frozen one-outer source ablation panel. Preflight is zero-PES; execution is opt-in."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
COORD = HERE.parents[3]
SEEDS = (26101231, 26101232, 26101233, 26101234)
METHODS = {
    "full": (COORD, "6c70ef4b1a568866f084a9670a01eeae496d79fc"),
    "local_only": (Path("/home/gengjianrui/bin/pam-ssw-worktrees/direction-local-only-20261012"), "a13ab976a04aff821fa4a0352a23274f56ed8387"),
    "feedback_only": (Path("/home/gengjianrui/bin/pam-ssw-worktrees/direction-feedback-only-20261012"), "7e2a30945427a32dfd3b173bbf7ecdf0ee9a1d94"),
}
INPUTS = {
    "lj38": Path("/home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity/research/ga_ssw/evidence/lj38-source-20260912/optim-finish.extxyz"),
    "lj55": COORD / "research/ga_ssw/evidence/lj55-direction-target-20261007/prepared-1663031/lj55-seed26100721/prepared.extxyz",
    "c60": COORD / "research/ga_ssw/evidence/c60-source3-paper-ls-transfer-20261007/prepared-20261007-a/seed-26100791/ssw/initial-fresh.traj",
}
C60_PLAN = COORD / "research/ga_ssw/evidence/c60-source3-paper-ls-transfer-20261007/prepared-20261007-a/seed-26100791/ssw/plan.json"
C60_MODEL_SHA256 = "a522eb7f59c7879963d41586528f4980baf33e086c94aa92e3eafdeccad3be47"
_LJ_HELPER = None


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def panel_contract():
    return {"cases": ["lj38", "lj55", "c60"], "methods": list(METHODS),
            "seeds": list(SEEDS), "arm_count": 36, "outer_steps_per_arm": 1,
            "search_cap_per_arm": 2000, "fresh_cap_per_arm": 1,
            "max_search_requests": 72000, "max_fresh_requests": 36,
            "lj_case_wall_seconds": 600, "lj_total_cpu_minutes": 20,
            "c60_allocation_wall_seconds": 1800}


def second_gaussian_exposures(step):
    """A zero-release diagnostic still counts as entering Gaussian index >=1."""
    return [dict(item) for item in (step.get("climb") or [])
            if int(item.get("index", -1)) >= 1 and
            item.get("recovered_direction") is not None]


def load_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def source_probe(method):
    root, expected_head = METHODS[method]
    if not root.is_dir():
        raise FileNotFoundError(root)
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    core_tree = subprocess.check_output(["git", "-C", str(root), "rev-parse", f"{expected_head}:pamssw"], text=True).strip()
    actual_core_tree = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD:pamssw"], text=True).strip()
    if actual_core_tree != core_tree:
        raise RuntimeError(f"{method} expected core tree {core_tree} from {expected_head}, got {actual_core_tree}")
    sys.path.insert(0, str(root))
    import pamssw
    import pamssw.standalone.paper_reference as paper
    import pamssw.standalone.recovered_direction as recovered
    import pamssw.standalone.native_direction_control as native_direction
    actual = {"pamssw": Path(pamssw.__file__).resolve(),
              "paper_reference": Path(paper.__file__).resolve(),
              "recovered_direction": Path(recovered.__file__).resolve(),
              "native_direction_control": Path(native_direction.__file__).resolve()}
    for name, path in actual.items():
        if not path.is_relative_to(root.resolve()):
            raise RuntimeError(f"import hijack for {name}: {path} not under {root}")
    dirty = subprocess.run(["git", "-C", str(root), "diff", "--quiet", "HEAD", "--", "pamssw"]).returncode
    if dirty:
        raise RuntimeError(f"core files are dirty in source root {root}")
    from pamssw.standalone import ASESurface, NativeMCSettings, SSWConfig, run_ssw
    from pamssw.standalone.recovered_direction import RecoveredDirectionSettings
    if not all(callable(x) for x in (ASESurface, run_ssw, SSWConfig, NativeMCSettings,
                                     RecoveredDirectionSettings)):
        raise RuntimeError("required SSW APIs are not importable")
    if method == "full":
        from research.ga_ssw.full_pair_lj import FullPairLJ
        assert FullPairLJ is not None
    # Instantiate every frozen scientific settings object in this source-root
    # process. Dict serialization alone cannot catch API/field-name drift.
    settings_checks = {}
    for case in INPUTS:
        protocol = case_protocol(case)
        settings_checks[case] = {
            "ssw_config": asdict(SSWConfig(**protocol["ssw_config"])),
            "direction": asdict(RecoveredDirectionSettings(**protocol["direction"])),
            "native_mc": asdict(NativeMCSettings(**protocol["native_mc"])),
        }
    return {"root": str(root), "git_head": head, "expected_core_commit": expected_head,
            "core_tree": actual_core_tree,
            "actual_modules": {k: str(v) for k, v in actual.items()},
            "native_direction_control_sha256": sha256(actual["native_direction_control"]),
            "settings_instantiated": settings_checks}


def preflight():
    import numpy as np
    from ase.io import read
    rows = []
    for case, path in INPUTS.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        atoms = read(path, index=0)
        want = 60 if case == "c60" else int(case[2:])
        expected_symbols = {"Ar"} if case.startswith("lj") else {"C"}
        constraints = [type(c).__name__ for c in atoms.constraints]
        if len(atoms) != want or atoms.pbc.any() or constraints or set(atoms.get_chemical_symbols()) != expected_symbols:
            raise ValueError(f"{case}: atom/cell/constraint contract failed: n={len(atoms)}, pbc={atoms.pbc}, constraints={constraints}, symbols={set(atoms.get_chemical_symbols())}")
        if not np.isfinite(atoms.positions).all():
            raise ValueError(f"{case}: input has nonfinite positions")
        rows.append({"case": case, "input": str(path), "sha256": sha256(path),
                     "natoms": len(atoms), "symbols": sorted(set(atoms.get_chemical_symbols())),
                     "pbc": atoms.pbc.tolist(), "constraints": constraints,
                     "cell_A": atoms.cell.array.tolist()})
    probe = {}
    for method in METHODS:
        output = subprocess.check_output([sys.executable, str(Path(__file__).resolve()),
            "--source-probe", method], text=True)
        probe[method] = json.loads(output)
    plan = json.loads(C60_PLAN.read_text())
    model_hash = sha256(plan["model"])
    if model_hash != C60_MODEL_SHA256 or plan["model_sha256"] != C60_MODEL_SHA256:
        raise RuntimeError("frozen MH-1 model SHA256 mismatch")
    lj = case_protocol("lj38")
    c60 = case_protocol("c60")
    protocols = {"lj38": {k: v for k, v in lj.items() if k != "helper"},
                 "c60": {k: v for k, v in c60.items() if k != "plan"}}
    mace_check = json.loads(subprocess.check_output([sys.executable, str(Path(__file__).resolve()),
        "--mace-probe"], text=True))
    return {"status": "zero_pes_preflight_ok" if mace_check["available"] else "preflight_c60_backend_import_blocked",
            "pes_requests": 0,
            "contract": panel_contract(), "inputs": rows, "sources": probe,
            "c60_model": plan["model"], "c60_model_sha256": model_hash,
            "protocols": protocols, "c60_backend_import": mace_check}


def mace_probe():
    try:
        from mace.calculators import MACECalculator
        return {"available": MACECalculator is not None}
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def import_existing_lj_helpers():
    global _LJ_HELPER
    if _LJ_HELPER is not None:
        return _LJ_HELPER
    path = COORD / "research/ga_ssw/evidence/lj55-direction-target-20261007/run.py"
    _LJ_HELPER = load_file(path, "direction_panel_lj55_helpers")
    return _LJ_HELPER


def case_protocol(case):
    if case in ("lj38", "lj55"):
        helper = import_existing_lj_helpers()
        config, _rotation, direction, height, mc, temp = helper.PANEL.settings()
        direction = replace(direction, c1_radius_policy="per_atom")
        return {"backend": "FullPairLJ(epsilon=1,sigma=2.7,untruncated,nonperiodic)",
                "ssw_config": asdict(config), "direction": asdict(direction),
                "native_height": asdict(height), "native_mc": asdict(mc),
                "temperature_K": temp, "fmax": .05, "helper": helper,
                "target_energy_eV": -173.928427 if case == "lj38" else -279.248470,
                "target_energy_tolerance_eV": .001}
    plan = json.loads(C60_PLAN.read_text())
    native_mc = {"energy_tol": plan["native_mc"]["energy_tol_eV"],
                 "maxtrap": plan["native_mc"]["maxtrap"]}
    return {"backend": "MH-1 / omol / float64 / cuda", "ssw_config": plan["ssw_config"],
            "direction": plan["recovered_direction"], "native_mc": native_mc,
            "temperature_K": 150., "fmax": .03, "plan": plan,
            "target_energy_eV": plan["reference_energy_eV"], "target_energy_tolerance_eV": .01}


def create_calculator(case, protocol):
    if case == "c60":
        if sha256(protocol["plan"]["model"]) != C60_MODEL_SHA256:
            raise RuntimeError("C60 MH-1 model differs from frozen SHA256")
        import torch
        from mace.calculators import MACECalculator
        torch.set_num_threads(1)
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        return MACECalculator(model_paths=protocol["plan"]["model"], head="omol", device="cuda",
                              default_dtype="float64", enable_cueq=False, enable_oeq=False)
    from research.ga_ssw.full_pair_lj import FullPairLJ
    return FullPairLJ(epsilon=1., sigma=2.7)


def structural_diagnostics(case, atoms):
    import numpy as np
    if case.startswith("lj"):
        helper = import_existing_lj_helpers()
        comps = helper.PANEL.connected_components(atoms)
        row = {"connected_component_sizes_1p3sigma": comps, "full_cluster": comps == [len(atoms)]}
        if row["full_cluster"]:
            from ase import Atoms
            n = len(atoms)
            xyz = np.loadtxt(COORD / "research/ga_ssw/evidence/cluster-paper-reproduction-20260925/references" /
                             ("lj38.points" if case == "lj38" else "lj55.points"), dtype=float)
            reference = Atoms("Ar" + str(n), positions=2.7 * xyz, cell=atoms.cell.array, pbc=False)
            reference.positions += np.asarray(atoms.cell.array).diagonal() / 2.0 - reference.positions.mean(axis=0)
            row["published_GM_geometry"] = helper.ANALYSIS.compare_geometry(reference, atoms)
        return row
    path = COORD / "research/ga_ssw/evidence/c60-source3-paper-ls-transfer-20261007/prepared-20261007-a/validator.py"
    validator = load_file(path, "direction_panel_c60_validator")
    return {"graphs": {str(c): validator.graph_row(atoms.numbers, atoms.positions, c)
                       for c in (1.64, 1.70, 1.80)}}


def run_seed(case, method, seed, out, source, calc, protocol, deadline):
    import numpy as np
    from ase.io import read, write
    from pamssw.standalone import ASESurface, NativeMCSettings, SSWConfig, run_ssw
    from pamssw.standalone.recovered_direction import RecoveredDirectionSettings
    from direction_panel_lj55_helpers import CompactSurface
    out.mkdir(parents=True, exist_ok=False)
    atoms = read(INPUTS[case], index=0)
    surface = CompactSurface(calc, out / "search-ef-ledger.jsonl", cap=2000, deadline=deadline,
                             failures_dir=out / "failed-geometries")
    events, trajectory = [], out / "structures.extxyz"
    def frame(role, idx, value):
        if value is None:
            return None
        item = value.copy()
        item.info.update(role=role, index=int(idx))
        write(trajectory, item, format="extxyz", append=trajectory.exists())
        return f"structures.extxyz#{role}:{idx}"
    def progress(p):
        if p.kind == "initial":
            q = p.best
            events.append({"kind": "initial", "requests": int(p.evaluation_requests),
                "energy_eV": float(q.energy), "max_force_eV_A": float(q.max_force),
                "converged": bool(q.converged), "structure": frame("initial", -1, q.atoms),
                "structure_diagnostics": structural_diagnostics(case, q.atoms)})
            return False
        step = p.step
        row = {"kind": "outer_step", "index": int(step.index), "status": step.status,
               "accepted": bool(step.accepted), "requests": int(step.evaluation_requests),
               "cumulative_requests": int(p.evaluation_requests),
               "landing_energy_eV": None if step.landing is None else float(step.landing.energy),
               "landing_max_force_eV_A": None if step.landing is None else float(step.landing.max_force),
               "landing_structure": frame("landing", step.index, None if step.landing is None else step.landing.atoms),
               "current_structure": frame("current", step.index, p.current),
               "climb": [dict(x) for x in (step.climb or [])],
               "second_gaussian_exposures": second_gaussian_exposures({"climb": step.climb}),
               "landing_structure_diagnostics": None if step.landing is None else
                    structural_diagnostics(case, step.landing.atoms)}
        events.append(row)
        return False
    started = time.monotonic()
    result = error = None
    try:
        kwargs = {"config": SSWConfig(**protocol["ssw_config"]),
                  "rng": np.random.default_rng(seed), "progress_callback": progress,
                  "recovered_direction": RecoveredDirectionSettings(**protocol["direction"]),
                  "mc": NativeMCSettings(**protocol["native_mc"])}
        if case != "c60":
            kwargs["height_policy"] = protocol["helper"].PANEL.settings()[3]
        result = run_ssw(atoms, surface, steps=1, **kwargs)
    except Exception as exc:
        result = getattr(exc, "result", None)
        error = {"exception": repr(exc), "traceback": traceback.format_exc()}
    # A terminal cap/backend event can be returned without a progress callback.
    observed = {int(e["index"]) for e in events if e["kind"] == "outer_step"}
    if result is not None:
        cumulative = int(result.initial.evaluation_requests)
        for record in result.records:
            cumulative += int(record.evaluation_requests)
            idx = int(record.index)
            if idx in observed:
                continue
            landing = record.landing
            events.append({"kind": "outer_step", "index": idx, "status": record.status,
                "accepted": bool(record.accepted), "requests": int(record.evaluation_requests),
                "cumulative_requests": cumulative,
                "landing_energy_eV": None if landing is None else float(landing.energy),
                "landing_max_force_eV_A": None if landing is None else float(landing.max_force),
                "landing_structure": frame("terminal_landing", idx, None if landing is None else landing.atoms),
                "last_atoms_structure": frame("terminal_last_atoms", idx, record.last_atoms),
                "current_structure": frame("terminal_current", idx, result.current),
                "climb": [dict(x) for x in (record.climb or [])],
                "second_gaussian_exposures": second_gaussian_exposures({"climb": record.climb}),
                "landing_structure_diagnostics": None if landing is None else
                    structural_diagnostics(case, landing.atoms), "progress_callback_observed": False})
            observed.add(idx)
    # Validate a saved landing; result.best/current can conceal MC-rejected escape structures.
    landings = [] if result is None else [r.landing for r in result.records if r.landing is not None]
    landings = [q for q in landings if q.converged and q.max_force <= protocol["fmax"]]
    chosen = min(landings, key=lambda q: q.energy) if landings else None
    fresh = {"requests": 0, "selected_role": "lowest_energy_force_qualified_saved_landing" if chosen else None}
    if chosen is not None:
        if time.monotonic() >= deadline - 20.0:
            fresh["skipped_reason"] = "case_deadline_output_reserve"
        else:
            fresh_surface = ASESurface(calc)
            ledger_row = {"request": 1, "charged": True, "status": "pending",
                          "role": fresh["selected_role"]}
            try:
                calc.reset()
                before = surface._calculate_counter["calls"]
                energy, forces = fresh_surface.evaluate(chosen.atoms)
                fmax = float(np.linalg.norm(forces, axis=1).max())
                actual_fresh_calls = surface._calculate_counter["calls"] - before
                if actual_fresh_calls <= 0:
                    raise RuntimeError("fresh E/F did not trigger a calculator call after reset")
                fresh.update(requests=1, energy_eV=float(energy), max_force_eV_A=fmax,
                             passed_force_gate=fmax <= protocol["fmax"],
                             energy_delta_from_saved_landing_eV=float(energy - chosen.energy),
                             target_energy_delta_eV=float(energy - protocol["target_energy_eV"]),
                             target_energy_window=abs(energy - protocol["target_energy_eV"]) <= protocol["target_energy_tolerance_eV"],
                             actual_calculator_calls=actual_fresh_calls,
                             fresh_calculation_confirmed=actual_fresh_calls > 0,
                             structure_diagnostics=structural_diagnostics(case, chosen.atoms))
                ledger_row.update(status="ok", energy_eV=float(energy), fmax_eV_A=fmax,
                                  actual_calculator_calls=actual_fresh_calls)
                write(out / "fresh-validated-landing.extxyz", chosen.atoms)
            except Exception as exc:
                fresh.update(requests=1, error=repr(exc))
                ledger_row.update(status="failed", error=repr(exc))
            (out / "fresh-ef-ledger.jsonl").write_text(json.dumps(ledger_row, allow_nan=False) + "\n")
    payload = {"case": case, "method": method, "seed": seed, "status": "failed" if error else result.status,
        "error": error, "source": source, "input": str(INPUTS[case]), "input_sha256": sha256(INPUTS[case]),
        "outer_steps_requested": 1, "search_request_cap": 2000,
        "search_requests": surface.paid, "actual_calculator_calls": surface.actual_calls,
        "denied_requests": surface.denied, "fresh": fresh,
        "initial": next((e for e in events if e["kind"] == "initial"), None),
        "outer_events": [e for e in events if e["kind"] == "outer_step"],
        "result_status": None if result is None else result.status,
        "elapsed_seconds": time.monotonic() - started,
        "protocol": {k: v for k, v in protocol.items() if k not in ("helper", "plan")}}
    if error:
        (out / "failure.json").write_text(json.dumps(error, indent=2) + "\n")
    cost = {"paid_search_requests": 0, "denied_requests": 0,
            "paid_evaluation_failures": 0, "actual_calculator_calls": 0}
    ledger = out / "search-ef-ledger.jsonl"
    if ledger.is_file():
        with ledger.open() as stream:
            for line in stream:
                item = json.loads(line)
                cost["paid_search_requests"] += bool(item.get("charged"))
                cost["denied_requests"] += item.get("status") == "denied"
                cost["paid_evaluation_failures"] += bool(item.get("charged") and item.get("status") == "failed")
                cost["actual_calculator_calls"] += int(item.get("actual_calculator_calls", 0))
    payload["search_cost_recount_from_ledger"] = cost
    if result is not None and result.status == "evaluation_failed":
        if cost["denied_requests"] and not cost["paid_evaluation_failures"]:
            payload["execution_outcome"] = "bounded_search_or_wall_truncation"
        else:
            payload["execution_outcome"] = "evaluation_failure_without_cap_denial"
    elif error is not None:
        payload["execution_outcome"] = "runner_exception"
    else:
        payload["execution_outcome"] = result.status if result is not None else "no_result"
    jsonable = import_existing_lj_helpers().PANEL.load_serializer()._jsonable(payload)
    (out / "result.json").write_text(json.dumps(jsonable, indent=2, allow_nan=False) + "\n")
    bounded_truncation = (result is not None and result.status == "evaluation_failed" and
                          cost["denied_requests"] > 0 and cost["paid_evaluation_failures"] == 0)
    hard_failure = (error is not None or result is None or fresh.get("error") is not None or
                    cost["paid_evaluation_failures"] > 0 or
                    (result is not None and result.status != "completed" and not bounded_truncation))
    return not hard_failure


def execute(case, method, seeds, output, wall_seconds):
    if case not in INPUTS or method not in METHODS:
        raise ValueError("unknown case or method")
    if wall_seconds <= 0 or wall_seconds > (1680 if case == "c60" else 540):
        raise ValueError("wall_seconds exceeds the frozen search window")
    deadline = time.monotonic() + wall_seconds
    source = source_probe(method)
    import_existing_lj_helpers()  # This helper also supplies counted E/F and the ledger adapter for C60.
    protocol = case_protocol(case)
    calc = create_calculator(case, protocol)
    root = Path(output) / case / method
    root.mkdir(parents=True, exist_ok=False)
    (root / "method-provenance.json").write_text(json.dumps({"source": source,
        "input": str(INPUTS[case]), "input_sha256": sha256(INPUTS[case]),
        "protocol": {k: v for k, v in protocol.items() if k not in ("helper", "plan")}},
        indent=2, allow_nan=False) + "\n")
    for seed in seeds:
        if time.monotonic() >= deadline:
            break
        if not run_seed(case, method, seed, root / f"seed-{seed}", source, calc, protocol, deadline):
            raise RuntimeError(f"arm failed; preserving outputs and stopping this method/case: {case}/{method}/{seed}")


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--source-probe", choices=METHODS)
    mode.add_argument("--mace-probe", action="store_true")
    parser.add_argument("--case", choices=INPUTS)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--wall-seconds", type=int)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.source_probe:
        print(json.dumps(source_probe(args.source_probe), allow_nan=False))
        return
    if args.mace_probe:
        print(json.dumps(mace_probe(), allow_nan=False))
        return
    if args.preflight:
        print(json.dumps(preflight(), indent=2, allow_nan=False))
        return
    if not args.case or not args.method or args.out is None:
        parser.error("--run requires --case, --method, and --out")
    seeds = tuple(int(x) for x in args.seeds.split(",") if x)
    if not seeds or len(set(seeds)) != len(seeds) or any(s not in SEEDS for s in seeds):
        parser.error("--seeds must be a nonempty subset of the four frozen seeds")
    max_wall = 1680 if args.case == "c60" else 540
    execute(args.case, args.method, seeds, args.out, args.wall_seconds or max_wall)


if __name__ == "__main__":
    main()
