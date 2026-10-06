#!/usr/bin/env python3
"""Bounded LJ55 direction-to-target study; PES work is explicit via CLI."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
PANEL_DIR = ROOT / "research/ga_ssw/evidence/ssw-native-lj-panel-20261007"
REFERENCE_PATH = ROOT / "research/ga_ssw/evidence/cluster-paper-reproduction-20260925/references/lj55.points"
N, SEEDS = 55, (26100721, 26100722)
SIGMA, EPSILON, FMAX, TARGET_E, HIT_TOL = 2.7, 1.0, 0.05, -279.248470, 0.001
SEARCH_CAP, OUTER_CAP, WALL_CAP = 160_000, 300, 600.0
PREP_CAP, FRESH_CAP = 1000, 2


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PANEL = load_module(PANEL_DIR / "run_panel.py", "lj55_target_panel_source")
ANALYSIS = load_module(PANEL_DIR / "analyze_panel.py", "lj55_target_panel_analysis")


def dump(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def source_info():
    from research.ga_ssw.full_pair_lj import FullPairLJ
    import pamssw.standalone.paper_reference as paper
    import pamssw.standalone.recovered_rotation as rotation
    import pamssw.standalone.recovered_direction as direction
    import pamssw.standalone.native_height_policy as height
    import pamssw.standalone.native_mc as mc
    import research.ga_ssw.full_pair_lj as lj
    return {
        "checkout": str(ROOT),
        "git_head": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "head_pamssw_tree": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD:pamssw"], text=True).strip(),
        "python": sys.executable,
        "FullPairLJ": str(Path(lj.__file__).resolve()),
        "FullPairLJ_class": f"{FullPairLJ.__module__}.{FullPairLJ.__name__}",
        "run_ssw": str(Path(paper.__file__).resolve()),
        "RecoveredRotationSettings": str(Path(rotation.__file__).resolve()),
        "RecoveredDirectionSettings": str(Path(direction.__file__).resolve()),
        "ConservativeNativeHeightPolicy": str(Path(height.__file__).resolve()),
        "NativeMCSettings": str(Path(mc.__file__).resolve()),
        "panel_runner": str(Path(PANEL.__file__).resolve()),
        "panel_runner_sha256": sha256(Path(PANEL.__file__).resolve()),
        "panel_analysis": str(Path(ANALYSIS.__file__).resolve()),
        "panel_analysis_sha256": sha256(Path(ANALYSIS.__file__).resolve()),
        "cambridge_reference": str(REFERENCE_PATH),
        "cambridge_reference_sha256": sha256(REFERENCE_PATH),
    }


def geometry_gate(atoms):
    """Nonperiodic domain; native-image diagnostics never bound the isolated PES."""
    diagnostics = PANEL.geometry_gate(atoms)
    finite = bool(np.isfinite(atoms.positions).all())
    nonperiodic = not bool(atoms.pbc.any())
    return {"eligible": finite and nonperiodic,
            "reason": "ok" if finite and nonperiodic else
                      "nonfinite_coordinates" if not finite else "periodic_input",
            "scope": "nonperiodic_full_pair_no_storage_bound_v2",
            "native_image_diagnostics_only": diagnostics}


def config_record():
    config, rotation, direction, height, mc, temp = PANEL.settings()
    direction = __import__("dataclasses").replace(direction, c1_radius_policy="per_atom")
    return {"protocol_revision": "v2-nonperiodic-domain",
            "potential": {"class": "FullPairLJ", "epsilon_eV": EPSILON,
                           "sigma_A": SIGMA, "cutoff_A": None, "periodic": False},
            "ssw": asdict(config), "rotation": asdict(rotation),
            "full_direction_per_atom": asdict(direction), "height": asdict(height),
            "mc": asdict(mc), "equivalent_temperature_K": temp,
            "fmax_search_eV_A": FMAX, "target_energy_eV": TARGET_E,
            "target_energy_tolerance_eV": HIT_TOL, "target_geometry":
            {"method": "existing analyze_panel.compare_geometry", "adjacency_cutoff_sigma": 1.3,
             "proper_rotation_rms_A": ANALYSIS.GEOM_RMS_TOL,
             "mapping_cap": ANALYSIS.MAX_MAPPINGS},
            "budgets": {"search_surface_requests_including_initial_quench": SEARCH_CAP,
                        "completed_outer_steps": OUTER_CAP, "wall_seconds": WALL_CAP,
                        "prepare_true_quench_requests": PREP_CAP,
                        "fresh_requests_per_seed_preparation": 1,
                        "fresh_reference_requests_total": 1,
                        "fresh_requests_per_arm": FRESH_CAP},
            "initialization": "uniform volume ball, bulk-density radius, no rejection/retry",
            "evaluation_domain": "finite nonperiodic positions; cell/span diagnostic only"}


def target_atoms():
    from ase import Atoms
    xyz = np.loadtxt(REFERENCE_PATH, dtype=float)
    if xyz.shape != (N, 3) or not np.isfinite(xyz).all():
        raise ValueError(f"invalid Cambridge reference shape/content: {xyz.shape}")
    atoms = Atoms("Ar55", positions=SIGMA * xyz, cell=np.eye(3) * 100.0, pbc=False)
    atoms.positions += 50.0 - atoms.positions.mean(axis=0)
    return atoms


def snapshot(out: Path):
    shutil.copy2(Path(__file__).resolve(), out / "run.py")
    protocol = HERE / "protocol.md"
    if protocol.is_file():
        shutil.copy2(protocol, out / "protocol.md")


def preflight():
    from research.ga_ssw.full_pair_lj import FullPairLJ
    from pamssw.standalone import run_ssw, ASESurface
    from pamssw.standalone.paper_reference import save_ssw_checkpoint
    from ase.calculators.calculator import Calculator, all_changes
    from ase import Atoms
    cfg = config_record()
    target = target_atoms()
    generated = []
    for seed in SEEDS:
        atoms, child = PANEL.uniform_volume_cluster(N, seed, "bulk-density")
        gate = geometry_gate(atoms)
        generated.append({"seed": seed, "natoms": len(atoms), "pbc": atoms.pbc.tolist(),
            "cell_A": atoms.cell.array.tolist(), "geometry_gate": gate,
            "raw_positions_sha256": hashlib.sha256(atoms.positions.tobytes()).hexdigest(),
            "search_rng_child_state": child.generate_state(4).tolist()})
    checks = {"target_points": str(REFERENCE_PATH), "target_shape": list(target.positions.shape),
              "target_geometry_gate": geometry_gate(target),
              "target_atom_count": len(target), "run_ssw": callable(run_ssw),
              "ASESurface": callable(ASESurface), "save_ssw_checkpoint": callable(save_ssw_checkpoint),
              "FullPairLJ": issubclass(FullPairLJ, Calculator),
              "slots": [{"slot": i, "seed": SEEDS[i // 2],
                         "arm": "rotation" if i % 2 == 0 else "full_per_atom"} for i in range(4)]}
    class Dummy(Calculator):
        implemented_properties = ["energy", "forces"]
        def __init__(self):
            super().__init__()
            self.calls = 0
        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.calls += 1
            self.results = {"energy": 0.0, "forces": np.zeros((len(self.atoms), 3))}
    with tempfile.TemporaryDirectory(prefix="lj55-target-preflight-") as tmp:
        dummy = Dummy()
        compact = CompactSurface(dummy, Path(tmp) / "ledger.jsonl", cap=2)
        compact.evaluate(Atoms("Ar", positions=[[50., 50., 50.]]))
        # Repeated requests are charged, but ASE may reuse the calculator cache.
        compact.evaluate(Atoms("Ar", positions=[[50., 50., 50.]]))
        denied = False
        try:
            compact.evaluate(Atoms("Ar", positions=[[50., 50., 50.]]))
        except RuntimeError as exc:
            denied = "request_cap_denied" in str(exc)
        ledger = [json.loads(line) for line in (Path(tmp) / "ledger.jsonl").read_text().splitlines()]
        checks["compact_surface_contract"] = bool(denied and compact.requests == 2 and
            compact.denied == 1 and dummy.calls == compact.actual_calls == 1
            and ledger[0]["charged"] and ledger[0]["actual_calculator_called"]
            and ledger[1]["charged"] and not ledger[1]["actual_calculator_called"]
            and ledger[0]["status"] == "ok" and not ledger[2]["charged"]
            and ledger[2]["status"] == "denied")
    # Observed v1 failures were finite nonperiodic configurations beyond 50 A.
    # This dummy check reproduces that domain boundary without evaluating any PES.
    wide = Atoms("Ar2", positions=[[-20., 0., 0.], [120., 0., 0.]], cell=[100.]*3, pbc=False)
    with tempfile.TemporaryDirectory(prefix="lj55-domain-preflight-") as tmp:
        dummy = Dummy()
        compact = CompactSurface(dummy, Path(tmp) / "ledger.jsonl", cap=2)
        compact.evaluate(wide)
        periodic_rejected = False
        wide.pbc = True
        try:
            compact.evaluate(wide)
        except RuntimeError as exc:
            periodic_rejected = "geometry_gate_failed:periodic_input" in str(exc)
        checks["nonperiodic_domain_regression"] = bool(periodic_rejected and dummy.calls == 1)
    print(json.dumps({"status": "preflight_ok", "pes_requests": 0, "protocol": cfg,
                      "checks": checks, "generated_inputs": generated,
                      "imports": source_info()}, indent=2, allow_nan=False))
    if (not all(x["geometry_gate"]["eligible"] for x in generated) or
            not checks["target_geometry_gate"]["eligible"] or not checks["compact_surface_contract"] or not checks["nonperiodic_domain_regression"]):
        raise SystemExit("preflight geometry-domain check failed")


class CompactSurface:
    """Counted ASE surface with bounded scalar ledger; no per-request force arrays."""
    def __init__(self, calculator, ledger: Path, *, cap: int, deadline: float | None = None,
                 failures_dir: Path | None = None):
        from pamssw.standalone import ASESurface
        self.inner = ASESurface(calculator)
        self._calculate_counter = PANEL.load_serializer().instrument_calculate(calculator)
        self.ledger, self.cap, self.deadline = ledger, int(cap), deadline
        self.failures_dir = failures_dir
        self.paid = self.actual_calls = 0
        self.denied = 0

    @property
    def requests(self):
        return self.paid

    def _append(self, row):
        with self.ledger.open("a") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")

    def evaluate(self, atoms):
        if self.paid >= self.cap:
            self.denied += 1
            self._append({"request": self.paid + 1, "charged": False, "status": "denied",
                          "reason": "request_cap_denied_before_evaluation",
                          "geometry_gate": geometry_gate(atoms)})
            raise RuntimeError("request_cap_denied_before_evaluation")
        if self.deadline is not None and time.monotonic() >= self.deadline:
            self.denied += 1
            self._append({"request": self.paid + 1, "charged": False, "status": "denied",
                          "reason": "wall_cap_denied_before_evaluation",
                          "geometry_gate": geometry_gate(atoms)})
            raise RuntimeError("wall_cap_denied_before_evaluation")
        self.paid += 1
        row = {"request": self.paid, "charged": True, "status": "pending",
               "actual_calculator_called": False, "energy_eV": None, "fmax_eV_A": None,
               "geometry_gate": geometry_gate(atoms)}
        calculate_before = self._calculate_counter["calls"]
        try:
            if not row["geometry_gate"]["eligible"]:
                raise RuntimeError("geometry_gate_failed:" + row["geometry_gate"]["reason"])
            energy, forces = self.inner.evaluate(atoms)
            fmax = float(np.linalg.norm(forces, axis=1).max())
            row.update(status="ok", energy_eV=float(energy), fmax_eV_A=fmax)
            return energy, forces
        except Exception as exc:
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            if self.failures_dir is not None:
                from ase.io import write
                self.failures_dir.mkdir(exist_ok=True)
                write(self.failures_dir / f"failed-request-{self.paid:06d}.extxyz", atoms)
            raise
        finally:
            actual = self._calculate_counter["calls"] - calculate_before
            self.actual_calls += actual
            row["actual_calculator_calls"] = actual
            row["actual_calculator_called"] = bool(actual)
            self._append(row)


def common_qualification(atoms, energy, fmax, converged, target):
    energy_gate = bool(np.isfinite(energy) and energy <= TARGET_E + HIT_TOL)
    force_gate = bool(np.isfinite(fmax) and fmax <= FMAX)
    candidate_prefilter = bool(converged and energy_gate and force_gate)
    connected = PANEL.connected_components(atoms) if candidate_prefilter else None
    geom = (ANALYSIS.compare_geometry(target, atoms) if connected == [N] else
            {"classification": "inconclusive", "reason": "not_checked_before_candidate_prefilter"})
    checks = {"converged": bool(converged), "energy_eV": float(energy),
              "energy_gate": energy_gate,
              "fmax_eV_A": float(fmax), "force_gate_0p05": bool(np.isfinite(fmax) and fmax <= FMAX),
              "force_gate_0p01_informational": bool(np.isfinite(fmax) and fmax <= 0.01),
              "connected_component_sizes_1p3sigma": connected,
              "connected_gate": connected == [N] if connected is not None else False,
              "geometry_to_Cambridge": geom,
              "geometry_gate": geom.get("classification") == "same"}
    checks["qualified_candidate"] = bool(checks["converged"] and checks["energy_gate"] and
        checks["force_gate_0p05"] and checks["connected_gate"] and checks["geometry_gate"])
    return checks


def prepare(output: Path):
    from ase.io import read, write
    from research.ga_ssw.full_pair_lj import FullPairLJ
    from pamssw.standalone import run_ssw
    from pamssw.standalone.paper_reference import save_ssw_checkpoint
    output.mkdir(parents=True, exist_ok=False)
    snapshot(output)
    runner_hash = sha256(output / "run.py")
    source = source_info()
    config = config_record()
    dump(output / "provenance.json", {"mode": "prepare", "imports": source,
        "runner_sha256": runner_hash, "protocol": config, "started_unix_time": time.time()})
    target = target_atoms()
    ref_fresh = CompactSurface(FullPairLJ(epsilon=EPSILON, sigma=SIGMA),
        output / "cambridge-reference-fresh-ledger.jsonl", cap=1,
        failures_dir=output / "failures")
    ref_row = None
    try:
        ref_energy, ref_forces = ref_fresh.evaluate(target)
        ref_component = float(np.abs(ref_forces).max())
        ref_row = {"energy_eV": float(ref_energy), "target_table_eV": TARGET_E,
                   "max_force_component_eV_A": ref_component,
                   "passed": bool(abs(ref_energy - TARGET_E) < 1e-5 and
                                  ref_component < 0.04 / SIGMA)}
    except Exception as exc:
        ref_row = {"passed": False, "error": repr(exc)}
    dump(output / "cambridge-reference-qualification.json", {
        "path": str(REFERENCE_PATH), "sha256": sha256(REFERENCE_PATH),
        "requests": ref_fresh.requests, "qualification": ref_row})
    cases = []
    for seed in SEEDS:
        case = output / f"lj55-seed{seed}"
        case.mkdir()
        surface = None
        try:
            raw, child = PANEL.uniform_volume_cluster(N, seed, "bulk-density")
            write(case / "raw.extxyz", raw)
            surface = CompactSurface(FullPairLJ(epsilon=EPSILON, sigma=SIGMA),
                case / "prepare-ef-ledger.jsonl", cap=PREP_CAP, failures_dir=case / "failures")
            cfg = PANEL.settings()[0]
            result = run_ssw(raw.copy(), surface, steps=0, config=cfg,
                rng=np.random.default_rng(child), progress_callback=lambda _p: False)
            cp = result.checkpoint
            if cp is not None:
                save_ssw_checkpoint(case / "prepare-checkpoint.pkl", cp)
            initial = result.initial
            best = cp.best if cp is not None else initial
            write(case / "initial.extxyz", initial.atoms)
            write(case / "prepared-candidate.extxyz", best.atoms)
            numerical = bool(initial.converged and np.isfinite(initial.energy) and initial.max_force <= FMAX
                             and best.converged and np.isfinite(best.energy) and best.max_force <= FMAX
                             and geometry_gate(best.atoms)["eligible"]
                             and PANEL.connected_components(best.atoms) == [N])
            fresh = CompactSurface(FullPairLJ(epsilon=EPSILON, sigma=SIGMA),
                case / "prepare-fresh-ledger.jsonl", cap=1, failures_dir=case / "failures")
            fresh_row = None
            if numerical:
                energy, forces = fresh.evaluate(best.atoms)
                fmax = float(np.linalg.norm(forces, axis=1).max())
                fresh_row = {"energy_eV": float(energy), "fmax_eV_A": fmax,
                             "passed": bool(np.isfinite(energy) and fmax <= FMAX)}
            qualified = bool(numerical and fresh_row and fresh_row["passed"] and ref_row["passed"])
            if qualified:
                prepared = best.atoms.copy()
                prepared.info.update(seed=seed, preparation_qualified=True,
                    preparation_energy_eV=float(best.energy), preparation_fmax_eV_A=float(best.max_force))
                write(case / "prepared.extxyz", prepared)
            row = {"seed": seed, "n": N, "initialization": "uniform-volume-bulk-density",
                "raw_input_sha256": sha256(case / "raw.extxyz"),
                "cambridge_reference_path": str(REFERENCE_PATH),
                "cambridge_reference_sha256": sha256(REFERENCE_PATH),
                "initial_requests": int(initial.evaluation_requests), "prepare_requests": surface.requests,
                "fresh_prepared_requests": fresh.requests,
                "initial": {"energy_eV": float(initial.energy), "fmax_eV_A": float(initial.max_force),
                            "converged": bool(initial.converged)},
                "prepared_candidate": {"energy_eV": float(best.energy), "fmax_eV_A": float(best.max_force),
                                       "connected_component_sizes": PANEL.connected_components(best.atoms)},
                "fresh_prepared": fresh_row, "cambridge_reference_fresh": ref_row,
                "search_rng_child_state": child.generate_state(4).tolist(),
                "qualified": qualified, "status": "qualified" if qualified else "not_qualified"}
            dump(case / "result.json", row)
            cases.append(row)
        except Exception as exc:
            fail = {"seed": seed, "status": "failed", "error": repr(exc),
                    "traceback": traceback.format_exc(), "prepare_requests": None if surface is None else surface.requests}
            dump(case / "failure.json", fail)
            cases.append(fail)
    dump(output / "prepare-summary.json", {"cases": cases,
        "cambridge_reference_fresh": ref_row,
        "qualified_seeds": [r["seed"] for r in cases if r.get("qualified")],
        "status": "complete_with_failures_preserved"})


def execute(prepared_dir: Path, slot: int, output: Path):
    from ase.io import read, write
    from research.ga_ssw.full_pair_lj import FullPairLJ
    from pamssw.standalone import run_ssw
    from pamssw.standalone.paper_reference import save_ssw_checkpoint
    if slot not in range(4):
        raise ValueError("slot must be 0..3")
    seed = SEEDS[slot // 2]
    arm = "rotation" if slot % 2 == 0 else "full_per_atom"
    prep_case = prepared_dir / f"lj55-seed{seed}"
    prep_result = json.loads((prep_case / "result.json").read_text())
    input_path = prep_case / "prepared.extxyz"
    if not prep_result.get("qualified") or not input_path.is_file():
        raise RuntimeError(f"seed {seed} is not qualified; this slot cannot search")
    atoms = read(input_path)
    if len(atoms) != N or not geometry_gate(atoms)["eligible"] or PANEL.connected_components(atoms) != [N]:
        raise RuntimeError("prepared input fails LJ55 domain/connectivity contract")
    output.mkdir(parents=True, exist_ok=False)
    snapshot(output)
    source = source_info()
    runner_hash = sha256(output / "run.py")
    config = config_record()
    seed_child = np.random.SeedSequence(seed).spawn(2)[1]
    dump(output / "provenance.json", {"mode": "execute", "slot": slot, "seed": seed,
        "arm": arm, "prepared_dir": str(prepared_dir.resolve()),
        "prepared_file": str(input_path.resolve()), "prepared_sha256": sha256(input_path),
        "cambridge_reference_path": str(REFERENCE_PATH),
        "cambridge_reference_sha256": sha256(REFERENCE_PATH),
        "prepared_result": prep_result, "imports": source, "runner_sha256": runner_hash,
        "protocol": config, "started_unix_time": time.time()})
    reference = target_atoms()
    calc = FullPairLJ(epsilon=EPSILON, sigma=SIGMA)
    surface = CompactSurface(calc, output / "search-ef-ledger.jsonl", cap=SEARCH_CAP,
        deadline=time.monotonic() + WALL_CAP, failures_dir=output / "failed-geometries")
    cfg, rotation, direction, height, mc, _ = PANEL.settings()
    if arm == "full_per_atom":
        direction = __import__("dataclasses").replace(direction, c1_radius_policy="per_atom")
    rng = np.random.default_rng(seed_child)
    events, candidate = [], None
    seen_steps, seen_minima = set(), 0
    previous_current = atoms.copy()
    from ase.io import write as ase_write
    trajectory = output / "outer-structures.extxyz"

    def record_event(row):
        events.append(row)
        with (output / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")

    def record_frame(role, index, structure):
        if structure is None:
            return None
        frame = structure.copy()
        frame.info.update(event_role=role, event_index=int(index))
        ase_write(trajectory, frame, format="extxyz", append=trajectory.exists())
        return f"outer-structures.extxyz#{role}:{index}"

    def progress(p):
        nonlocal candidate, previous_current, seen_minima
        if p.kind == "initial":
            if p.new_minimum is not None:
                seen_minima += 1
            record_event({"kind": "initial", "evaluation_requests": int(p.evaluation_requests),
                "energy_eV": float(p.best.energy), "fmax_eV_A": float(p.best.max_force),
                "converged": bool(p.best.converged),
                "structure": record_frame("initial", -1, p.best.atoms)})
            previous_current = p.current.copy()
            return False
        step = p.step
        idx = int(step.index)
        seen_steps.add(idx)
        start_ref = record_frame("start", idx, previous_current)
        landing_atoms = None if step.landing is None else step.landing.atoms
        landing_ref = record_frame("landing", idx, landing_atoms)
        minimum_ref = None
        candidate_gate = None
        if p.new_minimum is not None:
            seen_minima += 1
            minimum_ref = record_frame("minimum", idx, p.new_minimum.atoms)
            candidate_gate = common_qualification(p.new_minimum.atoms, p.new_minimum.energy,
                p.new_minimum.max_force, p.new_minimum.converged, reference)
            if candidate_gate["qualified_candidate"]:
                candidate = {"event_index": idx, "energy_eV": float(p.new_minimum.energy),
                    "fmax_eV_A": float(p.new_minimum.max_force),
                    "converged": bool(p.new_minimum.converged), "geometry": candidate_gate,
                    "atoms": p.new_minimum.atoms.copy()}
        current_ref = record_frame("current", idx, p.current)
        row = {"kind": "outer_step", "index": idx, "status": step.status,
            "accepted": bool(step.accepted), "step_requests": int(step.evaluation_requests),
            "cumulative_requests": int(p.evaluation_requests),
            "start_structure": start_ref, "landing_structure": landing_ref,
            "minimum_structure": minimum_ref, "current_structure": current_ref,
            "landing_energy_eV": None if step.landing is None else float(step.landing.energy),
            "landing_fmax_eV_A": None if step.landing is None else float(step.landing.max_force),
            "minimum_gate": candidate_gate,
            "climb_summary": [{k: item.get(k) for k in
                ("index", "status", "requests", "rotation_force_requests", "quench_requests", "error")
                if k in item} for item in (step.climb or [])]}
        record_event(row)
        previous_current = p.current.copy()
        if candidate is not None:
            write(output / "target-candidate.extxyz", candidate["atoms"])
            return True
        return False

    search_started = time.monotonic()
    result = None
    error = None
    try:
        result = run_ssw(atoms.copy(), surface, steps=OUTER_CAP, config=cfg, rng=rng,
            progress_callback=progress, recovered_rotation=rotation if arm == "rotation" else None,
            recovered_direction=direction if arm == "full_per_atom" else None,
            height_policy=height, mc=mc)
    except Exception as exc:
        error = {"error": repr(exc), "traceback": traceback.format_exc()}
    # Terminal budget/wall/backend failures can end inside an outer attempt,
    # before progress_callback is invoked. Preserve that paid tail from result.records.
    if result is not None:
        for rec in result.records:
            if rec.index in seen_steps:
                continue
            idx = int(rec.index)
            landing_atoms = None if rec.landing is None else rec.landing.atoms
            landing_ref = record_frame("landing", idx, landing_atoms)
            minimum_ref, minimum_gate = None, None
            if seen_minima < len(result.minima):
                minimum = result.minima[seen_minima]
                seen_minima += 1
                minimum_ref = record_frame("minimum", idx, minimum.atoms)
                minimum_gate = common_qualification(minimum.atoms, minimum.energy,
                    minimum.max_force, minimum.converged, reference)
                write(output / f"terminal-minimum-{idx:04d}.extxyz", minimum.atoms)
            last_ref = record_frame("terminal_last_atoms", idx, rec.last_atoms)
            current_ref = record_frame("terminal_current", idx, result.current)
            record_event({"kind": "outer_step", "index": idx, "status": rec.status,
                "accepted": bool(rec.accepted), "step_requests": int(rec.evaluation_requests),
                "start_structure": record_frame("terminal_start", idx, previous_current),
                "landing_structure": landing_ref, "minimum_structure": minimum_ref,
                "last_atoms_structure": last_ref, "current_structure": current_ref,
                "landing_energy_eV": None if rec.landing is None else float(rec.landing.energy),
                "landing_fmax_eV_A": None if rec.landing is None else float(rec.landing.max_force),
                "minimum_gate": minimum_gate, "progress_callback_observed": False})
            seen_steps.add(idx)
            previous_current = result.current.copy()
    if result is not None and result.checkpoint is not None:
        save_ssw_checkpoint(output / "last-checkpoint.pkl", result.checkpoint)
        write(output / "best.extxyz", result.best)
    elif result is not None:
        write(output / "best.extxyz", result.best)
    search_seconds = time.monotonic() - search_started
    initial_requests = int(result.initial.evaluation_requests) if result is not None else None
    outer_requests = sum(int(e.get("step_requests", 0)) for e in events if e.get("kind") == "outer_step")
    fresh_rows = []
    fresh = CompactSurface(FullPairLJ(epsilon=EPSILON, sigma=SIGMA),
        output / "fresh-ef-ledger.jsonl", cap=FRESH_CAP, failures_dir=output / "failed-geometries")
    if candidate is not None:
        try:
            energy, forces = fresh.evaluate(candidate["atoms"])
            fmax = float(np.linalg.norm(forces, axis=1).max())
            check = common_qualification(candidate["atoms"], energy, fmax, fmax <= FMAX, reference)
            fresh_rows.append({"role": "target_candidate", "energy_eV": float(energy),
                "fmax_eV_A": fmax, "passed": bool(check["qualified_candidate"]), "qualification": check})
        except Exception as exc:
            fresh_rows.append({"role": "target_candidate", "passed": False, "error": repr(exc)})
    elif result is not None:
        for role, q in (("initial", result.initial), ("best", result.checkpoint.best if result.checkpoint else None)):
            if q is None:
                continue
            try:
                energy, forces = fresh.evaluate(q.atoms)
                fmax = float(np.linalg.norm(forces, axis=1).max())
                fresh_rows.append({"role": role, "energy_eV": float(energy), "fmax_eV_A": fmax,
                    "force_gate_0p01_informational": bool(fmax <= 0.01), "requests": fresh.requests})
            except Exception as exc:
                fresh_rows.append({"role": role, "error": repr(exc), "requests": fresh.requests})
    candidate_fresh = next((r for r in fresh_rows if r.get("role") == "target_candidate"), None)
    pause_confirmed = bool(candidate is not None and result is not None and result.status == "paused")
    first_hit = bool(pause_confirmed and candidate_fresh and candidate_fresh.get("passed"))
    status = ("first_hit" if first_hit else "fresh_failed" if pause_confirmed else
              "candidate_unconfirmed" if candidate is not None else
              result.status if result is not None else "failed")
    payload = {"status": status, "slot": slot, "seed": seed, "arm": arm,
        "error": error, "search_requests_total": surface.requests,
        "initial_quench_requests": initial_requests, "outer_requests": outer_requests,
        "request_closure": None if initial_requests is None or result is None else
            surface.requests == initial_requests + outer_requests,
        "denied_requests": surface.denied, "actual_calculator_calls": surface.actual_calls,
        "search_wall_seconds": search_seconds, "outer_callbacks": sum(e.get("kind") == "outer_step" for e in events),
        "initial": events[0] if events else None, "outer_events": [e for e in events if e.get("kind") == "outer_step"],
        "first_candidate": None if candidate is None else {k: v for k, v in candidate.items() if k != "atoms"},
        "fresh": fresh_rows, "fresh_requests": fresh.requests,
        "candidate_pause_confirmed": pause_confirmed,
        "best_energy_eV": None if result is None else float(result.checkpoint.best.energy if result.checkpoint else result.initial.energy),
        "result_status": None if result is None else result.status,
        "first_hit_requires": "candidate + fresh FullPairLJ E/F + force/connected/geometry qualification",
        "settings": config, "rng_child_state": seed_child.generate_state(4).tolist()}
    dump(output / "result.json", payload)
    if error is not None:
        dump(output / "failure.json", error)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--execute", action="store_true")
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--slot", type=int, choices=range(4))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.preflight:
        if args.prepared or args.slot is not None or args.output:
            parser.error("--preflight takes no filesystem arguments")
        preflight()
    elif args.prepare:
        if args.output is None or args.prepared or args.slot is not None:
            parser.error("--prepare requires --output only")
        prepare(args.output)
    else:
        if args.prepared is None or args.slot is None or args.output is None:
            parser.error("--execute requires --prepared, --slot and --output")
        execute(args.prepared, args.slot, args.output)


if __name__ == "__main__":
    main()
