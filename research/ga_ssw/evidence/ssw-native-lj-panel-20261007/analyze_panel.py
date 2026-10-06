#!/usr/bin/env python3
"""Offline readout for the bounded native/Python LJ panel.

No calculator is constructed unless --fresh-native is explicitly supplied.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
LEGACY_READOUT = Path("/home/gengjianrui/bin/pam-ssw-worktrees/c60-periodic-python/research/ga_ssw/evidence/c60-native-long-readout-20260920/analyze_native_long.py")
LEGACY_COST = Path("/home/gengjianrui/bin/pam-ssw-worktrees/c60-periodic-python/research/ga_ssw/evidence/c60-native-long-cost-20260920/analyze_native_long_cost.py")
SIGMA = 2.7
FMAX = 0.05
GEOM_RMS_TOL = 0.03 * SIGMA
ENERGY_PRINT_TOL = 5.1e-7  # Minimum found prints energy to 6 decimal places.
FORCE_PRINT_TOL = 5.0001e-4  # Its force field prints only 3 decimal places.
MAX_MAPPINGS = 1000
ARMS = ("rotation", "full", "native")
SIZES = (55, 38)
EXPECTED_SLOTS = ((55, "rotation"), (55, "full"), (55, "native"),
                  (38, "rotation"), (38, "full"), (38, "native"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_revision():
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def load_module(path: Path, name: str):
    if not path.is_file():
        raise FileNotFoundError(path)
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def read_json(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def geometry_gate(positions):
    x = np.asarray(positions, dtype=float)
    if x.ndim != 2 or x.shape[1:] != (3,) or not np.isfinite(x).all():
        return {"eligible": False, "reason": "nonfinite_or_malformed_positions"}
    lo, hi = x.min(axis=0), x.max(axis=0)
    span = hi - lo
    inside = bool(np.all(lo >= 0.0) and np.all(hi < 100.0))
    eligible = inside and bool(np.all(span < 50.0))
    return {"eligible": eligible,
            "reason": "ok" if eligible else ("outside_storage_cell" if not inside else "axis_span_not_below_half_cell"),
            "min_A": lo.tolist(), "max_A": hi.tolist(), "axis_span_A": span.tolist(),
            "storage_cell_A": 100.0, "max_span_A": 50.0}


def connectivity(positions, cutoff=1.3 * SIGMA):
    x = np.asarray(positions, dtype=float)
    if x.ndim != 2 or x.shape[1:] != (3,) or not np.isfinite(x).all():
        return {"status": "unavailable_nonfinite_or_malformed"}
    import networkx as nx
    d = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(x)))
    graph.add_edges_from((i, j) for i in range(len(x)) for j in range(i + 1, len(x))
                         if 0.0 < d[i, j] < cutoff)
    sizes = sorted((len(c) for c in nx.connected_components(graph)), reverse=True)
    return {"status": "computed", "cutoff_A": cutoff, "components": len(sizes),
            "component_sizes_desc": sizes, "single_cluster": len(sizes) == 1,
            "meaning": "geometric graph connectivity only"}


def graph_and_positions(atoms):
    import networkx as nx
    x = np.asarray(atoms.positions, dtype=float)
    d = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(x)))
    graph.add_edges_from((i, j) for i in range(len(x)) for j in range(i + 1, len(x))
                         if 0.0 < d[i, j] < 1.3 * SIGMA)
    return graph, x


def compare_geometry(reference, candidate):
    """Graph-isomorphism plus proper-rotation RMS; capped nonmatch is inconclusive."""
    if reference is None or candidate is None:
        return {"classification": "inconclusive", "reason": "missing_geometry"}
    if len(reference) != len(candidate):
        return {"classification": "different", "reason": "atom_count_differs"}
    try:
        import networkx as nx
        g0, x0 = graph_and_positions(reference)
        g1, x1 = graph_and_positions(candidate)
        matcher = nx.algorithms.isomorphism.GraphMatcher(g0, g1)
        iterator = matcher.isomorphisms_iter()
        best = None
        checked = 0
        exhausted = False
        for mapping in iterator:
            checked += 1
            order = [mapping[i] for i in range(len(x0))]
            a = x0 - x0.mean(axis=0)
            b = x1[order] - x1[order].mean(axis=0)
            u, _, vt = np.linalg.svd(a.T @ b)
            sign = 1.0 if np.linalg.det(u @ vt) >= 0 else -1.0
            rotation = u @ np.diag([1.0, 1.0, sign]) @ vt
            rms = float(np.sqrt(np.mean(np.sum((a @ rotation - b) ** 2, axis=1))))
            best = rms if best is None else min(best, rms)
            if rms < GEOM_RMS_TOL:
                return {"classification": "same", "best_proper_rotation_rms_A": rms,
                        "rms_tolerance_A": GEOM_RMS_TOL, "mappings_checked": checked,
                        "mapping_cap": MAX_MAPPINGS, "capped": False}
            if checked >= MAX_MAPPINGS:
                try:
                    next(iterator)
                except StopIteration:
                    exhausted = True
                break
        if best is None:
            return {"classification": "different", "reason": "graphs_not_isomorphic",
                    "mappings_checked": checked, "mapping_cap": MAX_MAPPINGS, "capped": False}
        capped = checked >= MAX_MAPPINGS and not exhausted
        return {"classification": "inconclusive" if capped else "different",
                "reason": "mapping_cap_reached" if capped else "all_graph_mappings_exceeded_rms",
                "best_proper_rotation_rms_A": best, "rms_tolerance_A": GEOM_RMS_TOL,
                "mappings_checked": checked, "mapping_cap": MAX_MAPPINGS, "capped": capped}
    except Exception as exc:
        return {"classification": "inconclusive", "reason": f"geometry_error:{type(exc).__name__}:{exc}"}


def ase_read(path):
    from ase.io import read
    return read(path, index=0)


def native_readout(run_dir: Path, provenance, prepared_atoms):
    from ase import Atoms
    text_path = run_dir / "lasp.out"
    ef_path = run_dir / "external-ef.json"
    native_result = read_json(run_dir / "native-result.json")
    ext = read_json(ef_path)
    out = {"kind": "native", "path": str(run_dir),
           "status": "missing" if not run_dir.is_dir() else "partial",
           "provenance": provenance, "input": {}, "events": [], "costs": {},
           "native_result": native_result}
    input_path = run_dir / "input.extxyz"
    run_input = ase_read(input_path) if input_path.is_file() else None
    out["input"] = {"path": str(input_path) if input_path.is_file() else None,
                    "sha256": sha256(input_path) if input_path.is_file() else None,
                    "prepared_path": None, "max_position_delta_A": None,
                    "same_symbols_as_prepared": None, "same_cell_as_prepared": None,
                    "same_pbc_as_prepared": None,
                    "connectivity": None if run_input is None else connectivity(run_input.positions)}
    if run_input is not None and prepared_atoms is not None and len(run_input) == len(prepared_atoms):
        out["input"]["max_position_delta_A"] = float(np.max(np.abs(run_input.positions - prepared_atoms.positions)))
        out["input"]["prepared_path"] = "provided --prepared common file"
        out["input"]["same_symbols_as_prepared"] = run_input.get_chemical_symbols() == prepared_atoms.get_chemical_symbols()
        out["input"]["same_cell_as_prepared"] = bool(np.array_equal(run_input.cell.array, prepared_atoms.cell.array))
        out["input"]["same_pbc_as_prepared"] = bool(np.array_equal(run_input.pbc, prepared_atoms.pbc))
    if not text_path.is_file() or ext is None:
        out["status"] = "partial" if run_dir.exists() else "missing"
        out["missing"] = [str(p) for p in (text_path, ef_path) if not p.is_file()]
        return out, run_input, []
    parser = load_module(LEGACY_READOUT, "_panel_legacy_native_readout")
    cost_parser = load_module(LEGACY_COST, "_panel_legacy_native_cost")
    text = text_path.read_text(errors="replace")
    events = parser._parse_events(text)
    caller_labels, unknown, bad_adjacency = cost_parser._caller_rows(text)
    requests = ext.get("requests") or []
    by_request = {int(row.get("request", i + 1)): row for i, row in enumerate(requests)}
    parsed = []
    best_candidates = []
    for event in events:
        request_no = int(event["cumulative_paid_ef"])
        event_role = "initialization" if int(event.get("ordinal", -1)) == 0 else "landing"
        row = by_request.get(request_no)
        item = dict(event, event_role=event_role, matched_request=False, request_number=request_no,
                    match_reason="request_offset_missing")
        if row is not None:
            response = row.get("response") or {}
            pos = np.asarray(row.get("positions", []), dtype=float)
            force = np.asarray(response.get("forces", []), dtype=float)
            energy = response.get("energy")
            if response.get("ok") is True and pos.shape == (len(run_input) if run_input is not None else -1, 3) and force.shape == pos.shape and energy is not None:
                component = float(np.max(np.abs(force)))
                item.update(request_energy_eV=float(energy), request_force_component_eV_A=component,
                            energy_match=abs(float(energy) - event["event_energy_eV"]) <= ENERGY_PRINT_TOL,
                            force_component_match=abs(component - event["event_force_component_eV_A"]) <= FORCE_PRINT_TOL)
                if item["energy_match"] and item["force_component_match"]:
                    atoms = Atoms("Ar" * len(pos), positions=pos, cell=np.eye(3) * 100.0, pbc=False)
                    fmax = float(np.linalg.norm(force, axis=1).max())
                    gate = geometry_gate(pos)
                    item.update(matched_request=True, match_reason=None, fmax_atomnorm_eV_A=fmax,
                                force_qualified=bool(np.isfinite(fmax) and fmax <= FMAX),
                                geometry_gate=gate, connectivity=connectivity(pos),
                                geometry_to_initial=compare_geometry(prepared_atoms, atoms))
                    item["force_domain_qualified"] = bool(item["force_qualified"] and gate["eligible"])
                    item["qualified_landing"] = bool(event_role == "landing" and item["force_domain_qualified"])
                    item["intact_force_qualified_landing"] = bool(
                        item["qualified_landing"] and item["connectivity"].get("single_cluster") is True)
                    item["energy_eV"] = float(energy)
                    if item["intact_force_qualified_landing"]:
                        best_candidates.append((float(energy), pos.copy(), request_no))
                else:
                    item["match_reason"] = "printed_energy_or_force_mismatch"
            else:
                item["match_reason"] = "external_request_failed_or_shape_invalid"
        parsed.append(item)
    total = len(requests)
    errors = ext.get("errors") or []
    last_event = max((int(e["cumulative_paid_ef"]) for e in events), default=0)
    labels_count = Counter(caller_labels)
    out.update(status="complete" if native_result and native_result.get("native_returncode") == 0 and native_result.get("cleanup_survivors") == [] else "partial",
        events=parsed,
        costs={"external_successful_requests": total, "external_errors": len(errors),
            "external_attempted_rows": total + len(errors),
            "native_result_successful_requests": (native_result or {}).get("successful_requests"),
            "native_result_minus_external_successful_requests": (None if not native_result or native_result.get("successful_requests") is None else int(native_result["successful_requests"]) - total),
            "error_rows": errors, "event_request_sum": sum(int(e["cost_evaluation_requests"]) for e in events),
            "last_event_cumulative_requests": last_event,
            "paid_tail_after_last_event": max(0, total - last_event),
            "event_offset_exceeds_requests": max(0, last_event - total),
            "unmatched_events": sum(not e["matched_request"] for e in parsed),
            "parsed_event_denominator": len(parsed),
            "initialization_events": sum(e.get("event_role") == "initialization" for e in parsed),
            "landing_event_denominator": sum(e.get("event_role") == "landing" for e in parsed),
            "qualified_matched_landing_events": sum(bool(e.get("qualified_landing")) for e in parsed),
            "intact_force_qualified_landings": sum(bool(e.get("intact_force_qualified_landing")) for e in parsed),
            "caller_labels": dict(labels_count), "caller_unknown_rows": unknown,
            "caller_bad_adjacency_rows": bad_adjacency, "caller_rows": len(caller_labels),
            "caller_to_external_request_delta": len(caller_labels) - total,
            "caller_count_matches_external_requests": len(caller_labels) == total,
            "ssw_all_done_marker": "SSW all done" in text,
            "geometry_domain_failures": sum(not e.get("geometry_gate", {}).get("eligible", True) for e in parsed)})
    return out, run_input, best_candidates


def python_readout(run_dir: Path, provenance, prepared_atoms):
    from ase import Atoms
    result = read_json(run_dir / "result.json")
    progress_path = run_dir / "progress.jsonl"
    out = {"kind": "python", "path": str(run_dir), "provenance": provenance,
           "status": "missing" if not run_dir.is_dir() else "partial", "input": {},
           "events": [], "costs": {}, "minima": [], "best_energy_eV": None}
    input_path = run_dir / "input.extxyz"
    run_input = ase_read(input_path) if input_path.is_file() else None
    out["input"] = {"path": str(input_path) if input_path.is_file() else None,
                    "sha256": sha256(input_path) if input_path.is_file() else None,
                    "max_position_delta_A": (None if run_input is None or prepared_atoms is None or len(run_input) != len(prepared_atoms)
                                             else float(np.max(np.abs(run_input.positions - prepared_atoms.positions)))),
                    "same_symbols_as_prepared": (None if run_input is None or prepared_atoms is None else
                        run_input.get_chemical_symbols() == prepared_atoms.get_chemical_symbols()),
                    "same_cell_as_prepared": (None if run_input is None or prepared_atoms is None else
                        bool(np.array_equal(run_input.cell.array, prepared_atoms.cell.array))),
                    "same_pbc_as_prepared": (None if run_input is None or prepared_atoms is None else
                        bool(np.array_equal(run_input.pbc, prepared_atoms.pbc))),
                    "connectivity": None if run_input is None else connectivity(run_input.positions)}
    rows, initial_cost = [], None
    stage = {"rotation_force_requests": 0, "biased_quench_requests": 0,
        "outer_step_requests": 0, "outer_unassigned_requests": 0,
        "climb_events": 0, "climbs_missing_rotation_request_count": 0,
        "climbs_missing_quench_request_count": 0}
    if progress_path.is_file():
        for line_no, line in enumerate(progress_path.read_text().splitlines(), 1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                rows.append({"line": line_no, "status": "invalid_json", "error": str(exc)})
                continue
            rows.append(event)
            if event.get("kind") == "initial":
                initial_cost = int(event.get("evaluation_requests", 0))
            if event.get("kind") != "outer_step":
                continue
            step_cost = int(event.get("step_requests", 0))
            known_rot_bias = 0
            for climb in event.get("climb") or []:
                stage["climb_events"] += 1
                rot = climb.get("rotation_force_requests", climb.get("force_requests"))
                bias = climb.get("quench_requests")
                if rot is None:
                    stage["climbs_missing_rotation_request_count"] += 1
                    rot = 0
                if bias is None:
                    stage["climbs_missing_quench_request_count"] += 1
                    bias = 0
                stage["rotation_force_requests"] += int(rot)
                stage["biased_quench_requests"] += int(bias)
                known_rot_bias += int(rot) + int(bias)
            stage["outer_step_requests"] += step_cost
            # Outer requests include height/true-energy/landing and other work;
            # do not infer a complete climb subtotal from the partial fields.
            stage["outer_unassigned_requests"] += step_cost - known_rot_bias
    minima, candidates = [], []
    for rec in (result or {}).get("minima") or []:
        idx = rec.get("index")
        geom = run_dir / f"minimum-{int(idx):04d}.extxyz" if idx is not None else None
        atoms = ase_read(geom) if geom is not None and geom.is_file() else None
        role = "initial" if idx == 0 else "landing"
        cmp = compare_geometry(prepared_atoms, atoms) if role != "initial" else {"classification": "same", "reason": "reference_input"}
        qualified = bool(rec.get("converged") and rec.get("energy_eV") is not None and
                         np.isfinite(float(rec["energy_eV"])) and rec.get("fmax_eV_A") is not None and
                         float(rec["fmax_eV_A"]) <= FMAX and rec.get("geometry_gate", {}).get("eligible") is True)
        item = {**rec, "role": role, "geometry_path": str(geom) if atoms is not None else None,
                "connectivity": connectivity(atoms.positions) if atoms is not None else {"status": "missing_geometry"},
                "geometry_to_initial": cmp,
                "force_domain_qualified": bool(qualified),
                "qualified_landing": bool(qualified and role == "landing"),
                "intact_force_qualified_landing": bool(qualified and role == "landing" and atoms is not None and
                    connectivity(atoms.positions).get("single_cluster") is True)}
        minima.append(item)
        if item["intact_force_qualified_landing"] and atoms is not None:
            candidates.append((float(rec["energy_eV"]), atoms.positions.copy(), int(idx)))
    ledger_path = run_dir / "ef-ledger.jsonl"
    ledger = {"rows": None, "charged": None, "failed": None, "denied": None,
              "actual_calculator_calls": None}
    if ledger_path.is_file():
        ledger = {"rows": 0, "charged": 0, "failed": 0, "denied": 0,
                  "actual_calculator_calls": 0}
        for line in ledger_path.read_text().splitlines():
            if not line.strip():
                continue
            x = json.loads(line); ledger["rows"] += 1
            ledger["charged"] += bool(x.get("charged"))
            ledger["failed"] += x.get("status") == "failed"
            ledger["denied"] += x.get("status") == "denied"
            ledger["actual_calculator_calls"] += bool(x.get("actual_calculator_called"))
    result_total = None if result is None else result.get("evaluation_requests")
    surface_requests = None if result is None else result.get("surface_requests")
    charged = ledger["charged"]
    tail = (None if charged is None or initial_cost is None else
            charged - initial_cost - stage["outer_step_requests"])
    raw_geometry_evidence = []
    if result is None:
        for geom in sorted(run_dir.glob("minimum-*.extxyz")):
            try:
                atoms = ase_read(geom)
                raw_geometry_evidence.append({"path": str(geom), "sha256": sha256(geom),
                    "atom_count": len(atoms), "geometry_gate": geometry_gate(atoms.positions),
                    "connectivity_1p3sigma": connectivity(atoms.positions),
                    "classification": "unlinked_raw_geometry_not_a_qualified_minimum"})
            except Exception as exc:
                raw_geometry_evidence.append({"path": str(geom), "sha256": sha256(geom),
                    "classification": "unreadable_raw_geometry", "error": f"{type(exc).__name__}: {exc}"})
    failure = read_json(run_dir / "failure.json")
    status = ("failed_result_missing" if failure is not None else "partial_missing_result") if result is None else (
        "parsed" if progress_path.is_file() else "partial_missing_progress")
    missing = []
    if result is None:
        missing.append(str(run_dir / "result.json"))
    if not progress_path.is_file():
        missing.append(str(progress_path))
    out.update(status=status,
        missing=missing, failure=failure, execution_status=None if result is None else result.get("status"),
        minima=minima, raw_geometry_evidence=raw_geometry_evidence,
        ledger=ledger, progress_events=rows,
        best_energy_eV=None if result is None else result.get("best_energy_eV"),
        result_status=None if result is None else result.get("status"),
        costs={**stage, "initial_quench_requests": initial_cost,
            "result_total_requests": result_total,
            "surface_requests": surface_requests,
            "charged_ledger_rows": charged, "denied_ledger_rows": ledger["denied"],
            "ledger_rows_minus_surface_requests": (None if ledger["rows"] is None or surface_requests is None else ledger["rows"] - int(surface_requests)),
            "charged_ledger_rows_minus_surface_requests": (None if charged is None or surface_requests is None else charged - int(surface_requests)),
            "charged_ledger_minus_initial_outer": tail,
            "unassigned_tail_from_charged_ledger": tail,
            "result_total_minus_initial_outer": (None if result_total is None or initial_cost is None else
                int(result_total) - initial_cost - stage["outer_step_requests"]),
            "progress_step_count": sum(x.get("kind") == "outer_step" for x in rows),
            "minimum_denominator": len(minima),
            "initial_minimum_count": sum(x["role"] == "initial" for x in minima),
            "landing_minimum_count": sum(x["role"] == "landing" for x in minima),
            "force_domain_qualified_landing_count": sum(x["qualified_landing"] for x in minima),
            "intact_force_qualified_landing_count": sum(x["intact_force_qualified_landing"] for x in minima),
            "stage_closure_delta": (None if result_total is None or initial_cost is None else
                int(result_total) - initial_cost - stage["outer_step_requests"])})
    return out, run_input, candidates


def fresh_native(run_rows, output: Path):
    from ase import Atoms
    candidates_done = 0
    for row in run_rows:
        if row.get("kind") != "native" or row.get("status") == "missing":
            continue
        run_dir = Path(row["path"])
        input_path = run_dir / "input.extxyz"
        selected = row.get("_fresh_candidates") or []
        items = []
        if input_path.is_file():
            items.append(("native_initial", ase_read(input_path).positions.copy()))
        if selected:
            e, positions, event_index = min(selected, key=lambda x: x[0])
            items.append((f"lowest_qualified_matched_event_{event_index}_E{e:.8g}", positions))
        items = items[:2]
        ledger = output / f"fresh-native-{run_dir.name}-ef.jsonl"
        entry = {"status": "not_run_no_candidates", "calls": []}
        if items:
            imports = row.get("provenance", {}).get("imports", {})
            source = Path(imports.get("FullPairLJ", ""))
            try:
                module = load_module(source, f"_panel_full_pair_lj_{run_dir.name}")
                from ase import Atoms
                calc = module.FullPairLJ(epsilon=1.0, sigma=SIGMA)
                with ledger.open("x") as stream:
                    for label, positions in items:
                        call = {"label": label, "charged": True, "positions_A": positions.tolist()}
                        try:
                            atoms = Atoms("Ar" * len(positions), positions=positions,
                                          cell=np.eye(3) * 100.0, pbc=False)
                            atoms.calc = calc
                            energy = float(atoms.get_potential_energy())
                            forces = np.asarray(atoms.get_forces(), dtype=float)
                            call.update(status="ok", energy_eV=energy,
                                        max_force_component_eV_A=float(np.max(np.abs(forces))),
                                        fmax_atomnorm_eV_A=float(np.linalg.norm(forces, axis=1).max()))
                        except Exception as exc:
                            call.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                        stream.write(json.dumps(call, allow_nan=False) + "\n")
                        entry["calls"].append(call)
                        candidates_done += 1
                entry["status"] = "complete"
            except Exception as exc:
                entry["status"] = "failed"
                entry["error"] = f"{type(exc).__name__}: {exc}"
        row["fresh_native"] = entry
    return candidates_done


def md_report(payload):
    lines = ["# Native/Python LJ panel readout", "",
        "Offline accounting and geometry classification; no Hessian or reaction-network validation.",
        "Native is a bounded periodic-representation reference; it is not a causal controller comparison with the nonperiodic Python walkers.", "",
        "| Prepared N | Exists and qualified | Input components at 1.3σ |",
        "|---:|---|---:|"]
    for n, prep in payload["prepared"].items():
        conn = prep.get("connectivity_1p3sigma") or {}
        lines.append(f"| {n} | {prep.get('exists')} / {prep.get('preparation_qualified')} | {conn.get('components','NA')} ({conn.get('component_sizes_desc','NA')}) |")
    lines += ["", f"Run denominator: {payload['denominators']['supplied_slots']} supplied slots; statuses {payload['denominators']['status_counts']}.", "",
        "| N | Arm | Status | Total requests | Force/domain (intact) landings | Noninitial geometry evidence | Stage costs (E/F requests) | Best intact-qualified E (eV) | Connectivity |",
        "|---:|---|---|---:|---:|---|---|---:|---|"]
    for r in payload["runs"]:
        costs = r.get("costs", {})
        if r.get("kind") == "native":
            total = costs.get("external_successful_requests")
            qual = f"{costs.get('qualified_matched_landing_events', 0)} ({costs.get('intact_force_qualified_landings', 0)})"
            evs = r.get("events", [])
            evidence = Counter(e.get("geometry_to_initial", {}).get("classification", "unavailable")
                               for e in evs if e.get("matched_request"))
            if evs:
                evidence["parsed_unmatched_event_records"] = costs.get("unmatched_events", 0)
            best = min((e["energy_eV"] for e in evs if e.get("intact_force_qualified_landing")), default=None)
            stages = ", ".join(f"{k}:{v}" for k, v in costs.get("caller_labels", {}).items())
            stages += (f"; native_rc:{(r.get('native_result') or {}).get('native_returncode')}; "
                       f"event_match:{len(evs) - costs.get('unmatched_events', 0)}/{len(evs)}")
            con = Counter(str(e.get("connectivity", {}).get("single_cluster", "NA"))
                          for e in evs if e.get("matched_request"))
            input_con = r.get("input", {}).get("connectivity", {}).get("component_sizes_desc", "NA")
        elif r.get("kind") == "python":
            result_total = costs.get("result_total_requests")
            total = (result_total if result_total is not None else
                     f"ledger:{costs.get('charged_ledger_rows')} / result:NA")
            qual = f"{costs.get('force_domain_qualified_landing_count', 0)} ({costs.get('intact_force_qualified_landing_count', 0)})"
            evidence = Counter(m.get("geometry_to_initial", {}).get("classification", "unavailable")
                               for m in r.get("minima", []) if m.get("role") == "landing")
            if r.get("raw_geometry_evidence"):
                evidence["unlinked_raw_geometry_files"] = len(r["raw_geometry_evidence"])
            best = min((m["energy_eV"] for m in r.get("minima", []) if m.get("intact_force_qualified_landing")), default=None)
            stages = f"rot:{costs.get('rotation_force_requests')}, bias:{costs.get('biased_quench_requests')}, outer_unassigned:{costs.get('outer_unassigned_requests')}, charged_tail:{costs.get('unassigned_tail_from_charged_ledger')}"
            con = Counter(str(m.get("connectivity", {}).get("single_cluster", "NA"))
                          for m in r.get("minima", []) if m.get("role") == "landing")
            input_con = r.get("input", {}).get("connectivity", {}).get("component_sizes_desc", "NA")
        else:
            total, qual, best, stages, evidence, con, input_con = None, 0, None, "unavailable", Counter(), Counter(), "NA"
        lines.append(f"| {r.get('n','?')} | {r.get('arm','?')} | {r.get('status')} | {total if total is not None else 'NA'} | {qual} | {dict(evidence)} | {stages} | {best if best is not None else 'NA'} | input {input_con}; events {dict(con)} |")
    missing_result_failures = [r for r in payload["runs"] if r.get("kind") == "python" and r.get("result_status") is None and r.get("failure")]
    if missing_result_failures:
        lines += ["", f"Python result missing in {len(missing_result_failures)} runs; progress and charged-ledger costs are retained, while geometry files remain unlinked raw evidence. Result totals/surface-request fields are null, not inferred from those files."]
    native_unmatched = sum((r.get("costs") or {}).get("unmatched_events", 0) for r in payload["runs"] if r.get("kind") == "native")
    if native_unmatched:
        lines += [f"Native event-to-request matches left {native_unmatched} parsed events unmatched under E/F print tolerances; these do not qualify as landings."]
    lines += ["", "`same/different/inconclusive` uses the 1.3σ graph and proper-rotation RMS <0.03σ heuristic (at most 1000 mappings); this is not basin identity. Connectivity is the 1.3σ geometric graph only. A low force and an energy match do not verify Hessian index.",
              "Missing and partial run inputs remain in the JSON denominator. Censored tails, event offsets, caller mismatches, errors and unassigned costs are retained there.", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepared", type=Path, required=True)
    ap.add_argument("--runs", type=Path, nargs=6, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--fresh-native", action="store_true")
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    from ase.io import read
    prepared_atoms = {}
    for n in SIZES:
        path = args.prepared / f"lj{n}.extxyz"
        try:
            prepared_atoms[n] = read(path, index=0) if path.is_file() else None
        except Exception:
            prepared_atoms[n] = None
    prep_prov = args.prepared / "provenance.json"
    payload = {"scope": "bounded LJ development panel; same-coordinate E/F domain is checked, native periodic representation and Python nonperiodic proposals remain distinct",
        "analysis_source": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve()),
            "checkout": str(ROOT), "git_head": git_revision()},
        "prepared_source": {"directory": str(args.prepared), "provenance_path": str(prep_prov),
            "provenance_sha256": sha256(prep_prov) if prep_prov.is_file() else None},
        "parsers": {"event_parser": str(LEGACY_READOUT), "event_parser_sha256": sha256(LEGACY_READOUT) if LEGACY_READOUT.is_file() else None,
                    "caller_parser": str(LEGACY_COST), "caller_parser_sha256": sha256(LEGACY_COST) if LEGACY_COST.is_file() else None},
        "prepared": {str(n): {"path": str(args.prepared / f"lj{n}.extxyz"),
            "sha256": sha256(args.prepared / f"lj{n}.extxyz") if (args.prepared / f"lj{n}.extxyz").is_file() else None,
            "exists": (args.prepared / f"lj{n}.extxyz").is_file(),
            "seed": None if prepared_atoms[n] is None else prepared_atoms[n].info.get("seed"),
            "preparation_qualified": None if prepared_atoms[n] is None else prepared_atoms[n].info.get("preparation_qualified"),
            "connectivity_1p3sigma": None if prepared_atoms[n] is None else connectivity(prepared_atoms[n].positions),
            "prepared_search_eligible": bool(prepared_atoms[n] is not None and
                prepared_atoms[n].info.get("preparation_qualified") is True and
                connectivity(prepared_atoms[n].positions).get("single_cluster") is True)}
            for n in SIZES}, "fresh_native_enabled": args.fresh_native, "runs": []}
    for slot, run_dir in enumerate(args.runs):
        provenance = read_json(run_dir / "provenance.json")
        arm = (provenance or {}).get("arm")
        n = (provenance or {}).get("n")
        if arm not in ARMS or n not in SIZES:
            expected_n, expected_arm = EXPECTED_SLOTS[slot]
            row = {"path": str(run_dir), "status": "missing" if not run_dir.exists() else "incomplete_missing_provenance",
                   "kind": "native" if expected_arm == "native" else "python",
                   "arm": expected_arm if arm is None else arm, "n": expected_n if n is None else n,
                   "expected_slot": {"n": expected_n, "arm": expected_arm}, "provenance": provenance,
                   "missing": [str(run_dir / "provenance.json")] if provenance is None else []}
            payload["runs"].append(row)
            continue
        if arm == "native":
            row, _run_input, fresh_candidates = native_readout(run_dir, provenance, prepared_atoms.get(n))
            row["_fresh_candidates"] = fresh_candidates
        else:
            row, _run_input, _fresh_candidates = python_readout(run_dir, provenance, prepared_atoms.get(n))
        row.update(arm=arm, n=n)
        payload["runs"].append(row)
    payload["denominators"] = {"supplied_slots": len(args.runs),
        "status_counts": dict(Counter(r.get("status", "unknown") for r in payload["runs"])),
        "known_arm_size_counts": {f"{arm}/N{n}": sum(r.get("arm") == arm and r.get("n") == n for r in payload["runs"])
                                  for arm in ARMS for n in SIZES}}
    args.output.mkdir(parents=True, exist_ok=False)
    if args.fresh_native:
        payload["fresh_native_calls"] = fresh_native(payload["runs"], args.output)
    for row in payload["runs"]:
        row.pop("_fresh_candidates", None)
    with (args.output / "analysis.json").open("x") as f:
        json.dump(payload, f, indent=2, allow_nan=False, default=json_default)
        f.write("\n")
    (args.output / "README.md").write_text(md_report(payload))


if __name__ == "__main__":
    main()
