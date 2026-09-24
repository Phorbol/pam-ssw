#!/usr/bin/env python3
"""Read-only analysis for the saved-path depth ablation outputs."""
from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path
from functools import lru_cache

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "ga_ssw"))


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else None


def finite_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def stream_ledger(path):
    """Count actual charged calls from JSONL, retaining missing/malformed state."""
    if not path.is_file():
        return {"status": "missing", "path": str(path), "charged_calls": None,
                "search": None, "search_failure": None, "search_denial": None,
                "line_count": None, "errors": ["ledger file missing"]}
    counts = {"search": 0, "search_failure": 0, "search_denial": 0}
    errors, line_count, last_request = [], 0, 0
    with path.open() as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            line_count += 1
            try:
                row = json.loads(line)
            except Exception as exc:
                errors.append(f"line {line_no}: invalid JSON ({exc!r})")
                continue
            kind = row.get("kind")
            if kind not in counts:
                errors.append(f"line {line_no}: unexpected kind {kind!r}")
                continue
            counts[kind] += 1
            request = row.get("request")
            if kind in ("search", "search_failure"):
                if not isinstance(request, int) or request != last_request + 1:
                    errors.append(f"line {line_no}: nonsequential charged request {request!r}")
                if isinstance(request, int):
                    last_request = request
            elif request != last_request:
                errors.append(f"line {line_no}: denial at {request!r}, charged total {last_request}")
    charged = counts["search"] + counts["search_failure"]
    return {"status": "valid" if not errors else "invalid", "path": str(path),
            "charged_calls": charged, "search": counts["search"],
            "search_failure": counts["search_failure"],
            "search_denial": counts["search_denial"], "line_count": line_count,
            "last_charged_request": last_request, "errors": errors}


def geometry(atoms):
    if not isinstance(atoms, dict) or not isinstance(atoms.get("numbers"), list) \
            or not isinstance(atoms.get("positions"), list):
        return None
    if len(atoms["numbers"]) != len(atoms["positions"]):
        return None
    return atoms


def build_graph(atoms, system, cutoff=None):
    import networkx as nx
    import numpy as np

    numbers = [int(z) for z in atoms["numbers"]]
    positions = np.asarray(atoms["positions"], dtype=float)
    graph = nx.Graph()
    graph.add_nodes_from((i, {"number": z}) for i, z in enumerate(numbers))
    if system == "C4H6":
        from pamssw.standalone.native_ls import HC_BOND_LENGTHS
        for i, j in itertools.combinations(range(len(numbers)), 2):
            key = tuple(sorted((numbers[i], numbers[j])))
            threshold = HC_BOND_LENGTHS.get(key)
            if threshold is not None and float(np.linalg.norm(positions[i] - positions[j])) <= threshold + 0.1:
                graph.add_edge(i, j)
    else:
        for i, j in itertools.combinations(range(len(numbers)), 2):
            if float(np.linalg.norm(positions[i] - positions[j])) <= cutoff:
                graph.add_edge(i, j)
    return graph


def graph_summary(atoms, system):
    import networkx as nx
    if atoms is None:
        return None, {"status": "missing_geometry"}
    cutoffs = (1.8, 1.64) if system == "C60" else (None,)
    summaries, graphs = {}, {}
    for cutoff in cutoffs:
        g = build_graph(atoms, system, cutoff)
        label = f"{cutoff:.2f}A" if cutoff is not None else "covalent_plus_0.1A"
        graphs[label] = g
        summaries[label] = {
            "connected": nx.is_connected(g) if len(g) else False,
            "component_count": nx.number_connected_components(g),
            "degree_sequence": sorted(int(d) for _, d in g.degree()),
            "edge_count": g.number_of_edges(),
        }
    return graphs, {"status": "computed", "graphs": summaries}


@lru_cache(maxsize=1)
def torsion_helper():
    import importlib.util
    from analyze_c4h6_ls_reaction_coverage import atoms_from_dict
    path = HERE.parent / "c4h6-torsion-audit-20260924" / "analyze.py"
    spec = importlib.util.spec_from_file_location("saved_path_torsion_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return atoms_from_dict, module.torsion


def torsion_summary(atoms):
    if atoms is None:
        return {"status": "missing_geometry"}
    try:
        atoms_from_dict, torsion = torsion_helper()
        obj = atoms_from_dict(atoms)
        indices, angle, cosine = torsion(obj)
        return {"status": "computed", "carbon_path": list(indices),
                "raw_degrees": angle, "cosine": cosine,
                "cosine_sign_region": "positive" if cosine > 0 else "negative" if cosine < 0 else "zero"}
    except Exception as exc:
        return {"status": "unavailable", "error": repr(exc)}


def proper_kabsch_rms(left, right, left_graph, right_graph, require_connected=True):
    """Minimum RMS over graph-compatible mappings and proper rotations only."""
    import networkx as nx
    import numpy as np

    if left is None or right is None:
        return {"status": "missing_geometry"}
    if require_connected and (not nx.is_connected(left_graph) or not nx.is_connected(right_graph)):
        return {"status": "not_attempted_disconnected"}
    node_match = nx.algorithms.isomorphism.categorical_node_match("number", None)
    matcher = nx.algorithms.isomorphism.GraphMatcher(left_graph, right_graph, node_match=node_match)
    p = np.asarray(left["positions"], dtype=float)
    q = np.asarray(right["positions"], dtype=float)
    best = None
    mappings = 0
    for mapping in matcher.isomorphisms_iter():
        order = [mapping[i] for i in range(len(p))]
        q_ordered = q[order]
        pc, qc = p.mean(axis=0), q_ordered.mean(axis=0)
        x, y = p - pc, q_ordered - qc
        u, _, vt = np.linalg.svd(x.T @ y)
        correction = np.eye(3)
        correction[-1, -1] = 1.0 if np.linalg.det(u @ vt) >= 0 else -1.0
        rotation = u @ correction @ vt
        delta = x @ rotation - y
        rms = float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))
        mappings += 1
        if best is None or rms < best:
            best = rms
    if best is None:
        return {"status": "graphs_not_isomorphic", "isomorphic_mappings": 0}
    return {"status": "computed", "rms_A": best, "isomorphic_mappings": mappings,
            "rotation_constraint": "proper_only_det_plus_1", "similarity_threshold": None}


def independent_fresh(result, label, fmax_limit):
    if not isinstance(result, dict):
        return {"status": "missing_result", "energy_eV": None, "force_qualified": None}
    for item in result.get("fresh", []):
        if item.get("label") == label:
            status = item.get("status", "missing")
            fmax = finite_float(item.get("fmax"))
            reported = item.get("force_qualified")
            qualified = (fmax is not None and fmax_limit is not None and fmax <= fmax_limit) \
                if status == "completed" else None
            return {"status": status, "energy_eV": finite_float(item.get("energy")),
                    "fmax_eV_per_A": fmax, "force_qualified": qualified,
                    "protocol_fmax_eV_per_A": fmax_limit,
                    "reported_force_qualified": reported,
                    "reported_qualification_matches_fmax": reported == qualified if status == "completed" else None,
                    "criterion": "independent fresh fmax <= protocol fmax"}
    return {"status": "missing", "energy_eV": None, "force_qualified": None}


def continuous_summary(values):
    values = sorted(float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(v))
    if not values:
        return {"n": 0, "median": None, "min": None, "max": None}
    middle = len(values) // 2
    median = values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2
    return {"n": len(values), "median": median, "min": values[0], "max": values[-1]}


def state(atoms, energy=None, source=None, system=None):
    graphs, summary = graph_summary(atoms, system) if system else (None, None)
    return {"source": source, "energy_eV": finite_float(energy),
            "atom_count": len(atoms["numbers"]) if atoms else None,
            "graphs": summary, "_graphs": graphs}


def structure_comparison(a_atoms, b_atoms, system):
    import networkx as nx
    ga, sa = graph_summary(a_atoms, system)
    gb, sb = graph_summary(b_atoms, system)
    if a_atoms is None or b_atoms is None:
        return {"status": "missing_geometry", "left_graphs": sa, "right_graphs": sb}
    if system == "C60":
        result = {"cutoff_graphs": {}, "geometry": {}}
        for key in ("1.80A", "1.64A"):
            left_g, right_g = ga[key], gb[key]
            iso = nx.is_isomorphic(left_g, right_g,
                node_match=nx.algorithms.isomorphism.categorical_node_match("number", None))
            result["cutoff_graphs"][key] = {"isomorphic": bool(iso),
                                              "left_connected": nx.is_connected(left_g),
                                              "right_connected": nx.is_connected(right_g)}
            if key == "1.80A":
                result["geometry"][key] = proper_kabsch_rms(a_atoms, b_atoms, left_g, right_g)
        return result
    left_g, right_g = ga["covalent_plus_0.1A"], gb["covalent_plus_0.1A"]
    iso = nx.is_isomorphic(left_g, right_g,
        node_match=nx.algorithms.isomorphism.categorical_node_match("number", None))
    return {"graph": {"isomorphic": bool(iso),
                       "left_component_count": nx.number_connected_components(left_g),
                       "right_component_count": nx.number_connected_components(right_g)},
            "geometry": proper_kabsch_rms(a_atoms, b_atoms, left_g, right_g, require_connected=False),
            "torsion_left": torsion_summary(a_atoms), "torsion_right": torsion_summary(b_atoms),
            "interpretation": "same connectivity graph does not imply same conformational basin"}


def self_check_alignment(rows):
    """Verify proper, permutation-aware alignment on one frozen real input."""
    import numpy as np
    row = next((r for r in rows if r.get("start", {}).get("numbers")), None)
    if row is None:
        return {"status": "unavailable_no_real_input"}
    original = row["start"]
    xyz = np.asarray(original["positions"], dtype=float)
    # Fixed nontrivial proper rotation and translation; reorder atom rows legally.
    angle = 0.713
    rotation = np.array([[math.cos(angle), -math.sin(angle), 0.0],
                         [math.sin(angle), math.cos(angle), 0.0],
                         [0.0, 0.0, 1.0]])
    order = np.arange(len(xyz))[::-1]
    transformed = dict(original)
    transformed["numbers"] = np.asarray(original["numbers"])[order].tolist()
    transformed["positions"] = (xyz[order] @ rotation + np.array([3.1, -2.7, 1.4])).tolist()
    system = row["system"]
    ga, _ = graph_summary(original, system)
    gb, _ = graph_summary(transformed, system)
    key = "1.80A" if system == "C60" else "covalent_plus_0.1A"
    aligned = proper_kabsch_rms(original, transformed, ga[key], gb[key])
    rms = aligned.get("rms_A")
    passed = aligned.get("status") == "computed" and rms is not None and rms <= 1e-8
    return {"status": "passed" if passed else "failed", "case_id": row.get("case_id"),
            "operation": "proper rigid rotation + translation + reversed atom order",
            "alignment": aligned, "numerical_self_check_tolerance_A": 1e-8}


def analyze_row(item, folder):
    result_path = folder / "result.json"
    result = read_json(result_path)
    row = {"case_id": item["case_id"], "system": item["system"], "arm": item["arm"],
           "seed": item["seed"], "record_index": item["record_index"], "depth": item["depth"],
           "full_depth": item["full_depth"], "status": "missing_run_result" if result is None else result.get("status", "unknown"),
           "result_path": str(result_path), "force_qualification_rule": "fresh only; missing/failed stays missing/failed"}
    if result is None:
        search_ledger = stream_ledger(folder / "requests.jsonl")
        fresh_ledger = stream_ledger(folder / "fresh.jsonl")
        search_calls = search_ledger.get("charged_calls")
        fresh_calls = fresh_ledger.get("charged_calls")
        row.update(search_ledger=search_ledger, fresh_ledger=fresh_ledger,
                   actual_new_quench_calls=search_calls, actual_new_fresh_calls=fresh_calls,
                   cost={"saved_prefix_requests": item.get("prefix_requests"),
                         "new_quench_calls": search_calls,
                         "truncated_prefix_plus_quench": (item.get("prefix_requests", 0) + search_calls)
                            if isinstance(search_calls, int) else None,
                         "original_full_cost": item.get("original_full_requests"),
                         "difference_vs_full": (item.get("prefix_requests", 0) + search_calls - item.get("original_full_requests"))
                            if isinstance(search_calls, int) and isinstance(item.get("original_full_requests"), int) else None,
                         "new_fresh_calls_separate": fresh_calls})
        row["fresh"] = {"truncated": {"status": "missing", "force_qualified": None},
                        "full": {"status": "missing", "force_qualified": None},
                        "start": {"status": "not_independently_checked", "force_qualified": None}}
        row["quench_convergence"] = {"status": "missing_result", "converged": None}
        return row

    search_ledger = stream_ledger(folder / "requests.jsonl")
    fresh_ledger = stream_ledger(folder / "fresh.jsonl")
    search_reported = result.get("search_requests")
    fresh_reported = result.get("fresh_requests")
    search_calls = search_ledger.get("charged_calls")
    fresh_calls = fresh_ledger.get("charged_calls")
    closure = {
        "search_ledger_matches_result": search_calls == search_reported if search_calls is not None else None,
        "fresh_ledger_matches_result": fresh_calls == fresh_reported if fresh_calls is not None else None,
        "search_ledger_rows_close": search_ledger.get("line_count") == search_calls + search_ledger.get("search_denial", 0) if search_calls is not None else None,
        "fresh_ledger_rows_close": fresh_ledger.get("line_count") == fresh_calls + fresh_ledger.get("search_denial", 0) if fresh_calls is not None else None,
    }
    row.update(search_ledger=search_ledger, fresh_ledger=fresh_ledger,
               ledger_closure=closure, search_requests_reported=search_reported,
               fresh_requests_reported=fresh_reported, actual_new_quench_calls=search_calls,
               actual_new_fresh_calls=fresh_calls)
    fmax_limit = finite_float(item.get("fmax"))
    row["fresh"] = {"truncated": independent_fresh(result, "truncated", fmax_limit),
                    "full": independent_fresh(result, "full", fmax_limit),
                    "start": {"status": "not_independently_checked", "force_qualified": None}}

    quench = result.get("quench") if isinstance(result.get("quench"), dict) else None
    q_converged = quench.get("converged") if quench is not None else None
    row["quench_convergence"] = {
        "status": "converged" if q_converged is True else "not_converged" if q_converged is False
            else "missing_flag" if quench is not None else "missing_quench",
        "converged": q_converged if isinstance(q_converged, bool) else None,
        "result_status": result.get("status"),
    }
    trunc_atoms = geometry(quench.get("atoms")) if quench else None
    start_atoms = geometry(item.get("start"))
    full_atoms = geometry(item.get("full", {}).get("atoms"))
    states = {
        "start": {"geometry": start_atoms, "energy_eV": None,
                  "force_qualification": row["fresh"]["start"]},
        "truncated": {"geometry": trunc_atoms,
                       "energy_eV": row["fresh"]["truncated"]["energy_eV"],
                       "archived_energy_eV": finite_float(quench.get("energy")) if quench else None,
                       "force_qualification": row["fresh"]["truncated"]},
        "original_full": {"geometry": full_atoms,
                           "energy_eV": row["fresh"]["full"]["energy_eV"],
                           "archived_energy_eV": finite_float(item.get("full", {}).get("energy")),
                           "force_qualification": row["fresh"]["full"]},
    }
    row["states"] = {}
    for name, data in states.items():
        atoms = data["geometry"]
        g, graph_info = graph_summary(atoms, item["system"])
        row["states"][name] = {"energy_eV": data["energy_eV"],
                                "archived_energy_eV": data.get("archived_energy_eV"),
                                "atom_count": len(atoms["numbers"]) if atoms else None,
                                "fresh_force_qualification": data["force_qualification"],
                                "graph": graph_info,
                                "torsion": torsion_summary(atoms) if item["system"] == "C4H6" else None}
    delta_e = None
    if (row["fresh"]["truncated"].get("force_qualified") is True
            and row["fresh"]["full"].get("force_qualified") is True):
        e_truncated = row["fresh"]["truncated"].get("energy_eV")
        e_full = row["fresh"]["full"].get("energy_eV")
        if e_truncated is not None and e_full is not None:
            delta_e = e_truncated - e_full
    row["fresh_energy_difference_truncated_minus_full_eV"] = delta_e
    row["comparisons"] = {
        "truncated_vs_start": structure_comparison(start_atoms, trunc_atoms, item["system"]),
        "truncated_vs_original_full": structure_comparison(trunc_atoms, full_atoms, item["system"]),
        "start_vs_original_full": structure_comparison(start_atoms, full_atoms, item["system"]),
    }
    row["comparison_scope"] = {
        "truncated_vs_start": {"fresh_qualified_pair": False,
            "scope": "archived geometry diagnostic; start has no independent fresh check"},
        "start_vs_original_full": {"fresh_qualified_pair": False,
            "scope": "archived geometry diagnostic; start has no independent fresh check"},
        "truncated_vs_original_full": {"fresh_qualified_pair":
            row["fresh"]["truncated"]["force_qualified"] is True
            and row["fresh"]["full"]["force_qualified"] is True,
            "scope": "fresh-qualified scientific endpoint comparison" if
            row["fresh"]["truncated"]["force_qualified"] is True
            and row["fresh"]["full"]["force_qualified"] is True else "geometry retained; scientific comparison unassessed"},
    }

    if search_calls is None:
        row["cost"] = {"saved_prefix_requests": item.get("prefix_requests"),
                        "new_quench_calls": None, "truncated_prefix_plus_quench": None,
                        "original_full_cost": item.get("original_full_requests"),
                        "new_fresh_calls_separate": fresh_calls}
    else:
        combined = int(item.get("prefix_requests", 0)) + int(search_calls)
        full_cost = item.get("original_full_requests")
        row["cost"] = {"saved_prefix_requests": item.get("prefix_requests"),
                        "new_quench_calls": search_calls,
                        "truncated_prefix_plus_quench": combined,
                        "original_full_cost": full_cost,
                        "difference_vs_full": combined - full_cost if isinstance(full_cost, int) else None,
                        "new_fresh_calls_separate": fresh_calls}
    return row


def render_report(analysis):
    def fmt_range(summary, digits=3):
        if not summary or summary.get("n", 0) == 0:
            return "not assessed"
        return f"{summary['median']:.{digits}f} [{summary['min']:.{digits}f}, {summary['max']:.{digits}f}] (n={summary['n']})"

    lines = [
        "# Saved-path depth ablation readout", "",
        "This readout asks whether a truncated true quench returns near the saved outer-step start, and how both compare with the original full landing. Start-related geometry comparisons are diagnostics because starts have no independent fresh force check.", "",
        f"Rows: {analysis['summary']['rows_expected']} expected; {analysis['summary']['rows_with_result']} have result files; {analysis['summary']['rows_missing']} are missing. Observed ledger calls: {analysis['summary']['observed_charged_requests']}; protocol cap: {analysis['summary']['total_cap']}.", "",
        "Fresh force qualification uses independent checks. Structural and energy comparisons are scientific endpoint comparisons only when both endpoints pass fresh force qualification; otherwise measurements are archived-geometry diagnostics or unassessed. C4H6 graph equality does not imply the same conformational basin. C60 graph counts use 1.80 Å and 1.64 Å cutoffs.", "",
        "| System | Rows | Missing results | Fresh-qualified truncated/full | Quench converged | Not converged | Quench/flag missing | Ledger issues |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for system, summary in analysis["summary"]["by_system"].items():
        lines.append(
            f"| {system} | {summary['rows']} | {summary['missing_runs']} | "
            f"{summary['fresh_qualified_truncated']}/{summary['fresh_qualified_full']} | "
            f"{summary['quench_converged']} | {summary['quench_not_converged']} | "
            f"{summary['quench_missing']} | {summary['ledger_issues']} |")
    lines += [
        "", "Fresh energy difference is truncated minus original full in eV. Cost difference is saved prefix plus actual new quench requests minus original full requests; fresh calls are accounted separately.", "",
        "| System | Depth | N | Quench converged / not / missing | Fresh ΔE median [min, max] | Cost Δ median [min, max] requests |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in analysis["summary"]["by_system_depth"]:
        q = item["quench_convergence"]
        lines.append(
            f"| {item['system']} | {item['depth']} | {item['rows']} | "
            f"{q['converged']} / {q['not_converged']} / {q['missing_quench_or_flag']} | "
            f"{fmt_range(item['fresh_energy_difference_truncated_minus_full_eV'], 3)} | "
            f"{fmt_range(item['cost_delta_vs_full_requests'], 0)} |")
    lines += [
        "", "Structural diagnostics by endpoint pair. Start-related rows use archived geometries and have no fresh-qualified start endpoint; they do not establish physical qualification.", "",
        "| System | Depth | Pair | Geometry pairs / source rows | Fresh-qualified pairs | Same / different / unassessed graph | C4H6 RMS median [min,max] Å | C4H6 absolute Δtorsion median [min,max]° | C60 1.80Å same/diff/unassessed; both connected | C60 1.64Å same/diff/unassessed; both connected |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {"truncated_vs_start": "truncated vs start",
              "start_vs_original_full": "start vs original full",
              "truncated_vs_original_full": "truncated vs original full"}
    for item in analysis["summary"]["structural_pairs"]:
        if item["system"] == "C4H6":
            graph = f"{item['graph_same']}/{item['graph_different']}/{item['graph_unassessed']}"
            rms = fmt_range(item["proper_kabsch_rms_A"], 3)
            torsion = fmt_range(item["absolute_circular_torsion_difference_deg"], 1)
            c60_18 = c60_164 = "n/a"
        else:
            graph = rms = torsion = "n/a"
            c18 = item["c60_cutoff_graphs"]["1.80A"]
            c164 = item["c60_cutoff_graphs"]["1.64A"]
            c60_18 = f"{c18['same_graph']}/{c18['different_graph']}/{c18['unassessed']}; conn={c18['both_connected']}"
            c60_164 = f"{c164['same_graph']}/{c164['different_graph']}/{c164['unassessed']}; conn={c164['both_connected']}"
        lines.append(
            f"| {item['system']} | {item['depth']} | {labels[item['comparison']]} | "
            f"{item['geometry_pairs_compared']}/{item['source_rows']} | "
            f"{item['fresh_qualified_pairs']} | {graph} | {rms} | {torsion} | {c60_18} | {c60_164} |")
    lines += [
        "", "C4H6 torsion differences are shortest absolute circular differences of the detected CCCC dihedral; RMS is a continuous graph-compatible proper-Kabsch diagnostic with no similarity threshold. C60 geometric alignment is attempted only for connected graph-compatible pairs.", "",
        "No basin count, universal ranking, or end-to-end stopping policy is inferred. These saved-state diagnostics can motivate a separate depth-policy comparison; starting-geometry physical qualification remains unmeasured.", "",
    ]
    return "\n".join(lines)


def main():
    input_path = HERE / "inputs.json"
    inputs = read_json(input_path)
    if inputs is None:
        raise FileNotFoundError(input_path)
    rows = inputs.get("rows", [])
    if len(rows) != 32:
        raise ValueError(f"expected the frozen 32 input rows, found {len(rows)}")
    output_rows = [analyze_row(item, HERE / "runs" / item["case_id"]) for item in rows]
    total_cap = inputs.get("total_cap")
    observed = 0
    known = True
    for row in output_rows:
        for key in ("search_ledger", "fresh_ledger"):
            count = row[key].get("charged_calls")
            if count is None:
                known = False
            else:
                observed += count
    if isinstance(total_cap, int) and observed > total_cap:
        raise ValueError(f"observed charged requests {observed} exceed total cap {total_cap}")

    by_system = {}
    for system in sorted({r["system"] for r in rows}):
        subset = [r for r in output_rows if r["system"] == system]
        by_system[system] = {
            "rows": len(subset),
            "missing_runs": sum(r["status"] == "missing_run_result" for r in subset),
            "fresh_qualified_truncated": sum(r.get("fresh", {}).get("truncated", {}).get("force_qualified") is True for r in subset),
            "fresh_qualified_full": sum(r.get("fresh", {}).get("full", {}).get("force_qualified") is True for r in subset),
            "ledger_mismatches": sum(any(v is False for v in r.get("ledger_closure", {}).values()) for r in subset),
            "ledger_issues": sum(
                r.get("search_ledger", {}).get("status") != "valid"
                or r.get("fresh_ledger", {}).get("status") != "valid"
                or any(v is False for v in r.get("ledger_closure", {}).values())
                for r in subset),
            "quench_converged": sum(r.get("quench_convergence", {}).get("status") == "converged" for r in subset),
            "quench_not_converged": sum(r.get("quench_convergence", {}).get("status") == "not_converged" for r in subset),
            "quench_missing": sum(r.get("quench_convergence", {}).get("status") not in ("converged", "not_converged") for r in subset),
        }
    by_system_depth = []
    structural_pairs = []
    pair_names = ("truncated_vs_start", "start_vs_original_full", "truncated_vs_original_full")
    for system in sorted({r["system"] for r in rows}):
        for depth in sorted({r["depth"] for r in rows if r["system"] == system}):
            subset = [r for r in output_rows if r["system"] == system and r["depth"] == depth]
            deltas = [r.get("cost", {}).get("difference_vs_full") for r in subset]
            qstates = [r.get("quench_convergence", {}).get("status", "missing_result") for r in subset]
            energy_diffs = [r.get("fresh_energy_difference_truncated_minus_full_eV") for r in subset]
            by_system_depth.append({
                "system": system, "depth": depth, "rows": len(subset),
                "cost_delta_vs_full_requests": continuous_summary(deltas),
                "fresh_energy_difference_truncated_minus_full_eV": continuous_summary(energy_diffs),
                "quench_convergence": {
                    "converged": sum(x == "converged" for x in qstates),
                    "not_converged": sum(x == "not_converged" for x in qstates),
                    "missing_quench_or_flag": sum(x in ("missing_result", "missing_quench", "missing_flag") for x in qstates),
                    "denominator": len(subset),
                },
            })

            for pair_name in pair_names:
                eligible = []
                for row in subset:
                    comp = row.get("comparisons", {}).get(pair_name, {})
                    if comp.get("status") == "missing_geometry":
                        continue
                    if pair_name == "truncated_vs_original_full":
                        if row.get("comparison_scope", {}).get(pair_name, {}).get("fresh_qualified_pair") is not True:
                            continue
                    # Start comparisons intentionally remain archived-geometry diagnostics;
                    # their start endpoint has no fresh force qualification.
                    eligible.append((row, comp))

                entry = {"system": system, "depth": depth, "comparison": pair_name,
                         "source_rows": len(subset), "geometry_pairs_compared": len(eligible),
                         "fresh_qualified_pairs": len(eligible) if pair_name == "truncated_vs_original_full" else 0,
                         "scope": "fresh-qualified endpoints" if pair_name == "truncated_vs_original_full"
                            else "archived geometry diagnostic; not force-qualified at both endpoints"}
                if system == "C4H6":
                    same = different = 0
                    rms_values, torsion_deltas = [], []
                    torsion_same_region = torsion_different_region = 0
                    for _, comp in eligible:
                        isomorphic = comp.get("graph", {}).get("isomorphic")
                        same += isomorphic is True
                        different += isomorphic is False
                        rms = comp.get("geometry", {}).get("rms_A")
                        if isinstance(rms, (int, float)) and math.isfinite(rms):
                            rms_values.append(float(rms))
                        left, right = comp.get("torsion_left", {}), comp.get("torsion_right", {})
                        a, b = finite_float(left.get("raw_degrees")), finite_float(right.get("raw_degrees"))
                        if a is not None and b is not None:
                            torsion_deltas.append(abs((a - b + 180.0) % 360.0 - 180.0))
                        sa, sb = left.get("cosine_sign_region"), right.get("cosine_sign_region")
                        if sa is not None and sb is not None:
                            if sa == sb:
                                torsion_same_region += 1
                            else:
                                torsion_different_region += 1
                    entry.update(graph_same=same, graph_different=different,
                                 graph_unassessed=len(eligible) - same - different,
                                 proper_kabsch_rms_A=continuous_summary(rms_values),
                                 absolute_circular_torsion_difference_deg=continuous_summary(torsion_deltas),
                                 torsion_region_same=torsion_same_region,
                                 torsion_region_different=torsion_different_region)
                else:
                    cutoff_stats = {}
                    for cutoff in ("1.80A", "1.64A"):
                        same = different = both_connected = 0
                        for _, comp in eligible:
                            metrics = comp.get("cutoff_graphs", {}).get(cutoff, {})
                            same += metrics.get("isomorphic") is True
                            different += metrics.get("isomorphic") is False
                            both_connected += metrics.get("left_connected") is True and metrics.get("right_connected") is True
                        cutoff_stats[cutoff] = {"same_graph": same, "different_graph": different,
                                                "unassessed": len(eligible) - same - different,
                                                "both_connected": both_connected}
                    rms_values = [comp.get("geometry", {}).get("1.80A", {}).get("rms_A")
                                  for _, comp in eligible]
                    entry.update(c60_cutoff_graphs=cutoff_stats,
                                 proper_kabsch_rms_A=continuous_summary(rms_values))
                structural_pairs.append(entry)
    analysis = {
        "scope": "saved-path depth ablation; no rerun or general basin/performance claim",
        "input": str(input_path), "inputs_sources": inputs.get("sources", []),
        "alignment_self_check": self_check_alignment(rows),
        "summary": {"rows_expected": len(rows),
                    "rows_with_result": sum(r["status"] != "missing_run_result" for r in output_rows),
                    "rows_missing": sum(r["status"] == "missing_run_result" for r in output_rows),
                    "observed_charged_requests": observed,
                    "all_ledger_totals_known": known, "total_cap": total_cap,
                    "cap_respected_by_observed_calls": observed <= total_cap if known and isinstance(total_cap, int) else None,
                    "by_system": by_system, "by_system_depth": by_system_depth,
                    "structural_pairs": structural_pairs},
        "rows": output_rows,
        "interpretation_limits": [
            "Only independent fresh force checks confer force qualification; missing and failed checks remain distinct.",
            "C4H6 connectivity graph equality does not imply basin identity; torsion is a conformational diagnostic.",
            "C60 uses 1.80 A and 1.64 A graphs; geometric RMS uses graph-compatible atom mappings and proper rotations only, with no RMS decision threshold.",
            "Request ledgers count search/search_failure as charged calls; denials are retained but not charged.",
            "Saved prefix plus a new quench is a cost attribution, not a rerun of the full outer-step algorithm.",
        ]}
    (HERE / "analysis.json").write_text(json.dumps(analysis, indent=2, allow_nan=False) + "\n")
    (HERE / "report.md").write_text(render_report(analysis))
    print(json.dumps(analysis["summary"], indent=2))
    print("alignment self-check:", json.dumps(analysis["alignment_self_check"]))


if __name__ == "__main__":
    main()
