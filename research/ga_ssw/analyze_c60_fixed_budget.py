"""Offline fixed-budget analysis for the two random-C60 development arms."""
import argparse
import json
import re
from pathlib import Path

import numpy as np

from analyze_c60_random_development import EREF, ETOL, FMAX, graph_row


BUDGETS = (6000, 15000, 30000, 60000)
EVENT_RE = re.compile(
    r"Minimum found\s+(\d+)\s+(\d+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)"
    r".*?\s(F|T)\s+[-+0-9.eE]+\s+([-+0-9.eE]+).*?\s(\d+)\s*$",
    re.M,
)


def case_dirs(root):
    """Find exactly one result directory for each required seed."""
    found = {}
    for path in root.iterdir():
        if not path.is_dir():
            continue
        m = re.search(r"(?:seed|c60[_-]?)(17093|17094)", path.name)
        if m:
            seed = int(m.group(1))
            if seed in found:
                raise RuntimeError(f"multiple result directories for seed {seed}")
            found[seed] = path
    for seed in (17093, 17094):
        found.setdefault(seed, root / f"seed{seed}")
    return found


def load_reference(root, reference_path=None):
    if reference_path is None:
        candidates = sorted(root.parent.glob("c60-reference-qualification*/final.extxyz"))
        if not candidates:
            raise FileNotFoundError("C60 reference final.extxyz not found beside input evidence")
        reference_path = candidates[-1]
    from ase.io import read
    atoms = read(reference_path)
    import networkx as nx
    d = np.linalg.norm(atoms.positions[:, None] - atoms.positions[None, :], axis=2)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(atoms)))
    graph.add_edges_from(zip(*np.where(np.triu((d < 1.8) & (d > 0), 1))))
    return atoms, graph


def python_case(path, reference_graph, reference_energy):
    if not (path / "result.json").exists():
        return {"status": "missing_result", "search_requests": None}, []
    result = json.loads((path / "result.json").read_text())
    summary = json.loads((path / "summary.json").read_text())
    checks = {int(c["index"]): c for c in summary.get("fresh_checks", summary.get("checks", []))}
    events = []
    cumulative = int(result['initial']['evaluation_requests'])
    candidates = [(0, cumulative, result['initial'])]
    minimum_index = 1
    for record in result.get('records', []):
        cumulative += int(record['evaluation_requests'])
        landing = record.get('landing')
        if landing is not None and landing['converged']:
            candidates.append((minimum_index, cumulative, landing))
            minimum_index += 1
    if cumulative != result['evaluation_requests']:
        raise ValueError('outer-step cost accounting mismatch')
    if len(candidates) != len(result['minima']):
        raise ValueError('qualified landing to minima index mismatch')
    for index, cost, minimum in candidates:
        atoms = minimum['atoms']
        check = checks.get(index, {})
        geometry = {str(cut): graph_row(atoms['numbers'], atoms['positions'], cut, reference_graph)
                    for cut in (1.64, 1.7, 1.8)}
        qualified = bool(check.get('qualified', False))
        events.append(dict(index=index, cumulative_search_calls=cost,
            energy_eV=minimum['energy'], fmax_eV_per_A=check.get('fmax_eV_per_A', check.get('fmax')),
            fresh_qualified=qualified, fresh_check_present=bool(check),
            positions_A=atoms['positions'], geometry=geometry,
            cage_candidate=geometry['1.8']['graph_cage_candidate'],
            energy_success=qualified and minimum['energy'] <= reference_energy + ETOL))
    return summary, events


def native_case(path, native_root, reference_graph, reference_energy):
    """Parse LASP events while retaining only request coordinates needed by events."""
    process = path / 'process.json'
    summary = {'process': json.loads(process.read_text()) if process.exists() else None}
    events = []
    cumulative = 0
    if not (path / 'lasp.out').exists() or not (native_root / 'requests.jsonl').exists():
        return dict(summary, status='missing_native_output'), [], None
    text = (path / 'lasp.out').read_text(errors='replace')
    for match in EVENT_RE.finditer(text):
        delta = int(match.group(7))
        cumulative += delta
        events.append(dict(index=int(match.group(1)), cumulative_search_calls=cumulative,
            cost_evaluation_requests=delta, energy_eV=float(match.group(4)),
            event_force_component=float(match.group(6)), fresh_qualified=False,
            fresh_check_present=False, online_force_qualification=False,
            matched_request=False, positions_A=[]))
    wanted = {row['cumulative_search_calls']: row for row in events}
    count = 0
    with (native_root / 'requests.jsonl').open() as stream:
        for line in stream:
            request = json.loads(line)
            if request.get('case') != path.name or not request.get('response', {}).get('ok'):
                continue
            count += 1
            row = wanted.get(count)
            if row is None:
                continue
            forces = np.asarray(request['forces'], float)
            energy_match = abs(request['energy'] - row['energy_eV']) <= 5.1e-7
            force_match = abs(float(abs(forces).max()) - row['event_force_component']) <= .00051
            row.update(energy_match=energy_match, force_match=force_match,
                       matched_request=energy_match and force_match)
            if not row['matched_request']:
                continue
            fmax = float(np.linalg.norm(forces, axis=1).max())
            geometry = {str(cut): graph_row([6] * 60, request['positions'], cut, reference_graph)
                        for cut in (1.64, 1.7, 1.8)}
            qualified = bool(np.isfinite(fmax) and fmax <= FMAX)
            row.update(positions_A=request['positions'], fmax_eV_per_A=fmax,
                energy_eV=request['energy'], online_force_qualification=qualified,
                geometry=geometry, cage_candidate=geometry['1.8']['graph_cage_candidate'],
                energy_success=qualified and request['energy'] <= reference_energy + ETOL)
    summary['unmatched_events'] = sum(not e['matched_request'] for e in events)
    return summary, events, count


def cut_summary(events, budget, *, native=False, actual_calls=None, native_fresh=False):
    reached = [e for e in events if e["cumulative_search_calls"] <= budget]
    if native and not native_fresh:
        eligible = [e for e in reached if e.get("online_force_qualification")]
        qualification = "online force qualification only; no independent fresh check"
    else:
        eligible = [e for e in reached if e.get("fresh_qualified")]
        qualification = "independent fresh qualification"
    return {
        "budget_search_calls": budget,
        "budget_reached": actual_calls is not None and actual_calls >= budget,
        "actual_search_calls": actual_calls,
        "events_within_budget": len(reached),
        "qualified_minima": len(eligible),
        "best_energy_eV": min((e.get("energy_eV") for e in eligible if e.get("energy_eV") is not None), default=None),
        "cage_candidates": sum(bool(e.get("cage_candidate")) for e in eligible),
        "energy_successes": sum(bool(e.get("energy_success")) for e in eligible),
        "cage_energy_intersections": sum(bool(e.get("cage_candidate") and e.get("energy_success")) for e in eligible),
        "qualification_basis": qualification,
        "missing_or_truncated": not bool(reached) or (actual_calls is not None and actual_calls < budget),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", type=Path, required=True)
    ap.add_argument("--native", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--native-fresh", type=Path)
    ap.add_argument("--reference", type=Path)
    ap.add_argument("--reference-energy", type=float)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    if (args.reference is None) != (args.reference_energy is None):
        ap.error("--reference and --reference-energy must be provided together")

    def has_mh1_backend(plan_root):
        for plan in sorted(plan_root.glob("**/plan.json")):
            try:
                text = plan.read_text(errors="replace").lower()
            except OSError:
                continue
            if "mh1" in text or "mace-mh" in text:
                return True
        return False

    mh1 = has_mh1_backend(args.python) or has_mh1_backend(args.native)
    if mh1 and args.reference is None:
        ap.error("MH1 backend detected in plan; explicit --reference and --reference-energy are required")
    reference_energy = EREF if args.reference_energy is None else args.reference_energy
    py_cases = case_dirs(args.python)
    native_cases = case_dirs(args.native)
    reference, reference_graph = load_reference(args.native, args.reference)
    output = {"scope": "fixed-budget development curves; not an independent success rate", "budgets": list(BUDGETS), "reference": {"path": str(args.reference) if args.reference else "legacy OMAT reference discovered beside native evidence", "energy_eV": reference_energy, "energy_tolerance_eV": ETOL, "fmax_eV_per_A": FMAX, "explicit": args.reference is not None}, "cases": {}}
    for seed in (17093, 17094):
        py_summary, py_events = python_case(py_cases[seed], reference_graph, reference_energy)
        na_summary, na_events, native_calls = native_case(native_cases[seed], args.native, reference_graph, reference_energy)
        fresh_complete = False
        if args.native_fresh is not None:
            fresh_path = args.native_fresh / 'events' / f'seed{seed}.fresh.jsonl'
            if fresh_path.exists():
                fresh = {row['cumulative_request']: row for line in fresh_path.open()
                         if line.strip() for row in [json.loads(line)] if 'cumulative_request' in row}
                for event in na_events:
                    check = fresh.get(event['cumulative_search_calls'])
                    event['fresh_check_present'] = check is not None
                    if check is not None:
                        event['fresh_qualified'] = bool(check.get('qualified'))
                        event['fresh_energy_eV'] = check.get('fresh_energy_eV')
                        event['energy_success'] = bool(event['fresh_qualified'] and check['fresh_energy_eV'] <= reference_energy + ETOL)
                fresh_complete = True  # Missing/failed checks remain unqualified.
        py_calls = py_summary["search_requests"]
        output["cases"][str(seed)] = {"python": {"summary": py_summary, "events": py_events, "budgets": [cut_summary(py_events, b, actual_calls=py_calls) for b in BUDGETS]}, "native": {"summary": na_summary, "events": na_events, "actual_search_calls": native_calls, "budgets": [cut_summary(na_events, b, native=True, actual_calls=native_calls, native_fresh=fresh_complete) for b in BUDGETS]}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(output, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
