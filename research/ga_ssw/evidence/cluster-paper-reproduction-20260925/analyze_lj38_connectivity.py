"""Read saved LJ38 minima and summarize geometric connectivity only.

No calculator, potential, optimizer, or graph-isomorphism code is used.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np
from ase.io import read


HERE = Path(__file__).resolve().parent
RUN_DIRS = (
    HERE / "runs" / "lj38-seed25092501",
    HERE / "runs" / "lj38-seed25092502",
    HERE / "paper-direction-runs" / "lj38-seed25092501",
    HERE / "paper-direction-runs" / "lj38-seed25092502",
)
OUTPUT = HERE / "lj38-connectivity.json"
SIGMA_A = 2.7
CUTOFF_SCALES = (1.3, 1.5)


def graph_stats(path: Path, cutoff: float) -> dict:
    atoms = read(path, index=0)
    positions = np.asarray(atoms.positions, dtype=float)
    if len(atoms) != 38 or not np.isfinite(positions).all():
        raise ValueError(f"expected 38 finite atoms: {path}")
    if np.asarray(atoms.pbc, dtype=bool).any():
        raise ValueError(f"expected nonperiodic saved cluster: {path}")
    distances = atoms.get_all_distances(mic=False)
    adjacency = distances < cutoff
    np.fill_diagonal(adjacency, False)
    unseen = set(range(len(atoms)))
    sizes = []
    while unseen:
        stack = [unseen.pop()]
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            neighbors = set(np.flatnonzero(adjacency[node])) & unseen
            unseen.difference_update(neighbors)
            stack.extend(neighbors)
        sizes.append(size)
    sizes.sort(reverse=True)
    return {"components": len(sizes), "largest_component": sizes[0],
            "component_sizes_desc": sizes, "single_cluster": len(sizes) == 1}


def load_events(folder: Path) -> tuple[list[dict], list[dict]]:
    path = folder / "outer-steps.jsonl"
    events, errors = [], []
    if not path.is_file():
        return events, [{"kind": "missing_outer_steps", "path": str(path)}]
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        try:
            events.append(json.loads(line))
        except (json.JSONDecodeError, TypeError) as error:
            errors.append({"kind": "invalid_jsonl_line", "line": line_number,
                           "error": repr(error)})
    return events, errors


def summarize_group(rows: list[dict], cutoff_key: str) -> dict:
    classified = [row for row in rows if row.get(cutoff_key) is not None]
    return {
        "expected_minima": len(rows),
        "classified_minima": len(classified),
        "missing_or_failed_minima": len(rows) - len(classified),
        "component_count_histogram": dict(sorted(Counter(
            str(row[cutoff_key]["components"]) for row in classified).items(),
            key=lambda item: int(item[0]))),
        "largest_component_histogram": dict(sorted(Counter(
            str(row[cutoff_key]["largest_component"]) for row in classified).items(),
            key=lambda item: int(item[0]))),
        "single_cluster_count": sum(row[cutoff_key]["single_cluster"] for row in classified),
        "single_cluster_fraction_among_classified": (
            None if not classified else sum(row[cutoff_key]["single_cluster"]
                                             for row in classified) / len(classified)),
    }


def analyze_run(folder: Path, root: Path) -> dict:
    events, errors = load_events(folder)
    summary_path = folder / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else None
    records, missing_files, seen_indices = [], [], set()
    missing_landing_rows = []
    for event in events:
        step = event.get("step")
        new_minima = event.get("new_minima") or []
        if step is not None and step >= 0 and not new_minima:
            missing_landing_rows.append({"step": step, "status": event.get("status"),
                                         "landing_energy_eV": event.get("landing_energy_eV")})
        for scalar in new_minima:
            index = scalar.get("index")
            row = {"minimum_index": index, "step": step,
                   "role": "initial_true_quench" if step == -1 else "landing",
                   "energy_eV": scalar.get("energy_eV"),
                   "fmax_eV_A": scalar.get("fmax_eV_A"),
                   "converged": scalar.get("converged"),
                   "event_status": event.get("status")}
            if index in seen_indices:
                row["error"] = "duplicate_minimum_index_in_scalar_events"
                errors.append({"kind": "duplicate_minimum_index", "minimum_index": index})
            seen_indices.add(index)
            geometry_path = folder / "minima" / f"minimum-{int(index):04d}.extxyz" if index is not None else None
            if geometry_path is None or not geometry_path.is_file():
                row["geometry_status"] = "missing"
                row["geometry_path"] = None if geometry_path is None else str(geometry_path)
                missing_files.append({"minimum_index": index, "step": step,
                                      "path": None if geometry_path is None else str(geometry_path)})
            else:
                row["geometry_status"] = "classified"
                row["geometry_path"] = str(geometry_path)
                for scale in CUTOFF_SCALES:
                    key = f"cutoff_{scale:.1f}sigma_A"
                    try:
                        row[key] = graph_stats(geometry_path, scale * SIGMA_A)
                    except Exception as error:
                        row[key] = None
                        row["geometry_error"] = repr(error)
                        errors.append({"kind": "geometry_error", "minimum_index": index,
                                       "step": step, "error": repr(error)})
            records.append(row)

    initial = next((row for row in records if row["role"] == "initial_true_quench"), None)
    initial_event = next((event for event in events if event.get("step") == -1), None)
    landings = [row for row in records if row["role"] == "landing"]
    scalar_best = min((row for row in records
                       if isinstance(row.get("energy_eV"), (int, float)) and
                       np.isfinite(row["energy_eV"])),
                      key=lambda row: row["energy_eV"], default=None)
    summary_best_energy = None if summary is None else summary.get("best_energy_eV")
    best_match = (scalar_best is not None and summary_best_energy is not None and
                  abs(scalar_best["energy_eV"] - summary_best_energy) <= 1e-8)
    result = {
        "run": str(folder.relative_to(root)),
        "status": None if summary is None else summary.get("status"),
        "seed": None if summary is None else summary.get("seed"),
        "search_requests": None if summary is None else summary.get("search_requests"),
        "checkpoint_requests": None if summary is None else summary.get("checkpoint_requests"),
        "scalar_event_count": len(events),
        "outer_step_count": sum(isinstance(event.get("step"), int) and event["step"] >= 0
                                 for event in events),
        "outer_steps_without_saved_landing": missing_landing_rows,
        "initial_quench_event": (None if initial_event is None else {
            key: initial_event.get(key) for key in
            ("status", "requests", "cumulative_requests", "energy_eV", "fmax_eV_A", "converged", "error")
            if key in initial_event}),
        "initial_quench_event_missing": initial_event is None,
        "initial_true_quench": initial,
        "landing_minima": landings,
        "lowest_energy_minimum_from_scalar_events": scalar_best,
        "summary_best_energy_eV": summary_best_energy,
        "scalar_best_matches_summary_best": best_match,
        "missing_geometry_files": missing_files,
        "errors": errors,
        "landing_summary_by_cutoff": {
            f"{scale:.1f}sigma": summarize_group(landings, f"cutoff_{scale:.1f}sigma_A")
            for scale in CUTOFF_SCALES},
    }
    return result


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    runs = []
    for folder in RUN_DIRS:
        if folder.is_dir():
            runs.append(analyze_run(folder, HERE))
        else:
            runs.append({"run": str(folder.relative_to(HERE)),
                         "status": "missing_run", "errors": ["missing_run_directory"]})
    result = {
        "scope": "Saved LJ38 structures; connectivity only, not basin identity or success.",
        "potential_or_calculator_calls": 0,
        "source": "Per-run minima/*.extxyz and scalar outer-steps.jsonl; summary.json for status/cost cross-check.",
        "connectivity_definition": {
            "sigma_A": SIGMA_A,
            "cutoff_scales": list(CUTOFF_SCALES),
            "cutoffs_A": {f"{scale:.1f}sigma": scale * SIGMA_A
                           for scale in CUTOFF_SCALES},
            "edge_rule": "nonperiodic pair distance strictly below cutoff",
        },
        "runs": runs,
    }
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
