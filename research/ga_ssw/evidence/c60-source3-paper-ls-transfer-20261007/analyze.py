"""Read the fixed local C60 LS experiment; no calculator or PES evaluations."""
from __future__ import annotations
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys

CUTOFFS = (1.64, 1.70, 1.80)
PREFIXES = (5000, 10000, 20000)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def graph(atoms, cutoff):
    import networkx as nx
    import numpy as np
    distances = np.linalg.norm(atoms.positions[:, None] - atoms.positions[None, :], axis=2)
    g = nx.Graph()
    g.add_nodes_from(range(len(atoms)))
    g.add_edges_from(zip(*np.where(np.triu((distances < cutoff) & (distances > 0), 1))))
    return g


def analyze(prepared, out):
    import networkx as nx
    import numpy as np
    from ase.io import read, write
    sys.path.insert(0, str(prepared / "source"))
    from pamssw.standalone import load_ssw_checkpoint
    from pamssw.standalone.softening import FrozenBondSoftening, LSResponseState
    spec = importlib.util.spec_from_file_location("local_c60_validator", prepared / "validator.py")
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    manifest = json.loads((prepared / "search-plan-manifest.json").read_text())
    out.mkdir(parents=True, exist_ok=False)
    rows, observations_by_arm = [], {}
    for entry in sorted(manifest["arm_map"], key=lambda x: x["array_task_id"]):
        folder = prepared / entry["run_dir"]
        plan = json.loads((folder / "plan.json").read_text())
        row = dict(seed=entry["seed"], arm=entry["arm"], path=str(folder))
        if not (folder / "budget.json").exists():
            rows.append(dict(row, status="not_started", search_requests=0, fresh_requests=0))
            continue
        budget = json.loads((folder / "budget.json").read_text())
        raw = ([json.loads(line) for line in (folder / "requests.jsonl").read_text().splitlines()]
               if (folder / "requests.jsonl").exists() else [])
        if [r["id"] for r in raw] != list(range(1, len(raw) + 1)):
            raise ValueError(f"nonsequential search ledger: {folder}")
        uncertain = budget.get("unconfirmed_search_reservations", 0)
        if budget["status"] == "running":
            uncertain += max(0, budget.get("search_reserved", budget["search"]) - budget["search"])
        if budget["status"] != "running" and not uncertain and len(raw) != budget["search"]:
            raise ValueError(f"terminal search cost mismatch: {folder}")
        if budget["fresh"] != len(budget["fresh_checks"]):
            raise ValueError(f"fresh cost mismatch: {folder}")
        row.update(status=budget["status"], search_requests=budget["search"],
            fresh_requests=budget["fresh"], failed_pes_requests=sum("error" in r for r in raw),
            raw_requests=len(raw), uncertain_reservations=uncertain,
            actual_search_calculate=sum(s.get("calculate_calls", 0) for s in budget["segments"]),
            actual_fresh_calculate="not_instrumented_by_existing_runner",
            elapsed_seconds=sum(s.get("elapsed_seconds", 0) for s in budget["segments"]),
            fresh_checks=budget["fresh_checks"])
        cp_path = folder / "last-result.pkl"
        if not cp_path.exists():
            cp_path = folder / "checkpoint.pkl"
        if not cp_path.exists():
            rows.append(dict(row, checkpoint_status="missing"))
            continue
        cp = load_ssw_checkpoint(cp_path)
        ih = read(folder / plan["reference_input"])
        ih_graphs = {c: graph(ih, c) for c in CUTOFFS}
        cumulative = cp.initial.evaluation_requests
        observed = [(None, cumulative, cp.initial, None)]
        ls_events, prequench_cost, failed_prequench_cost = [], 0, 0
        ls_spec = plan.get("paper_ls")
        frozen = response = None
        current = cp.initial.atoms.copy()
        if ls_spec is not None:
            energies = {tuple(map(int, k.split(","))): v for k, v in ls_spec["bond_energies"].items()}
            lengths = {tuple(map(int, k.split(","))): v for k, v in ls_spec["bond_lengths"].items()}
            frozen = FrozenBondSoftening.from_atoms(current, bond_energies=energies,
                bond_lengths=lengths, initial_fraction=ls_spec["initial_fraction"], xi=ls_spec["xi"])
            response = LSResponseState(ls_spec["target_per_atom"], ls_spec["learning_rate"])
        for step in cp.records:
            cumulative += step.evaluation_requests
            # A failed soft quench is stored for diagnosis, never a bare-PES landing.
            if step.landing is not None and step.landing.converged and step.status != "ls_prequench_failed":
                observed.append((step.index, cumulative, step.landing, step.accepted))
            preparation = step.ls_preparation
            if step.accepted and step.landing is not None:
                current = step.landing.atoms.copy()
            if step.status == "ls_prequench_failed":
                failed_prequench_cost += step.evaluation_requests
                ls_events.append(dict(outer_index=step.index, status=step.status,
                    requests=step.evaluation_requests, error=step.error))
            if preparation is not None:
                prequench_cost += preparation["evaluation_requests"]
            if preparation is not None and step.status != "mc_failed":
                event = dict(outer_index=step.index, requests=preparation["evaluation_requests"],
                    qualification=preparation["qualification"],
                    softened_fmax=preparation["soft_quench"].max_force,
                    true_response_eV_atom=step.energy_response,
                    strength_before_eV=float(sum(frozen.strengths)),
                    pairs_before=len(frozen.pairs))
                try:
                    following = response.update(frozen, current,
                        energy_before=preparation["true_energy_before"],
                        energy_after=preparation["true_energy_after"],
                        bond_energies=energies, bond_lengths=lengths)
                except ValueError as error:
                    if step.status != "ls_update_failed":
                        raise
                    event.update(update_status="domain_failure", error=str(error))
                else:
                    frozen = following
                    event.update(update_status="replayed", strength_after_eV=float(sum(frozen.strengths)),
                                 pairs_after=len(frozen.pairs))
                ls_events.append(event)
        if cumulative != cp.evaluation_requests or cumulative > budget["search"]:
            raise ValueError(f"checkpoint lineage cost mismatch: {folder}")
        if frozen is not None:
            if (frozen.pairs != cp.frozen.pairs or
                    not np.allclose(frozen.strengths, cp.frozen.strengths, rtol=1e-12, atol=1e-12)):
                raise ValueError(f"LS response replay differs from saved final state: {folder}")
        observations, frames, graph_classes = [], [], []
        initial_graph = graph(cp.initial.atoms, 1.7)
        for index, cost, minimum, accepted in observed:
            atoms = minimum.atoms.copy()
            atoms.calc = None
            graphs = {str(c): validator.graph_row(atoms.numbers, atoms.positions, c, ih_graphs[c]) for c in CUTOFFS}
            force_ok = bool(minimum.converged and minimum.max_force <= plan["ssw_config"]["fmax"])
            connected = all(r["components"] == 1 for r in graphs.values())
            ih_ok = all(r["graph_cage_candidate"] and r["ih_graph_match"] for r in graphs.values())
            energy_ok = minimum.energy <= plan["reference_energy_eV"] + .01
            candidate = bool(force_ok and ih_ok and energy_ok)
            if force_ok and connected:
                g = graph(atoms, 1.7)
                if not nx.is_isomorphic(g, initial_graph) and not any(nx.is_isomorphic(g, h) for h in graph_classes):
                    graph_classes.append(g)
            cold_qualified = False
            for label, check in budget["fresh_checks"].items():
                fresh_path = folder / f"{label}-fresh.traj"
                if check.get("numerical_qualified") and fresh_path.is_file():
                    fresh = read(fresh_path)
                    if np.array_equal(fresh.positions, atoms.positions):
                        cold_qualified = True
            record = dict(outer_index=index, search_cost=cost, energy_eV=minimum.energy,
                relative_to_ih_eV=minimum.energy - plan["reference_energy_eV"],
                fmax_eV_A=minimum.max_force, force_qualified=force_ok,
                connected_all_cutoffs=connected, graphs=graphs, accepted=accepted,
                ih_all_cutoffs=ih_ok, energy_target=energy_ok,
                target_candidate=candidate, cold_qualified=cold_qualified,
                target_requires_3d_review=bool(candidate and cold_qualified))
            observations.append(record)
            atoms.info.update({k: v for k, v in record.items() if k != "graphs" and v is not None})
            frames.append(atoms)
        key = (entry["seed"], entry["arm"])
        observations_by_arm[key] = observations
        stem = f"seed-{entry['seed']}-{entry['arm']}"
        dump(out / f"{stem}-observations.json", observations)
        dump(out / f"{stem}-ls-events.json", ls_events)
        if frames:
            write(out / f"{stem}-landings.extxyz", frames)
        row.update(outer_attempts=len(cp.records), lineage_requests=cp.evaluation_requests,
            statuses=dict(Counter(s.status for s in cp.records)),
            climb_stages=sum(len(s.climb) for s in cp.records),
            generated_gaussians=sum("weight" in event for s in cp.records for event in s.climb),
            qualified_observations=sum(r["force_qualified"] for r in observations),
            fragmented_observations=sum(r["force_qualified"] and not r["connected_all_cutoffs"] for r in observations),
            new_connected_graph_classes=len(graph_classes),
            ls_completed_prequench_requests=prequench_cost,
            ls_failed_prequench_requests=failed_prequench_cost,
            ls_total_prequench_requests=prequench_cost + failed_prequench_cost,
            target_candidate=any(r["target_candidate"] for r in observations),
            cold_target_candidate=any(r["target_candidate"] and r["cold_qualified"] for r in observations))
        rows.append(row)
    comparisons = []
    for seed in sorted({r["seed"] for r in rows}):
        arms = [r for r in rows if r["seed"] == seed]
        common_cost = min(r["search_requests"] for r in arms)
        for prefix in PREFIXES:
            horizon = min(prefix, common_cost)
            paired = {}
            for r in arms:
                valid = [v for v in observations_by_arm.get((seed, r["arm"]), [])
                         if v["force_qualified"] and v["search_cost"] <= horizon]
                connected = [v for v in valid if v["connected_all_cutoffs"]]
                hits = [v for v in valid if v["target_candidate"]]
                paired[r["arm"]] = dict(qualified_observations=len(valid),
                    best_connected_energy_eV=min((v["energy_eV"] for v in connected), default=None),
                    first_ih_energy_force_cost=hits[0]["search_cost"] if hits else None,
                    cold_target_candidate=any(v["cold_qualified"] for v in hits))
            comparisons.append(dict(seed=seed, requested_prefix=prefix, common_horizon=horizon,
                both_horizon_reached=common_cost >= prefix, arms=paired))
    if sum(r["search_requests"] for r in rows) > 80000 or sum(r["fresh_requests"] for r in rows) > 12:
        raise ValueError("aggregate protocol budget exceeded")
    result = dict(rows=rows, common_cost_comparisons=comparisons,
        scope="One new published defect cage, two search RNG seeds, MH-1; candidates need 3D review; not random-start global success rate.",
        qualified_input=json.loads((prepared / manifest["qualification_summary"]).read_text()))
    dump(out / "analysis.json", result)
    print(json.dumps([{k: r.get(k) for k in ("seed", "arm", "status", "search_requests", "target_candidate", "cold_target_candidate")} for r in rows]))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("prepared", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    analyze(args.prepared.resolve(), args.out.resolve())
