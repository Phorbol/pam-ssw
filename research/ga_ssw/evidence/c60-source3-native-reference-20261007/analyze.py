"""Offline native minimum-event readout; never evaluates a calculator."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import re

import numpy as np
from ase import Atoms
from ase.io import read, write

CUTS = (1.64, 1.7, 1.8)
PREFIXES = (5000, 10000, 15000)
EVENT_RE = re.compile(
    r"Minimum found\s+(\d+)\s+(\d+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)"
    r".*?\s(F|T)\s+[-+0-9.eE]+\s+([-+0-9.eE]+).*?\s(\d+)\s*$", re.M)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_events(text):
    cumulative, events = 0, []
    for m in EVENT_RE.finditer(text):
        cumulative += int(m.group(7))
        events.append(dict(ordinal=int(m.group(1)), event_index=int(m.group(2)),
            printed_energy_eV=float(m.group(4)), printed_force_component=float(m.group(6)),
            search_cost=cumulative, initial=int(m.group(1)) == 0))
    return events


def stream_ledger(path, wanted):
    counts = dict(raw=0, paid=0, failed_paid=0, denied=0, unexpected_denied=0, geometry_bad=0,
        metadata_bad=0, nonfinite_paid=0)
    selected = {}
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            counts['raw'] += 1
            if row['request'] != counts['raw']:
                raise ValueError('Nonsequential native request IDs')
            if row.get('paid_ef') is not True:
                counts['denied'] += 1
                if row.get('error_kind') != 'request_cap':
                    counts['unexpected_denied'] += 1
                continue
            counts['paid'] += 1
            # A prevalidation denial followed by paid work makes native offsets ambiguous.
            if row['request'] != counts['paid']:
                raise ValueError('Paid requests do not form a consecutive prefix')
            if row.get('ok') is False or 'energy' not in row or 'forces' not in row:
                counts['failed_paid'] += 1
            elif (not np.isfinite(float(row['energy'])) or
                  not np.isfinite(np.asarray(row['forces'], dtype=float)).all() or
                  not np.isfinite(np.asarray(row.get('positions', []), dtype=float)).all()):
                counts['nonfinite_paid'] += 1
            if row.get('geometry_gate', {}).get('eligible') is not True:
                counts['geometry_bad'] += 1
            if (row.get('cell') != (np.eye(3)*50).tolist() or
                row.get('pbc') != [True]*3 or
                np.asarray(row.get('positions', [])).shape != (60, 3) or
                np.asarray(row.get('forces', [])).shape != (60, 3) or
                float(row.get('model_r_max_A', -1)) != 6.0):
                counts['metadata_bad'] += 1
            if row['request'] in wanted:
                selected[row['request']] = row
    return counts, selected


def graph(atoms, cutoff):
    import networkx as nx
    d = np.linalg.norm(atoms.positions[:, None] - atoms.positions[None, :], axis=2)
    g = nx.Graph()
    g.add_nodes_from(range(len(atoms)))
    g.add_edges_from(zip(*np.where(np.triu((d < cutoff) & (d > 0), 1))))
    return g


def match_event(event, row, vacuum, validator, reference, reference_energy):
    import networkx as nx
    result = dict(event, matched=False, numerical_qualified=False, joint_target=False)
    if row is None or row.get('ok') is False or 'energy' not in row or 'forces' not in row:
        result['reason'] = 'missing_or_failed_paid_request'
        return result, None
    forces = np.asarray(row['forces'], dtype=float)
    positions = np.asarray(row['positions'], dtype=float)
    energy = float(row['energy'])
    if (forces.shape != (60, 3) or positions.shape != (60, 3) or
        not np.isfinite(forces).all() or not np.isfinite(positions).all() or not np.isfinite(energy)):
        result['reason'] = 'nonfinite_or_bad_shape'
        return result, None
    if (abs(energy - event['printed_energy_eV']) > 1e-4 or
        abs(float(np.abs(forces).max()) - event['printed_force_component']) > 5e-4):
        result['reason'] = 'printed_event_does_not_match_paid_request'
        return result, None
    atoms = Atoms('C60', positions=positions, cell=np.eye(3)*50, pbc=True)
    canonical, gate = vacuum.inspect_vacuum(atoms, 6.0)
    if canonical is None or not gate['eligible']:
        result['reason'] = 'periodic_image_gate_failed'
        return result, None
    fmax = float(np.linalg.norm(forces, axis=1).max())
    graphs = {str(c): validator.graph_row(canonical.numbers, canonical.positions, c) for c in CUTS}
    ih = all(nx.is_isomorphic(graph(canonical, c), graph(reference, c)) for c in CUTS)
    connected = all(g['components'] == 1 for g in graphs.values())
    result.update(matched=True, energy_eV=energy, fmax_eV_A=fmax,
        numerical_qualified=fmax <= .03, connected_all_cutoffs=connected,
        ih_all_cutoffs=ih, graphs=graphs, energy_target=energy <= reference_energy+.01,
        joint_target=bool(fmax <= .03 and ih and energy <= reference_energy+.01),
        cold_qualified=False, physical_review='not_performed')
    canonical.set_pbc(False)
    canonical.calc = None
    return result, canonical


def analyze(prepared, output):
    manifest = json.loads((prepared/'manifest.json').read_text())
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for entry in manifest['cases']:
        folder = prepared/entry['run_dir']
        if not (folder/'summary.json').exists():
            rows.append(dict(seed=entry['seed'], execution='not_started_or_missing_summary'))
            continue
        plan = json.loads((folder/'plan.json').read_text())
        summary = json.loads((folder/'summary.json').read_text())
        vacuum = load_module(folder/'vacuum_geometry.py', 'frozen_native_vacuum')
        validator = load_module(folder/'graph_helper.py', 'frozen_native_graph')
        reference = read(folder/'reference.extxyz')
        events = parse_events((folder/'lasp.out').read_text(errors='replace'))
        counts, selected = stream_ledger(folder/'request.jsonl', {e['search_cost'] for e in events})
        if counts['paid'] != summary['paid_ef'] or counts['raw'] != summary['requests']:
            raise ValueError('Summary and raw ledger disagree')
        observed, frames = [], {}
        for event in events:
            row, atoms = match_event(event, selected.get(event['search_cost']), vacuum,
                validator, reference, plan['reference_energy_eV'])
            observed.append(row)
            if atoms is not None:
                frames[event['ordinal']] = atoms
        process = summary.get('process') or {}
        clean = (not any(counts[k] for k in ['failed_paid', 'geometry_bad', 'metadata_bad', 'nonfinite_paid', 'unexpected_denied'])
            and bool(observed) and observed[0]['initial']
            and all(e['matched'] for e in observed)
            and not process.get('cleanup_survivors')
            and process.get('state') in ['completed', 'timeout'])
        qualified = [e for e in observed if e['numerical_qualified']]
        best = min((e for e in qualified if e['connected_all_cutoffs']),
            key=lambda e:e['energy_eV'], default=None)
        target = next((e for e in qualified if e['joint_target'] and not e['initial']), None)
        initial = next((e for e in qualified if e['initial']), None)
        selected_fresh = {}
        for label, event in [('initial', initial), ('best', best), ('first_target', target)]:
            if event is not None:
                filename = f"seed-{entry['seed']}-{label}.extxyz"
                write(output/filename, frames[event['ordinal']])
                selected_fresh[label] = dict(path=filename, event=event['ordinal'],
                    search_cost=event['search_cost'], energy_eV=event['energy_eV'])
        prefixes = []
        for cost in PREFIXES:
            available = [e for e in qualified if e['search_cost'] <= min(cost, counts['paid'])]
            prefixes.append(dict(requested=cost, charged_horizon=min(cost, counts['paid']),
                charged_horizon_reached=counts['paid'] >= cost,
                last_observed_minimum_cost=max((e['search_cost'] for e in available), default=None),
                best_connected_energy_eV=min((e['energy_eV'] for e in available if e['connected_all_cutoffs']), default=None),
                first_target_cost=next((e['search_cost'] for e in available if e['joint_target'] and not e['initial']), None)))
        dump(output/f"seed-{entry['seed']}-observations.json", observed)
        if frames:
            write(output/f"seed-{entry['seed']}-minima.extxyz", list(frames.values()))
        rows.append(dict(seed=entry['seed'], counts=counts, actual_search_calculate=summary.get('actual_calculate_calls'),
            process=process, comparison_eligible=clean, ssw_done=summary.get('ssw_done'),
            cap_reached=counts['paid'] >= plan['request_cap'], minimum_events=len(observed),
            qualified_observations=len(qualified), prefixes=prefixes, fresh_selection=selected_fresh))
    dump(output/'analysis.json', dict(rows=rows, scope='native stack on MH1; no fresh/PES calls',
        limitations=['Different RNG/MC/optimizer and periodic vacuum; not a component ablation',
            'Targets require independent cold and three-dimensional physical qualification',
            'Completed minima only; unfinished paid tails remain charged']))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepared', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    analyze(args.prepared.resolve(), args.output.resolve())
