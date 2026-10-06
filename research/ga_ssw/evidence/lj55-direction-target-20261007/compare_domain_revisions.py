#!/usr/bin/env python3
"""Offline domain-fix regression and paired cost prefixes; no PES evaluation."""
import argparse
import json
from pathlib import Path


def ledger(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def signature(row):
    return tuple(row.get(k) for k in ('charged', 'status', 'energy_eV',
                 'fmax_eV_A', 'actual_calculator_called', 'actual_calculator_calls'))


def curve(result):
    initial = result['initial']
    cost = result['initial_quench_requests']
    points = [(cost, initial['energy_eV'])] if initial['converged'] else []
    for event in result['outer_events']:
        cost += event['step_requests']
        if 'cumulative_requests' in event:
            assert cost == event['cumulative_requests']
        gate = event.get('minimum_gate')
        # Target prefilter deliberately does not compute all landing connectivity.
        # General connected-landing checks come from the saved geometry below.
        if gate and gate['force_gate_0p05'] and gate['converged']:
            points.append((cost, gate['energy_eV'], event['minimum_structure']))
    return points


def main():
    from ase.io import read
    import importlib.util
    import sys
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location('lj55_domain_runner', here/'run.py')
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--v1', type=Path, required=True)
    parser.add_argument('--v2', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    old = json.loads(args.v1.read_text())
    new = json.loads(args.v2.read_text())
    args.output.mkdir()
    checks, curves, structural_hits = [], {}, []
    for before, after in zip(old['rows'], new['rows'], strict=True):
        assert (before['slot'],before['seed'],before['arm']) == (after['slot'],after['seed'],after['arm'])
        a, b = Path(before['directory']), Path(after['directory'])
        old_rows = list(ledger(a/'search-ef-ledger.jsonl'))
        new_rows = list(ledger(b/'search-ef-ledger.jsonl'))
        is_full = before['arm']=='full_per_atom'
        prefix = len(old_rows) if is_full else len(old_rows)-1
        same = len(new_rows)>=prefix and all(signature(x)==signature(y)
               for x,y in zip(old_rows[:prefix],new_rows[:prefix],strict=True))
        row = dict(slot=before['slot'],arm=before['arm'],seed=before['seed'],
                   identical_successful_prefix_requests=prefix, numerical_prefix_unchanged=same)
        assert same, row
        if is_full:
            row['complete_trace_unchanged'] = len(old_rows)==len(new_rows)
            assert row['complete_trace_unchanged'], row
        else:
            failed = old_rows[-1]
            assert failed['status']=='failed' and not failed['actual_calculator_called']
            row['old_research_gate_failure'] = failed.get('error')
            if len(new_rows)>prefix:
                continued = new_rows[prefix]
                row['replayed_boundary_request'] = continued
                assert continued['geometry_gate']['eligible']
                assert not continued['geometry_gate']['native_image_diagnostics_only']['eligible']
        checks.append(row)
        result = json.loads((b/'result.json').read_text())
        frames = read(b/'outer-structures.extxyz', ':')
        frame_map = {(f.info['event_role'],int(f.info['event_index'])):f for f in frames}
        points = []
        first_shape = None
        for point in curve(result):
            if len(point)==2:
                structure=frame_map[('initial',-1)]
            else:
                role_index=point[2].split('#')[1].split(':')
                structure=frame_map[(role_index[0],int(role_index[1]))]
            if runner.PANEL.connected_components(structure)==[55]:
                points.append(point[:2])
                if first_shape is None:
                    match=runner.ANALYSIS.compare_geometry(runner.target_atoms(), structure)
                    if match.get('classification')=='same':
                        first_shape=dict(cost=point[0],energy_eV=point[1],geometry=match,
                            energy_window_passed=point[1]<=runner.TARGET_E+runner.HIT_TOL)
        structural_hits.append(dict(slot=before['slot'],seed=before['seed'],arm=before['arm'],
            first_connected_force_qualified_reference_geometry=first_shape,
            scope='Post hoc structure-only diagnostic, not a replacement of joint predeclared hit.'))
        curves[before['slot']]=points
    prefixes=[]
    for seed_slot in (0,2):
        paired=new['rows'][seed_slot:seed_slot+2]
        common=min(r['search']['charged'] for r in paired)
        for cost in sorted(set(x for x in (1000,5000,10000,25000,common) if x<=common)):
            energies={r['arm']:min((e for c,e in curves[r['slot']] if c<=cost),default=None) for r in paired}
            prefixes.append(dict(seed=paired[0]['seed'],prefix=cost,best_connected_qualified_energy=energies))
    payload=dict(source_v1=str(args.v1.resolve()),source_v2=str(args.v2.resolve()),
                 checks=checks, matched_development_prefixes=prefixes, structure_only_diagnostics=structural_hits,
                 scope='Full repetitions are regressions, not additional independent trajectories; no PES.')
    (args.output/'comparison.json').write_text(json.dumps(payload,indent=2)+'\n')
    lines=['# LJ55 domain correction: trace regression and matched prefixes','',
           'All successful v1 prefixes reproduced; full-direction repetitions are regression checks, not new seeds.','',
           '| Seed | Paid prefix | Rotation connected best E | Full connected best E |',
           '|---:|---:|---:|---:|']
    for p in prefixes:
        e=p['best_connected_qualified_energy']
        lines.append(f"| {p['seed']} | {p['prefix']} | {e['rotation']} | {e['full_per_atom']} |")
    lines+=['', '| Seed | Arm | First structure-only Ih cost | E | Joint energy window at that frame |',
            '|---:|---|---:|---:|---|']
    for r in structural_hits:
        hit=r['first_connected_force_qualified_reference_geometry'] or {}
        lines.append(f"| {r['seed']} | {r['arm']} | {hit.get('cost')} | {hit.get('energy_eV')} | {hit.get('energy_window_passed')} |")
    lines+=['','Structure-only rows are post hoc diagnostics; first structural and joint hits remain distinct.',
            'Endpoints credited after full true-quench cost, with connected 1.3-sigma geometry and configured force certificate.',
            'Common observed horizon is a post hoc development diagnostic; no extrapolated first-hit cost.']
    (args.output/'README.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':
    main()
