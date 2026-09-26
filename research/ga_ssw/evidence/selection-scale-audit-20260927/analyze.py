"""Conditional probabilities on archived MC proposals, never a counterfactual chain."""
import json
import math
from pathlib import Path
import numpy as np
from ase import units, Atoms
import networkx as nx


def bond_graph(d, cutoffs):
    a = Atoms(numbers=d['numbers'], positions=d['positions'])
    g = nx.Graph()
    g.add_nodes_from((i, {'Z': int(z)}) for i, z in enumerate(a.numbers))
    for i in range(len(a)):
        for j in range(i):
            key = ','.join(map(str, sorted((int(a.numbers[i]), int(a.numbers[j])))))
            if np.linalg.norm(a.positions[i] - a.positions[j]) <= cutoffs[key]:
                g.add_edge(i, j)
    return g

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
plan = json.loads((ROOT / 'ls-pool-routing-20260926/plan.json').read_text())
results = []
for case_index, arm_index in [(0, 0), (1, 3)]:
    case = plan['cases'][case_index]
    path = ROOT / f'ls-pool-routing-20260926/run-1500677-{arm_index}'
    raw = json.loads((path / 'search-result.json').read_text())
    result = raw['result']
    fresh = {c['record_index']: c for c in json.loads((path / 'fresh-checks.json').read_text())['checks']}
    assert fresh[-1]['certified']
    current = result['initial']['energy']
    current_atoms = result['initial']['atoms']
    temperature = case['config']['temperature_K']
    rows = []
    for r in result['records']:
        assert r.get('starter_selection') is None, 'Only actual MC chain is supported'
        landing = r['landing']
        if landing is None or not landing['converged']:
            assert not r['accepted']
            continue
        assert fresh[r['index']]['certified']
        delta = landing['energy'] - current
        ordinary = math.exp(-max(0., delta) / (units.kB * temperature))
        native_fixed = math.exp(-max(0., delta) * 96485. / (20. * 8.314 * temperature))
        telemetry = r.get('mc_telemetry')
        if telemetry is not None:
            assert abs(telemetry['delta_energy_eV'] - delta) < 1e-8
            assert telemetry['temperature_increment_K'] == 0
            assert math.isclose(telemetry['acceptance_probability'], native_fixed, rel_tol=1e-10, abs_tol=1e-300)
            assert r['accepted'] == (telemetry['uniform'] <= native_fixed or delta <= 0)
        classification = {}
        if case_index == 0:
            cg = bond_graph(current_atoms, case['graph_cutoffs_A'])
            lg = bond_graph(landing['atoms'], case['graph_cutoffs_A'])
            classification = dict(connected=nx.is_connected(lg), graph_changed=not nx.is_isomorphic(cg, lg, node_match=lambda a,b: a['Z']==b['Z']))
        rows.append(dict(**classification, index=r['index'],delta_eV=delta,accepted=r['accepted'],ordinary_probability=ordinary,native_fixed_probability=native_fixed))
        if r['accepted']:
            current = landing['energy']
            current_atoms = landing['atoms']
    assert np.allclose(current_atoms['positions'], result['current']['positions'], rtol=0, atol=1e-8)
    uphill = [r for r in rows if r['delta_eV'] > 0]
    groups = {}
    if case_index == 0:
        for group in ['connected_changed', 'connected_same', 'disconnected']:
            selected = [r for r in rows if ('disconnected' if not r['connected'] else 'connected_changed' if r['graph_changed'] else 'connected_same') == group]
            groups[group] = dict(count=len(selected), actual_accepts=sum(r['accepted'] for r in selected), ordinary_conditional_sum=sum(r['ordinary_probability'] for r in selected), native_conditional_sum=sum(r['native_fixed_probability'] for r in selected))
    results.append(dict(groups=groups, case=case['name'],source=str(path.relative_to(ROOT)),temperature_K=temperature,
        actual_policy='native_no_heating' if case['native_mc'] else 'ordinary',
        search_requests=raw['search_requests'],outer_attempts=len(result['records']),qualified_landings=len(rows),
        missing_or_failed_landings=len(result['records'])-len(rows),actual_accepts=sum(r['accepted'] for r in rows),
        uphill=len(uphill),downhill_or_equal=len(rows)-len(uphill),
        uphill_delta_quantiles_eV=np.quantile([r['delta_eV'] for r in uphill],[0,.25,.5,.75,1]).tolist(),
        ordinary_expected_uphill_accepts_on_fixed_pairs=sum(r['ordinary_probability'] for r in uphill),
        native_expected_uphill_accepts_on_fixed_pairs=sum(r['native_fixed_probability'] for r in uphill),rows=rows))
output=dict(scope='Conditional decision audit on fixed observed (current, candidate) pairs; no alternative trajectory, success rate, or efficiency prediction.',pes_requests=0,cases=results)
(OUT / 'analysis.json').write_text(json.dumps(output,indent=2)+'\n')
for result in results:
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
