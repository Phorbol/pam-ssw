"""Paired graph and geometry readout, no potential evaluations."""
import json
import sys
from pathlib import Path
import importlib.util
import numpy as np
import networkx as nx
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'research/ga_ssw')]
from analyze_c4h6_ls_reaction_coverage import atoms_from_dict,graph
spec=importlib.util.spec_from_file_location('torsion_helpers',HERE.parent/'c4h6-torsion-audit-20260924/analyze.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
nm=nx.algorithms.isomorphism.categorical_node_match('number',None)

def compare(a,b):
    ga,gb=graph(a),graph(b)
    matches=nx.algorithms.isomorphism.GraphMatcher(ga,gb,node_match=nm)
    best=None;count=0
    x=a.positions-a.positions.mean(axis=0)
    for mapping in matches.isomorphisms_iter():
        y=b.positions[[mapping[i] for i in range(len(a))]];y=y-y.mean(axis=0)
        u,_,vt=np.linalg.svd(x.T@y);sign=np.linalg.det(u@vt)
        rotation=u@np.diag([1.,1.,sign])@vt
        distances=np.linalg.norm(x@rotation-y,axis=1)
        rms=float(np.sqrt(np.mean(distances**2)));count+=1
        if best is None or rms<best['rms_A']:best=dict(rms_A=rms,max_atom_A=float(distances.max()))
    return dict(graph_isomorphic=count>0,graph_permutations=count,aligned_geometry=best,
                components_a=nx.number_connected_components(ga),components_b=nx.number_connected_components(gb))

def torsion(a):
    try:return helper.torsion(a)[1:]
    except (ValueError,ZeroDivisionError):return None

manifest=json.loads((HERE/'preflight.json').read_text());rows=[]
# Check the geometry comparison using a real saved structure under a legal reordering and rigid transform.
first=json.loads((HERE/'inputs'/f"{manifest['inputs'][0]['case_id']}.json").read_text())
a=atoms_from_dict(first['step_start_atoms']);b=a[::-1];b.positions=b.positions@np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])+[2.,-3.,1.]
invariance=compare(a,b);assert invariance['graph_isomorphic'] and invariance['aligned_geometry']['rms_A']<1e-10
for item in manifest['inputs']:
    cid=item['case_id'];path=HERE/'runs'/cid/'result.json'
    if not path.exists():rows.append(dict(case_id=cid,status='missing'));continue
    d=json.loads(path.read_text());r=d['result'];inp=d['input'];q=r['prequench_only_quench']
    row=dict(case_id=cid,status=r['status'],requests=r['total_requests'],fresh=r['fresh_checks'])
    if 'terminal_atoms' in q:
        start=atoms_from_dict(inp['step_start_atoms']);pre=atoms_from_dict(q['terminal_atoms']);full=atoms_from_dict(inp['ssw_landing_atoms'])
        row.update(pre_vs_start=compare(pre,start),full_vs_start=compare(full,start),pre_vs_full=compare(pre,full),
                   torsion_degrees_cos=dict(start=torsion(start),pre=torsion(pre),full=torsion(full)),
                   pre_minus_full_eV=q['energy_eV']-inp['landing_reported']['energy_eV'],
                   original_full_requests=inp['evaluation_requests'],
                   original_preparation_requests=inp['ls_preparation']['evaluation_requests'],
                   new_pre_only_quench_requests=r['quench_requests'])
    rows.append(row)
result=dict(rows=rows,invariance_check=invariance,requests=sum(r.get('requests',0) for r in rows),
    scope='12 fixed saved states; geometric similarity and force qualification, not certified basin identity or global-search ranking')
assert result['requests']<=12024
(HERE/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
for r in rows:
    print(r['case_id'],r['status'],r.get('pre_vs_start'),r.get('full_vs_start'),r.get('pre_vs_full'),r.get('pre_minus_full_eV'))
