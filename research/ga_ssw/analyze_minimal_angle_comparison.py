"""Zero-PES fixed-denominator analysis of the frozen eight-run height comparison."""
from pathlib import Path
import json,math
from collections import Counter
import numpy as np
import networkx as nx
from ase import Atoms
from pamssw.standalone.native_ls import HC_BOND_LENGTHS

BASE=Path('research/ga_ssw/evidence/conservative-native-height-two-system')
NEWBASE=Path('research/ga_ssw/evidence/minimal-angle-height-two-system')
REF=Path('research/ga_ssw/evidence/cu13-safe-total/strict-validation')

def fingerprint(a):return np.sort(a.get_all_distances()[np.triu_indices(len(a),1)])
def graph(a):
 g=nx.Graph();g.add_nodes_from((i,dict(Z=int(z))) for i,z in enumerate(a.numbers));edges=[];margin=[]
 for i in range(len(a)):
  for j in range(i):
   pair=tuple(sorted((int(a.numbers[i]),int(a.numbers[j]))));cutoff=HC_BOND_LENGTHS[pair]+.1;distance=float(np.linalg.norm(a.positions[i]-a.positions[j]));margin.append(abs(distance-cutoff))
   if distance<cutoff:g.add_edge(i,j);edges.append([j,i])
 return g,dict(edges=sorted(edges),component_formulas=sorted(a[list(c)].get_chemical_formula() for c in nx.connected_components(g)),minimum_cutoff_margin_A=min(margin))

def main():
 summary=json.loads((BASE/'summary.json').read_text());plan=json.loads((BASE/'plan.json').read_text())
 expected=[(system,seed,arm) for system in ('c4h6','cu13') for seed in (3,17) for arm in ('forward_force','native_conservative','minimal_angle')]
 newsummary=json.loads((NEWBASE/'summary.json').read_text())
 actual={(r['system'],r['seed'],r['arm']):r for r in summary['runs']+newsummary['runs']}
 if len(actual)!=12 or any(k not in actual or actual[k]['status']=='running' for k in expected):raise RuntimeError('all twelve terminal runs required; never publish partial denominator')
 refs={};refplan=json.loads((REF/'plan.json').read_text());refsummary=json.loads((REF/'summary.json').read_text())
 for run in refsummary['runs']:
  source=REF/run['source']
  for row in json.loads(source.read_text()):
   if row['qualified'] and row['fingerprint_group'] not in refs:
    refs[row['fingerprint_group']]=dict(fp=fingerprint(Atoms('Cu13',positions=row['positions'])),source=str(source),source_index=row['index'],energy=row['energy'])
 graphs=[];rows=[]
 for key in expected:
  raw=actual[key];name=f'{key[0]}-{key[2]}-seed{key[1]}';directory=(NEWBASE if key[2]=='minimal_angle' else BASE)/name
  result=json.loads((directory/'result.json').read_text());calls=[json.loads(s) for s in (directory/'calls.jsonl').read_text().splitlines()]
  freshcalls=[c for c in calls if c['fresh']];searchcalls=[c for c in calls if not c['fresh']]
  assert len(calls)==raw['total_requests'] and len(searchcalls)==result['evaluation_requests']==raw['search_requests']
  assert [c['request'] for c in calls]==list(range(1,len(calls)+1))
  initcost=result['evaluation_requests']-sum(s['evaluation_requests'] for s in result['records'])
  seq=[dict(role='initial',accepted=None,request=initcost)];cumulative=initcost
  for step in result['records']:
   cumulative+=step['evaluation_requests']
   if step['landing'] is not None and step['landing']['converged']:seq.append(dict(role='landing',accepted=step['accepted'],outer_step=step['index'],request=cumulative))
  assert len(seq)==len(raw['checks'])==len(result['minima'])==len(freshcalls)
  checks=[]
  for i,(check,cost,freshcall) in enumerate(zip(raw['checks'],seq,freshcalls)):
   a=Atoms(**check['atoms']);entry=dict(index=i,**cost,fresh_certificate_request=freshcall['request'],energy=check['energy'],energy_error=check['energy_error'],fmax=check['fmax'],fresh_force_pass=check['fmax']<=plan['configs'][key[0]]['fmax'],components=check['components'])
   if key[0]=='c4h6':
    g,geometry=graph(a);group=None
    for j,ref in enumerate(graphs):
     if nx.is_isomorphic(g,ref,node_match=nx.algorithms.isomorphism.categorical_node_match('Z',0)):group=j;break
    if group is None:group=len(graphs);graphs.append(g)
    carbons=[j for j,z in enumerate(a.numbers) if z==6]
    if i==0:
     cg=g.subgraph(carbons);ends=sorted(n for n,d in cg.degree if d==1)
     carbon_order=nx.shortest_path(cg,ends[0],ends[-1]) if len(ends)==2 else carbons
    dihedral=float(a.get_dihedral(*carbon_order)) if len(carbon_order)==4 else None
    entry.update(geometry,graph_group=group,carbon_order=carbon_order,initial_chain_dihedral_degrees=dihedral,initial_carbon_path_still_bonded=all(g.has_edge(u,v) for u,v in zip(carbon_order,carbon_order[1:])))
   else:
    fp=fingerprint(a);dist={k:float(np.max(abs(fp-v['fp']))) for k,v in refs.items()};nearest=min(dist,key=dist.get);tol=refplan['fingerprint_max_abs_tolerance_A'];matches=[k for k,d in dist.items() if d<=tol]
    entry.update(nearest_existing_strict_group=nearest,nearest_strict_fingerprint_max_abs_A=dist[nearest],strict_tolerance_matches=matches,fingerprint_range_A=[float(fp[0]),float(fp[-1])],all_strict_fingerprint_distances_A=dist)
   checks.append(entry)
  heights=[s for step in result['records'] for s in step['climb'] if 'weight' in s]
  prepared=[s['height_preparation'] for s in heights if 'height_preparation' in s]
  rows.append(dict(name=name,system=key[0],seed=key[1],arm=key[2],status=raw['status'],censored=raw.get('censored',False),total_requests=len(calls),search_requests=len(searchcalls),fresh_requests=len(freshcalls),initial_requests=initcost,seconds=raw['seconds'],step_statuses=raw['steps'],step_requests=[s['evaluation_requests'] for s in result['records']],accepted=raw['accepted'],observations=checks,weights=[s['weight'] for s in heights],height_stop_reasons=dict(Counter(p.get('stop_reason',p.get('status')) for p in prepared)),height_growth_updates=sum(len(p.get('update_trace',())) for p in prepared),height_angles_degrees=[p['angle_degrees'] for p in prepared],history_rewrite_count=sum(len(p.get('changed_history',())) for p in prepared),min_observed_energy=min(c['energy'] for c in checks)))
 aggregates=[]
 for system in ('c4h6','cu13'):
  for arm in ('forward_force','native_conservative','minimal_angle'):
   local=[r for r in rows if r['system']==system and r['arm']==arm];obs=[c for r in local for c in r['observations']]
   aggregates.append(dict(system=system,arm=arm,runs=len(local),total_requests=sum(r['total_requests'] for r in local),initial_requests=sum(r['initial_requests'] for r in local),fresh_requests=sum(r['fresh_requests'] for r in local),seconds=sum(r['seconds'] for r in local),outer_attempts=sum(len(r['step_statuses']) for r in local),valid_landings=sum(c['role']=='landing' for c in obs),accepted_landings=sum(c['accepted'] is True for c in obs),fresh_checks=len(obs),fresh_pass=sum(c['fresh_force_pass'] for c in obs),fragmented=sum(c['components']>1 for c in obs),lowest_observed_energy=min(c['energy'] for c in obs),height_stop_reasons=dict(sum((Counter(r['height_stop_reasons']) for r in local),Counter())),height_growth_updates=sum(r['height_growth_updates'] for r in local),history_rewrite_count=sum(r['history_rewrite_count'] for r in local)))
 report=dict(denominator=12,new_run_denominator=4,original_baseline_requests=summary["total_requests"],new_experiment_requests=newsummary["total_requests"],terminal_runs=len(rows),analysis_PES_requests=0,total_experiment_requests=sum(r['total_requests'] for r in rows),reference_tolerance_A=refplan['fingerprint_max_abs_tolerance_A'],cu13_strict_reference_source=str(REF),cu13_reference_count=len(refs),cu13_reference_records={str(k):{kk:vv for kk,vv in v.items() if kk!='fp'} for k,v in refs.items()},molecular_graph_cutoffs_A={str(k):v+.1 for k,v in HC_BOND_LENGTHS.items()},molecular_graph_groups=len(graphs),runs=rows,aggregates=aggregates,limits='Fixed two-outer-step development comparison, unequal realized cost; no new quench or Hessian; Cu strict refs do not transfer qualification to these finite-force snapshots; sorted-distance fingerprint noninjective, no force-qualified-new-minimum/global efficiency claim.')
 qualification_path=Path('research/ga_ssw/evidence/height-cu13-qualification/result.json')
 if qualification_path.exists():
  qualification=json.loads(qualification_path.read_text());assert len(qualification['runs'])==6 and len(qualification['endpoints'])==18
  groups_by_arm={arm:set() for arm in ('forward_force','native_conservative','minimal_angle')}
  for endpoint in qualification['endpoints']:
   runname=Path(endpoint['source']).parent.name
   for row in rows:
    if row['name']==runname:
     row['observations'][endpoint['index']]['posthoc_refinement']=dict(group=endpoint['fingerprint_group'],refined_energy=endpoint['refined_energy'],refined_fmax=endpoint['refined_fmax'],raw_to_refined_fingerprint_max_A=endpoint['raw_to_refined_fingerprint_max_A'],source=str(qualification_path))
     groups_by_arm[row['arm']].add(endpoint['fingerprint_group'])
  report['posthoc_cu13_qualification']=dict(source=str(qualification_path),requests=qualification['requests'],seconds=qualification['seconds'],groups_by_arm={k:sorted(v) for k,v in groups_by_arm.items()},endpoints=18,scope='Qualification belongs to newly refined geometries; raw-to-refined same-basin claim not established; separate group namespace from old14 strict references')
 (NEWBASE/'offline-analysis.json').write_text(json.dumps(report,indent=2)+'\n');(NEWBASE/'offline-analyzer.py').write_text(Path(__file__).read_text());print(json.dumps(aggregates,indent=2))
if __name__=='__main__':main()
