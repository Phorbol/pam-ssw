"""Derived, non-search analysis for the random-C60 development runs."""
import argparse,json,re
from collections import Counter
from pathlib import Path
import networkx as nx
import numpy as np
from ase.io import read

EREF=-514.5727735075726; ETOL=.01; FMAX=.03
def graph_row(numbers,positions,cutoff,reference_graph=None):
    x=np.asarray(positions,float); g=nx.Graph(); g.add_nodes_from(range(len(x)))
    d=np.linalg.norm(x[:,None]-x[None,:],axis=2)
    g.add_edges_from(zip(*np.where(np.triu((d<cutoff)&(d>0),1))))
    planar,emb=nx.check_planarity(g); faces=[]
    if planar:
        seen=set()
        for u,v in emb.edges():
            if (u,v) not in seen: faces.append(emb.traverse_face(u,v,seen))
    degrees=dict(g.degree()); deg=Counter(degrees.values()); three_connected=bool(len(g) >= 3 and nx.node_connectivity(g) >= 3) if nx.is_connected(g) else False
    cage=bool(len(x)==60 and all(z==6 for z in numbers) and nx.is_connected(g) and len(g.edges())==90 and all(v==3 for v in degrees.values()) and three_connected and planar and Counter(map(len,faces))==Counter({5:12,6:20}))
    ih=bool(reference_graph is not None and nx.is_isomorphic(g,reference_graph))
    return dict(cutoff_A=cutoff,all_carbon=bool(len(x)==60 and all(z==6 for z in numbers)),components=nx.number_connected_components(g),edges=g.number_of_edges(),degree_counts=dict(deg),three_connected=three_connected,planar=planar,face_counts=dict(Counter(map(len,faces))),graph_cage_candidate=cage,ih_graph_match=ih)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--python',type=Path,required=True); ap.add_argument('--native',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--corrected',type=Path); a=ap.parse_args()
    if not a.python.is_dir() or not a.native.is_dir(): raise FileNotFoundError('both completed evidence directories required')
    summaries=sorted(a.python.glob('*/summary.json'))
    if len(summaries)!=4: raise RuntimeError(f'expected 4 completed runs, found {len(summaries)}')
    native_cases=sorted(p for p in a.native.iterdir() if p.is_dir() and p.name.startswith('seed'))
    if {p.name for p in native_cases} != {'seed17093','seed17094'}: raise RuntimeError('native two-case outputs are incomplete')
    nsummary=json.loads((a.native/'summary.json').read_text()) if (a.native/'summary.json').exists() else []
    if len(nsummary)!=2:
        nsummary=[]
        for p in native_cases:
            proc=json.loads((p/'process.json').read_text()) if (p/'process.json').exists() else {}
            if not proc: raise RuntimeError('native case process status is incomplete')
            nsummary.append({'case':p.name,'process':proc,'requests':sum(1 for line in (a.native/'requests.jsonl').read_text().splitlines() if json.loads(line).get('case')==p.name),'ssw_done':'SSW all done' in ((p/'lasp.out').read_text(errors='replace') if (p/'lasp.out').exists() else '')})
    if len(nsummary)!=2: raise RuntimeError('native two-case status is incomplete')
    ref=[read(a.native.parent/'c60-reference-qualification-v2-20260917'/'final.extxyz')]
    dref=np.linalg.norm(ref[0].positions[:,None]-ref[0].positions[None,:],axis=2); reference_graph=nx.Graph(); reference_graph.add_nodes_from(range(60)); reference_graph.add_edges_from(zip(*np.where(np.triu((dref<1.8)&(dref>0),1))))
    if a.corrected is not None:
        extra=sorted(a.corrected.glob('*/summary.json'))
        if len(extra)!=2: raise RuntimeError('expected two completed corrected runs')
        summaries+=extra
    rows=[]
    for p in summaries:
        s=json.loads(p.read_text()); result=json.loads((p.parent/'result.json').read_text()) if (p.parent/'result.json').exists() else {}
        checks=json.loads((p.parent/'qualification.json').read_text()) if (p.parent/'qualification.json').exists() else []
        mins=result.get('minima',[]); candidates=[]
        for i,m in enumerate(mins[:101]):
            atoms=m.get('atoms',{}); nums=atoms.get('numbers',[]); pos=atoms.get('positions',[])
            if len(nums)==60:
                q=checks[i] if i < len(checks) else {}
                candidates.append(dict(index=i,energy=m.get('energy'),fmax=q.get('fmax'),public_qualified=bool(q.get('qualified') and q.get('composition_match') and q.get('fixed_cell')),target=bool(q.get('qualified') and q.get('composition_match') and q.get('fixed_cell') and q.get('energy',1e99)<=EREF+ETOL),graph_1p8=graph_row(nums,pos,1.8,reference_graph),graph_1p64=graph_row(nums,pos,1.64,reference_graph),graph_1p7=graph_row(nums,pos,1.7,reference_graph)))
        rows.append(dict(run=p.parent.name+('-per_atom' if a.corrected is not None and p.is_relative_to(a.corrected) else ''),summary=s,qualification=checks,candidates=candidates))
    out={'scope':'development analysis only; no winner or formal success-rate claim','reference':{'energy_eV':EREF,'energy_operational_tolerance_eV':ETOL,'calibration_note':'reference .01->.001 changes by .000103 eV'},'python':rows,'native':{}}
    native_rows=[]
    all_requests=[json.loads(line) for line in (a.native/'requests.jsonl').read_text().splitlines()]
    for case_dir in native_cases:
        records=[r for r in all_requests if r.get('case')==case_dir.name and r.get('response',{}).get('ok')]
        txt=(case_dir/'lasp.out').read_text(errors='replace') if (case_dir/'lasp.out').exists() else ''
        local=[]; cumulative=0
        for m in re.finditer(r'Minimum found\s+(\d+)\s+(\d+)\s+([-+0-9.]+)\s+([-+0-9.]+).*?\s(F|T)\s+[-+0-9.]+\s+([-+0-9.]+).*?\s(\d+)\s*$',txt,re.M):
            delta=int(m.group(7)); cumulative+=delta; matches=[records[cumulative-1]] if 0 < cumulative <= len(records) else []
            row={'ordinal':int(m.group(1)),'event_energy_eV':float(m.group(4)),'event_force_component':float(m.group(6)),'cumulative_move_calls':cumulative,'matched_requests':len(matches)}
            if len(matches)==1:
                r=matches[0]; comp=float(np.max(np.abs(np.asarray(r['forces'])))); row.update(request_energy_eV=r['energy'],request_force_component=comp,energy_match=abs(r['energy']-row['event_energy_eV'])<=5.1e-7,force_match=abs(comp-row['event_force_component'])<=.00051,matched=abs(r['energy']-row['event_energy_eV'])<=5.1e-7 and abs(comp-row['event_force_component'])<=.00051)
                if row['matched']:
                    fm=float(np.linalg.norm(np.asarray(r['forces']),axis=1).max())
                    row.update(fmax=fm,public_qualified=bool(np.isfinite(fm) and fm<=FMAX),
                               target=bool(np.isfinite(fm) and fm<=FMAX and r['energy']<=EREF+ETOL),
                               graph_1p8=graph_row([6]*60,r['positions'],1.8,reference_graph),
                               graph_1p64=graph_row([6]*60,r['positions'],1.64,reference_graph),
                               graph_1p7=graph_row([6]*60,r['positions'],1.7,reference_graph))
            else: row['matched']=False
            local.append(row)
        native_rows.append(dict(case=case_dir.name,successful_callbacks=len(records),
                                minimum_events=local,
                                unmatched_events=sum(not r['matched'] for r in local)))
    out['native'].update({'summary':nsummary,'cases':native_rows,'frame_semantics':'allfor.arc contains local-optimization intermediate frames; only lasp.out Minimum found events are native minimum records. Event coordinates are matched only by cumulative request number and E/component-force tolerances; unmatched events remain unmatched.','reference_positive_candidate':graph_row(ref[0].numbers,ref[0].positions,1.8,reference_graph),'reference_sensitivity_1p64':graph_row(ref[0].numbers,ref[0].positions,1.64,reference_graph),'reference_sensitivity_1p7':graph_row(ref[0].numbers,ref[0].positions,1.7,reference_graph)})
    # Explicit positive/negative checks validate analysis, not search efficacy.
    positive=out['native']['reference_positive_candidate']
    escape_root=a.native.parent/'lasp-external-mace-escape-20260917'
    broken=json.loads((escape_root/'requests.jsonl').read_text().splitlines()[-1])
    negative=graph_row([6]*60,broken['positions'],1.8,reference_graph)
    assert positive['graph_cage_candidate'] and positive['ih_graph_match']
    assert not negative['graph_cage_candidate'] and not negative['ih_graph_match']
    out['validator_checks']=dict(positive_reference=positive,negative_broken_cage=negative)
    out['run_summary']=[]
    for row in rows:
        candidates=row['candidates'];eligible=[r for r in candidates if r['public_qualified']]
        out['run_summary'].append(dict(run=row['run'],search_requests=row['summary']['search_requests'],
            fresh_requests=row['summary']['fresh_requests'],status=row['summary']['execution'],
            boundary=row['summary']['boundary'],qualified_minima=len(eligible),
            best_energy=min((r['energy'] for r in eligible),default=None),
            energy_success=any(r['target'] for r in eligible),
            graph_cage_candidate=any(r['graph_1p8']['graph_cage_candidate'] for r in eligible)))
    for row in native_rows:
        eligible=[r for r in row['minimum_events'] if r.get('public_qualified')]
        out['run_summary'].append(dict(run=row['case']+'-native',search_requests=row['successful_callbacks'],
            fresh_requests=0,qualified_minima=len(eligible),unmatched_events=row['unmatched_events'],
            best_energy=min((r['request_energy_eV'] for r in eligible),default=None),
            energy_success=any(r['target'] for r in eligible),
            graph_cage_candidate=any(r['graph_1p8']['graph_cage_candidate'] for r in eligible)))
    a.output.mkdir(parents=False,exist_ok=False); (a.output/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__': main()
