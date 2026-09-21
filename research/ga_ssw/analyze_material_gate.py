"""Offline eight-run material gate analysis: no calculators and no PES calls.

Run using mace_env with PYTHONNOUSERSITE=1 for pymatgen. Partial results retain
all eight planned slots. Matchers/coordination are approximate geometry checks,
not Hessian certificates or efficacy conclusions. Does not mutate run results.
"""
import argparse
import collections
import itertools
import json
import importlib.metadata
from datetime import datetime, timezone
from pathlib import Path
from research.ga_ssw.material_budget_outcomes import classify_material_events
import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.structure_matcher import StructureMatcher

def geometry(evaluation):
    a=Atoms(**evaluation['atoms']);symbols=np.asarray(a.get_chemical_symbols())
    i,j,d,S=neighbor_list('ijdS',a,6.,self_interaction=False)
    out={'formula':a.get_chemical_formula(),'composition':dict(collections.Counter(symbols)),
         'volume_A3':a.get_volume(),'density_g_cm3':float(a.get_masses().sum()/a.get_volume()*1.66053906660),
         'min_periodic_distance_A':float(d.min()),'pair_minima_A':{},'coordination':{},'H_nearest_O':[]}
    for x,y in itertools.combinations_with_replacement(sorted(set(symbols)),2):
        ds=d[(symbols[i]==x)&(symbols[j]==y)]
        assert len(ds), (x,y)
        out['pair_minima_A'][x+'-'+y]=float(ds.min())
    for x,y,cutoffs in [('Ti','O',[2.2,2.3,2.4]),('Al','O',[2.2,2.3,2.4]),('H','O',[1.1,1.2,1.3])]:
        centers=np.flatnonzero(symbols==x)
        if not len(centers):continue
        entry={}
        for c in cutoffs:
            counts=[int(np.sum((i==k)&(symbols[j]==y)&(d<c))) for k in centers]
            entry[str(c)]={'indices_zero_based':centers.tolist(),'counts':counts,'histogram':dict(collections.Counter(counts))}
        out['coordination'][x+'-'+y]=entry
    for k in np.flatnonzero(symbols=='H'):
        mask=np.flatnonzero((i==k)&(symbols[j]=='O'));pick=mask[np.argmin(d[mask])]
        out['H_nearest_O'].append({'H':int(k),'O':int(j[pick]),'shift':S[pick].tolist(),'distance_A':float(d[pick])})
    return a,out


def matching(a, b):
    outputs=[]
    for ltol,stol,angle in [(.1,.15,2),(.2,.3,5),(.3,.5,10)]:
        settings=dict(ltol=ltol,stol=stol,angle_tol=angle,scale=False,
                      primitive_cell=True,attempt_supercell=True)
        try:
            sm=StructureMatcher(**settings)
            match=bool(sm.fit(AseAtomsAdaptor.get_structure(a),
                              AseAtomsAdaptor.get_structure(b),symmetric=True))
            outputs.append(dict(settings=settings,match=match))
        except Exception as exc:
            outputs.append(dict(settings=settings,match=None,error=repr(exc)))
    return outputs


def errors(value):
    if isinstance(value,dict):
        for key,item in value.items():
            if key=='error' and item:yield str(item)
            else:yield from errors(item)
    elif isinstance(value,list):
        for item in value:yield from errors(item)


def analyze(gate):
    manifest=json.loads((gate/'manifest.json').read_text())
    commands=manifest['commands']
    if len(commands)!=8:raise ValueError('This gate requires exactly eight declared runs')
    execution=json.loads((gate/'execution.json').read_text()) if (gate/'execution.json').exists() else {}
    execution_by_name={r['name']:r for r in execution.get('runs',[])}
    rows=[];starts={}
    for command in commands:
        declared=Path(command[command.index('--output')+1]);name=declared.name
        path=gate/'results'/name/'result.json'
        arm=command[command.index('--arm')+1]; seed=int(command[command.index('--seed')+1])
        system=Path(command[command.index('--input')+1]).stem
        row=dict(name=name,system=system,arm=arm,seed=seed,
                 objective_domain='fixed' if arm=='fixed' else 'variable',
                 result_path=str(path),process=execution_by_name.get(name),
                 status='not_available',cost_known=False,landings=[])
        rows.append(row)
        if not path.exists():continue
        try:data=json.loads(path.read_text())
        except (json.JSONDecodeError,OSError) as exc:
            row.update(status='unreadable_or_incomplete_json',read_error=repr(exc));continue
        row.update(status=data['status'],cost_known=True,requests=data['requests'],
                   initial_requests=data['initial_requests'],wall_seconds=data.get('wall_seconds'),
                   requests_reconciled=data['requests_reconciled'],
                   steps_requested=data['steps_requested'],reported_valid_proposals=data['valid_proposals'],
                   domain=data['domain'],objective=data['objective'],pressure=data['pressure'])
        initial=next((p for p in data['landings'] if p['index']==-1),None)
        if initial is not None:
            a,g=geometry(initial);row['common_start_geometry']=g
            row['common_start_energy']=initial['energy']; starts[name]=(system,a)
        else:a=None
        valid_by_index={p['index']:p for p in data['landings'] if p['index']>=0}
        events=[]
        classified=classify_material_events(data)
        for event in data['records']:
            if 'index' not in event:continue
            index=event['index'];err=list(errors(event))
            outcome=classified[index]['outcome']
            events.append(dict(index=index,status=event['status'],outcome=outcome,
                               requests=event['requests'],errors=err,
                               request_interval=classified[index]['request_interval'],
                               budget_ledger_evidence=classified[index]['budget_ledger_evidence'],
                               atomic_scheduled=event.get('atomic_scheduled')))
        row['events']=events;row['outcome_counts']=dict(collections.Counter(e['outcome'] for e in events))
        row['initial_errors']=list(errors(data['records'][0])) if data['records'] else []
        row['not_attempted_proposals']=max(0,data['steps_requested']-len(events))+sum(e['outcome']=='not_started_no_budget' for e in events)
        row['all_errors']=list(errors(data['records']))
        row['cost_recomputed_from_records']=sum(e['requests'] for e in data['records'])
        for index,p in valid_by_index.items():
            entry=dict(index=index,accepted=p['accepted'],energy=p['energy'],objective=p['objective'],
                       certificate=p['certificate'],
                       delta_objective_from_common_start=p['objective']-initial['objective'] if initial else None)
            try:
                b,g=geometry(p);entry['geometry']=g
                entry['composition_matches_common_start']=bool(np.array_equal(a.numbers,b.numbers)) if a is not None else None
                entry['matches_common_start']=matching(a,b) if a is not None else []
                entry['cell_difference_frobenius_A']=float(np.linalg.norm(a.cell.array-b.cell.array)) if a is not None else None
            except Exception as exc:entry['geometry_error']=repr(exc)
            row['landings'].append(entry)
    # Verify preparation equivalence independently; raw input is not the common
    # prepared starting point and fixed/VC final domains remain separate.
    start_comparisons=[]
    for (n1,(s1,a)),(n2,(s2,b)) in itertools.combinations(starts.items(),2):
        if s1==s2:start_comparisons.append(dict(first=n1,second=n2,matching=matching(a,b)))
    present=[r for r in rows if r['cost_known']]
    domains={}
    for domain in ['fixed','variable']:
        selected=[r for r in present if r['objective_domain']==domain]
        counts=collections.Counter()
        for r in selected:counts.update(r['outcome_counts'])
        domains[domain]=dict(planned_runs=sum(r['objective_domain']==domain for r in rows),
            available_runs=len(selected),requests=sum(r['requests'] for r in selected),
            proposal_outcomes=dict(counts))
    return dict(generated_utc=datetime.now(timezone.utc).isoformat(),planned_run_denominator=8,
                available_results=len(present),unavailable_results=8-len(present),
                extra_PES_requests=0,execution_status=execution.get('status'),
                total_recorded_requests=sum(r['requests'] for r in present),
                cost_unknown_run_count=8-len(present),
                recorded_wall_seconds_sum=sum(r.get('wall_seconds') or 0 for r in present),
                elapsed_missing_count=sum(r.get('wall_seconds') is None for r in present),
                packages={p:importlib.metadata.version(p) for p in ['ase','pymatgen','numpy']},
                domains=domains,runs=rows,prepared_start_cross_arm_matching=start_comparisons)


def markdown(report):
    lines=['# Eight-run complex-material gate: offline snapshot','',
           f"Available results: **{report['available_results']}/8**. Missing results: {report['unavailable_results']}/8. Execution status: `{report['execution_status']}`.",'',
           f"Recorded E/F/stress requests: {report['total_recorded_requests']}; {report['cost_unknown_run_count']} run costs remain unknown. Analysis adds 0 PES requests.",'',
           '| Run | Domain | Result status | Valid accepted/rejected | Failed/censored/not started | Requests | Seconds |',
           '|---|---|---|---|---|---:|---:|']
    for r in report['runs']:
        c=r.get('outcome_counts',{})
        lines.append(f"| {r['name']} | {r['objective_domain']} | {r['status']} | {c.get('valid_accepted',0)}/{c.get('valid_rejected',0)} | {c.get('failed',0)}/{c.get('budget_censored',0)}/{c.get('not_started_no_budget',0)} | {r.get('requests','unknown')} | {r.get('wall_seconds','unknown')} |")
    lines+=['','Counts retain rejected valid landings and all failed or budget-censored attempts. Missing rows are not failures or zero-cost successes. Initial preparation is included once per run; raw stage errors and request reconciliation are retained in JSON. Fixed-cell outcomes optimize a different domain and are not pooled with variable-cell outcomes for a success rate.','',
            '## Available landing geometry','',
            '| Run / proposal | Accepted | Δobjective from prepared start, eV | Volume, Å³ | Density, g/cm³ | Matches start (three settings) |',
            '|---|---|---:|---:|---:|---|']
    for r in report['runs']:
        for p in r['landings']:
            g=p.get('geometry',{})
            matches=[v['match'] for v in p.get('matches_common_start',[])]
            lines.append(f"| {r['name']} / {p['index']} | {p['accepted']} | {p['delta_objective_from_common_start']} | {g.get('volume_A3','unknown')} | {g.get('density_g_cm3','unknown')} | {matches} |")
    lines+=['','Geometry JSON contains per-site H–O, Al–O and Ti–O coordination at explicitly stated cutoffs, periodic pair minima, hydrogen partners and volumes. Neighbor radius is 6 Å, counting periodic images. Matching uses scale=False, primitive_cell=True, attempt_supercell=True, symmetric matching, with (ltol,stol,angle_tol)=(.1,.15,2),(.2,.3,5),(.3,.5,10); stol is normalized, not Å. These are diagnostic sensitivity settings, not optimized bond definitions.','',
            'No Hessian or fresh physical evaluations are performed by this script. Stored certificates remain in-run certificates unless the source explicitly records otherwise. Structural non-matching and small force/stress do not establish a stable new phase. One seed per arm/system and a feasibility gate cannot establish search efficiency or generality.']
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gate',type=Path,default=Path(__file__).resolve().parent/'prospective/complex-vc-feasibility')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();gate=args.gate.resolve();out=args.output or gate/'offline-analysis'
    report=analyze(gate);out.mkdir(parents=True,exist_ok=True)
    for name,content in [('analysis.json',json.dumps(report,indent=2)),('report.md',markdown(report))]:
        path=out/name;temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(content);temp.replace(path)
    print(f"Analyzed {report['available_results']}/8 results; {report['total_recorded_requests']} recorded requests; 0 new PES requests. {out}")


if __name__=='__main__':main()
