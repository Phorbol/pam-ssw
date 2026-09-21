"""Reconcile Python C60 request ledgers with recorded solver-stage telemetry."""
import argparse,json
from pathlib import Path
import numpy as np

PY_CAMPAIGNS=[Path('research/ga_ssw/evidence/c60-random-python-development-20260917'),Path('research/ga_ssw/evidence/c60-random-c1-per-atom-development-20260918')]
def one_run(p):
    result=json.loads((p/'result.json').read_text()); lines=[json.loads(z) for z in (p/'requests.jsonl').read_text().splitlines()]
    qualification=json.loads((p/'qualification.json').read_text()) if (p/'qualification.json').exists() else []
    search=sum(z.get('kind')=='search' for z in lines); denials=sum(z.get('kind')=='search_denial' for z in lines)
    initial=int(result['initial']['evaluation_requests']); records=[]; stage={'initial_quench':initial,'rotation':0,'biased_quench':0,'true_landing':0,'event_residual':0,'record_residual':0}
    for r in result['records']:
        climb=[]
        for e in r.get('climb',[]):
            rot=int(e.get('rotation_force_requests',0)); bias=int(e.get('quench_requests',0)); total=int(e.get('requests',0)); residual=total-rot-bias
            stage['rotation']+=rot; stage['biased_quench']+=bias; stage['event_residual']+=residual
            climb.append({'index':e.get('index'),'status':e.get('status'),'termination_reason':e.get('termination_reason'),'requests':total,'rotation_force_requests':rot,'biased_quench_requests':bias,'residual_requests':residual,'rotation_stop_reason':e.get('rotation_stop_reason'),'rotation_converged':e.get('rotation_converged'),'optimizer_termination':(e.get('optimizer_telemetry') or {}).get('termination_reason')})
        land=int((r.get('landing') or {}).get('evaluation_requests',0)); stage['true_landing']+=land
        accounted=sum(x['requests'] for x in climb)+land; record_res=int(r.get('evaluation_requests',0))-accounted; stage['record_residual']+=record_res
        records.append({'index':r.get('index'),'status':r.get('status'),'accepted':r.get('accepted'),'record_requests':r.get('evaluation_requests'),'climb':climb,'true_landing_requests':land,'record_residual':record_res,'error':r.get('error')})
    accounted=sum(stage.values())
    qualified_landing_requests=[m.get('evaluation_requests') for m,q in zip(result.get('minima',[]),qualification) if q.get('qualified') and m.get('evaluation_requests') is not None]
    gaussian_events=[g for r in records for g in r['climb']]
    return {'source':str(p.resolve()),'campaign':p.parent.name,'run':p.name,'status':result.get('status'),'ledger_search_requests':search,'ledger_denials':denials,'result_evaluation_requests':result.get('evaluation_requests'),'stage':stage,'accounted_total':accounted,'ledger_reconciliation_residual':search-accounted,'records':records,'minima_count':len(result.get('minima',[])),'qualified_landing_requests':qualified_landing_requests,'gaussian_event_count':len(gaussian_events),'gaussian_rotation_requests':[g['rotation_force_requests'] for g in gaussian_events],'gaussian_biased_quench_requests':[g['biased_quench_requests'] for g in gaussian_events],'gaussian_event_residuals':[g['residual_requests'] for g in gaussian_events],'rotation_stop_reasons':[g['rotation_stop_reason'] for g in gaussian_events],'biased_quench_exit_reasons':[g['optimizer_termination'] for g in gaussian_events]}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args(); runs=[]
    for root in PY_CAMPAIGNS:
        if not root.is_dir(): continue
        for p in sorted(root.glob('seed*-*/')):
            if (p/'result.json').exists() and (p/'requests.jsonl').exists(): runs.append(one_run(p))
    if len(runs)!=6: raise RuntimeError(f'expected six Python runs, found {len(runs)}')
    native=Path('research/ga_ssw/evidence/c60-random-native-development-20260917'); native_rows=[]
    if native.is_dir() and (native/'requests.jsonl').exists():
        allreq=[json.loads(z) for z in (native/'requests.jsonl').read_text().splitlines()]
        for p in sorted(native.glob('seed*')):
            if p.is_dir():
                proc=json.loads((p/'process.json').read_text()) if (p/'process.json').exists() else None
                native_rows.append({'case':p.name,'process':proc,'request_records':sum(r.get('case')==p.name for r in allreq),'boundary':'request ledger only; allfor.arc frames excluded from cost'})
    payload={'scope':'Python stage allocation and ledger reconciliation; no efficiency or solver ranking claim','python_runs':runs,'native_boundary':native_rows,'definitions':{'rotation':'event rotation_force_requests','biased_quench':'event quench_requests/optimizer telemetry','true_landing':'SSWStep.landing.evaluation_requests','event_residual':'event requests minus rotation and biased quench','record_residual':'SSWStep evaluation_requests minus climb requests and landing','ledger_denials':'request ledger denials excluded from paid search count'}}
    a.output.mkdir(parents=False,exist_ok=False); (a.output/'analysis.json').write_text(json.dumps(payload,indent=2)+'\n')
if __name__=='__main__': main()
