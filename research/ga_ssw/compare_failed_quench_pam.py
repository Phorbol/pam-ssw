"""Same frozen 31 Cu13 stages through existing PAM numerical optimizers only.
No PAM walker is invoked. All methods retain the complete E+B objective.
"""
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.calculators.emt import EMT
from pamssw.relax import Relaxer, RelaxEvaluation
from pamssw.state import State
from pamssw.standalone.gaussian import ProjectedGaussian
from .compare_failed_quench_linesearch import CappedSurface, RequestLimit


def main():
    base=Path('research/ga_ssw/evidence/cu13-direction-only')
    out=Path('research/ga_ssw/evidence/cu13-failed-quench-pam');out.mkdir(exist_ok=False)
    (out/'script.py').write_text(Path(__file__).read_text())
    (out/'pam-relax-source.py').write_text(Path('pamssw/relax.py').read_text())
    (out/'plan.json').write_text(json.dumps(dict(denominator=31,methods=['safe-lbfgs-total','bias-separated-lbfgs'],
        selection='all 31 previously failed biased quenches, original starts, frozen Gaussian history',
        max_requests=201,maxiter=200,fmax=.01,limits='conditional subproblem comparison, no full SSW or LS; no tuning or native parity'),indent=2))
    rows=[]
    for source in sorted(base.glob('[0-9]*-*.json')):
        data=json.loads(source.read_text())
        for record in data['result']['records']:
            if record['status']!='biased_quench_failed':continue
            g=record['climb'][-1];a=Atoms(**record['last_atoms']);a.positions=np.array(g['center'])+g['width']*np.array(g['direction'])
            state=State(a.numbers,a.positions)
            terms=[ProjectedGaussian(np.array(x['center']),np.array(x['direction']),x['width'],x['weight']) for x in record['climb']]
            for method in ['safe-lbfgs-total','bias-separated-lbfgs']:
                surface=CappedSurface(EMT());evaluations=[];trace=[]
                def parts(flat,template):
                    candidate=Atoms(numbers=template.numbers,positions=np.asarray(flat).reshape(-1,3))
                    e,f=surface.evaluate(candidate);be=0.;bf=np.zeros_like(f)
                    for term in terms:
                        de,df=term.evaluate(candidate);be+=de;bf+=df
                    part=RelaxEvaluation(e,-f.ravel(),be,-bf.ravel(),0.,np.zeros(f.size),e+be,-(f+bf).ravel())
                    evaluations.append(dict(positions=candidate.positions.tolist(),energy=e+be,max_force=float(np.linalg.norm(f+bf,axis=1).max()),requests=surface.requests))
                    return part
                def total(flat,template):
                    p=parts(flat,template);return p.total_energy,p.total_gradient
                def observe(current):
                    if not evaluations or not np.array_equal(current.positions,np.array(evaluations[-1]['positions'])):
                        raise RuntimeError('accepted-point reporting cannot trigger extra physical calls')
                    trace.append(evaluations[-1].copy())
                error=None;telemetry=None
                try:
                    result=Relaxer(total,optimizer=method,component_evaluator=parts).relax(state,fmax=.01,maxiter=200,trajectory_callback=observe)
                    status=result.telemetry.termination_reason
                    telemetry=asdict(result.telemetry)
                except RequestLimit as exc:status='request_limit';error=str(exc)
                row=dict(source=source.name,step=record['index'],optimizer=method,status=status,error=error,requests=surface.requests,
                    last_accepted_force=trace[-1]['max_force'] if trace else None,accepted_iterates=len(trace),telemetry=telemetry)
                (out/f'{source.stem}-step{record["index"]}-{method}.json').write_text(json.dumps(dict(**row,accepted_trace=trace,evaluations=evaluations),indent=2)+'\n')
                rows.append(row)
    result=dict(runs=rows,by_optimizer={})
    for method in ['safe-lbfgs-total','bias-separated-lbfgs']:
        selected=[r for r in rows if r['optimizer']==method]
        result['by_optimizer'][method]=dict(attempts=len(selected),converged=sum(r['status']=='converged' for r in selected),
            requests=sum(r['requests'] for r in selected),statuses={s:sum(r['status']==s for r in selected) for s in sorted(set(r['status'] for r in selected))})
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['by_optimizer'],indent=2))
if __name__=='__main__':main()
