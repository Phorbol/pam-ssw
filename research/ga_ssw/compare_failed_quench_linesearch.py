"""Replay every failed Cu13 biased quench with equal physical E/F request caps.

Frozen Gaussians and original STARTS, not recovery from failed endpoints.
No outcome changes the original 31/60 failure denominator.
"""
import inspect,json
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.calculators.emt import EMT
from ase.optimize import LBFGS,LBFGSLineSearch
from pamssw.standalone import ASESurface
from pamssw.standalone.surface import SurfaceCalculator
from pamssw.standalone.gaussian import ProjectedGaussian


class RequestLimit(RuntimeError):pass
class CappedSurface(ASESurface):
    def evaluate(self,atoms):
        if self.requests>=201:raise RequestLimit('201 E/F requests exhausted')
        return super().evaluate(atoms)


def main():
    base=Path('research/ga_ssw/evidence/cu13-direction-only')
    out=Path('research/ga_ssw/evidence/cu13-failed-quench-linesearch');out.mkdir(exist_ok=False)
    (out/'script.py').write_text(Path(__file__).read_text())
    (out/'plan.json').write_text(json.dumps(dict(denominator=31,selection='all biased_quench_failed from direction-only experiment',
        requests_per_quench=201,fmax=.01,optimizers=['LBFGS','LBFGSLineSearch'],
        starts='original last Gaussian center + width * direction',
        scope='frozen modified-surface replay; not new complete SSW campaign; original failures remain failures'),indent=2))
    rows=[]
    for source in sorted(base.glob('[0-9]*-*.json')):
        data=json.loads(source.read_text())
        for record in data['result']['records']:
            if record['status']!='biased_quench_failed':continue
            g=record['climb'][-1];initial=Atoms(**record['last_atoms'])
            initial.positions=np.array(g['center'])+g['width']*np.array(g['direction'])
            terms=[ProjectedGaussian(np.array(x['center']),np.array(x['direction']),x['width'],x['weight']) for x in record['climb']]
            for method in [LBFGS,LBFGSLineSearch]:
                surface=CappedSurface(EMT());a=initial.copy();a.calc=SurfaceCalculator(surface,terms=terms)
                opt=method(a,logfile=None);trace=[]
                def observe():
                    f=a.get_forces();e=a.get_potential_energy();x=a.positions.copy()
                    trace.append(dict(positions=x.tolist(),energy=e,max_force=float(np.linalg.norm(f,axis=1).max()),
                        negative_curvature_pairs=int(np.sum(np.array(opt.state.rho)<0)),requests=surface.requests,
                        forces=f.tolist()))
                opt.attach(observe,interval=1);error=None
                try:
                    opt.run(fmax=.01,steps=200)
                    status='converged' if trace[-1]['max_force']<=.01 else 'step_limit'
                except RequestLimit as exc:status='request_limit';error=str(exc)
                except (RuntimeError,ValueError,FloatingPointError) as exc:status='optimizer_error';error=str(exc)
                ascents=[]
                for previous,current in zip(trace,trace[1:]):
                    displacement=np.array(current['positions'])-np.array(previous['positions'])
                    ascents.append(float(np.sum(-np.array(previous['forces'])*displacement)))
                row=dict(source=source.name,step=record['index'],optimizer=method.__name__,status=status,error=error,
                    requests=surface.requests,accepted_iterates=len(trace),last_accepted_force=trace[-1]['max_force'] if trace else None,
                    uphill_direction_steps=sum(v>0 for v in ascents),max_negative_pairs=max((t['negative_curvature_pairs'] for t in trace),default=0),
                    original_endpoint_difference=float(np.max(abs(a.positions-np.array(record['last_atoms']['positions'])))) if method is LBFGS else None,
                    trace=trace)
                (out/f"{source.stem}-step{record['index']}-{method.__name__}.json").write_text(json.dumps(row,indent=2)+'\n')
                rows.append({k:v for k,v in row.items() if k!='trace'})
    summary=dict(runs=rows,by_optimizer={})
    for name in ['LBFGS','LBFGSLineSearch']:
        selected=[r for r in rows if r['optimizer']==name]
        summary['by_optimizer'][name]=dict(attempts=len(selected),converged=sum(r['status']=='converged' for r in selected),
            requests=sum(r['requests'] for r in selected),with_negative_pairs=sum(r['max_negative_pairs']>0 for r in selected),
            with_uphill_steps=sum(r['uphill_direction_steps']>0 for r in selected),statuses={s:sum(r['status']==s for r in selected) for s in sorted(set(r['status'] for r in selected))})
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary['by_optimizer'],indent=2))
if __name__=='__main__':main()
