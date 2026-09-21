"""All paired Cu13 endpoints: independent refinement and internal curvature."""
from pathlib import Path
import json,time,shutil
import numpy as np
from ase import Atoms
from ase.calculators.emt import EMT
from pamssw.standalone import ASESurface
from pamssw.standalone.surface import quench
from pamssw.standalone.cluster_frame import ClusterFrame
from research.ga_ssw.compare_vc_arms import serial
from research.ga_ssw.validate_cu13_eckart import fingerprint

BASE=Path('research/ga_ssw/evidence')
OUT=BASE/'height-cu13-qualification'
PLAN=dict(run_denominator=6,total_EF_cap=4000,seconds=120,threads=1,
    fmax=1e-5,relax_steps=300,fingerprint_tolerance_A=1e-4,hessian_steps_A=[1e-4,5e-5],
    domain='isolated Cu13 EMT internal coordinates, six rigid modes excluded',
    rule='all initial/landings incl rejects, all fresh first, then all refinement, then first-seen fingerprint representative Hessians; never search feedback',
    provenance='settings from validate_cu13_eckart.py; pair fingerprint not injective; before/after changes retained')

def main():
    sources=sorted((BASE/'conservative-native-height-two-system').glob('cu13-*/result.json'))
    sources+=sorted((BASE/'minimal-angle-height-two-system').glob('cu13-*/result.json'))
    if len(sources)!=6:raise ValueError('all six Cu13 runs must be terminal before qualification')
    if any(not (p.parent/'summary.json').exists() for p in sources):raise ValueError('source still running')
    OUT.mkdir(exist_ok=False);shutil.copy2(__file__,OUT/'script.py')
    (OUT/'plan.json').write_text(json.dumps(dict(PLAN,sources=list(map(str,sources))),indent=2))
    shutil.copytree('pamssw',OUT/'source'/'pamssw',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    begin=time.monotonic();report=dict(plan=PLAN,runs=[],endpoints=[],representatives=[],requests=0,status='running')
    class Counted(ASESurface):
        def evaluate(self,a):
            if self.requests>=PLAN['total_EF_cap'] or time.monotonic()-begin>=PLAN['seconds']:raise RuntimeError('declared qualification budget')
            return super().evaluate(a)
    surface=Counted(EMT())
    def save():
        report.update(requests=surface.requests,seconds=time.monotonic()-begin)
        (OUT/'result.json').write_text(json.dumps(serial(report),indent=2,allow_nan=False))
    for path in sources:
        d=json.loads(path.read_text());report['runs'].append(dict(source=str(path),search_status=d['status'],search_requests=d['evaluation_requests'],minima=len(d['minima'])))
        accepted=[None]+[r['accepted'] for r in d['records'] if r.get('landing') and r['landing'].get('converged')]
        if len(accepted)!=len(d['minima']):raise ValueError('landing/acceptance provenance mismatch')
        for index,m in enumerate(d['minima']):
            report['endpoints'].append(dict(source=str(path),index=index,accepted=accepted[index],original=Atoms(**m['atoms']),original_energy=m['energy'],status='pending_fresh'))
    for row in report['endpoints']:
        try:
            surface.calculator=EMT();e,f=surface.evaluate(row['original'])
            row.update(status='fresh_checked',fresh_energy=e,fresh_fmax=float(np.linalg.norm(f,axis=1).max()),energy_error=e-row['original_energy'])
        except Exception as error:row.update(status='fresh_failed_or_pending',error=repr(error))
        save()
    if any(r['status']!='fresh_checked' for r in report['endpoints']):
        report['status']='partial';save();return
    representatives=[];fps=[]
    for row in report['endpoints']:
        before=surface.requests
        try:
            q=quench(row['original'],surface,fmax=PLAN['fmax'],steps=PLAN['relax_steps'],optimizer='safe-lbfgs-total')
            surface.calculator=EMT();e,f=surface.evaluate(q.atoms)
            fmax=float(np.linalg.norm(f,axis=1).max());fp=fingerprint(q.atoms)
            row.update(refinement=q,refined=q.atoms,refined_energy=e,refined_fmax=fmax,
                raw_to_refined_fingerprint_max_A=float(np.max(abs(fingerprint(row['original'])-fp))),
                status='refined' if q.converged and fmax<=PLAN['fmax'] else 'refinement_failed')
            if row['status']=='refined':
                group=next((i for i,old in enumerate(fps) if np.max(abs(old-fp))<=PLAN['fingerprint_tolerance_A']),None)
                if group is None:
                    group=len(fps);fps.append(fp);representatives.append(q.atoms.copy())
                row['fingerprint_group']=group
        except Exception as error:row.update(status='refinement_failed_or_pending',error=repr(error))
        row['refinement_requests']=surface.requests-before;save()
    for group,a in enumerate(representatives):
        item=dict(group=group,atoms=a,status='pending_hessian',spectra=[]);report['representatives'].append(item);before=surface.requests
        try:
            frame=ClusterFrame(a);u,_,_=np.linalg.svd(frame.basis,full_matrices=True);basis=u[:,6:]
            for h in PLAN['hessian_steps_A']:
                spec=dict(h=h,columns=[],status='partial');item['spectra'].append(spec)
                for direction in basis.T:
                    plus=a.copy();minus=a.copy();delta=h*direction.reshape(-1,3)
                    plus.positions+=delta;minus.positions-=delta
                    _,fp=surface.evaluate(plus);_,fm=surface.evaluate(minus)
                    spec['columns'].append(-basis.T@(fp-fm).ravel()/(2*h));save()
                H=np.asarray(spec['columns']).T
                spec.update(status='completed',eigenvalues=np.linalg.eigvalsh((H+H.T)/2),antisymmetry=float(np.linalg.norm(H-H.T)))
            item['status']='completed'
        except Exception as error:item.update(status='partial_or_failed',error=repr(error))
        item['requests']=surface.requests-before;save()
    report['status']='completed' if all(r['status']=='refined' for r in report['endpoints']) and all(r['status']=='completed' for r in report['representatives']) else 'partial'
    save();print(json.dumps(dict(status=report['status'],requests=surface.requests,groups=len(representatives),endpoints=len(report['endpoints']))))

if __name__=='__main__':main()
