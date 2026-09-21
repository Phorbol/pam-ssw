"""Complete saved partial Hessian columns; retain the original timed-out run."""
import json, shutil, signal, time, hashlib
from pathlib import Path
import numpy as np
from ase import Atoms
from pamssw.standalone.vc_geometry import ASEStressSurface, SymmetricLogStrainChart
from research.ga_ssw.xxxii_replicated_calculator import XXXIIReplicatedCalculator

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'research/ga_ssw/evidence/xxxii-replicated-hessian-qualification'
OUT=ROOT/'research/ga_ssw/evidence/xxxii-replicated-hessian-completion'
previous=json.loads((SRC/'result.json').read_text())
assert previous['status']!='running'
OUT.mkdir(exist_ok=False);shutil.copytree(SRC/'matrices',OUT/'matrices')
shutil.copy2(__file__,OUT/'runner-executed.py')
shutil.copytree(SRC/'source',OUT/'source')
for relative in ('pamssw/standalone/vc_geometry.py','research/ga_ssw/xxxii_replicated_calculator.py'):
    assert (OUT/'source'/relative).read_bytes()==(ROOT/relative).read_bytes()
plans=[]
for endpoint in (0,1):
    stages=[]
    for h in (1e-4,5e-5):
        path=OUT/'matrices'/f'endpoint-{endpoint}-h-{h:.0e}.npy'
        if path.exists():
            matrix=np.load(path);complete=np.isfinite(matrix).all(axis=0)
        else:
            complete=np.zeros(519,dtype=bool)
            matrix=np.lib.format.open_memmap(path,mode='w+',dtype=np.float64,shape=(519,519),fortran_order=True)
            matrix[:]=np.nan;matrix.flush()
        stages.append(dict(h=h,completed_columns=np.flatnonzero(complete).tolist(),missing_columns=np.flatnonzero(~complete).tolist()))
    plans.append(dict(endpoint=endpoint,stages=stages,max_new_API=3+2*sum(len(s['missing_columns']) for s in stages)))
plan=dict(source=str(SRC),original_API=previous['total_API'],endpoints=plans,seconds_per_endpoint=180,
    reason='Original90s cap reached at907/943API; measured~0.1s/API. Extend only wall allocation to180s for remaining columns, retaining all prior costs. No geometry, model, finite-difference spacing or mathematical method change.',
    verification='one fresh center and one overlapping central column per endpoint (3API); fill only incomplete columns from original matrices')
(OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
report=dict(status='running',endpoints=[],original_API=previous['total_API'])
for ep in plans:
    index=ep['endpoint'];record=json.loads((SRC/'endpoint-inputs'/f'endpoint-{index}.json').read_text())
    a=Atoms(**record);chart=SymmetricLogStrainChart(a,strain_length=5.)
    center=np.load(OUT/'matrices'/f'endpoint-{index}-center.npy');basis=np.load(OUT/'matrices'/f'endpoint-{index}-basis.npy')
    model=OUT/'source/research/ga_ssw'
    calc=XXXIIReplicatedCalculator(data_path=model/'lmp.data',input_path=model/'in.simple',model_manifest=model/'manifest.json',reference_atoms=a,repetitions=(1,1,2))
    surf=ASEStressSurface(calc);start=time.monotonic();row=dict(endpoint=index,status='running',spectra=[])
    report['endpoints'].append(row)
    def evaluate(atoms):
        if surf.requests>=ep['max_new_API']:raise RuntimeError('declared remaining-column API cap')
        return surf.evaluate(atoms)
    def timeout(*args):raise RuntimeError('declared180s remaining-column wall cap')
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,180)
    try:
        energy,forces,stress=evaluate(a)
        row['fresh']=dict(energy=energy,fmax=float(np.linalg.norm(forces,axis=1).max()),stress_max=float(abs(stress).max()))
        k=ep['stages'][0]['completed_columns'][0];direction=basis[:,k];h=1e-4
        plus=chart.evaluate(center+h*direction,evaluate);minus=chart.evaluate(center-h*direction,evaluate)
        check=basis.T@(plus.gradient-minus.gradient)/(2*h)
        old=np.load(OUT/'matrices'/f'endpoint-{index}-h-{h:.0e}.npy')[:,k]
        row['overlap']=dict(column=k,max_error=float(abs(check-old).max()),norm_error=float(np.linalg.norm(check-old)))
        for stage in ep['stages']:
            h=stage['h'];path=OUT/'matrices'/f'endpoint-{index}-h-{h:.0e}.npy'
            matrix=np.load(path,mmap_mode='r+')
            for k in stage['missing_columns']:
                direction=basis[:,k]
                plus=chart.evaluate(center+h*direction,evaluate);minus=chart.evaluate(center-h*direction,evaluate)
                matrix[:,k]=basis.T@(plus.gradient-minus.gradient)/(2*h);matrix.flush()
                (OUT/f'endpoint-{index}-progress.json').write_text(json.dumps(dict(h=h,last_column=k,completed_columns=int(np.isfinite(matrix).all(axis=0).sum()),new_API=surf.requests,seconds=time.monotonic()-start))+'\n')
            sym=(matrix+matrix.T)/2
            eig=np.linalg.eigvalsh(sym);np.save(path.with_name(path.stem+'-eigenvalues.npy'),eig)
            row['spectra'].append(dict(h=h,min_eigenvalue=float(eig[0]),max_eigenvalue=float(eig[-1]),skew_frobenius=float(np.linalg.norm(matrix-matrix.T))))
        row['status']='completed'
    except Exception as exc:row.update(status='failed',error=repr(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);calc.close()
        row.update(new_API=surf.requests,engine_calls=calc.engine_calls,atoms_evaluated=calc.atoms_evaluated,wall_seconds=time.monotonic()-start)
        (OUT/'result.json').write_text(json.dumps(report,indent=2)+'\n')
report.update(status='completed' if all(r['status']=='completed' for r in report['endpoints']) else 'completed_with_failures',total_API=sum(r['new_API'] for r in report['endpoints']),engine_calls=sum(r['engine_calls'] for r in report['endpoints']),atoms_evaluated=sum(r['atoms_evaluated'] for r in report['endpoints']))
(OUT/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
