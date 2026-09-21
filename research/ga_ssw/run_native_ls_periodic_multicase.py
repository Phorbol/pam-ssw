"""Prepare/execute bounded native-LS MIC versus periodic-image EMT arms."""
import argparse, dataclasses, json, os, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT

CAP=4000; WALL=60.; SEED=11

def systems():
    out={}
    for symbol,lattice in (('Cu',3.6),('Al',4.05)):
        conv=bulk(symbol,'fcc',a=lattice,cubic=True); conv.positions[0]+=[.12,-.08,.06]
        out[f'{symbol}4']=conv
        vac=bulk(symbol,'fcc',a=lattice,cubic=True).repeat((2,2,2)); del vac[0]; vac.positions[0]+=[.12,-.08,.06]
        out[f'{symbol}31']=vac
    return out

def config():
    from pamssw.standalone.paper_reference import SSWConfig
    return SSWConfig(width=.1,rotation_bias=100.,max_gaussians=14,temperature_K=150.,fmax=.01,relax_steps=200,fd_step=1e-4,rotation_hvp=100,rotation_tol=.02,direction_sampling='global',rotation_solver='dimer',cluster_frame='translation_only',quench_optimizer='safe-lbfgs-total',bias_fmax=.1)

def settings(symbol,geometry):
    from pamssw.standalone.ls_native_reference import NativeLSSettings
    z={'Cu':29,'Al':13}[symbol]; length={'Cu':2.9,'Al':3.0}[symbol]
    return NativeLSSettings({(z,z):1.},{(z,z):length},scale=5.,target_mev_per_atom=20.,bond_geometry=geometry)

class LedgerSurface:
    def __init__(self,path):
        from pamssw.standalone.surface import ASESurface
        self.surface=ASESurface(EMT()); self.path=Path(path); self.start=time.monotonic(); self.denials=0
        self.path.parent.mkdir(parents=True,exist_ok=True)
    @property
    def requests(self): return self.surface.requests
    def evaluate(self,atoms):
        before=self.surface.requests; reason=None
        if before>=CAP: reason='request_cap'
        elif time.monotonic()-self.start>=WALL: reason='wall_cap'
        payload={'positions':np.asarray(atoms.positions).tolist(),'cell':atoms.cell.array.tolist(),'pbc':atoms.pbc.tolist(),'request':before+1}
        try:
            if reason: raise RuntimeError(reason)
            energy,forces=self.surface.evaluate(atoms); payload.update(kind='paid',energy=float(energy),forces=np.asarray(forces).tolist(),charged=True); return energy,forces
        except Exception as error:
            self.denials+=int(reason is not None); payload.update(kind='failure' if reason is None else 'denial',reason=str(error),charged=bool(self.surface.requests>before)); raise
        finally:
            with self.path.open('a') as handle: handle.write(json.dumps(payload)+'\n')

def serial(value):
    from ase import Atoms
    if isinstance(value,Atoms): return {'numbers':value.numbers.tolist(),'positions':value.positions.tolist(),'cell':value.cell.array.tolist(),'pbc':value.pbc.tolist()}
    if dataclasses.is_dataclass(value): return {f.name:serial(getattr(value,f.name)) for f in dataclasses.fields(value)}
    if isinstance(value,dict): return {str(k):serial(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [serial(v) for v in value]
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    return value if value is None or isinstance(value,(str,int,float,bool)) else repr(value)

def fresh(minimum,source):
    from pamssw.standalone.surface import ASESurface
    surface=ASESurface(EMT()); energy,forces=surface.evaluate(minimum.atoms.copy()); d=minimum.atoms.get_all_distances(mic=True)
    tri=d[np.triu_indices(len(d),1)] if len(d)>1 else np.array([]); mindistance=float(tri.min()) if len(tri) else None
    return {'stored_energy':float(minimum.energy),'fresh_energy':float(energy),'energy_error':abs(float(energy)-float(minimum.energy)),'fresh_fmax':float(np.linalg.norm(forces,axis=1).max()),'min_distance_mic':mindistance,'cell_exact':bool(np.array_equal(minimum.atoms.cell.array,source.cell.array))}

def child(out,names):
    import pamssw
    import ase
    from ase.io import write
    from pamssw.standalone.ls_native_reference import run_native_ls_ssw
    source=Path(out)/'source'; assert str(Path(pamssw.__file__).resolve()).startswith(str(source.resolve()))
    (Path(out)/'runtime-import.json').write_text(json.dumps({'pamssw_file':str(Path(pamssw.__file__).resolve()),'source':str(source.resolve())},indent=2)+'\n')
    allsystems=systems(); report={'seed':SEED,'cap':CAP,'wall_seconds':WALL,'arms':[],'source':str(source)}
    (Path(out)/'config.json').write_text(json.dumps({'config':serial(config()),'seed':SEED,'cap':CAP,'wall_seconds':WALL,'model':'ASE EMT','units':{'energy':'eV','force':'eV/A'},'versions':{'ase':ase.__version__,'numpy':np.__version__}},indent=2)+'\n')
    plan={'systems':{name:serial(allsystems[name]) for name in names},
          'ls_settings':{name:{mode:serial(settings(name[:2],mode)) for mode in ('native-mic','periodic-images')} for name in names},
          'config':serial(config()),'seed':SEED,'cap':CAP,'wall_seconds':WALL}
    (Path(out)/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    for name in names:
        for geometry in ('native-mic','periodic-images'):
            arm=Path(out)/f'{name}-{geometry}'; arm.mkdir(); atoms=allsystems[name].copy(); write(arm/'input.extxyz',atoms,format='extxyz'); surface=LedgerSurface(arm/'ledger.jsonl'); result=None
            try:
                result=run_native_ls_ssw(atoms,surface,steps=2,config=config(),rng=np.random.default_rng(SEED),ls=settings(name[:2],geometry)); fresh_rows=[]
                for minimum in result.minima:
                    try: fresh_rows.append(fresh(minimum,atoms))
                    except Exception as error: fresh_rows.append({'error':f'{type(error).__name__}: {error}'})
                row={'case':name,'bond_geometry':geometry,'status':result.status,'evaluation_requests':result.evaluation_requests,'paid_requests':surface.requests,'denials':surface.denials,'records':serial(result.records),'raw_result':serial(result),'fresh':fresh_rows}
            except Exception as error:
                row={'case':name,'bond_geometry':geometry,'status':'failed','error':f'{type(error).__name__}: {error}','evaluation_requests':getattr(result,'evaluation_requests',None),'paid_requests':surface.requests,'denials':surface.denials,'raw_result':serial(result),'records':serial(getattr(result,'records',()))}
            (arm/'result.json').write_text(json.dumps(row,indent=2)+'\n'); (arm/'summary.json').write_text(json.dumps({k:row.get(k) for k in ('case','bond_geometry','status','evaluation_requests','paid_requests','denials')},indent=2)+'\n'); report['arms'].append({k:row.get(k) for k in ('case','bond_geometry','status','evaluation_requests','paid_requests','denials')})
    (Path(out)/'summary.json').write_text(json.dumps(report,indent=2)+'\n'); print('\n'.join(f"{r['case']} {r['bond_geometry']} {r['status']} eval={r.get('evaluation_requests')} paid={r.get('paid_requests')}" for r in report['arms']))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True,type=Path); p.add_argument('--cases'); p.add_argument('--execute',action='store_true'); a=p.parse_args(); out=a.output.resolve(); names=list(systems()) if not a.cases else [x for x in a.cases.split(',') if x]
    if not a.execute:
        if out.exists(): raise FileExistsError(out)
        out.mkdir(parents=True); shutil.copy2(Path(__file__).resolve(),out/'runner.py'); shutil.copytree('pamssw',out/'source'/'pamssw',ignore=shutil.ignore_patterns('__pycache__','*.pyc')); (out/'PREPARED-NO-PES.md').write_text('Snapshot prepared; rerun with --execute.\n'); return
    env=os.environ.copy(); env['PYTHONPATH']=str(out/'source')+os.pathsep+str(Path.cwd().resolve()); env['PAM_NATIVE_LS_CHILD']='1'; env['PAM_NATIVE_LS_OUTPUT']=str(out); env['PAM_NATIVE_LS_CASES']=','.join(names)
    subprocess.run([sys.executable,str(out/'runner.py'),'--output',str(out),'--execute'],env=env,check=True)

if __name__=='__main__':
    if os.environ.get('PAM_NATIVE_LS_CHILD')=='1': child(Path(os.environ['PAM_NATIVE_LS_OUTPUT']),[x for x in os.environ['PAM_NATIVE_LS_CASES'].split(',') if x])
    else: main()
