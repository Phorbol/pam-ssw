"""Paper-derived periodic inputs: direction qualification and paired cost pilot."""
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('ledger',HERE.parent/'periodic-rotation-priority-20260923/ledger.py')
ledger=importlib.util.module_from_spec(spec);spec.loader.exec_module(ledger)


def main():
    from ase import Atoms
    from ase.io import read,write
    from mace.calculators import MACECalculator
    import torch
    from pamssw.standalone.paper_reference import SSWConfig,run_ssw,load_ssw_checkpoint
    from pamssw.standalone.recovered_direction import RecoveredDirectionSettings
    from pamssw.standalone.recovered_rotation import RecoveredRotationSettings
    from pamssw.standalone.surface import ASESurface
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    out=HERE/'runs';out.mkdir(exist_ok=False)
    model='/home/gengjianrui/.cache/mace/mace-omat-0-small.model'
    kwargs=dict(model_paths=model,head='omat_pbe',device='cuda',default_dtype='float64',enable_cueq=False,enable_oeq=False)
    calc=MACECalculator(**kwargs);fresh_calc=MACECalculator(**kwargs)
    config=SSWConfig(width=.1,rotation_bias=100.,max_gaussians=25,temperature_K=150.,
        fmax=.03,bias_fmax=.1,relax_steps=400,fd_step=.001,rotation_hvp=100,rotation_tol=.02,
        direction_sampling='global',cluster_frame='translation_only',quench_optimizer='safe-lbfgs-total',
        lbfgs_memory=500,rotation_exit_policy='force_or_budget')
    direction=RecoveredDirectionSettings(50,.5,.5,5,15,.2,.02,'euclidean',40,
        startup_order='randomized',geometry='periodic_local')
    rotation=RecoveredRotationSettings(5,15,.2,.02,'euclidean',40)
    ledger.dump(out/'protocol.json',dict(config=config,direction=direction,rotation=rotation,
        seed=26092631,steps=3,search_cap_per_arm=6000,fresh_total_cap=24,
        live_total_cap=36000,replay_total_cap=18000,wall_per_arm=360,
        scope='paper-material functional qualification and cost pilot; not efficacy acceptance',model=model))
    rows=[];fresh_count=0
    for case in json.loads((HERE/'cases.json').read_text())['cases']:
        if case['source'].endswith('.json'):
            a=Atoms(**json.loads(Path(case['source']).read_text())['initial']['atoms'])
        else:a=read(case['source'])
        for method in ('global','local_memory'):
            folder=out/(case['case']+'-'+method);folder.mkdir()
            write(folder/'input.extxyz',a)
            tape=[];started=time.monotonic()
            class Counted(ASESurface):
                live=0;cursor=0;replay=False
                def evaluate(self,atoms):
                    if self.replay:
                        if self.cursor>=len(tape):raise RuntimeError('replay exceeds recorded stream')
                        p,e,f=tape[self.cursor]
                        np.testing.assert_array_equal(atoms.positions,p)
                        np.testing.assert_array_equal(atoms.cell.array,a.cell.array)
                        np.testing.assert_array_equal(atoms.pbc,a.pbc)
                        self.cursor+=1;self.requests+=1
                        return e,f.copy()
                    if self.live>=6000 or time.monotonic()-started>=360:
                        raise RuntimeError('frozen qualification budget exhausted')
                    self.live+=1
                    e,f=super().evaluate(atoms)
                    ledger.append(folder/'requests.jsonl',dict(atoms=atoms,energy=e,forces=f))
                    if method=='local_memory':tape.append((atoms.positions.copy(),e,f.copy()))
                    return e,f
            surface=Counted(calc)
            option=dict(recovered_direction=direction) if method=='local_memory' else dict(recovered_rotation=rotation)
            row=dict(case=case['case'],method=method,qualified=False)
            try:
                result=run_ssw(a,surface,steps=3,config=config,rng=np.random.default_rng(26092631),
                               checkpoint_path=folder/'continuous.pkl',**option)
                ledger.dump(folder/'result.json',result)
                row.update(status=result.status,landings=len(result.minima)-1,
                    stages=sum(len(r.climb) for r in result.records),best_energy=result.best.energy)
                if method=='local_memory':
                    from collections import Counter
                    row['routes']=dict(Counter(s.get('recovered_direction',{}).get('route','unreported')
                        for r in result.records for s in r.climb))
                assert result.status=='completed',result.status
                for minimum in result.minima:
                    b=minimum.atoms.copy();b.calc=fresh_calc
                    energy=b.get_potential_energy();forces=b.get_forces();fresh_count+=1
                    assert fresh_count<=24
                    ledger.append(folder/'fresh.jsonl',dict(atoms=b,energy=energy,forces=forces))
                    assert abs(energy-minimum.energy)<1e-6
                    assert np.max(np.linalg.norm(forces,axis=1))<=config.fmax
                    np.testing.assert_array_equal(b.cell.array,a.cell.array)
                if method=='local_memory':
                    assert all('recovered_direction' in s for r in result.records for s in r.climb)
                    row['history_used']=any(len(r.climb)>1 for r in result.records)
                    surface.replay=True
                    run_ssw(a,surface,steps=1,config=config,rng=np.random.default_rng(26092631),
                            checkpoint_path=folder/'split.pkl',**option)
                    cp=load_ssw_checkpoint(folder/'split.pkl')
                    run_ssw(a,surface,steps=2,config=config,rng=np.random.default_rng(999),checkpoint=cp,
                            checkpoint_path=folder/'split.pkl',**option)
                    assert surface.cursor==len(tape)
                    continuous=ledger._jsonable(load_ssw_checkpoint(folder/'continuous.pkl'))
                    resumed=ledger._jsonable(load_ssw_checkpoint(folder/'split.pkl'))
                    assert continuous==resumed,'full checkpoint state differs under identical oracle stream'
                    row['replay_state_equal']=True
                row['qualified']=True
            except Exception as error:
                row['error']=repr(error)
            row.update(live_requests=surface.live,replayed=surface.cursor)
            rows.append(row);ledger.dump(out/'summary.json',dict(rows=rows,fresh=fresh_count))
            print(json.dumps(row),flush=True)
    assert all(row['qualified'] for row in rows),'qualification failure; inspect preserved rows'

if __name__=='__main__':main()
