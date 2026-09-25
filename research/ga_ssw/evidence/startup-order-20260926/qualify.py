"""Bounded real Cu13/EMT state qualification, not efficiency evidence."""
import importlib.util
import json
import pickle
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))


class RestartSelector:
    def __init__(self): self.calls = 0
    def checkpoint_contract(self): return {'identity': 'startup-order-test', 'version': 1}
    def export_state(self): return {'calls': self.calls}
    def restore_state(self, state): self.calls = state['calls']
    def __call__(self, snapshot, rng):
        self.calls += 1
        rng.random()
        return next((o.index for o in snapshot.observations if o.index != snapshot.current_index), None)


def main():
    from ase.calculators.emt import EMT
    from ase.cluster import Icosahedron
    from ase.io import write
    from pamssw.standalone import ASESurface, SSWConfig, run_ssw
    from pamssw.standalone.recovered_direction import RecoveredDirectionSettings
    from pamssw.standalone.paper_reference import load_ssw_checkpoint
    spec = importlib.util.spec_from_file_location('ledger', ROOT / 'research/ga_ssw/evidence/periodic-rotation-priority-20260923/ledger.py')
    ledger = importlib.util.module_from_spec(spec); spec.loader.exec_module(ledger)
    out = HERE / 'runs'; out.mkdir(exist_ok=False)
    atoms = Icosahedron('Cu', 2)
    atoms.positions += np.random.default_rng(25092530).normal(0, .04, (13, 3))
    write(out / 'input.extxyz', atoms)
    cfg = SSWConfig(width=.6, rotation_bias=1., max_gaussians=6, temperature_K=300.,
        fmax=.03, bias_fmax=.1, relax_steps=1000, fd_step=.001, rotation_hvp=39,
        rotation_tol=.02, forward_force=.1, direction_sampling='global',
        rotation_solver='broyden-euclidean', cluster_frame='direction_only',
        quench_optimizer='safe-lbfgs-total', lbfgs_memory=500, rotation_exit_policy='force_or_budget')
    direction = RecoveredDirectionSettings(50,.5,.5,5,15,.2,.02,'euclidean',40,startup_order='randomized')
    started=time.monotonic(); total=[0]; rows=[]
    class Counted(ASESurface):
        def evaluate(self, candidate):
            if total[0]>=40000 or time.monotonic()-started>=540:
                raise RuntimeError('qualification budget exhausted')
            total[0]+=1
            return super().evaluate(candidate)
    ledger.dump(out/'protocol.json', dict(config=cfg,direction=direction,seed=25092531,
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    # Read an actual pre-feature schema4 archive; no model or PES request needed.
    oldpath=ROOT/'research/ga_ssw/evidence/c60-local-defect-20260925/direction-probe/runs/native_ls-1101/checkpoint.pkl'
    old=load_ssw_checkpoint(oldpath)
    assert old.schema_version==4 and 'startup_order' not in vars(old.recovered_direction_state.settings)
    original_rng=old.rng_state
    old_surface=Counted(EMT())
    replay=run_ssw(old.current,old_surface,steps=0,config=old.config,ls=old.ls,mc=old.mc_settings,
        rng=np.random.default_rng(998),checkpoint=old,progress_callback=lambda event: False,
        height_policy=old.height_policy,gaussian_policy=old.gaussian_policy,
        height_update_budget=old.height_update_budget,reconnect_distance=old.reconnect_distance)
    assert old_surface.requests==0 and replay.checkpoint.rng_state==original_rng
    ledger.dump(out/'old-schema4.json',dict(path=str(oldpath),schema=4,default='legacy',zero_pes=True))
    for pool in (False,True):
        baseline=None
        for pause in (None,0,1):
            selector=RestartSelector() if pool else None
            kw=dict(starter_selector=selector,selector_rng=np.random.default_rng(31)) if pool else {}
            surface=Counted(EMT()); before=total[0]
            result=run_ssw(atoms,surface,steps=2,config=cfg,rng=np.random.default_rng(25092531),
                recovered_direction=direction,progress_callback=lambda e:e.next_index==pause,**kw)
            if pause is not None:
                assert result.status=='paused'
                cp=pickle.loads(pickle.dumps(result.checkpoint))
                kw=dict(starter_selector=RestartSelector(),selector_rng=np.random.default_rng(999)) if pool else {}
                result=run_ssw(atoms,Counted(EMT()),steps=2-cp.next_index,config=cfg,
                    rng=np.random.default_rng(999),checkpoint=cp,progress_callback=lambda e:False,**kw)
            name=f'pool{int(pool)}-pause{pause}'
            ledger.dump(out/f'{name}-result.json',result)
            assert result.status=='completed' and len(result.records)==2, (name,result.status)
            state=ledger._jsonable(dict(records=result.records,minima=result.minima,current=result.current,
                best=result.best,requests=result.evaluation_requests,rng=result.checkpoint.rng_state,
                direction=result.checkpoint.recovered_direction_state,pool=result.checkpoint.pool_state))
            if baseline is None: baseline=state
            assert state==baseline, name
            if pool: assert any(r.starter_selection['restarted'] for r in result.records)
            fresh=ASESurface(EMT()); energy,forces=fresh.evaluate(result.best)
            assert np.linalg.norm(forces,axis=1).max()<=cfg.fmax
            rows.append(dict(name=name,exact_state_match=True,requests=total[0]-before,
                fresh_requests=1,energy_eV=energy,fmax_eV_A=float(np.linalg.norm(forces,axis=1).max())))
            ledger.dump(out/'summary.json',dict(rows=rows,search_requests=total[0],complete=len(rows)==6))
            print(json.dumps(rows[-1]),flush=True)


if __name__=='__main__': main()
