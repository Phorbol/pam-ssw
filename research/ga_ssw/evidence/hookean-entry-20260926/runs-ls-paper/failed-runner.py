"""Real-calculator qualification of approved entry; no efficacy claims."""
import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("output", nargs="?", default="runs")
    parser.add_argument("--ls", choices=("none", "paper", "native"), default="none")
    args = parser.parse_args()
    from ase.calculators.emt import EMT
    from ase.constraints import Hookean
    from ase.io import read, write
    from pamssw.standalone import ASESurface, SSWConfig, run_ssw
    from pamssw.standalone.ase_constraints import normalize_constraints
    from pamssw.standalone.recovered_direction import RecoveredDirectionSettings
    import subprocess
    old = load('startup_qualification', HERE.parent / 'startup-order-20260926/qualify.py')
    ledger = load('ledger', HERE.parent / 'periodic-rotation-priority-20260923/ledger.py')
    out = HERE / args.output
    out.mkdir(exist_ok=False)
    atoms = read(HERE.parent / 'startup-order-20260926/runs/input.extxyz')
    threshold = .9 * float(np.linalg.norm(atoms.positions[1] - atoms.positions[0]))
    atoms.set_constraint(Hookean(0, 1, k=1., rt=threshold))
    constraints = normalize_constraints(atoms)
    write(out / 'input.extxyz', constraints.clean_atoms(atoms))  # specs saved in protocol
    cfg = SSWConfig(width=.6, rotation_bias=1., max_gaussians=6, temperature_K=300.,
        fmax=.03, bias_fmax=.1, relax_steps=1000, fd_step=.001, rotation_hvp=39,
        rotation_tol=.02, forward_force=.1, direction_sampling='global',
        rotation_solver='broyden-euclidean', cluster_frame='direction_only',
        quench_optimizer='safe-lbfgs-total', lbfgs_memory=500, rotation_exit_policy='force_or_budget')
    direction = RecoveredDirectionSettings(50,.5,.5,5,15,.2,.02,'euclidean',40,
                                           startup_order='randomized')
    # Existing interface-test settings, not fitted Cu search parameters.
    from pamssw.standalone.paper_reference import LSSettings
    from pamssw.standalone.ls_native_reference import NativeLSSettings
    ls = (NativeLSSettings({(29, 29): 3.}, {(29, 29): 2.8}, scale=.1)
          if args.ls == 'native' else
          LSSettings({(29, 29): 1.}, {(29, 29): 3.}, target_per_atom=.001)
          if args.ls == 'paper' else None)
    started = time.monotonic()
    calls = [0]
    class Counted(ASESurface):
        def evaluate(self, candidate):
            if calls[0] >= 6000 or time.monotonic()-started >= 270:
                raise RuntimeError('qualification budget exhausted')
            calls[0] += 1
            return super().evaluate(candidate)
    ledger.dump(out / 'protocol.json', dict(config=cfg, direction=direction,
        ls=ls, ls_mode=args.ls, specs=constraints.hookean_specs, seed=25092531, selector_seed=31,
        search_cap=6000, fresh_cap=4, wall_seconds=270,
        source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
    rows=[]
    try:
        for pool in (False, True):
            reference=None
            for pause in (False, True):
                before=calls[0]
                kw=dict(starter_selector=old.RestartSelector(), selector_rng=np.random.default_rng(31)) if pool else {}
                result=run_ssw(atoms,Counted(EMT()),steps=2,config=cfg,rng=np.random.default_rng(25092531),
                    ls=ls,recovered_direction=direction,progress_callback=lambda e: pause and e.next_index==1,**kw)
                if pause:
                    assert result.status=='paused'
                    cp=pickle.loads(pickle.dumps(result.checkpoint))
                    (out/f'pool{pool}-boundary.pkl').write_bytes(pickle.dumps(cp))
                    kw=dict(starter_selector=old.RestartSelector(),selector_rng=np.random.default_rng(999)) if pool else {}
                    result=run_ssw(atoms,Counted(EMT()),steps=2-cp.next_index,config=cfg,
                        rng=np.random.default_rng(999),checkpoint=cp,
                        progress_callback=lambda e:False,**kw)
                name=f'pool{pool}-pause{pause}'
                ledger.dump(out/f'{name}-result.json',result)
                assert result.status=='completed' and len(result.records)==2
                assert result.checkpoint.schema_version==6
                assert result.checkpoint.base_schema_version == (5 if pool else 4)
                assert result.checkpoint.hookean_specs==constraints.hookean_specs
                assert not result.best.constraints and len(atoms.constraints)==1
                state=ledger._jsonable(dict(records=result.records,minima=result.minima,current=result.current,
                    best=result.best,requests=result.evaluation_requests,rng=result.checkpoint.rng_state,
                    direction=result.checkpoint.recovered_direction_state,pool=result.checkpoint.pool_state,
                    restraints=result.checkpoint.hookean_specs,
                    ls_settings=result.checkpoint.ls, ls_frozen=result.checkpoint.frozen,
                    ls_response=result.checkpoint.response))
                if reference is None: reference=state
                assert state==reference
                restarted=any(r.starter_selection and r.starter_selection['restarted'] for r in result.records)
                if pool: assert restarted
                if ls is not None:
                    assert all(r.ls_preparation is not None for r in result.records)
                    assert all(r.energy_response is not None for r in result.records)
                    if args.ls == "native":
                        assert all(r.ls_update is not None for r in result.records)
                fresh=constraints.attach(result.best)
                fresh.calc=EMT()
                energy=fresh.get_potential_energy()
                forces=fresh.get_forces()
                fmax=float(np.linalg.norm(forces,axis=1).max())
                assert fmax <= cfg.fmax
                assert abs(energy-result.checkpoint.best.energy)<=1e-10
                rows.append(dict(name=name,search=calls[0]-before,fresh=1,exact_state=True,
                                 restarted=restarted,energy=energy,fmax=fmax))
                ledger.dump(out/'summary.json',dict(rows=rows,search=calls[0],complete=len(rows)==4))
        assert rows[0]['search']==rows[1]['search'] and rows[2]['search']==rows[3]['search']
        print(json.dumps(dict(rows=rows,search=calls[0],fresh=4,complete=True)),flush=True)
    except Exception as error:
        ledger.dump(out/'failure.json',dict(error=repr(error),search=calls[0],rows=rows))
        raise


if __name__=='__main__': main()
