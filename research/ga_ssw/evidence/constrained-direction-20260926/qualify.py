"""Bounded real fixed-atom / point-restraint direction qualification."""
import argparse
import importlib.util
import json
import pickle
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('ledger',HERE.parent/'periodic-rotation-priority-20260923/ledger.py')
ledger=importlib.util.module_from_spec(spec);spec.loader.exec_module(ledger)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('case',choices=('cu-fixed','c4h6-point'))
    parser.add_argument('--output',required=True)
    parser.add_argument('--oracle-replay',action='store_true')
    parser.add_argument('--native-ls',action='store_true')
    args=parser.parse_args()
    from ase.cluster import Icosahedron
    from ase.calculators.emt import EMT
    from ase.constraints import FixAtoms,Hookean
    from ase.io import read,write
    from pamssw.standalone.constrained_reference import ConstrainedSSWConfig,run_constrained_ssw
    from pamssw.standalone.recovered_direction import RecoveredDirectionSettings
    from pamssw.standalone.ase_constraints import normalize_constraints
    from pamssw.standalone.surface import ASESurface
    out=HERE/args.output;out.mkdir(exist_ok=False)
    if args.case=='cu-fixed':
        atoms=Icosahedron('Cu',2)
        atoms.positions+=np.random.default_rng(250926).normal(0,.03,atoms.positions.shape)
        atoms.set_constraint(FixAtoms(indices=[0,1,2]))
        calc=EMT();fresh_calc=EMT();model='ASE EMT'
        cap,wall=4000,240
    else:
        import torch
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        from mace.calculators import MACECalculator
        atoms=read(HERE.parent/'c4h6-model-qualification-20260924-r2/butadiene/terminal.extxyz')
        # Explicit nonzero point term, not a confinement recommendation.
        center=atoms.positions[0]+np.array([.2,0,0])
        atoms.set_constraint(Hookean(0,center,k=1.,rt=.1))
        model='/home/gengjianrui/.cache/mace/mace-mh-1.model'
        kwargs=dict(model_paths=model,head='omol',device='cuda',default_dtype='float64',enable_cueq=False,enable_oeq=False)
        calc=MACECalculator(**kwargs);fresh_calc=MACECalculator(**kwargs)
        cap,wall=4000,1080
    if args.oracle_replay:
        cap=3462  # Previous molecular attempt spent538 of the original4000 bound.
    constraints=normalize_constraints(atoms)
    write(out/'input.extxyz',constraints.clean_atoms(atoms))
    cfg=ConstrainedSSWConfig(width=.6,rotation_bias=1.,max_gaussians=3,
        fmax=.03,gradient_tol=.1,relax_steps=1000,fd_step=.001,
        lbfgs_memory=500,rotation_exit_policy='force_or_budget')
    direction=RecoveredDirectionSettings(50,.5,.5,5,15,.2,.02,'euclidean',40,startup_order='randomized')
    from pamssw.standalone.ls_native_reference import NativeLSSettings
    ls=NativeLSSettings({(29,29):3.},{(29,29):2.8}) if args.native_ls else None
    ledger.dump(out/'protocol.json',dict(case=args.case,config=cfg,direction=direction,
        hookean=constraints.hookean_specs,fixed=constraints.fixed_indices,ls=ls,oracle_replay=args.oracle_replay,
        model=model,model_head='omol' if args.case=='c4h6-point' else None,
        seed=25092611,search_cap=cap,fresh_cap=3 if args.oracle_replay else 6,wall_seconds=wall,
        source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
    started=time.monotonic();calls=[0];fresh_calls=0
    tape=[]
    class Counted(ASESurface):
        replay=False
        cursor=0
        live_requests=0
        def evaluate(self,a):
            if calls[0]>=cap or time.monotonic()-started>=wall:raise RuntimeError('qualification budget exhausted')
            calls[0]+=1
            if self.replay:
                self.requests+=1
                position,energy,forces=tape[self.cursor]
                np.testing.assert_array_equal(a.positions,position)
                np.testing.assert_array_equal(a.numbers,atoms.numbers)
                np.testing.assert_array_equal(a.cell.array,atoms.cell.array)
                np.testing.assert_array_equal(a.pbc,atoms.pbc)
                self.cursor+=1
                return energy,forces.copy()
            energy,forces=super().evaluate(a)
            self.live_requests+=1
            if args.oracle_replay:tape.append((a.positions.copy(),energy,forces.copy()))
            return energy,forces
    surface=Counted(calc);rows=[]
    try:
        reference=None
        for split in (False,True):
            surface.replay=bool(args.oracle_replay and split)
            before=calls[0]
            r=run_constrained_ssw(atoms,surface,steps=1 if split else 2,config=cfg,
                rng=np.random.default_rng(25092611),recovered_direction=direction,ls=ls)
            if split:
                cp=pickle.loads(pickle.dumps(r.checkpoint))
                (out/'boundary.pkl').write_bytes(pickle.dumps(cp))
                r=run_constrained_ssw(atoms,surface,steps=1,config=cfg,
                    rng=np.random.default_rng(999),checkpoint=cp,recovered_direction=direction,ls=ls)
            ledger.dump(out/f'split{split}-result.json',r)
            assert r.status=='completed',r.status
            assert r.checkpoint.next_index==2 and r.checkpoint.schema_version==2
            assert all(event['status']=='valid_landing' for event in r.records[1:])
            state=ledger._jsonable(dict(initial=r.initial,current=r.current,best=r.best,
                minima=r.minima,records=r.records,rng=r.checkpoint.rng_state,
                direction=r.checkpoint.recovered_direction_state,ls=r.checkpoint.ls_state,requests=r.requests))
            if reference is None:reference=state
            else:assert state==reference,'continuous/resume mismatch'
            fixed=np.asarray(constraints.fixed_indices,dtype=int)
            for minimum in ([] if args.oracle_replay and split else r.minima):
                a=constraints.attach(minimum.atoms)
                np.testing.assert_array_equal(a.positions[fixed],atoms.positions[fixed])
                a.calc=fresh_calc
                energy=a.get_potential_energy();force=a.get_forces();fresh_calls+=1
                assert fresh_calls<=6
                assert abs(energy-minimum.energy)<1e-8
                assert np.linalg.norm(force,axis=1).max()<=cfg.fmax
            stages=[stage for event in r.records[1:] for stage in event['climb']]
            assert stages and all('recovered_direction' in stage for stage in stages)
            rows.append(dict(split=split,search=calls[0]-before,status=r.status,
                exact_state=True,stages=len(stages),landings=len(r.minima)-1,
                fresh=fresh_calls,has_continuation=any(stage['index']>0 for stage in stages)))
        assert rows[0]['search']==rows[1]['search']
        if args.oracle_replay:
            assert surface.cursor==len(tape)
            np.savez_compressed(out/'oracle-tape.npz',positions=np.array([t[0] for t in tape]),
                energies=np.array([t[1] for t in tape]),forces=np.array([t[2] for t in tape]))
        ledger.dump(out/'summary.json',dict(complete=True,rows=rows,search=calls[0],fresh=fresh_calls,
            live_search=surface.live_requests,replayed=surface.cursor))
        print(json.dumps(dict(complete=True,rows=rows,search=calls[0],fresh=fresh_calls)),flush=True)
    except Exception as error:
        ledger.dump(out/'failure.json',dict(error=repr(error),rows=rows,search=calls[0],fresh=fresh_calls))
        raise


if __name__=='__main__':main()
