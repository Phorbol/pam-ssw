"""Bounded true quenches of frozen saved biased endpoints; explicit execution only."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from ase.io import read, write

import run_lj_pilot as base
from analyze_lj_pilot import geometry
from analyze_lj38_stages import connectivity
from pamssw.standalone import ASESurface
from pamssw.standalone.paper_reference import quench
from research.ga_ssw.full_pair_lj import FullPairLJ

HERE = Path(__file__).resolve().parent
SOURCE = HERE/'stage-probe-repaired-runs'
OUT = HERE/'early-quench-runs'
INPUTS = [(seed, outer, stage) for seed, outer, last in ((25092501,0,7),(25092502,2,8))
          for stage in (0,3,6,last)]


def input_path(seed, outer, stage):
    return SOURCE/f'lj38-paper-seed{seed}'/'biased-endpoints'/f'outer-{outer:02d}-gaussian-{stage:02d}.extxyz'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    assert not OUT.exists(), 'output already exists'
    assert json.loads((SOURCE/'analysis.json').read_text())['all_prefixes_qualified']
    for seed, outer, stage in INPUTS:
        atoms = read(input_path(seed,outer,stage))
        assert len(atoms)==38 and not atoms.pbc.any() and np.isfinite(atoms.positions).all()
    print('Eight saved inputs qualified, zero PES preflight', flush=True)
    if not args.execute:
        return
    OUT.mkdir()
    shutil.copy2(__file__,OUT/'runner.py')
    shutil.copy2(HERE/'early-quench-plan.md',OUT/'plan.md')
    ledger=base.load_ledger()
    config,_=base.make_settings()
    ledger.dump(OUT/'execution.json',dict(head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        config=config,inputs=INPUTS,search_cap=12000,fresh_cap=8,wall_seconds=240))
    deadline=time.monotonic()+240
    rows=[]
    for seed,outer,stage in INPUTS:
        row=dict(seed=seed,outer=outer,gaussian=stage,search_requests=0,fresh_requests=0)
        if time.monotonic()>=deadline:
            row['status']='not_run_wall_cap';rows.append(row);continue
        name=f'seed{seed}-outer{outer}-g{stage}'
        folder=OUT/name;folder.mkdir()
        atoms=read(input_path(seed,outer,stage));write(folder/'input.extxyz',atoms)
        surface=base.BoundedSurface(FullPairLJ(epsilon=1.,sigma=2.7),arm_cap=1500,
            arm_deadline=deadline,total_deadline=deadline)
        try:
            result=quench(atoms,surface,fmax=config.fmax,steps=config.relax_steps,
                optimizer=config.quench_optimizer,lbfgs_memory=config.lbfgs_memory)
            write(folder/'final.extxyz',result.atoms)
            start=read(SOURCE/f'lj38-paper-seed{seed}'/f'outer-{outer:02d}-start.extxyz')
            row.update(status='converged' if result.converged else 'quench_failed',energy=result.energy,
                max_force=result.max_force,geometry_to_start=geometry(result.atoms,start),
                connectivity={str(scale):connectivity(result.atoms,scale*2.7) for scale in (1.3,1.5)})
            if time.monotonic()<deadline:
                fresh=ASESurface(FullPairLJ(epsilon=1.,sigma=2.7))
                energy,forces=fresh.evaluate(result.atoms)
                fmax=float(np.linalg.norm(forces,axis=1).max())
                row.update(fresh_requests=fresh.requests,fresh_energy=energy,fresh_fmax=fmax,
                           fresh_force_qualified=fmax<=config.fmax)
        except Exception as error:
            row.update(status='exception',error=f'{type(error).__name__}: {error}')
        row.update(search_requests=surface.requests,boundary=surface.boundary)
        rows.append(row);ledger.dump(folder/'summary.json',row)
        ledger.dump(OUT/'summary.json',dict(rows=rows,status='running'))
    ledger.dump(OUT/'summary.json',dict(rows=rows,status='complete_or_censored',
        search_requests=sum(r['search_requests'] for r in rows),fresh_requests=sum(r['fresh_requests'] for r in rows)))


if __name__=='__main__':main()
