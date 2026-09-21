"""Prepare a paired full-walker rotation comparison; execution requires allocation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / 'research/ga_ssw/evidence/c60-mh1-matched-cbd-20260919-r2'
LONG_RUN = ROOT / 'research/ga_ssw/evidence/c60-mh1-python-20260919'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    parent = json.loads((PARENT / 'plan.json').read_text())
    original = json.loads((LONG_RUN / 'plan.json').read_text())
    plan = dict(scope='Four development states, two rotation mechanisms, one full outer attempt with up to 12 Gaussians; not a global success-rate study.',
                parent=str(PARENT), backend=parent['backend'], runtime=parent['runtime'],
                config=original['config'], native_mc=original['native_mc'],
                inputs={}, arms=['baseline', 'recovered_cbd'], steps=1,
                search_cap_per_arm=2998, fresh_cap_per_arm=2,
                total_cap_per_arm=3000, wall_seconds_per_arm=210,
                recovered_rotation=dict(pre_rotmax=5, rotmax=15, pre_ftol=.2,
                                        ftol=.02, metric='euclidean', max_force_calls=40),
                parameter_source='Matched single-Gaussian protocol: native dr=.005, ftol=1/.1 rescaled to common dr=.001; original complete SSW config otherwise unchanged.',
                budget_status='planned_not_authorized_for_new_group_account_allocation')
    (out/'inputs').mkdir()
    for name, meta in parent['inputs'].items():
        src=Path(meta['prepared']); target=out/'inputs'/f'{name}.traj'
        if digest(src)!=meta['prepared_sha256']: raise ValueError('parent input changed')
        shutil.copy2(src,target)
        plan['inputs'][name]=dict(path=str(target.relative_to(out)), sha256=digest(target),
            seed=parent['direction']['seed_by_case'][name.split('-')[0]],
            source_energy_eV=meta['source_energy_eV'], source=str(src))
    shutil.copytree(ROOT/'pamssw',out/'source'/'pamssw',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(ROOT/'research/ga_ssw/run_public_broyden_ssw.py',out/'ledger_helpers.py')
    shutil.copy2(__file__,out/'runner.py')
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')


def execute(out):
    plan=json.loads((out/'plan.json').read_text())
    if (out/'summary.json').exists() or (out/'execution-started.json').exists():
        raise FileExistsError('preserve prior attempt; prepare a new run')
    (out/'execution-started.json').write_text(json.dumps(dict(job_id=os.getenv('SLURM_JOB_ID'), account=os.getenv('SLURM_JOB_ACCOUNT')))+'\n')
    sys.path[:0]=[str(out/'source'),str(out)]
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',plan['runtime']['CUBLAS_WORKSPACE_CONFIG'])
    import numpy as np
    import torch
    from ase.io import read
    from mace.calculators import MACECalculator
    from ledger_helpers import CountedSurface, dump
    from pamssw.standalone import ASESurface, NativeMCSettings, SSWConfig, RecoveredRotationSettings, run_ssw
    torch.set_num_threads(1);torch.manual_seed(0);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    backend=plan['backend']
    if digest(backend['model'])!=backend['model_sha256']:raise ValueError('model changed')
    calc=MACECalculator(model_paths=backend['model'],head=backend['head'],device=backend['device'],
                        default_dtype=backend['dtype'],enable_cueq=False,enable_oeq=False)
    config=SSWConfig(**plan['config']);mc=NativeMCSettings(plan['native_mc']['energy_tol_eV'],plan['native_mc']['maxtrap'])
    rows=[]
    for name,meta in plan['inputs'].items():
        path=out/meta['path']
        if digest(path)!=meta['sha256']:raise ValueError('frozen input changed')
        for arm in plan['arms']:
            folder=out/f'{name}-{arm}';folder.mkdir()
            atoms=read(path);calc.reset()
            surface=CountedSurface(calc,folder/'requests.jsonl',cap=plan['search_cap_per_arm'],wall=plan['wall_seconds_per_arm'])
            row=dict(state=name,arm=arm,status='started')
            fresh=None;checks=[]
            try:
                result=run_ssw(atoms,surface,steps=plan['steps'],config=config,
                    rng=np.random.default_rng(meta['seed']),mc=mc,
                    recovered_rotation=(RecoveredRotationSettings(**plan['recovered_rotation']) if arm=='recovered_cbd' else None),
                    checkpoint_path=folder/'checkpoint.pkl')
                dump(folder/'result.json',result)
                checks=[];fresh=ASESurface(calc)
                for i,minimum in enumerate(result.minima[:plan['fresh_cap_per_arm']]):
                    calc.reset()
                    try:
                        e,f=fresh.evaluate(minimum.atoms);fm=float(np.linalg.norm(f,axis=1).max())
                        checks.append(dict(index=i,energy_eV=e,fmax_eV_A=fm,qualified=bool(fm<=config.fmax),
                                           delta_energy_from_input=e-meta['source_energy_eV']))
                    except Exception as exc:checks.append(dict(index=i,qualified=False,error=repr(exc)))
                row.update(status=result.status,records=len(result.records),
                    gaussian_counts=[len(r.climb) for r in result.records],
                    record_statuses=[r.status for r in result.records],
                    search_calls=surface.requests,fresh_calls=fresh.requests,fresh_checks=checks)
            except Exception as exc:
                row.update(status='exception',error=repr(exc))
            row.update(search_calls=surface.requests, fresh_calls=0 if fresh is None else fresh.requests,
                       fresh_checks=checks, boundary=surface.boundary,denials=surface.denials)
            rows.append(row);dump(out/'summary.json',rows);print(row,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    g=p.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--execute',action='store_true')
    a=p.parse_args();(prepare if a.prepare else execute)(a.output.resolve())
