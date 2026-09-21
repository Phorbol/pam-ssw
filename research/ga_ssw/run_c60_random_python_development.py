"""Prepare/execute the bounded random-C60 Python development comparison."""
import argparse, hashlib, json, shutil, subprocess, sys, time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL = Path('/home/gengjianrui/.cache/mace/mace-omat-0-small.model')
INPUTS = ROOT / 'research/ga_ssw/evidence/c60-random-inputs-development-20260917'
OUTNAME = 'c60-random-python-development-20260917'
sys.path.insert(0, str(ROOT))

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def enc(x):
    from ase import Atoms
    if isinstance(x, Atoms): return dict(numbers=x.numbers.tolist(), positions=x.positions.tolist(), cell=x.cell.array.tolist(), pbc=x.pbc.tolist())
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)): return x.item()
    if isinstance(x, Path): return str(x)
    if isinstance(x, dict): return {str(k): enc(v) for k,v in x.items()}
    if isinstance(x, (list, tuple)): return [enc(v) for v in x]
    if hasattr(x, '__dict__'): return enc(vars(x))
    return x
def dump(p, x): Path(p).write_text(json.dumps(enc(x), indent=2, allow_nan=False) + '\n')

def make_plan():
    from pamssw.standalone.paper_reference import SSWConfig
    from pamssw.standalone.recovered_direction import RecoveredDirectionSettings
    base = SSWConfig(width=.6, rotation_bias=1., max_gaussians=12, temperature_K=150.,
        fmax=.03, relax_steps=1000, fd_step=.001, rotation_hvp=39,
        rotation_tol=.02, direction_sampling='global', rotation_solver='broyden-euclidean',
        cluster_frame='direction_only', quench_optimizer='safe-lbfgs-total',
        lbfgs_memory=500, bias_fmax=.1, rotation_exit_policy='force_or_budget')
    recovered = RecoveredDirectionSettings(ratio_local=50, local_probability=.5,
        group_threshold=.5, pre_rotmax=2, rotmax=8, pre_ftol=.01, ftol=.01,
        metric='euclidean', max_force_calls=40)
    return dict(status='prepared_not_executed', purpose='bounded Python development comparison; no ranking',
        cases=['seed17093','seed17094'], arms=['paper','recovered'], seeds=[17093,17094], steps=100,
        per_arm_request_cap=6000, per_arm_wall_seconds=600, fresh_request_cap=101,
        backend=dict(name='MACE-OMAT-0-small', kind='mace', model=str(MODEL), model_sha256=sha(MODEL), device='cuda', dtype='float64'),
        config=asdict(base), recovered=asdict(recovered),
        excluded=['LS','reconnect_distance','gaussian_policy','height_policy'],
        parameter_basis='recovered-ls-materials-continuation config; width=.6 and NG12 from C60 SI; relax_steps1000 and history500 explicit candidates; no tuning',
        qualification='fresh MACE evaluation of at most 101 returned minima per run; fmax uses max atom-force norm; retain failures and truncation',
        scientific_scope='development wiring/cost evidence only; no winner or success-rate inference',
        source_script=str(Path(__file__).resolve()))

def prepare(out):
    if out.exists(): raise FileExistsError(out)
    if not MODEL.is_file(): raise FileNotFoundError(MODEL)
    files=[INPUTS/f'seed{i}.extxyz' for i in (17093,17094)]
    if not all(p.is_file() for p in files): raise FileNotFoundError('shared random input files are not present')
    out.mkdir(parents=True)
    shutil.copytree(ROOT/'pamssw', out/'source/pamssw', ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(ROOT/'research/ga_ssw/run_public_broyden_ssw.py', out/'ledger_helpers.py')
    shutil.copy2(__file__, out/'runner.py')
    (out/'inputs').mkdir()
    for p in files: shutil.copy2(p, out/'inputs'/p.name)
    plan=make_plan(); plan['input_sources']={p.stem:{'path':str(p),'sha256':sha(p)} for p in files}
    dump(out/'plan.json',plan)
    dump(out/'source-manifest.json', {'sha256':{str(p.relative_to(out/'source')):sha(p) for p in sorted((out/'source').rglob('*.py'))}})
    job=f'''#!/bin/bash\n#SBATCH --partition=4V100\n#SBATCH --qos=rush-1o2gpu\n#SBATCH --nodes=1\n#SBATCH --ntasks=1\n#SBATCH --gres=gpu:1\n#SBATCH --time=00:45:00\n#SBATCH --job-name=c60-py-dev\n#SBATCH --output={out}/slurm-%j.out\nset -euo pipefail\nexport OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONNOUSERSITE=1\ncd {out}\nPYTHONPATH={out}/source /home/gengjianrui/.conda/envs/mace_env/bin/python runner.py --execute --output {out}\n'''
    (out/'job.sh').write_text(job)
    dump(out/'prepared.json', {'plan_sha256':sha(out/'plan.json'),'source_manifest_sha256':sha(out/'source-manifest.json'),'job':str(out/'job.sh')})

def execute(out):
    def calculator(backend, model):
        from mace.calculators import MACECalculator
        return MACECalculator(model_paths=str(model), device='cuda', default_dtype='float64', enable_cueq=False, enable_oeq=False)
    from pamssw.standalone.paper_reference import SSWConfig, run_ssw
    from pamssw.standalone.recovered_direction import RecoveredDirectionSettings
    from pamssw.standalone.surface import ASESurface
    from ledger_helpers import CountedSurface
    plan=json.loads((out/'plan.json').read_text()); sys.path.insert(0,str(out/'source'))
    cfg=SSWConfig(**plan['config']); rec=RecoveredDirectionSettings(**plan['recovered']); rows=[]
    for case,seed in zip(plan['cases'],plan['seeds']):
        from ase.io import read
        initial=read(out/'inputs'/f'{case}.extxyz')
        for arm in plan['arms']:
            folder=out/f'{case}-{arm}'; folder.mkdir(); ledger=folder/'requests.jsonl'; ledger.touch()
            surf=CountedSurface(calculator('mace',MODEL),ledger,cap=6000,wall=600); validation=ASESurface(calculator('mace',MODEL)); started=time.monotonic()
            row={'case':case,'seed':seed,'arm':arm,'execution':'started','numerical':'unassessed'}; checks=[]
            try:
                result=run_ssw(initial.copy(),surf,steps=100,config=cfg,rng=np.random.default_rng(seed),recovered_direction=(rec if arm=='recovered' else None))
                dump(folder/'result.json',result)
                for i,m in enumerate(result.minima[:101]):
                    try:
                        e,f=validation.evaluate(m.atoms); fm=float(np.linalg.norm(f,axis=1).max()); checks.append({'index':i,'energy':e,'energy_error':e-m.energy,'fmax':fm,'qualified':fm<=cfg.fmax,'composition_match':bool(np.array_equal(m.atoms.numbers,initial.numbers)),'fixed_cell':bool(np.array_equal(m.atoms.cell.array,initial.cell.array))})
                    except Exception as exc: checks.append({'index':i,'qualified':False,'error':repr(exc)})
                dump(folder/'qualification.json',checks); row.update(execution=result.status,minima=len(result.minima),outer_statuses=dict(Counter(r.status for r in result.records)))
            except Exception as exc: row.update(execution='exception',error=repr(exc))
            row.update(search_requests=surf.requests,fresh_requests=validation.requests,denials=surf.denials,boundary=surf.boundary,checks=checks,elapsed=time.monotonic()-started)
            dump(folder/'summary.json',row); rows.append(row); dump(out/'summary.json',rows); print(case,arm,row['execution'],surf.requests,flush=True)

def main():
    p=argparse.ArgumentParser(); g=p.add_mutually_exclusive_group(required=True); g.add_argument('--prepare',action='store_true'); g.add_argument('--execute',action='store_true'); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); out=a.output.resolve(); prepare(out) if a.prepare else execute(out)
if __name__=='__main__': main()
